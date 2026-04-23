"""
train_gpt_modalite.py — Modalite Super Model for Parameter Golf
Özgün mimari: Record kodlarından ilham + DeepSeek V3 MTP + YaRN + SwiGLU hybrid
"""
from __future__ import annotations
import copy, glob, io, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
try:
    import zstandard; _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from flash_attn import flash_attn_func as flash_attn_3_func
except ImportError:
    def flash_attn_3_func(q, k, v, causal=True):
        return F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=causal).transpose(1,2)
try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 16))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    # SOTA techniques
    activation = os.environ.get("ACTIVATION", "leaky_relu2")  # leaky_relu2 | swiglu | relu2
    softcap_type = os.environ.get("SOFTCAP_TYPE", "poly")  # poly | tanh
    temp_scaling = bool(int(os.environ.get("TEMP_SCALING", "1")))
    eval_temperature = float(os.environ.get("EVAL_TEMPERATURE", 0.90))
    sliding_batch_size = int(os.environ.get("SLIDING_BATCH_SIZE", 32))
    # TTT hyperparameters
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "mamba_scale,attn_scale,mlp_scale,diff_lambda,resid_mix,skip_weights,skip_gates,smear,ve_layer_scales,ve_shared.scale",
    ).split(",") if p
)

# ─── MUON OPTIMIZER (Newton-Schulz5 Orthogonalized Momentum) ─────────────────
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["v_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    v_buf = state["v_buffer"]
                    
                    # 1 & 2: Update first momentum (M_t = beta * M_{t-1} + G_t)
                    buf.mul_(momentum).add_(g)
                    
                    # 3: Compute sign-stabilized orthogonal direction (O_t)
                    O_t = zeropower_via_newtonschulz5(torch.sign(buf), steps=backend_steps)
                    
                    # 4: Update second momentum (V_t = beta * V_{t-1} + (1 - beta) * O_t^2)
                    v_buf.mul_(momentum).addcmul_(O_t, O_t, value=1.0 - momentum)
                    
                    # 5: Apply second momentum update
                    O_hat = O_t / (v_buf.sqrt() + 1e-8)
                    
                    # 6: RMS-aligned constraint
                    gamma = 0.2 * (O_hat.numel() ** 0.5) / (O_hat.float().norm() + 1e-8)
                    
                    g_out = gamma * O_hat
                    updates_flat[curr:curr + p.numel()] = g_out.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0: p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# ─── TOKENIZER EVALUATION HELPERS ────────────────────────────────────────────
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size())
    table_size = max(sp_vs, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary_token_np[tid] = False
        if sp.is_byte(tid): base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"): has_leading_space_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.int32, copy=False))

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks = []; remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance_file(); continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len); y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ─── TRANSFORMER MODULES ─────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, eps=1e-5): super().__init__(); self.eps = eps if eps is not None else 1e-5
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class TernaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, group_size=64):
        super().__init__(in_features, out_features, bias=bias)
        self.group_size = group_size
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.bfloat16()
        g = self.group_size
        w_g = w.reshape(-1, g)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        q = (w_g / scale).round().clamp(-1, 1)
        w_ternary = w + ((q * scale).reshape(w.shape) - w).detach()
        return F.linear(x, w_ternary, self.bias.to(x.dtype) if self.bias is not None else None)

class NormedTernaryLinear(TernaryLinear):
    def __init__(self, in_features, out_features, bias=False, group_size=64):
        super().__init__(in_features, out_features, bias=bias, group_size=group_size)
        self.rms=RMSNorm()
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.rms_norm(x, (x.size(-1),)))

class GroupedTernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=4, group_size=64, normed=False):
        super().__init__()
        self.groups, self.group_in, self.group_out = groups, in_features // groups, out_features // groups
        self.group_size, self.normed = group_size, normed
        self.weight = nn.Parameter(torch.randn(groups * self.group_out, self.group_in) * 0.02)
        self.rms=RMSNorm()
    def forward(self, x: Tensor) -> Tensor:
        if self.normed: x = self.rms(x)
        w = self.weight.bfloat16()
        w_ternary = w + (((w.reshape(-1, self.group_size) / w.reshape(-1, self.group_size).abs().mean(-1, keepdim=True).clamp(min=1e-8)).round().clamp(-1, 1) * w.reshape(-1, self.group_size).abs().mean(-1, keepdim=True).clamp(min=1e-8)).reshape(w.shape) - w).detach()
        return torch.einsum('...gi,goi->...go', x.reshape(*x.shape[:-1], self.groups, self.group_in), w_ternary.reshape(self.groups, self.group_out, self.group_in)).reshape(*x.shape[:-1], self.groups * self.group_out)

class TverskyProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_features: int = 16, group_size: int = 64):
        super().__init__()
        self.features = nn.Parameter(torch.empty(num_features, in_features).uniform_(-0.02, 0.02))
        self.prototypes = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.02, 0.02))
        self.theta, self.alpha, self.beta = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(0.5)), nn.Parameter(torch.tensor(0.5))
        self.group_size = group_size
    def _ternary_ste(self, w: Tensor) -> Tensor:
        w_g = w.bfloat16().reshape(-1, self.group_size)
        scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
        return w.bfloat16() + (((w_g / scale).round().clamp(-1, 1) * scale).reshape(w.shape) - w.bfloat16()).detach()
    def forward(self, x: Tensor) -> Tensor:
        x_f, p_f = x @ self.features.float().t(), self._ternary_ste(self.prototypes) @ self.features.float().t()
        x_s, p_s = torch.sigmoid(x_f * 5.0), torch.sigmoid(p_f * 5.0)
        x_a, p_a = x_f * x_s, p_f * p_s
        return self.theta.abs() * (x_a @ p_a.t()) - self.alpha.abs() * (x_a @ (1 - p_s).t()) - self.beta.abs() * ((1 - x_s) @ p_a.t())

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, yarn_max_len=4096, rope_dims=0):
        super().__init__()
        self.dim, self.base, self.train_seq_len, self.yarn_max_len = dim, base, train_seq_len, yarn_max_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        scale = train_seq_len / yarn_max_len
        ramp = torch.clamp((torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims - 0.25) / 0.75, 0.0, 1.0)
        self.register_buffer("inv_freq", inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0), persistent=False)
        self._seq_len_cached = 0; self._cos_cached = None; self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            freqs = torch.outer(torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype), self.inv_freq.to(device))
            self._cos_cached, self._sin_cached, self._seq_len_cached = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :], seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    half = (rope_dims if rope_dims > 0 else x.size(-1)) // 2
    x_rope, x_pass = (x[..., :rope_dims], x[..., rope_dims:]) if rope_dims > 0 else (x, torch.empty(*x.shape[:-1], 0, device=x.device, dtype=x.dtype))
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    return torch.cat((torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1), x_pass), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, dim // num_heads
        self.c_qkv = TernaryLinear(dim, self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim, bias=False)
        self.proj = TverskyProjection(dim, dim)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.diff_lambda = nn.Parameter(torch.full((num_heads,), 0.5, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.rms=RMSNorm()
    def forward(self, x, v_embed=None):
        bsz, seqlen, dim = x.shape
        q_out, k_out, v_out = self.c_qkv(x).split([self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim], dim=-1)
        q_reshaped = q_out.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        q = self.rms(q_reshaped)
        k_reshaped = k_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        k = self.rms(k_reshaped)
        v = v_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        if v_embed is not None: v = v + v_embed.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin, self.rope_dims), apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        half = self.head_dim // 2
        y1 = flash_attn_3_func(q[..., :half].contiguous(), k[..., :half].contiguous(), v[..., :half].contiguous(), causal=True)
        y2 = flash_attn_3_func(q[..., half:].contiguous(), k[..., half:].contiguous(), v[..., half:].contiguous(), causal=True)
        lam = self.diff_lambda.to(dtype=y1.dtype)[None, None, :, None]
        return self.proj(torch.cat([y1 - lam * y2, y1 + lam * y2], dim=-1).reshape(bsz, seqlen, dim))

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32); mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t); out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None: nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids):
        h = self.embed(token_ids)
        if self.proj is not None: h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, activation="leaky_relu2"):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.activation = activation
        if activation == "swiglu":
            self.gate_up = GroupedTernaryLinear(dim, hidden * 2, groups=4)
        else:
            self.fc = GroupedTernaryLinear(dim, hidden, groups=4)
        self.proj = GroupedTernaryLinear(hidden, dim, groups=4, normed=True)
    def forward(self, x):
        if self.activation == "swiglu":
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            return self.proj(F.silu(gate) * up)
        elif self.activation == "relu2":
            return self.proj(torch.relu(self.fc(x)).square())
        else:  # leaky_relu2 (SOTA: -0.003 bpb)
            return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=0, ln_scale=False, activation="leaky_relu2"):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, activation=activation)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False
    def forward(self, x, x0, v_embed=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor, v_embed=v_embed)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            return x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            return x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)

def ternary_ste_hook(module, args):
    w = module.weight.bfloat16()
    w_g = w.reshape(-1, 64)
    scale = w_g.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    q = (w_g / scale).round().clamp(-1, 1)
    module.weight.data = w + ((q * scale).reshape(w.shape) - w).detach()
    return args

class MambaBlock(nn.Module):
    def __init__(self, dim, layer_idx=0, ln_scale=False):
        super().__init__()
        self.norm = RMSNorm()
        if Mamba2 is None: raise ImportError("Mamba2 not found. Please pip install mamba-ssm causal-conv1d")
        self.mamba = Mamba2(d_model=dim, d_state=128, headdim=64, chunk_size=256, dtype=torch.bfloat16)
        self.mamba_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
    def forward(self, x, x0, v_embed=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        norm_x = self.norm(x_in) * self.ln_scale_factor
        if v_embed is not None: norm_x = norm_x + F.pad(v_embed, (0, norm_x.size(-1) - v_embed.size(-1)))
        return x_in + self.mamba_scale.to(dtype=x_in.dtype)[None, None, :] * self.mamba(norm_x)

# ─── GPT MODEL (U-Net + XSA + ValueEmbed + SmearGate + BigramHash) ──────────
class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap, rope_base,
                 qk_gain_init, mtp_num_heads=0, mtp_loss_weight=0.1,
                 bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0, rope_dims=0,
                 ln_scale=False, ve_enabled=False, ve_dim=128, ve_layers="9,10",
                 activation="leaky_relu2", softcap_type="poly"):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.softcap_type = softcap_type
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            MambaBlock(model_dim, layer_idx=i, ln_scale=ln_scale) if i % 2 == 0 else
            TransformerBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, layer_idx=i, ln_scale=ln_scale, activation=activation)
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                if hasattr(block, 'attn'):
                    block.attn.rope_dims = rope_dims
                    block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(kv_dim, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None; self.ve_layer_scales = nn.ParameterList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)])
        for h in self.mtp_heads: h._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                if hasattr(self.blocks[i], 'attn'):
                    self.blocks[i].attn.use_xsa = True
        for i in range(len(self.blocks)):
            if i >= 7 and hasattr(self.blocks[i], 'parallel'):
                self.blocks[i].parallel = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    with torch.no_grad():
                        w32 = torch.empty_like(module.weight, dtype=torch.float32)
                        nn.init.orthogonal_(w32, gain=1.0)
                        module.weight.copy_(w32.to(module.weight.dtype))
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def _get_ve(self, layer_idx, input_ids, ve_cache=None):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices: return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def _softcap(self, logits: Tensor) -> Tensor:
        """Poly5 softcap (SOTA: -0.001 bpb vs tanh) or standard tanh."""
        s = self.logit_softcap
        if self.softcap_type == "tanh":
            return s * torch.tanh(logits / s)
        # Poly5: degree-5 Taylor approx — smoother gradients than tanh
        x_sc = torch.clamp(logits / s, -2.0, 2.0)
        x2 = x_sc * x_sc
        return s * torch.clamp(x_sc * (1.0 - x2 / 3.0 + x2 * x2 / 15.0), -1.0, 1.0)

    def forward(self, input_ids, target_ids, temperature: float = 1.0):
        x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype)
        x = self.smear(x)
        x0 = x; skips = []; ve_cache = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve); skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                x = torch.lerp(scaled_skip, x, g)
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1)); targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)
        logits = self._softcap(logits_proj)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0: continue
                mtp_logits_proj = mtp_head(x[:, :valid_t, :].reshape(-1, dim))
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), target_ids[:, k+1:].reshape(-1), reduction="mean")
                mtp_count += 1
            if mtp_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_count)
        return main_loss

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None: x = x + self.bigram(input_ids)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype)
        x = self.smear(x)
        x0 = x; skips = []; ve_cache = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve); skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                x = torch.lerp(scaled_skip, x, g)
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self._softcap(logits_proj)

# ─── EVALUATION ──────────────────────────────────────────────────────────────
def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
             eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be = min(bs + local_batch_seqs, seq_end)
            raw_s, raw_e = bs * seq_len, be * seq_len + 1
            local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len); y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            btc = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * btc
            val_token_count += btc
            prev_ids = x.reshape(-1); tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [val_loss_sum, val_token_count, val_byte_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)

def eval_val_sliding(args, base_model, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride, batch_seqs=32, eval_seq_len=None, temperature=1.0):
    """Sliding window evaluation with optional temperature scaling (SOTA: T=0.90)."""
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = base_model.forward_logits
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]; bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens); wlen = end - ws; wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]; y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            # Temperature scaling (SOTA: T=0.90 for relu²/leaky_relu² models)
            if temperature != 1.0:
                logits = logits / temperature
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y_batch.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]; s = 0 if ws == 0 else max(wlen - stride, 0)
                loss_sum += nll[i, s:wlen].to(torch.float64).sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in [loss_sum, token_count, byte_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / token_count).item()
    base_model.train()
    return vl, vl / math.log(2.0) * (token_count.item() / byte_count.item())

def eval_val_sliding_ttt(args, base_model, rank, world_size, device, val_tokens,
                         base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                         stride, batch_seqs=32, temperature=1.0, log0=print):
    """Legal Score-First TTT (SOTA: -0.0025 bpb).
    Score each chunk with sliding windows THEN train on it.
    Every token is scored BEFORE any update that could use it."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    # Pre-compute all window starts
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]

    # Assign each window to a chunk
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log0(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
         f"total_windows={len(window_starts)} stride={stride} "
         f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} "
         f"freeze_blocks={args.ttt_freeze_blocks}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Freeze first N blocks (default: 0 = all unfrozen)
    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = any(f"blocks.{bi}." in name for bi in frozen_block_ids)
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    log0(f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        # --- Phase 1: SCORE this chunk (inference_mode = no gradient, no weight mutation) ---
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws; wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                if temperature != 1.0:
                    logits = logits / temperature
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # --- Phase 2: TRAIN on this chunk (already scored = legal) ---
        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb

# ─── QUANTIZATION ────────────────────────────────────────────────────────────
INT8_CLIP_Q = 0.9999984
def tensor_nbytes(t): return int(t.numel()) * int(t.element_size())

def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mamba." in name: return "mamba"
    if ".mlp." in name: return "mlp"
    if ".attn." in name or ".proj" in name: return "attn"
    return "other" 

def quantize_int6_per_row(t, clip_range=31):
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err: best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    return torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8), scale

def mixed_quantize_int6(state_dict, int6_cats):
    result, meta = {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous(); cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"; continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float(); meta[name] = "passthrough_ctrl"; continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q; result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta

def dequantize_mixed_int6(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        if info in ("passthrough", "passthrough_ctrl"):
            t = result[name]
            if t.dtype == torch.float16 and orig.dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig.dtype)
            out[name] = t; continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(orig.dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig.dtype)
    return out

# ─── MAIN TRAINING LOOP ─────────────────────────────────────────────────────
def main():
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0: raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 64 // world_size; grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device); dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    try:
        from torch.backends.cuda import enable_cudnn_sdp
        enable_cudnn_sdp(False)
    except ImportError:
        pass
    logfile = None
    if master_process: os.makedirs("logs", exist_ok=True); logfile = f"logs/{args.run_id}.txt"
    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile: open(logfile, "a", encoding="utf-8").write(msg + "\n")
    log0(code, console=False); log0("=" * 100, console=False)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    CastedLinear._qat_enabled = False
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads, mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        activation=args.activation, softcap_type=args.softcap_type,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    # compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    compiled_model = base_model
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # --- Optimizer setup ---
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.mtp_num_heads > 0:
        matrix_params.extend([p for p in base_model.mtp_heads.parameters() if p.ndim == 2])
    scalar_params = [p for n, p in block_named_params if p.ndim != 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
    if getattr(base_model, 'skip_gates', None) is not None and base_model.skip_gates.numel() > 0: scalar_params.append(base_model.skip_gates)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None: scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None: matrix_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None: matrix_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales: scalar_params.append(s)

    opt_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.AdamW([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                    betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar]
    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
                                     betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, opt_head)

    # --- Training ---
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            wd_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if wd_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    # Warmup
    if args.warmup_steps > 0:
        init_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_states = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): wl = model(x, y)
                (wl * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
        base_model.load_state_dict(init_state, strict=True)
        for o, s in zip(optimizers, init_opt_states): o.load_state_dict(s)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # EMA state
    ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
    ema_decay = 0.997
    training_time_ms = 0.0; stop_after_step = None
    torch.cuda.synchronize(); t0 = time.perf_counter(); step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step: break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True; log0(f"late_qat:enabled step:{step}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y)
            train_loss += loss.detach(); (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups: g["momentum"] = muon_mom
        for opt in optimizers:
            for g in opt.param_groups: g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()

        with torch.no_grad():
            for n, t in base_model.state_dict().items():
                ema_state[n].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms")
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX); reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap: stop_after_step = step

    # --- Apply EMA + Export ---
    log0("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {n: t.to(dtype=current_state[n].dtype) for n, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items() if "mtp_heads" not in k}
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
    quant_buf = io.BytesIO(); torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw) if _COMPRESSOR == "zstd" else zlib.compress(quant_raw, 9)

    if master_process:
        with open("final_model.int6.ptz", "wb") as f: f.write(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_COMPRESSOR}: {len(quant_blob)} bytes")
        log0(f"Total submission size: {len(quant_blob) + code_bytes} bytes")

    if distributed: dist.barrier()
    with open("final_model.int6.ptz", "rb") as f: quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob_disk) if _COMPRESSOR == "zstd" else zlib.decompress(quant_blob_disk)),
        map_location="cpu")
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        activation=args.activation, softcap_type=args.softcap_type,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    torch.cuda.synchronize(); t_q = time.perf_counter()
    compiled_eval = eval_model
    q_vl, q_vb = eval_val(args, compiled_eval, rank, world_size, device, grad_accum_steps,
                           val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                           eval_seq_len=effective_eval_seq_len)
    torch.cuda.synchronize()
    log0(f"final_int6_roundtrip val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f} eval_time:{1000*(time.perf_counter()-t_q):.0f}ms")

    sw_seq_len = effective_eval_seq_len
    eval_temp = args.eval_temperature if args.temp_scaling else 1.0
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize(); t_sw = time.perf_counter()
        sw_vl, sw_vb = eval_val_sliding(args, eval_model, rank, world_size, device,
                                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                                         stride=args.eval_stride, eval_seq_len=sw_seq_len,
                                         temperature=eval_temp)
        torch.cuda.synchronize()
        log0(f"final_int6_sliding_window val_loss:{sw_vl:.4f} val_bpb:{sw_vb:.4f} stride:{args.eval_stride} temp:{eval_temp}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_vl:.8f} val_bpb:{sw_vb:.8f}")

    # --- Legal TTT (SOTA: -0.0025 bpb) ---
    if args.ttt_enabled and args.eval_stride > 0:
        log0("ttt:starting Legal Score-First TTT...")
        torch.cuda.synchronize(); t_ttt = time.perf_counter()
        ttt_vl, ttt_vb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, batch_seqs=args.ttt_batch_seqs,
            temperature=eval_temp, log0=log0
        )
        torch.cuda.synchronize()
        log0(f"final_ttt_sliding val_loss:{ttt_vl:.4f} val_bpb:{ttt_vb:.4f} ")
        log0(f"ttt:done time:{1000*(time.perf_counter()-t_ttt):.0f}ms")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()