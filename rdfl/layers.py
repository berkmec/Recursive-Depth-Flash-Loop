"""Core layers of Recursive Depth Flash Loop (RDFL).

Implements:
- RMSNorm & SwiGLU
- CausalShortConv (depthwise k=4 causal convolution)
- GatedDeltaNetLayer: chunk-parallel WY solver with Householder-range beta in (0, 2),
  multi-scale slow decay alpha_0 in [0.953, 0.999], and bit-exact O(1) step() recurrence
- BdhSparseGate: ReLU-gated sparse associative interaction
- Fp32SlotCoreBoundary: FP32-locked chunk-boundary RLT-2 working-memory slots (S=4)
- BottleneckAdapter: zero-initialized residual adapter for K3 early-exit and MTP-4 heads
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    """Gated SwiGLU feed-forward network."""

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.out = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.up(x).chunk(2, dim=-1)
        return cast(Tensor, self.out(F.silu(a) * b))


class CausalShortConv(nn.Module):
    """Depthwise causal short convolution (k=4) with streaming state [B, D, k-1]."""

    def __init__(self, d_model: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, groups=d_model, bias=True
        )

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        b, _, d = x.shape
        pad = self.kernel_size - 1
        x_t = x.transpose(1, 2)
        hist = (
            x.new_zeros(b, d, pad)
            if state is None
            else state.to(device=x.device, dtype=x.dtype)
        )
        inp = torch.cat((hist, x_t), dim=2)
        out = F.silu(self.conv(inp)).transpose(1, 2)
        new_state = inp[:, :, -pad:] if x.requires_grad else inp[:, :, -pad:].detach()
        return out, new_state


def _chunk_delta_wy(
    q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, beta: Tensor, s0: Tensor
) -> tuple[Tensor, Tensor]:
    """Exact chunk-parallel Gated DeltaNet WY solver (Householder beta in (0, 2))."""
    c_len = q.shape[2]
    cum = log_alpha.cumsum(dim=-1)
    causal = torch.ones(c_len, c_len, device=q.device, dtype=torch.bool).tril()
    strict_lower = torch.ones(c_len, c_len, device=q.device, dtype=torch.bool).tril(-1)
    decay = (cum[..., :, None] - cum[..., None, :]).masked_fill(~causal, -torch.inf).exp()
    carry = cum.exp()

    kk = torch.matmul(k, k.transpose(-1, -2))
    lower = (beta[..., :, None] * decay * kk).masked_fill(~strict_lower, 0.0)
    eye = torch.eye(c_len, device=q.device, dtype=q.dtype)
    rhs = beta[..., None] * (v - carry[..., None] * torch.matmul(k, s0.transpose(-1, -2)))
    u = torch.linalg.solve_triangular(eye + lower, rhs, upper=False, unitriangular=True)

    qk = torch.matmul(q, k.transpose(-1, -2)) * decay
    out = carry[..., None] * torch.matmul(q, s0.transpose(-1, -2)) + torch.matmul(qk, u)
    end_decay = (cum[..., -1:] - cum).exp()
    s_end = cum[..., -1].exp()[..., None, None] * s0 + torch.matmul(
        (u * end_decay[..., None]).transpose(-1, -2), k
    )
    return out, s_end


class GatedDeltaNetLayer(nn.Module):
    """O(1)-state Gated DeltaNet with multi-scale slow decay and exact chunk/step parity.

    Note: This reference implementation uses pure PyTorch chunk-parallel WY (FP32 state
    update outside FP16 autocast). For large-scale from-scratch pretraining (200M+),
    replace `_chunk_delta_wy` with a fused Triton / Flash Linear Attention kernel.
    """

    def __init__(self, d_model: int, num_heads: int, chunk_size: int = 64) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        self.short_conv = CausalShortConv(d_model, kernel_size=4)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, 2 * num_heads, bias=True)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self._init_multiscale_decay()

    def _init_multiscale_decay(self) -> None:
        # Multi-scale slow decay alpha_0 in [0.953, 0.999] (half-lives ~14 to ~688 tokens)
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            biases = torch.linspace(3.0, 6.9, self.num_heads)
            self.gate_proj.bias[: self.num_heads].copy_(biases)
            self.gate_proj.bias[self.num_heads :].zero_()

    def _project(
        self, x: Tensor, conv_state: Tensor | None, calc_dtype: torch.dtype
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, t, _ = x.shape
        x_c, new_conv = self.short_conv(x, conv_state)
        qkv = self.qkv(x_c).to(calc_dtype)
        gates = self.gate_proj(x_c).to(calc_dtype)
        q, k, v = (
            p.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            for p in qkv.chunk(3, dim=-1)
        )
        q = F.normalize(F.silu(q), p=2, dim=-1, eps=1e-6)
        k = F.normalize(F.silu(k), p=2, dim=-1, eps=1e-6)
        a_logit, b_logit = gates.chunk(2, dim=-1)
        log_alpha = F.logsigmoid(a_logit).transpose(1, 2)
        beta = (2.0 * torch.sigmoid(b_logit)).transpose(1, 2)
        return q, k, v, log_alpha, beta, F.silu(self.g_proj(x)), new_conv

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor | None] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        b, t, d = x.shape
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        s_in = None if state is None else state[0]
        c_in = None if state is None else state[1]
        q, k, v, log_alpha, beta, out_gate, new_conv = self._project(x, c_in, dt)
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = (
                x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim, dtype=dt)
                if s_in is None
                else s_in.to(dt)
            )
            chunks: list[Tensor] = []
            for start in range(0, t, self.chunk_size):
                sl = slice(start, start + self.chunk_size)
                o_c, s = _chunk_delta_wy(
                    q[:, :, sl], k[:, :, sl], v[:, :, sl], log_alpha[:, :, sl], beta[:, :, sl], s
                )
                chunks.append(o_c)
            y = torch.cat(chunks, dim=2).transpose(1, 2).reshape(b, t, d)
        return self.out_proj(self.o_norm(y.to(x.dtype)) * out_gate), (s, new_conv)

    def step(
        self, x_t: Tensor, state: tuple[Tensor, Tensor | None] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Exact single-step O(1) recurrence matching chunk-parallel forward()."""
        b, _, d = x_t.shape
        dt = torch.float64 if x_t.dtype == torch.float64 else torch.float32
        s_in = None if state is None else state[0]
        c_in = None if state is None else state[1]
        q, k, v, log_alpha, beta, out_gate, new_conv = self._project(x_t, c_in, dt)
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            s = (
                x_t.new_zeros(b, self.num_heads, self.head_dim, self.head_dim, dtype=dt)
                if s_in is None
                else s_in.to(dt)
            )
            alpha = log_alpha[:, :, 0].exp()[..., None, None]
            b_t = beta[:, :, 0][..., None]
            k_t, v_t, q_t = k[:, :, 0], v[:, :, 0], q[:, :, 0]
            s_decayed = alpha * s
            u_t = b_t * (v_t - torch.matmul(s_decayed, k_t[..., None]).squeeze(-1))
            s = s_decayed + torch.matmul(u_t[..., None], k_t[..., None, :])
            o = torch.matmul(s, q_t[..., None]).squeeze(-1).reshape(b, 1, d)
        return self.out_proj(self.o_norm(o.to(x_t.dtype)) * out_gate), (s, new_conv)


class BdhSparseGate(nn.Module):
    """BDH-inspired ReLU sparse associative interaction gate."""

    def __init__(self, d_model: int, expansion: int = 2) -> None:
        super().__init__()
        hid = d_model * expansion
        self.enc_x = nn.Linear(d_model, hid, bias=False)
        self.enc_y = nn.Linear(d_model, hid, bias=False)
        self.dec = nn.Linear(hid, d_model, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        act = F.relu(self.enc_x(x)) * F.relu(self.enc_y(y))
        return cast(Tensor, self.dec(act))


class Fp32SlotCoreBoundary(nn.Module):
    """FP32-locked chunk-boundary RLT-2 working-memory slots (S=4, 0 FP16 NaN)."""

    def __init__(self, d_model: int, num_slots: int = 4, alpha: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.alpha = alpha
        self.pn = RMSNorm(d_model)
        self.gate = nn.Linear(2 * d_model, d_model, bias=True)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.slot_norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.slot_update = nn.Linear(2 * d_model, d_model, bias=True)
        self.init_slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)

    def initial_state(self, batch: int, dtype: torch.dtype | None = None) -> Tensor:
        dt = torch.float64 if dtype == torch.float64 else torch.float32
        return self.init_slots.to(dtype=dt).expand(batch, -1, -1).clone()

    def apply_feedback(self, chunk: Tensor, slots: Tensor) -> Tensor:
        pooled = slots.mean(dim=1).to(device=chunk.device, dtype=chunk.dtype)
        r = self.pn(pooled).unsqueeze(1).expand_as(chunk)
        g = torch.sigmoid(self.gate(torch.cat([chunk, r], dim=-1)))
        return cast(Tensor, chunk + self.alpha * g * self.merge(r))

    def _slot_message_fp32(self, slots: Tensor, dt: torch.dtype) -> Tensor:
        s = slots.shape[1]
        norm = slots * torch.rsqrt(slots.pow(2).mean(dim=-1, keepdim=True) + self.slot_norm.eps)
        yn = norm * self.slot_norm.weight.to(dt)
        q = F.linear(yn, self.q_proj.weight.to(dt))
        k = F.linear(yn, self.k_proj.weight.to(dt))
        v = F.linear(yn, self.v_proj.weight.to(dt))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)
        diag = torch.eye(s, device=slots.device, dtype=torch.bool).unsqueeze(0)
        weights = torch.softmax(scores.masked_fill(diag, -1e9), dim=-1)
        return F.linear(torch.matmul(weights, v), self.o_proj.weight.to(dt))

    def commit_boundary(self, slots: Tensor, chunk_end_state: Tensor) -> Tensor:
        b, s, d = slots.shape
        is_f64 = slots.dtype == torch.float64 or chunk_end_state.dtype == torch.float64
        dt = torch.float64 if is_f64 else torch.float32
        with torch.autocast(device_type=slots.device.type, enabled=False):
            sl_f = slots.to(dt)
            summary = chunk_end_state.to(dt).unsqueeze(1).expand(b, s, d)
            cat_in = torch.cat([sl_f, summary], dim=-1)
            upd_bias = None if self.slot_update.bias is None else self.slot_update.bias.to(dt)
            delta = torch.tanh(F.linear(cat_in, self.slot_update.weight.to(dt), upd_bias))
            updated = sl_f + 0.1 * delta
            if s > 1:
                updated = updated + 0.15 * self._slot_message_fp32(updated, dt)
        return updated


class BottleneckAdapter(nn.Module):
    """Zero-initialized residual bottleneck adapter for K3 early-exit and MTP-4 heads."""

    def __init__(self, d_model: int, divisor: int = 4) -> None:
        super().__init__()
        hid = max(16, d_model // divisor)
        self.down = nn.Linear(d_model, hid, bias=False)
        self.up = nn.Linear(hid, d_model, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.up.weight)

    def forward(self, z: Tensor) -> Tensor:
        return cast(Tensor, z + self.up(F.silu(self.down(z))))
