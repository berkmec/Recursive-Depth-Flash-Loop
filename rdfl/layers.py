"""Core layers of Recursive Depth Flash Loop (RDFL).

Implements:
- RMSNorm (with FP32/FP64 upcast safety) & SwiGLU
- CausalShortConv (depthwise k=4 causal convolution)
- GatedDeltaNetLayer: chunk-parallel WY solver with Householder-range beta in (0, 2),
  multi-scale slow decay alpha_0 in [0.953, alpha_max], per-head FP32 o_norm,
  and bit-exact O(1) step() recurrence
- BdhSparseGate: ReLU-gated sparse associative interaction
- Fp32RLT2Boundary: lightweight damped + FP32 RMSNorm single-vector (S=1) RLT-2 boundary
- Fp32SlotCoreBoundary: FP32-locked chunk-boundary RLT-2 working-memory slots (S=4)
- BottleneckAdapter: zero-initialized residual adapter for K3 early-exit and MTP-4 heads
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _calc_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with FP32/FP64 upcast safety."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        dt = _calc_dtype(x.dtype)
        x_f = x.to(dt)
        norm = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight.to(dt)).to(x.dtype)


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

    def forward_with_buffer(
        self, x: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
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
        return out, new_state, inp

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        out, new_state, _ = self.forward_with_buffer(x, state)
        return out, new_state


def _chunk_delta_wy_core(
    q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, beta: Tensor, s0: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Exact chunk-parallel Gated DeltaNet WY solver returning (out, s_end, u, cum)."""
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
    return out, s_end, u, cum


def _chunk_delta_wy(
    q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, beta: Tensor, s0: Tensor
) -> tuple[Tensor, Tensor]:
    """Exact chunk-parallel Gated DeltaNet WY solver (Householder beta in (0, 2))."""
    out, s_end, _, _ = _chunk_delta_wy_core(q, k, v, log_alpha, beta, s0)
    return out, s_end


def _extract_chunk_states(
    k: Tensor, u: Tensor, cum: Tensor, s0: Tensor, s_end: Tensor
) -> list[Tensor]:
    """Extract exact per-token GDN state tensors within a single chunk."""
    c_len = k.shape[2]
    states: list[Tensor] = []
    for p in range(c_len):
        if p == c_len - 1:
            states.append(s_end)
        else:
            end_decay_p = (cum[..., p : p + 1] - cum[..., : p + 1]).exp()
            s_p = cum[..., p].exp()[..., None, None] * s0 + torch.matmul(
                (u[:, :, : p + 1] * end_decay_p[..., None]).transpose(-1, -2),
                k[:, :, : p + 1],
            )
            states.append(s_p)
    return states


class GatedDeltaNetLayer(nn.Module):
    """O(1)-state Gated DeltaNet with multi-scale slow decay and exact chunk/step parity.

    Note: This reference implementation uses pure PyTorch chunk-parallel WY (FP32 state
    update and per-head FP32 o_norm outside FP16 autocast). For large-scale pretraining
    (200M+), replace `_chunk_delta_wy` with a fused Triton / Flash Linear Attention kernel.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        chunk_size: int = 64,
        alpha_max: float | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if alpha_max is not None and not (0.5 < alpha_max < 1.0):
            raise ValueError("alpha_max must be in (0.5, 1.0)")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        self.alpha_max = alpha_max

        self.short_conv = CausalShortConv(d_model, kernel_size=4)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, 2 * num_heads, bias=True)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        # Per-head FP32 RMSNorm prevents slow heads from drowning fast heads and FP16 overflow
        self.o_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self._init_multiscale_decay()

    def _init_multiscale_decay(self) -> None:
        # Multi-scale slow decay alpha_0 in [0.953, alpha_max] (default 0.999 -> bias 6.9)
        nn.init.zeros_(self.gate_proj.weight)
        max_bias = (
            6.9
            if self.alpha_max is None
            else math.log(self.alpha_max / (1.0 - self.alpha_max))
        )
        with torch.no_grad():
            biases = torch.linspace(3.0, max_bias, self.num_heads)
            self.gate_proj.bias[: self.num_heads].copy_(biases)
            self.gate_proj.bias[self.num_heads :].zero_()

    def _project(
        self,
        x: Tensor,
        conv_state: Tensor | None,
        calc_dtype: torch.dtype,
        use_short_conv: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        b, t, _ = x.shape
        new_conv: Tensor | None
        conv_inp: Tensor | None
        if use_short_conv:
            x_c, new_conv, conv_inp = self.short_conv.forward_with_buffer(x, conv_state)
        else:
            x_c, new_conv, conv_inp = x, conv_state, None
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
        return q, k, v, log_alpha, beta, F.silu(self.g_proj(x)), new_conv, conv_inp

    def _forward_impl(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor | None] | None,
        use_short_conv: bool,
        collect_snapshots: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor | None], list[tuple[Tensor, Tensor | None]]]:
        b, t, d = x.shape
        dt = _calc_dtype(x.dtype)
        s_in = None if state is None else state[0]
        c_in = None if state is None else state[1]
        q, k, v, log_alpha, beta, out_gate, new_conv, conv_inp = self._project(
            x, c_in, dt, use_short_conv=use_short_conv
        )
        pad = self.short_conv.kernel_size - 1
        snapshots: list[tuple[Tensor, Tensor | None]] = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = (
                x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim, dtype=dt)
                if s_in is None
                else s_in.to(dt)
            )
            chunks: list[Tensor] = []
            for start in range(0, t, self.chunk_size):
                sl = slice(start, start + self.chunk_size)
                s_chunk_in = s
                o_c, s, u_c, cum_c = _chunk_delta_wy_core(
                    q[:, :, sl], k[:, :, sl], v[:, :, sl], log_alpha[:, :, sl], beta[:, :, sl], s
                )
                chunks.append(o_c)
                if collect_snapshots:
                    s_list = _extract_chunk_states(k[:, :, sl], u_c, cum_c, s_chunk_in, s)
                    for p, s_p in enumerate(s_list):
                        tok_idx = start + p
                        if conv_inp is None:
                            c_p = c_in
                        else:
                            sl_c = conv_inp[:, :, tok_idx + 1 : tok_idx + 1 + pad]
                            c_p = sl_c if x.requires_grad else sl_c.detach()
                        snapshots.append((s_p, c_p))
            y_heads = torch.cat(chunks, dim=2).transpose(1, 2)
            y_norm = self.o_norm(y_heads).reshape(b, t, d).to(x.dtype)
        return self.out_proj(y_norm * out_gate), (s, new_conv), snapshots

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor | None] | None = None,
        use_short_conv: bool = True,
    ) -> tuple[Tensor, tuple[Tensor, Tensor | None]]:
        out, new_state, _ = self._forward_impl(
            x, state=state, use_short_conv=use_short_conv, collect_snapshots=False
        )
        return out, new_state

    def forward_with_snapshots(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor | None] | None = None,
        use_short_conv: bool = True,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor | None]]]:
        """Run chunk-parallel forward and return per-token (S_t, conv_t) snapshots."""
        out, _, snapshots = self._forward_impl(
            x, state=state, use_short_conv=use_short_conv, collect_snapshots=True
        )
        return out, snapshots

    def step(
        self,
        x_t: Tensor,
        state: tuple[Tensor, Tensor | None] | None = None,
        use_short_conv: bool = True,
    ) -> tuple[Tensor, tuple[Tensor, Tensor | None]]:
        """Exact single-step O(1) recurrence matching chunk-parallel forward()."""
        b, _, d = x_t.shape
        dt = _calc_dtype(x_t.dtype)
        s_in = None if state is None else state[0]
        c_in = None if state is None else state[1]
        q, k, v, log_alpha, beta, out_gate, new_conv, _ = self._project(
            x_t, c_in, dt, use_short_conv=use_short_conv
        )
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
            o = torch.matmul(s, q_t[..., None]).squeeze(-1)
            y_norm = self.o_norm(o.unsqueeze(1)).reshape(b, 1, d).to(x_t.dtype)
        return self.out_proj(y_norm * out_gate), (s, new_conv)


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


class Fp32RLT2Boundary(nn.Module):
    """Lightweight damped + FP32-RMSNorm single-vector (S=1) RLT-2 chunk-boundary module."""

    def __init__(
        self, d_model: int, alpha: float = 0.1, init_decay_logit: float = 2.2
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_slots = 1
        self.alpha = alpha
        self.pn = RMSNorm(d_model)
        self.gate = nn.Linear(2 * d_model, d_model, bias=True)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.update_proj = nn.Linear(2 * d_model, d_model, bias=True)
        self.decay_logit = nn.Parameter(torch.full((d_model,), init_decay_logit))
        self.init_slots = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def initial_state(self, batch: int, dtype: torch.dtype | None = None) -> Tensor:
        dt = _calc_dtype(dtype if dtype is not None else torch.float32)
        return self.init_slots.to(dtype=dt).expand(batch, -1, -1).clone()

    def apply_feedback(self, chunk: Tensor, slots: Tensor) -> Tensor:
        is_f64 = slots.dtype == torch.float64 or chunk.dtype == torch.float64
        dt = torch.float64 if is_f64 else torch.float32
        s_vec = slots[:, 0].to(dt)
        norm = s_vec * torch.rsqrt(s_vec.pow(2).mean(dim=-1, keepdim=True) + self.pn.eps)
        r = (norm * self.pn.weight.to(dt)).to(device=chunk.device, dtype=chunk.dtype)
        r_exp = r.unsqueeze(1).expand_as(chunk)
        g = torch.sigmoid(self.gate(torch.cat([chunk, r_exp], dim=-1)))
        return cast(Tensor, chunk + self.alpha * g * self.merge(r_exp))

    def commit_boundary(self, slots: Tensor, chunk_end_state: Tensor) -> Tensor:
        is_f64 = slots.dtype == torch.float64 or chunk_end_state.dtype == torch.float64
        dt = torch.float64 if is_f64 else torch.float32
        with torch.autocast(device_type=slots.device.type, enabled=False):
            sl_f = slots.to(dt)
            summary = chunk_end_state.to(dt).unsqueeze(1)
            sum_norm = summary * torch.rsqrt(
                summary.pow(2).mean(dim=-1, keepdim=True) + self.pn.eps
            )
            cat_in = torch.cat([sl_f, sum_norm], dim=-1)
            upd_bias = None if self.update_proj.bias is None else self.update_proj.bias.to(dt)
            cand = torch.tanh(F.linear(cat_in, self.update_proj.weight.to(dt), upd_bias))
            gamma = torch.sigmoid(self.decay_logit.to(dt)).view(1, 1, -1)
            updated = gamma * sl_f + (1.0 - gamma) * cand
        return updated


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
        dt = _calc_dtype(dtype if dtype is not None else torch.float32)
        return self.init_slots.to(dtype=dt).expand(batch, -1, -1).clone()

    def apply_feedback(self, chunk: Tensor, slots: Tensor) -> Tensor:
        is_f64 = slots.dtype == torch.float64 or chunk.dtype == torch.float64
        dt = torch.float64 if is_f64 else torch.float32
        pooled = slots.mean(dim=1).to(dt)
        norm = pooled * torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + self.pn.eps)
        r = (norm * self.pn.weight.to(dt)).to(device=chunk.device, dtype=chunk.dtype)
        r_exp = r.unsqueeze(1).expand_as(chunk)
        g = torch.sigmoid(self.gate(torch.cat([chunk, r_exp], dim=-1)))
        return cast(Tensor, chunk + self.alpha * g * self.merge(r_exp))

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
