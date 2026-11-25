"""RDFL v0.5: Unified Pure O(1) Core with Non-Collapsing Step Reasoning & Learned Halting.

Pure RDFL Red-Line Guarantees:
  - Zero Attention / Softmax(QK^T) / Self-Attention / KV-cache
  - Zero external tokenizers (BPE/SentencePiece); raw 256-byte stream (0..255)
  - Strict O(1) constant-size inference state across T, R in {1..1024}, n_h, and tasks
  - FP64 forward() == step() parity <= 1e-13 (slowOFF/ON, chunked splits, random_matched)
  - Future-byte causality leakage == 0.0 (both update_slow=False and True)
  - Lossless ecosystem compatibility: K3 (OY-1), MSP + T* (OY-3), MTP-3 (OY-4), reverse KL (OY-7)

v0.5 Architecture:
  - Stage 0 (P3 + P2, P=1): Byte embedding + calibrated MSNgramHashStem (2/3/4/6-gram) +
      dual-regime Micro-DeltaProduct-GDN (n_h=4, Mod-1 WY C_tok=16, SoftRMSNorm).
  - Stage 1 (P3 + P1, P=4): Fixed-shape patch packer + Macro RDFL v0.5 core:
      * r=0: single write_and_read (K_mem=1) + Fp32RLT2Boundary commit ONLY from r=0 h1_seg[:, -1].
      * r>=1 (default reasoning_mode="step_halting"): read-only memory query +
        non-collapsing norm-bounded residual update with learned halting gate g_step in (0, 1):
          h_{r+1} = h_r + g_step(h_r, o_r) * (norm(h_r + delta_r) - h_r)
        Guarantees exact no-op preservation (h_{r+1} == h_r) as g_step -> 0 and bounded norm
        for all R up to 1024, plus dynamic early exit on eps_halt > 0.
      * Optional backward-compatible reasoning_mode="adfp" (v0.4.1 contractive fixed-point).
      * Unified reason_from_state() / forward_step_reasoning() on the exact same RDFLv05Model
        for silent multi-hop reasoning and Jacobian Global Workspace (J-Space) analysis.
  - Stage 2 (P4 + OY-3): Zero-loop causal surprise gate from logits_fast (NO S_slow),
      with symmetric pos==0 / pos>0 return_aux outputs.
  - Stage 3 (P4): OrthogonalSlowGDN (M2 OWM-GDN, zero-drift, outside P1 loop, SoftRMSNorm)
      + ConsolidationManager (Post-AdamW OGP + scale-normalized Online EWC + reverse KL).
  - Stage 4 (P3 + OY-4): Single-pass dec_ffn + untied head + MTP-3 lookahead heads.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal, cast, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rdfl.calibration import reverse_kl_per_token
from rdfl.layers import BottleneckAdapter, Fp32RLT2Boundary, RMSNorm, SwiGLU

SlowMode = Literal["m1", "m2"]
GateMode = Literal["surprise", "always_one", "random_matched", "off"]
ReasoningMode = Literal["step_halting", "adfp"]

_VALID_SLOW_MODES = {"m1", "m2"}
_VALID_GATE_MODES = {"surprise", "always_one", "random_matched", "off"}
_VALID_REASONING_MODES = {"step_halting", "adfp"}
_VALID_BOUNDARY_MODES = {"none", "rlt2_s1"}


def _calc_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _apply_rmsnorm(z: Tensor, weight: Tensor, eps: float) -> Tensor:
    dt = _calc_dtype(z.dtype)
    z_f = z.to(dt)
    norm = z_f * torch.rsqrt(z_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (norm * weight.to(dt)).to(z.dtype)


def rms_norm_tau(z: Tensor, tau: float = 1.0) -> Tensor:
    """Global 1/tau-Lipschitz floored RMSNorm with zero singularity at z=0."""
    dt = _calc_dtype(z.dtype)
    zf = z.to(dt)
    denom = torch.sqrt(zf.pow(2).mean(dim=-1, keepdim=True) + (tau * tau))
    return (zf / denom).to(z.dtype)


def bounded_q_norm(q: Tensor, tau_q: float = 0.25) -> Tensor:
    """Bounded per-head query norm: ||q||_2 < 1 with zero singularity at q=0."""
    dt = _calc_dtype(q.dtype)
    qf = q.to(dt)
    denom = torch.sqrt(qf.pow(2).sum(dim=-1, keepdim=True) + (tau_q * tau_q))
    return (qf / denom).to(q.dtype)


def _causal_msp_uncertainty(logits: Tensor, temperature: float = 1.0) -> Tensor:
    dt = _calc_dtype(logits.dtype)
    probs = F.softmax(logits.to(dt) / float(temperature), dim=-1)
    return 1.0 - probs.max(dim=-1).values


_WY_MASK_CACHE: dict[tuple[int, str, torch.dtype], tuple[Tensor, Tensor]] = {}


def _wy_masks(c_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    key = (c_len, str(device), dtype)
    cached = _WY_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    causal = torch.ones(c_len, c_len, device=device, dtype=torch.bool).tril()
    eye = torch.eye(c_len, device=device, dtype=dtype)
    _WY_MASK_CACHE[key] = (causal, eye)
    return causal, eye


def _chunk_delta_wy_core(
    q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor, beta: Tensor, s0: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Exact chunk-parallel Gated DeltaNet WY solver returning (out, s_end, u, cum)."""
    c_len = q.shape[2]
    cum = log_alpha.cumsum(dim=-1)
    causal, eye = _wy_masks(c_len, q.device, q.dtype)
    decay = (cum[..., :, None] - cum[..., None, :]).masked_fill(~causal, -torch.inf).exp()
    carry = cum.exp()

    kk = torch.matmul(k, k.transpose(-1, -2))
    lower = torch.tril(beta[..., :, None] * decay * kk, diagonal=-1)
    s0_t = s0.transpose(-1, -2)
    rhs = beta[..., None] * (v - carry[..., None] * torch.matmul(k, s0_t))
    u = torch.linalg.solve_triangular(eye + lower, rhs, upper=False, unitriangular=True)

    qk = torch.matmul(q, k.transpose(-1, -2)) * decay
    out = carry[..., None] * torch.matmul(q, s0_t) + torch.matmul(qk, u)
    end_decay = (cum[..., -1:] - cum).exp()
    s_end = carry[..., -1, None, None] * s0 + torch.matmul(
        (u * end_decay[..., None]).transpose(-1, -2), k
    )
    return out, s_end, u, cum


def _chunk_orth_delta_wy_core(
    q: Tensor,
    k: Tensor,
    p: Tensor,
    v: Tensor,
    log_alpha_eff: Tensor,
    eta: Tensor,
    s0: Tensor,
) -> tuple[Tensor, Tensor]:
    """Exact chunk-parallel Asymmetric Orthogonalized Delta-WY solver."""
    c_len = q.shape[2]
    cum = log_alpha_eff.cumsum(dim=-1)
    causal, eye = _wy_masks(c_len, q.device, q.dtype)
    decay = (cum[..., :, None] - cum[..., None, :]).masked_fill(~causal, -torch.inf).exp()
    carry = cum.exp()

    kp_mat = torch.matmul(k, p.transpose(-1, -2))
    lower = torch.tril(eta[..., :, None] * decay * kp_mat, diagonal=-1)
    rhs = eta[..., None] * (v - carry[..., None] * torch.matmul(k, s0.transpose(-1, -2)))
    u = torch.linalg.solve_triangular(eye + lower, rhs, upper=False, unitriangular=True)

    qp_mat = torch.matmul(q, p.transpose(-1, -2)) * decay
    out = carry[..., None] * torch.matmul(q, s0.transpose(-1, -2)) + torch.matmul(qp_mat, u)
    end_decay = (cum[..., -1:] - cum).exp()
    s_end = carry[..., -1, None, None] * s0 + torch.matmul(
        (u * end_decay[..., None]).transpose(-1, -2), p
    )
    return out, s_end


def _extract_deltaproduct_chunk_states(
    k: Tensor, u: Tensor, cum: Tensor, s0: Tensor, n_h: int
) -> list[Tensor]:
    """Extract per-token [B, H, d_h, d_h] states from a Mod-1 DeltaProduct WY chunk."""
    m_len = k.shape[2]
    c_tok = m_len // n_h
    states: list[Tensor] = []
    for tok in range(c_tok):
        m_end = (tok + 1) * n_h
        end_decay_p = (cum[..., m_end - 1 : m_end] - cum[..., :m_end]).exp()
        s_p = cum[..., m_end - 1].exp()[..., None, None] * s0 + torch.matmul(
            (u[:, :, :m_end] * end_decay_p[..., None]).transpose(-1, -2),
            k[:, :, :m_end],
        )
        states.append(s_p)
    return states


# ---------------------------------------------------------------------------
# Building Blocks (P1, P2, P3, P4)
# ---------------------------------------------------------------------------


class BiasFreeCausalShortConv(nn.Module):
    """Depthwise causal short convolution (k=4, strictly bias=False) with O(1) state."""

    def __init__(self, d_model: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, groups=d_model, bias=False
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
        with torch.no_grad():
            self.conv.weight[:, 0, -1].add_(0.5)

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
        out = self.conv(inp).transpose(1, 2)
        new_state = inp[:, :, -pad:] if x.requires_grad else inp[:, :, -pad:].detach()
        return out, new_state, inp


class SoftRMSNorm(nn.Module):
    """Energy-preserving soft-floor RMSNorm preventing empty-state noise amplification."""

    def __init__(
        self, dim: int, init_tau: float = 0.05, learnable: bool = True, eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.eps = eps
        self.learnable = learnable
        self.weight = nn.Parameter(torch.ones(dim))
        if learnable:
            self.theta = nn.Parameter(torch.tensor(math.log(math.expm1(init_tau))))
            self.kappa = nn.Parameter(torch.tensor(4.0))
            self.w_abs = nn.Parameter(torch.zeros(dim))
        else:
            self.tau_sq = init_tau * init_tau

    def forward(self, x: Tensor) -> Tensor:
        dt = _calc_dtype(x.dtype)
        xf = x.to(dt)
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        if self.learnable:
            tau = F.softplus(self.theta.to(dt))
            r = torch.sqrt(ms + self.eps)
            normed = xf * torch.rsqrt(ms + tau * tau + self.eps) * self.weight.to(dt)
            gate = torch.sigmoid(-self.kappa.to(dt) * (r - tau))
            return (normed + gate * self.w_abs.to(dt)).to(x.dtype)
        normed = xf * torch.rsqrt(ms + self.tau_sq) * self.weight.to(dt)
        return normed.to(x.dtype)


class RMSNormTauLayer(nn.Module):
    """Per-feature floored RMSNorm without learnable unbounded scale."""

    def __init__(self, dim: int, tau: float = 0.5) -> None:
        super().__init__()
        self.tau = float(tau)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm_tau(x, tau=self.tau)


class SoftSpectralLinear(nn.Module):
    """Linear layer with soft spectral cap ||W||_2 <= c_sn (zero mutation in forward)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        c_sn: float = 2.5,
        bias: bool = False,
        init_scale: float = 0.7,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c_sn = float(c_sn)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=init_scale / math.sqrt(float(in_features)))
        u_init = F.normalize(torch.randn(in_features), dim=0)
        self.register_buffer("u_spec", u_init)

    def _effective_weight(self) -> Tensor:
        if self.c_sn <= 0.0:
            return self.weight
        w = self.weight
        with torch.no_grad():
            u = cast(Tensor, self.u_spec).to(dtype=w.dtype, device=w.device)
            v = F.normalize(torch.mv(w, u), dim=0, eps=1e-12)
            wu = torch.mv(w.t(), v)
            u2 = F.normalize(wu, dim=0, eps=1e-12)
            v2 = F.normalize(torch.mv(w, u2), dim=0, eps=1e-12)
        sigma = torch.dot(v2, torch.mv(w, u2)).abs()
        scale = torch.clamp(sigma / self.c_sn, min=1.0)
        return w / scale

    @torch.no_grad()
    def update_spectral_vec(self) -> None:
        w = self.weight.detach().float()
        u_buf = cast(Tensor, self.u_spec)
        u = u_buf.float()
        for _ in range(3):
            v = F.normalize(torch.mv(w, u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(w.t(), v), dim=0, eps=1e-12)
        u_buf.copy_(u.to(dtype=u_buf.dtype))

    def forward(self, x: Tensor) -> Tensor:
        w_eff = self._effective_weight().to(dtype=x.dtype)
        b_eff = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, w_eff, b_eff)


class BoundedBdhGate(nn.Module):
    """BDH sparse associative gate with bounded ReLU(tanh(.)) branches."""

    def __init__(
        self, d_model: int, expansion: int = 2, c_sn: float = 1.35, c_sn_out: float = 0.85
    ) -> None:
        super().__init__()
        hid = d_model * expansion
        self.enc_x = SoftSpectralLinear(d_model, hid, c_sn=c_sn, bias=False)
        self.enc_y = SoftSpectralLinear(d_model, hid, c_sn=c_sn, bias=False)
        self.dec = SoftSpectralLinear(hid, d_model, c_sn=c_sn_out, bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        act = F.relu(torch.tanh(self.enc_x(x))) * F.relu(torch.tanh(self.enc_y(y)))
        return cast(Tensor, self.dec(act))


class BoundedSwiGLU(nn.Module):
    """Bounded gated FFN: SiLU(W1 x) * tanh(W2 x)."""

    def __init__(
        self, d_model: int, ffn_dim: int, c_sn: float = 1.35, c_sn_out: float = 0.85
    ) -> None:
        super().__init__()
        self.w1 = SoftSpectralLinear(d_model, ffn_dim, c_sn=c_sn, bias=False)
        self.w2 = SoftSpectralLinear(d_model, ffn_dim, c_sn=c_sn, bias=False)
        self.out = SoftSpectralLinear(ffn_dim, d_model, c_sn=c_sn_out, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.out(F.silu(self.w1(x)) * torch.tanh(self.w2(x))))


class MSNgramHashStem(nn.Module):
    """Calibrated Multi-Scale 2/3/4/6-gram Count-Sketch +-1 + digit-protecting gate."""

    def __init__(
        self,
        d_out: int,
        buckets: tuple[int, int, int, int] = (2048, 4096, 4096, 4096),
        d_h: int = 16,
        alpha_init: float = 0.008,
    ) -> None:
        super().__init__()
        for nb in buckets:
            if nb <= 0 or (nb & (nb - 1)) != 0:
                raise ValueError(f"bucket count must be power of 2, got {nb}")
        self.buckets = buckets
        self.d_h = d_h
        self.d_out = d_out
        self.emb2 = nn.Embedding(buckets[0], d_h)
        self.emb3 = nn.Embedding(buckets[1], d_h)
        self.emb4 = nn.Embedding(buckets[2], d_h)
        self.emb6 = nn.Embedding(buckets[3], d_h)
        for e in (self.emb2, self.emb3, self.emb4, self.emb6):
            nn.init.normal_(e.weight, mean=0.0, std=0.02)
        gl = torch.zeros(256, 4)
        gl[:, 3].fill_(-2.5)
        self.gate_lut = nn.Parameter(gl)
        self.ng_proj = nn.Linear(4 * d_h, d_out, bias=False)
        nn.init.normal_(self.ng_proj.weight, mean=0.0, std=0.02)
        self.ng_norm = RMSNorm(d_out)
        self.alpha_ng = nn.Parameter(torch.tensor(alpha_init))

    def embed_delta(
        self,
        x: Tensor,
        prev_bytes: Tensor | None,
        collect_snapshots: bool = False,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        b, t = x.shape
        if prev_bytes is None:
            pb = torch.full((b, 5), 256, dtype=torch.long, device=x.device)
        else:
            pb = prev_bytes.to(device=x.device, dtype=torch.long)
        seq_ext = torch.cat([pb, x.to(torch.long)], dim=1)
        b0 = seq_ext[:, 5:]
        b1 = seq_ext[:, 4:-1]
        b2 = seq_ext[:, 3:-2]
        b3 = seq_ext[:, 2:-3]
        b4 = seq_ext[:, 1:-4]
        b5 = seq_ext[:, :-5]

        raw2 = (b1 * 1315423911 + b0 * 2654435761 + 0x9E3779B9) & 0x7FFFFFFFFFFFFFFF
        raw3 = (
            b2 * 2246822519 + b1 * 3266489917 + b0 * 668265263 + 0x517CC1B7
        ) & 0x7FFFFFFFFFFFFFFF
        raw4 = (
            b3 * 374761393 + b2 * 668265263 + b1 * 2246822519 + b0 * 3266489917 + 0x2545F491
        ) & 0x7FFFFFFFFFFFFFFF
        raw6 = (
            b5 * 1140071481
            + b4 * 2654435761
            + b3 * 374761393
            + b2 * 668265263
            + b1 * 2246822519
            + b0 * 3266489917
            + 0x165667B1
        ) & 0x7FFFFFFFFFFFFFFF

        m2, m3, m4, m6 = [nb - 1 for nb in self.buckets]
        i2, i3, i4, i6 = raw2 & m2, raw3 & m3, raw4 & m4, raw6 & m6

        dt = self.ng_proj.weight.dtype
        s2 = (((raw2 >> 31) & 1) * 2 - 1).to(dt).unsqueeze(-1)
        s3 = (((raw3 >> 31) & 1) * 2 - 1).to(dt).unsqueeze(-1)
        s4 = (((raw4 >> 31) & 1) * 2 - 1).to(dt).unsqueeze(-1)
        s6 = (((raw6 >> 31) & 1) * 2 - 1).to(dt).unsqueeze(-1)

        gamma = torch.sigmoid(self.gate_lut[x.to(torch.long)]).to(dt)
        v2 = self.emb2(i2) * s2 * gamma[..., 0:1]
        v3 = self.emb3(i3) * s3 * gamma[..., 1:2]
        v4 = self.emb4(i4) * s4 * gamma[..., 2:3]
        v6 = self.emb6(i6) * s6 * gamma[..., 3:4]

        cat = torch.cat([v2, v3, v4, v6], dim=-1)
        u_ng = self.ng_norm(self.ng_proj(cat))
        delta = self.alpha_ng.to(dt) * u_ng
        new_prev = seq_ext[:, -5:]
        snaps: list[Tensor] = []
        if collect_snapshots:
            for p in range(t):
                snaps.append(seq_ext[:, p + 1 : p + 6])
        return delta, new_prev, snaps


class MicroDeltaProductGDNLayer(nn.Module):
    """Stage 0 (P2+P3): Dual-regime n_h-step Householder Gated DeltaNet (Mod-1 WY)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        n_h: int = 4,
        chunk_size: int = 64,
        conv_init: float = 0.05,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if n_h < 1:
            raise ValueError("n_h must be >= 1")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.n_h = n_h
        self.chunk_size = chunk_size

        self.short_conv = BiasFreeCausalShortConv(d_model, kernel_size=4)
        self.n_assoc = num_heads // 2
        self.n_hh = num_heads - self.n_assoc
        da = self.n_assoc * self.head_dim
        dh_tot = self.n_hh * self.head_dim
        self.conv_gate_hh = nn.Parameter(torch.full((d_model,), conv_init))

        self.q_a = nn.Linear(d_model, da, bias=False)
        self.k_a = nn.Linear(d_model, n_h * da, bias=False)
        self.v_a = nn.Linear(d_model, n_h * da, bias=False)
        self.g_a = nn.Linear(d_model, self.n_assoc * (1 + n_h), bias=True)

        self.q_h = nn.Linear(d_model, dh_tot, bias=False)
        self.k_h = nn.Linear(d_model, n_h * dh_tot, bias=False)
        self.v_h = nn.Linear(d_model, n_h * dh_tot, bias=False)
        self.g_h = nn.Linear(d_model, self.n_hh * (1 + n_h), bias=True)

        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_norm = SoftRMSNorm(self.head_dim, init_tau=0.05, learnable=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.s0_hh = nn.Parameter(torch.randn(1, self.n_hh, self.head_dim, self.head_dim) * 0.25)

        nn.init.zeros_(self.g_a.weight)
        nn.init.zeros_(self.g_h.weight)
        with torch.no_grad():
            if self.n_assoc > 0:
                self.g_a.bias[: self.n_assoc].copy_(torch.linspace(3.0, 6.0, self.n_assoc))
                ba = self.g_a.bias[self.n_assoc :].view(self.n_assoc, n_h)
                ba[:, 0] = 0.0
                if n_h > 1:
                    ba[:, 1:].fill_(-3.5)
                    self.v_a.weight[da:].mul_(0.05)
            if self.n_hh > 0:
                self.g_h.bias[: self.n_hh].copy_(torch.linspace(5.5, 7.8, self.n_hh))
                self.g_h.bias[self.n_hh :].fill_(2.5)

    def _project(
        self,
        x: Tensor,
        conv_state: Tensor | None,
        dt: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        b, t, _ = x.shape
        x_sc, new_conv, conv_inp = self.short_conv.forward_with_buffer(x, conv_state)
        x_hh = x + self.conv_gate_hh.to(x.dtype) * x_sc

        qa = F.normalize(
            self.q_a(x_sc).to(dt).reshape(b, t, self.n_assoc, self.head_dim).transpose(1, 2),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        ka = F.normalize(
            self.k_a(x_sc)
            .to(dt)
            .reshape(b, t, self.n_h, self.n_assoc, self.head_dim)
            .permute(0, 3, 1, 2, 4),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        va = (
            self.v_a(x_sc)
            .to(dt)
            .reshape(b, t, self.n_h, self.n_assoc, self.head_dim)
            .permute(0, 3, 1, 2, 4)
        )
        ga = self.g_a(x_sc).to(dt)
        laa = F.logsigmoid(ga[..., : self.n_assoc]).transpose(1, 2)
        bta = (
            2.0 * torch.sigmoid(ga[..., self.n_assoc :].reshape(b, t, self.n_assoc, self.n_h))
        ).permute(0, 2, 1, 3)

        qh = F.normalize(
            self.q_h(x_hh).to(dt).reshape(b, t, self.n_hh, self.head_dim).transpose(1, 2),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        kh = F.normalize(
            self.k_h(x_hh)
            .to(dt)
            .reshape(b, t, self.n_h, self.n_hh, self.head_dim)
            .permute(0, 3, 1, 2, 4),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        vh = (
            self.v_h(x_hh)
            .to(dt)
            .reshape(b, t, self.n_h, self.n_hh, self.head_dim)
            .permute(0, 3, 1, 2, 4)
        )
        gh = self.g_h(x_hh).to(dt)
        lah = F.logsigmoid(gh[..., : self.n_hh]).transpose(1, 2)
        bth = (
            2.0 * torch.sigmoid(gh[..., self.n_hh :].reshape(b, t, self.n_hh, self.n_h))
        ).permute(0, 2, 1, 3)

        q = torch.cat([qa, qh], dim=1)
        k = torch.cat([ka, kh], dim=1)
        v = torch.cat([va, vh], dim=1)
        log_alpha = torch.cat([laa, lah], dim=1)
        beta = torch.cat([bta, bth], dim=1)
        out_gate = F.silu(self.g_proj(x))
        return q, k, v, log_alpha, beta, out_gate, new_conv, conv_inp

    def init_s(self, b: int, device: torch.device, dt: torch.dtype, s_in: Tensor | None) -> Tensor:
        if s_in is not None:
            return s_in.to(dt)
        s0_a = torch.zeros(b, self.n_assoc, self.head_dim, self.head_dim, device=device, dtype=dt)
        s0_h = self.s0_hh.to(device=device, dtype=dt).expand(b, -1, -1, -1).clone()
        return torch.cat([s0_a, s0_h], dim=1)

    def forward_impl(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor | None] | None,
        collect_snapshots: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], list[tuple[Tensor, Tensor]]]:
        b, t, d = x.shape
        dt = _calc_dtype(x.dtype)
        s_in = None if state is None else state[0]
        c_in = None if state is None else state[1]
        q, k, v, log_alpha, beta, out_gate, new_conv, conv_inp = self._project(x, c_in, dt)
        pad = self.short_conv.kernel_size - 1
        snapshots: list[tuple[Tensor, Tensor]] = []

        with torch.autocast(device_type=x.device.type, enabled=False):
            s = self.init_s(b, x.device, dt, s_in)
            tok_chunk = max(1, self.chunk_size // self.n_h) if self.n_h > 1 else self.chunk_size
            chunks: list[Tensor] = []
            for start in range(0, t, tok_chunk):
                end = min(start + tok_chunk, t)
                c = end - start
                s_chunk_in = s
                if self.n_h == 1:
                    k_c = k[:, :, start:end, 0]
                    o_c, s, u_c, cum_c = _chunk_delta_wy_core(
                        q[:, :, start:end],
                        k_c,
                        v[:, :, start:end, 0],
                        log_alpha[:, :, start:end],
                        beta[:, :, start:end, 0],
                        s,
                    )
                    if collect_snapshots:
                        for p in range(c):
                            tok_idx = start + p
                            end_dec_p = (cum_c[..., p : p + 1] - cum_c[..., : p + 1]).exp()
                            s_p = cum_c[..., p].exp()[..., None, None] * s_chunk_in + torch.matmul(
                                (u_c[:, :, : p + 1] * end_dec_p[..., None]).transpose(-1, -2),
                                k_c[:, :, : p + 1],
                            )
                            sl_c = conv_inp[:, :, tok_idx + 1 : tok_idx + 1 + pad]
                            snapshots.append((s_p, sl_c if x.requires_grad else sl_c.detach()))
                else:
                    q_c = (
                        q[:, :, start:end]
                        .unsqueeze(3)
                        .expand(b, self.num_heads, c, self.n_h, self.head_dim)
                        .reshape(b, self.num_heads, c * self.n_h, self.head_dim)
                    )
                    k_c = k[:, :, start:end].reshape(b, self.num_heads, c * self.n_h, self.head_dim)
                    v_c = v[:, :, start:end].reshape(b, self.num_heads, c * self.n_h, self.head_dim)
                    b_c = beta[:, :, start:end].reshape(b, self.num_heads, c * self.n_h)
                    la_c = F.pad(
                        log_alpha[:, :, start:end].unsqueeze(-1), (0, self.n_h - 1)
                    ).reshape(b, self.num_heads, c * self.n_h)
                    o_flat, s, u_c, cum_c = _chunk_delta_wy_core(q_c, k_c, v_c, la_c, b_c, s)
                    o_c = o_flat[:, :, self.n_h - 1 :: self.n_h]
                    if collect_snapshots:
                        s_list = _extract_deltaproduct_chunk_states(
                            k_c, u_c, cum_c, s_chunk_in, self.n_h
                        )
                        for p, s_p in enumerate(s_list):
                            tok_idx = start + p
                            sl_c = conv_inp[:, :, tok_idx + 1 : tok_idx + 1 + pad]
                            snapshots.append((s_p, sl_c if x.requires_grad else sl_c.detach()))
                chunks.append(o_c)

            y_heads = torch.cat(chunks, dim=2).transpose(1, 2)
            y_norm = self.o_norm(y_heads).reshape(b, t, d).to(x.dtype)
        return cast(Tensor, self.out_proj(y_norm * out_gate)), (s, new_conv), snapshots

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor | None] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        out, st, _ = self.forward_impl(x, state, collect_snapshots=False)
        return out, st

    def step(
        self, x_t: Tensor, state: tuple[Tensor, Tensor | None] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        b, _, d = x_t.shape
        dt = _calc_dtype(x_t.dtype)
        s_in = None if state is None else state[0]
        c_in = None if state is None else state[1]
        q, k, v, log_alpha, beta, out_gate, new_conv, _ = self._project(x_t, c_in, dt)
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            s = self.init_s(b, x_t.device, dt, s_in)
            alpha_t = log_alpha[:, :, 0].exp()[..., None, None]
            s = alpha_t * s
            for m in range(self.n_h):
                k_m = k[:, :, 0, m]
                v_m = v[:, :, 0, m]
                b_m = beta[:, :, 0, m][..., None]
                u_m = b_m * (v_m - torch.matmul(s, k_m[..., None]).squeeze(-1))
                s = s + torch.matmul(u_m[..., None], k_m[..., None, :])
            q_t = q[:, :, 0]
            o = torch.matmul(s, q_t[..., None]).squeeze(-1)
            y_norm = self.o_norm(o.unsqueeze(1)).reshape(b, 1, d).to(x_t.dtype)
        return cast(Tensor, self.out_proj(y_norm * out_gate)), (s, new_conv)


class GDNReadOnlyLayer(nn.Module):
    """Stage 1 (P1): Single-write (r=0) + zero-solver WY cache read-only (r>=1) GDN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        chunk_size: int = 64,
        c_sn: float = 1.35,
        c_sn_out: float = 0.85,
        tau_q: float = 0.25,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must divide num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        self.tau_q = float(tau_q)

        self.short_conv = BiasFreeCausalShortConv(d_model, kernel_size=4)
        self.qkv = SoftSpectralLinear(d_model, 3 * d_model, c_sn=c_sn, bias=False)
        self.q_read = SoftSpectralLinear(d_model, d_model, c_sn=c_sn, bias=False)
        self.gate_proj = nn.Linear(d_model, 2 * num_heads, bias=True)
        self.g_proj = SoftSpectralLinear(d_model, d_model, c_sn=c_sn, bias=False)
        self.o_norm = RMSNormTauLayer(self.head_dim, tau=0.5)
        self.out_proj = SoftSpectralLinear(d_model, d_model, c_sn=c_sn_out, bias=False)
        self._init_multiscale_decay()

    def _init_multiscale_decay(self) -> None:
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            biases = torch.linspace(3.0, 6.9, self.num_heads)
            self.gate_proj.bias[: self.num_heads].copy_(biases)
            self.gate_proj.bias[self.num_heads :].zero_()

    def write_and_read(
        self,
        u0: Tensor,
        state: tuple[Tensor, Tensor | None] | None,
        single_step: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Any]:
        b, t, d = u0.shape
        dt = _calc_dtype(u0.dtype)
        s_in = None if state is None else state[0]
        c_in = None if state is None else state[1]

        x_c, new_conv, _ = self.short_conv.forward_with_buffer(u0, c_in)
        qkv = self.qkv(x_c).to(dt)
        gates = self.gate_proj(x_c).to(dt)
        q, k, v = (
            p.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            for p in qkv.chunk(3, dim=-1)
        )
        q = bounded_q_norm(q, tau_q=self.tau_q)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        a_logit, b_logit = gates.chunk(2, dim=-1)
        log_alpha = F.logsigmoid(a_logit).transpose(1, 2)
        beta = (2.0 * torch.sigmoid(b_logit)).transpose(1, 2)
        out_gate = torch.sigmoid(self.g_proj(u0))

        with torch.autocast(device_type=u0.device.type, enabled=False):
            s0 = (
                u0.new_zeros(b, self.num_heads, self.head_dim, self.head_dim, dtype=dt)
                if s_in is None
                else s_in.to(dt)
            )
            if single_step:
                alpha = log_alpha[:, :, 0].exp()[..., None, None]
                b_t = beta[:, :, 0][..., None]
                k_t, v_t, q_t = k[:, :, 0], v[:, :, 0], q[:, :, 0]
                s_dec = alpha * s0
                u_t = b_t * (v_t - torch.matmul(s_dec, k_t[..., None]).squeeze(-1))
                s_new = s_dec + torch.matmul(u_t[..., None], k_t[..., None, :])
                o0 = torch.matmul(s_new, q_t[..., None]).squeeze(-1).unsqueeze(1)
                cache: Any = s_new
            else:
                o_c, s_new, u_c, cum_c = _chunk_delta_wy_core(q, k, v, log_alpha, beta, s0)
                o0 = o_c.transpose(1, 2)
                cache = (s0, k, u_c, cum_c)
            y_norm = self.o_norm(o0).reshape(b, t, d).to(u0.dtype)
        return cast(Tensor, self.out_proj(y_norm * out_gate)), (s_new, new_conv), cache

    def read_only(self, u_r: Tensor, cache: Any) -> Tensor:
        b, t, d = u_r.shape
        dt = _calc_dtype(u_r.dtype)
        q_r = self.q_read(u_r).to(dt).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        q_r = bounded_q_norm(q_r, tau_q=self.tau_q)
        out_gate = torch.sigmoid(self.g_proj(u_r))

        with torch.autocast(device_type=u_r.device.type, enabled=False):
            if isinstance(cache, Tensor):
                if t == 1:
                    o_r = torch.matmul(cache, q_r[:, :, 0, :, None]).squeeze(-1).unsqueeze(1)
                else:
                    o_r = torch.matmul(q_r, cache.transpose(-1, -2)).transpose(1, 2)
            else:
                s0, k_c, u_c, cum_c = cache
                c_len = q_r.shape[2]
                causal = torch.ones(c_len, c_len, device=q_r.device, dtype=torch.bool).tril()
                decay = (
                    (cum_c[..., :, None] - cum_c[..., None, :])
                    .masked_fill(~causal, -torch.inf)
                    .exp()
                )
                carry = cum_c.exp()
                qk = torch.matmul(q_r, k_c.transpose(-1, -2)) * decay
                o_r = (
                    carry[..., None] * torch.matmul(q_r, s0.transpose(-1, -2))
                    + torch.matmul(qk, u_c)
                ).transpose(1, 2)
            y_norm = self.o_norm(o_r).reshape(b, t, d).to(u_r.dtype)
        return cast(Tensor, self.out_proj(y_norm * out_gate))


class MacroRDFLv05Core(nn.Module):
    """Stage 1 (P1 + StepReasoner): Macro RDFL v0.5 core supporting step_halting and adfp."""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_loops: int = 2,
        chunk_size: int = 64,
        c_sn: float = 1.35,
        c_sn_out: float = 0.85,
        use_bdh: bool = True,
        boundary_mode: str = "rlt2_s1",
        reasoning_mode: ReasoningMode = "step_halting",
        halt_bias_init: float = 3.5,
    ) -> None:
        super().__init__()
        if reasoning_mode not in _VALID_REASONING_MODES:
            raise ValueError(f"reasoning_mode must be in {_VALID_REASONING_MODES}")
        if boundary_mode not in _VALID_BOUNDARY_MODES:
            raise ValueError(f"boundary_mode must be in {_VALID_BOUNDARY_MODES}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size
        self.num_loops = num_loops
        self.reasoning_mode: ReasoningMode = reasoning_mode

        self.n_in = RMSNorm(d_model)
        self.e_write = nn.Parameter(torch.randn(d_model) * 0.02)
        self.e_refine = nn.Parameter(torch.randn(d_model) * 0.02)
        self.gdn = GDNReadOnlyLayer(
            d_model, num_heads, chunk_size=chunk_size, c_sn=c_sn, c_sn_out=c_sn_out
        )
        self.bdh = (
            BoundedBdhGate(d_model, expansion=2, c_sn=c_sn, c_sn_out=c_sn_out)
            if use_bdh
            else None
        )
        self.ffn = BoundedSwiGLU(d_model, ffn_dim, c_sn=c_sn, c_sn_out=c_sn_out)
        self.boundary = Fp32RLT2Boundary(d_model) if boundary_mode == "rlt2_s1" else None

        # Step-halting & phase modules (non-collapsing v0.5 reasoning)
        self.phase_proj = SoftSpectralLinear(d_model, d_model, c_sn=c_sn, bias=True)
        halt_in = nn.Linear(2 * d_model, d_model, bias=True)
        halt_out = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(halt_out.weight, mean=0.0, std=0.02)
        nn.init.constant_(halt_out.bias, halt_bias_init)
        self.halt_gate = nn.Sequential(halt_in, nn.SiLU(), halt_out)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, max(1, d_model // 2), dtype=torch.float32)
            / float(max(1, d_model // 2))
        )
        self.register_buffer("phase_freqs", freqs)

        # Optional loop-internal scratchpad projections for multi-step state reasoning
        self.qw_proj = SoftSpectralLinear(d_model, d_model, c_sn=c_sn, bias=False)
        self.kw_proj = SoftSpectralLinear(2 * d_model, d_model, c_sn=c_sn, bias=False)
        self.vw_proj = SoftSpectralLinear(2 * d_model, d_model, c_sn=c_sn, bias=True)
        self.abw_proj = nn.Linear(d_model, 2 * num_heads, bias=True)
        nn.init.zeros_(self.abw_proj.weight)
        with torch.no_grad():
            self.abw_proj.bias[:num_heads].fill_(2.5)
            self.abw_proj.bias[num_heads:].zero_()
        self.ow_proj = SoftSpectralLinear(d_model, d_model, c_sn=c_sn_out, bias=False)
        self.reason_ffn_in = nn.Linear(3 * d_model, 2 * d_model, bias=True)
        self.reason_ffn_out = nn.Linear(2 * d_model, d_model, bias=True)
        self.reason_gate = nn.Linear(3 * d_model, d_model, bias=True)

        # Backward-compatible ADFP parameters
        self.beta_h_logit = nn.Parameter(torch.full((d_model,), 0.5))
        self.lambda_logit = nn.Parameter(torch.full((d_model,), 0.0))

        res_std = 0.8 / math.sqrt(float(d_model))
        nn.init.normal_(self.gdn.out_proj.weight, mean=0.0, std=res_std)
        nn.init.normal_(self.ffn.out.weight, mean=0.0, std=0.8 / math.sqrt(float(ffn_dim)))
        nn.init.normal_(self.ow_proj.weight, mean=0.0, std=res_std)
        if self.bdh is not None:
            nn.init.normal_(self.bdh.dec.weight, mean=0.0, std=0.6 / math.sqrt(float(2 * d_model)))

    def _phase_vec(self, r_idx: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        freqs = cast(Tensor, self.phase_freqs).to(device=device, dtype=torch.float32)
        ang = float(r_idx) * freqs
        vec = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if vec.shape[0] < self.d_model:
            vec = F.pad(vec, (0, self.d_model - vec.shape[0]))
        elif vec.shape[0] > self.d_model:
            vec = vec[: self.d_model]
        return vec.to(dtype=dtype)

    def _refine_step_adfp(self, h: Tensor, anchor: Tensor, cache: Any) -> Tensor:
        dt = h.dtype
        beta_h = 0.35 + 0.35 * torch.sigmoid(self.beta_h_logit.to(dt))
        u_r = rms_norm_tau(beta_h * h + (1.0 - beta_h) * anchor + self.e_refine.to(dt), tau=1.0)
        y_gdn = self.gdn.read_only(u_r, cache)
        y_bdh = self.bdh(u_r, y_gdn) if self.bdh is not None else 0.0
        v_r = rms_norm_tau(u_r + y_gdn + y_bdh, tau=1.0)
        y_ffn = self.ffn(v_r)
        core_sum = y_gdn + y_bdh + y_ffn
        f_val = rms_norm_tau(0.5 * anchor + 0.85 * core_sum, tau=1.0)
        lam = 0.25 + 0.25 * torch.sigmoid(self.lambda_logit.to(dt))
        return (1.0 - lam) * h + lam * f_val

    def _compute_halt_gate(self, h: Tensor, o_mem: Tensor) -> Tensor:
        g_in = torch.cat([rms_norm_tau(h, tau=1.0), rms_norm_tau(o_mem, tau=1.0)], dim=-1)
        return torch.sigmoid(self.halt_gate(g_in))

    def _refine_step_halting(
        self,
        h: Tensor,
        cache: Any,
        r_idx: int,
        *,
        gate_override: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Non-collapsing norm-bounded step update with learned halting gate g_step in (0, 1).

        Wait: o_r and g_step do NOT depend on r_idx/phase_r so that once h converges to a
        halted state (g_step -> 0), o_r and g_step remain fixed-point stable for any R_extra.
        Exact no-op guarantee: when g_step == 0, h_{r+1} == h_r bit-for-bit.
        """
        dt = h.dtype
        h_q = rms_norm_tau(h, tau=1.0)
        o_r = self.gdn.read_only(h_q, cache)
        if gate_override is None:
            g_step = self._compute_halt_gate(h, o_r)
        else:
            g_step = torch.full(
                (*h.shape[:-1], 1), float(gate_override), device=h.device, dtype=dt
            )

        ph = self.phase_proj(self._phase_vec(r_idx, h.device, dt))
        u_r = rms_norm_tau(h + ph + self.e_refine.to(dt), tau=1.0)
        y_bdh = self.bdh(u_r, o_r) if self.bdh is not None else 0.0
        v_r = rms_norm_tau(u_r + o_r + y_bdh, tau=1.0)
        delta_r = o_r + y_bdh + self.ffn(v_r)
        h_cand = rms_norm_tau(h + delta_r, tau=1.0)
        h_next = h + g_step * (h_cand - h)
        return h_next, g_step

    def reason_step_from_memory(
        self,
        h: Tensor,
        w_scratch: Tensor,
        s_mem: Tensor,
        r_idx: int,
        *,
        gate_override: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Single silent-reasoning step from read-only s_mem [B, H, d_h, d_h] + writable W_r.

        Uses the exact same MacroRDFLv05Core modules (gdn.read_only, bdh, ffn, halt_gate, phase_proj)
        plus optional O(1) Delta-Householder scratchpad W_r. When g_step == 0, both h and W_r
        satisfy exact no-op preservation (h_next == h, W_next == W).
        """
        b = h.shape[0]
        dt = h.dtype
        h_2d = h.unsqueeze(1) if h.ndim == 2 else h
        h_q = rms_norm_tau(h_2d, tau=1.0)
        ph = self.phase_proj(self._phase_vec(r_idx, h.device, dt))
        u_r = rms_norm_tau(h_2d + ph + self.e_refine.to(dt), tau=1.0)

        # Memory reads using phase-aware query u_r (zero RMSNorm division-by-zero on W=0)
        qs = F.normalize(
            self.gdn.q_read(u_r)
            .to(dt)
            .reshape(b, 1, self.num_heads, self.head_dim)
            .transpose(1, 2),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        o_s = torch.matmul(s_mem.to(dt), qs[:, :, 0, :, None]).squeeze(-1).reshape(b, 1, self.d_model)

        qw = F.normalize(
            self.qw_proj(u_r)
            .to(dt)
            .reshape(b, 1, self.num_heads, self.head_dim)
            .transpose(1, 2),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        o_w = (
            torch.matmul(w_scratch.to(dt), qw[:, :, 0, :, None])
            .squeeze(-1)
            .reshape(b, 1, self.d_model)
        )

        # Phase-independent halting gate so once halted, OOD R_extra never reopens the gate
        if gate_override is None:
            qs_h = F.normalize(
                self.gdn.q_read(h_q)
                .to(dt)
                .reshape(b, 1, self.num_heads, self.head_dim)
                .transpose(1, 2),
                p=2,
                dim=-1,
                eps=1e-6,
            )
            qw_h = F.normalize(
                self.qw_proj(h_q)
                .to(dt)
                .reshape(b, 1, self.num_heads, self.head_dim)
                .transpose(1, 2),
                p=2,
                dim=-1,
                eps=1e-6,
            )
            o_s_h = (
                torch.matmul(s_mem.to(dt), qs_h[:, :, 0, :, None])
                .squeeze(-1)
                .reshape(b, 1, self.d_model)
            )
            o_w_h = (
                torch.matmul(w_scratch.to(dt), qw_h[:, :, 0, :, None])
                .squeeze(-1)
                .reshape(b, 1, self.d_model)
            )
            o_halt = torch.tanh(o_s_h + o_w_h)
            g_step = torch.sigmoid(self.halt_gate(torch.cat([h_q, o_halt], dim=-1)))
        else:
            g_step = torch.full((b, 1, 1), float(gate_override), device=h.device, dtype=dt)

        v_cat = torch.cat([u_r, o_s, o_w], dim=-1)
        delta_r = self.reason_ffn_out(F.silu(self.reason_ffn_in(v_cat)))
        step_gate = torch.sigmoid(self.reason_gate(v_cat))
        h_cand = rms_norm_tau(h_2d + step_gate * delta_r, tau=1.0)
        h_next = h_2d + g_step * (h_cand - h_2d)

        # Delta-Householder scratchpad update gated by g_step
        kv_in = torch.cat([u_r[:, 0], o_s[:, 0]], dim=-1)
        kw = F.normalize(
            self.kw_proj(kv_in).to(dt).reshape(b, self.num_heads, self.head_dim),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        vw = torch.tanh(self.vw_proj(kv_in).to(dt).reshape(b, self.num_heads, self.head_dim))
        ab = self.abw_proj(u_r[:, 0]).to(dt)
        alpha_w = (0.95 + 0.049 * torch.sigmoid(ab[:, : self.num_heads]))[..., None, None]
        beta_w = (2.0 * torch.sigmoid(ab[:, self.num_heads :]))[..., None]
        wk = torch.matmul(w_scratch.to(dt), kw[..., None]).squeeze(-1)
        w_cand = alpha_w * w_scratch.to(dt) + torch.matmul(
            (beta_w * (vw - alpha_w.squeeze(-1) * wk))[..., None], kw[..., None, :]
        )
        g_mat = g_step[:, 0, :, None, None]
        w_next = w_scratch.to(dt) + g_mat * (w_cand - w_scratch.to(dt))
        return h_next[:, 0], w_next, g_step[:, 0]

    def _run_segment(
        self,
        h_seg: Tensor,
        n_loops: int,
        gdn_st: tuple[Tensor, Tensor | None],
        single_step: bool,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        k_bptt: int | None = None,
        collect_traj: bool = False,
        gate_override: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor], Tensor, Tensor, list[Tensor]]:
        dt = h_seg.dtype
        h_in = self.n_in(h_seg)
        anchor = rms_norm_tau(h_in, tau=1.0)
        u0 = rms_norm_tau(h_in + self.e_write.to(dt), tau=0.7)
        y0, new_gdn_st, cache = self.gdn.write_and_read(u0, gdn_st, single_step=single_step)
        yb0 = self.bdh(u0, y0) if self.bdh is not None else 0.0
        v0 = rms_norm_tau(h_in + y0 + yb0, tau=0.7)
        h1 = rms_norm_tau(h_in + y0 + yb0 + self.ffn(v0), tau=1.0)

        traj: list[Tensor] = [anchor, h1] if collect_traj else []
        h_cur = h1
        h_prev = h_in
        r_used = torch.full(h_seg.shape[:2], n_loops, dtype=torch.long, device=h_seg.device)
        active = torch.ones(h_seg.shape[:2], dtype=torch.bool, device=h_seg.device)
        g_step_last = torch.ones(h_seg.shape[:2], dtype=dt, device=h_seg.device)

        detach_until = (n_loops - k_bptt) if (k_bptt is not None and n_loops > k_bptt) else 0
        for r_idx in range(1, n_loops):
            if detach_until > 0 and r_idx == detach_until:
                h_cur = h_cur.detach()
            if self.reasoning_mode == "adfp":
                h_next = self._refine_step_adfp(h_cur, anchor, cache)
                g_step_val = (h_next - h_cur).norm(dim=-1) / (h_cur.norm(dim=-1) + 1e-8)
            else:
                h_next, g_step = self._refine_step_halting(
                    h_cur, cache, r_idx, gate_override=gate_override
                )
                g_step_val = g_step.squeeze(-1)

            g_step_last = torch.where(active, g_step_val, g_step_last)

            if eps_halt > 0.0 or eps_conv > 0.0:
                stop_cond = torch.zeros_like(active)
                if eps_halt > 0.0:
                    stop_cond = stop_cond | (g_step_val < eps_halt)
                if eps_conv > 0.0:
                    rel_diff = (h_next - h_cur).norm(dim=-1) / (h_cur.norm(dim=-1) + 1e-8)
                    stop_cond = stop_cond | (rel_diff < eps_conv)
                newly_stopped = active & stop_cond
                r_used = torch.where(newly_stopped, torch.full_like(r_used, r_idx), r_used)
                keep_updating = active & (~newly_stopped)
                h_prev = torch.where(keep_updating.unsqueeze(-1), h_cur, h_prev)
                h_cur = torch.where(keep_updating.unsqueeze(-1), h_next, h_cur)
                active = active & (~newly_stopped)
                if collect_traj:
                    traj.append(h_cur)
                if not bool(active.any().item()):
                    break
            else:
                h_prev = h_cur
                h_cur = h_next
                if collect_traj:
                    traj.append(h_cur)

        delta_conv = (h_cur - h_prev).norm(dim=-1) / (h_cur.norm(dim=-1) + 1e-6)
        return h1, h_cur, delta_conv, new_gdn_st, r_used, g_step_last, traj

    def forward_macro(
        self,
        u: Tensor,
        s_macro: Tensor,
        conv_macro: Tensor | None,
        slot_state: Tensor,
        macro_pos: int,
        n_loops: int,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        k_bptt: int | None = None,
        collect_traj: bool = False,
        gate_override: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor, Tensor, Tensor, list[Tensor]]:
        m_len = u.shape[1]
        dt = _calc_dtype(u.dtype)
        gdn_st: tuple[Tensor, Tensor | None] = (s_macro.to(dt), conv_macro)
        slot_st = slot_state.to(device=u.device, dtype=dt)

        r1_chunks: list[Tensor] = []
        fin_chunks: list[Tensor] = []
        dc_chunks: list[Tensor] = []
        ru_chunks: list[Tensor] = []
        gs_chunks: list[Tensor] = []
        traj_chunks: list[list[Tensor]] = []

        offset = 0
        while offset < m_len:
            pos = macro_pos + offset
            step_len = min(m_len - offset, self.chunk_size - (pos % self.chunk_size))
            h_seg = u[:, offset : offset + step_len]
            if self.boundary is not None:
                h_seg = self.boundary.apply_feedback(h_seg, slot_st)
            h1_seg, hfin_seg, dc_seg, gdn_st, ru_seg, gs_seg, seg_traj = self._run_segment(
                h_seg,
                n_loops=n_loops,
                gdn_st=gdn_st,
                single_step=(step_len == 1),
                eps_conv=eps_conv,
                eps_halt=eps_halt,
                k_bptt=k_bptt,
                collect_traj=collect_traj,
                gate_override=gate_override,
            )
            # CRITICAL P1<->K3 INVARIANT: commit boundary ONLY from r=0 output h1_seg[:, -1]!
            if self.boundary is not None and ((pos + step_len) % self.chunk_size) == 0:
                slot_st = self.boundary.commit_boundary(slot_st, h1_seg[:, -1])

            r1_chunks.append(h1_seg)
            fin_chunks.append(hfin_seg)
            dc_chunks.append(dc_seg)
            ru_chunks.append(ru_seg)
            gs_chunks.append(gs_seg)
            if collect_traj:
                if not traj_chunks:
                    traj_chunks = [[] for _ in range(len(seg_traj))]
                for ri in range(min(len(traj_chunks), len(seg_traj))):
                    traj_chunks[ri].append(seg_traj[ri])
            offset += step_len

        h_r1 = torch.cat(r1_chunks, dim=1)
        h_fin = torch.cat(fin_chunks, dim=1)
        delta_conv = torch.cat(dc_chunks, dim=1)
        r_used = torch.cat(ru_chunks, dim=1)
        g_step_last = torch.cat(gs_chunks, dim=1)
        h_traj = [torch.cat(ch, dim=1) for ch in traj_chunks] if collect_traj else []
        return h_fin, h_r1, delta_conv, gdn_st[0], gdn_st[1], slot_st, r_used, g_step_last, h_traj


class OrthogonalSlowGDN(nn.Module):
    """Stage 3 (P4): Surprise-Gated Zero-Drift Orthogonalized Slow Memory (M1 / M2)."""

    def __init__(self, cfg: RDFLv05Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.macro_dim
        self.num_heads = cfg.macro_heads
        self.head_dim = self.d_model // self.num_heads
        self.chunk_size = cfg.chunk_size
        self.mode: SlowMode = cfg.slow_mode

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.gate_proj = nn.Linear(self.d_model, 2 * self.num_heads, bias=True)
        self.g_out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_norm = SoftRMSNorm(self.head_dim, init_tau=0.25, learnable=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.gamma_slow = nn.Parameter(
            torch.full((self.d_model,), math.atanh(cfg.slow_gamma_init))
        )

        for mod in (self.q_proj, self.k_proj, self.v_proj, self.g_out, self.out_proj):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.q_proj.weight.copy_(self.k_proj.weight)
        nn.init.zeros_(self.gate_proj.weight)
        with torch.no_grad():
            self.gate_proj.bias[: self.num_heads].fill_(cfg.slow_init_alpha_bias)
            self.gate_proj.bias[self.num_heads :].zero_()

    def init_state(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        dt = _calc_dtype(dtype)
        s_slow = torch.zeros(
            batch, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dt
        )
        c_commit = torch.zeros_like(s_slow)
        c_pending = torch.zeros_like(s_slow)
        return s_slow, (c_commit, c_pending)

    def _compute_projector(self, c_commit: Tensor, dt: torch.dtype) -> Tensor:
        tr = c_commit.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        c_tilde = (float(self.head_dim) * c_commit) / (tr + self.cfg.slow_tau_c)
        eye = torch.eye(self.head_dim, device=c_commit.device, dtype=dt)
        mat = eye + self.cfg.slow_mu * c_tilde
        p_perp = torch.linalg.solve(mat, eye.expand_as(mat))
        return cast(Tensor, 0.5 * (p_perp + p_perp.transpose(-1, -2)))

    def forward_segment(
        self,
        h_fast: Tensor,
        g_surp: Tensor,
        s_slow: Tensor | None,
        c_slow: tuple[Tensor, Tensor] | None,
        macro_pos: int,
        *,
        update_slow: bool,
        return_delta: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        b, m_len, d = h_fast.shape
        dt = _calc_dtype(h_fast.dtype)
        if s_slow is None or c_slow is None:
            s_init, c_init = self.init_state(b, h_fast.device, h_fast.dtype)
            s_cur = s_init if s_slow is None else s_slow.to(dt)
            c_pair = c_init if c_slow is None else (c_slow[0].to(dt), c_slow[1].to(dt))
        else:
            s_cur = s_slow.to(dt)
            c_pair = (c_slow[0].to(dt), c_slow[1].to(dt))

        if m_len == 0:
            zero_or_fast = torch.zeros_like(h_fast) if return_delta else h_fast
            return zero_or_fast, s_cur, c_pair

        q = F.normalize(
            F.relu(self.q_proj(h_fast).to(dt))
            .reshape(b, m_len, self.num_heads, self.head_dim)
            .transpose(1, 2),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        k = F.normalize(
            F.relu(self.k_proj(h_fast).to(dt))
            .reshape(b, m_len, self.num_heads, self.head_dim)
            .transpose(1, 2),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        v = (
            self.v_proj(h_fast)
            .to(dt)
            .reshape(b, m_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        a_logit, b_logit = self.gate_proj(h_fast).to(dt).chunk(2, dim=-1)
        log_alpha_base = F.logsigmoid(a_logit).transpose(1, 2)
        beta_base = torch.sigmoid(b_logit).transpose(1, 2)

        g_gate = g_surp.to(dt) if update_slow else torch.zeros_like(g_surp, dtype=dt)
        log_alpha_eff = g_gate[:, None, :] * log_alpha_base
        beta_eff = g_gate[:, None, :] * beta_base

        out_chunks: list[Tensor] = []
        offset = 0
        with torch.autocast(device_type=h_fast.device.type, enabled=False):
            while offset < m_len:
                cur_pos = macro_pos + offset
                step_len = min(m_len - offset, self.chunk_size - (cur_pos % self.chunk_size))
                sl = slice(offset, offset + step_len)
                q_c, k_c, v_c = q[:, :, sl], k[:, :, sl], v[:, :, sl]
                la_c, be_c, g_c = (
                    log_alpha_eff[:, :, sl],
                    beta_eff[:, :, sl],
                    g_gate[:, sl],
                )

                if self.mode == "m1":
                    if step_len == 1:
                        alpha_t = la_c[:, :, 0].exp()[..., None, None]
                        b_t = be_c[:, :, 0][..., None]
                        kt, vt, qt = k_c[:, :, 0], v_c[:, :, 0], q_c[:, :, 0]
                        s_dec = alpha_t * s_cur
                        u_t = b_t * (vt - torch.matmul(s_dec, kt[..., None]).squeeze(-1))
                        s_cur = s_dec + torch.matmul(u_t[..., None], kt[..., None, :])
                        o_c = torch.matmul(s_cur, qt[..., None]).unsqueeze(2).squeeze(-1)
                    else:
                        o_c, s_cur, _, _ = _chunk_delta_wy_core(q_c, k_c, v_c, la_c, be_c, s_cur)
                else:
                    c_commit, c_pending = c_pair
                    p_perp = self._compute_projector(c_commit, dt)
                    p_c = torch.matmul(k_c, p_perp)
                    kp = (k_c * p_c).sum(dim=-1)
                    eta_c = be_c / (kp + self.cfg.slow_eps_reg)
                    if step_len == 1:
                        alpha_t = la_c[:, :, 0].exp()[..., None, None]
                        eta_t = eta_c[:, :, 0][..., None]
                        kt, pt, vt, qt = k_c[:, :, 0], p_c[:, :, 0], v_c[:, :, 0], q_c[:, :, 0]
                        s_dec = alpha_t * s_cur
                        u_t = eta_t * (vt - torch.matmul(s_dec, kt[..., None]).squeeze(-1))
                        s_cur = s_dec + torch.matmul(u_t[..., None], pt[..., None, :])
                        o_c = torch.matmul(s_cur, qt[..., None]).unsqueeze(2).squeeze(-1)
                    else:
                        o_c, s_cur = _chunk_orth_delta_wy_core(
                            q_c, k_c, p_c, v_c, la_c, eta_c, s_cur
                        )

                    if update_slow:
                        k_weighted = k_c * g_c[:, None, :, None]
                        c_pending = c_pending + torch.matmul(k_weighted.transpose(-1, -2), k_c)
                        if ((cur_pos + step_len) % self.chunk_size) == 0:
                            c_commit = self.cfg.slow_gamma_c * c_commit + c_pending
                            c_pending = torch.zeros_like(c_pending)
                    c_pair = (c_commit, c_pending)

                out_chunks.append(o_c)
                offset += step_len

        y_heads = torch.cat(out_chunks, dim=2).transpose(1, 2)
        y_norm = self.o_norm(y_heads).reshape(b, m_len, d).to(h_fast.dtype)
        out_gate = F.silu(self.g_out(h_fast))
        y_slow = self.out_proj(y_norm * out_gate)
        delta_slow = torch.tanh(self.gamma_slow).to(h_fast.dtype) * y_slow
        if return_delta:
            return delta_slow, s_cur, c_pair
        return h_fast + delta_slow, s_cur, c_pair


# ---------------------------------------------------------------------------
# Unified v0.5 Config, State, and Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RDFLv05Config:
    """Unified RDFL v0.5 configuration."""

    vocab_size: int = 256
    d_model: int | None = None
    # Stage 0 (P3 + P2: 1-byte Micro-DeltaProduct)
    d_byte: int = 64
    micro_heads: int = 4
    n_h: int = 4
    micro_conv_init: float = 0.05
    ngram_buckets: tuple[int, int, int, int] = (2048, 4096, 4096, 4096)
    ngram_dim: int = 16
    ngram_alpha_init: float = 0.008
    # Stage 1 (P3 + P1 + StepReasoner: Fixed-shape P-byte Macro RDFL)
    patch_size: int = 4
    patch_bottleneck: int = 48
    d_macro: int = 64
    macro_heads: int = 4
    ffn_macro: int = 128
    num_loops: int = 2
    chunk_size: int = 64
    c_sn: float = 1.35
    c_sn_out: float = 0.85
    use_bdh: bool = True
    boundary_mode: str = "rlt2_s1"
    reasoning_mode: ReasoningMode = "step_halting"
    halt_bias_init: float = 3.5
    # Stage 2 & 3 (P4: Causal Surprise Gate & M2 OWM-GDN)
    slow_mode: SlowMode = "m2"
    msp_temp: float = 1.0
    tau_delta: float = 0.5
    nll_low: float = 4.0
    nll_high: float = 6.5
    w_nll: float = 0.72
    w_msp: float = 0.18
    w_p1: float = 0.10
    tau_0: float = 0.25
    tau_surp: float = 0.45
    kappa_surp: float = 10.0
    slow_init_alpha_bias: float = 7.6
    slow_mu: float = 25.0
    slow_eps_reg: float = 0.10
    slow_gamma_c: float = 0.999
    slow_tau_c: float = 1.0
    slow_gamma_init: float = 0.03
    # Stage 4 (P3 Byte Decoder + K3 + MTP-3)
    dec_ffn: int = 64
    mtp_heads: int = 3
    mtp_divisor: int = 2
    mtp_stop_grad: bool = False

    @property
    def byte_dim(self) -> int:
        return self.d_model if self.d_model is not None else self.d_byte

    @property
    def macro_dim(self) -> int:
        return self.d_model if self.d_model is not None else self.d_macro


@dataclass
class RDFLv05State:
    """Strictly O(1) constant-size inference state across T, R_test, n_h, and tasks."""

    s_fast: Tensor
    conv_state: Tensor
    s_macro: Tensor
    conv_macro: Tensor | None
    slot_state: Tensor
    chunk_sum: Tensor | None = None
    s_slow: Tensor | None = None
    c_slow: tuple[Tensor, Tensor] | None = None
    prev_fast_logits: Tensor | None = None
    stem_state: tuple[Tensor, ...] | None = None
    pos: int = 0


class RDFLv05Model(nn.Module):
    """Unified Pure RDFL v0.5 Model (0 Attention, 0 KV-cache, 0 BPE, O(1) state)."""

    def __init__(self, cfg: RDFLv05Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else RDFLv05Config()
        if self.cfg.num_loops < 1:
            raise ValueError("num_loops must be >= 1")
        if self.cfg.patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if self.cfg.slow_mode not in _VALID_SLOW_MODES:
            raise ValueError(f"slow_mode must be in {_VALID_SLOW_MODES}")
        if self.cfg.reasoning_mode not in _VALID_REASONING_MODES:
            raise ValueError(f"reasoning_mode must be in {_VALID_REASONING_MODES}")

        d_b = self.cfg.byte_dim
        d_m = self.cfg.macro_dim
        p_sz = self.cfg.patch_size

        # Stage 0: 1-byte embedding + MS-Ngram + Micro-DeltaProduct
        self.byte_emb = nn.Embedding(self.cfg.vocab_size, d_b)
        nn.init.normal_(self.byte_emb.weight, mean=0.0, std=0.08)
        self.ngram = MSNgramHashStem(
            d_b,
            buckets=self.cfg.ngram_buckets,
            d_h=self.cfg.ngram_dim,
            alpha_init=self.cfg.ngram_alpha_init,
        )
        self.n_micro = RMSNorm(d_b)
        self.micro_gdn = MicroDeltaProductGDNLayer(
            d_b,
            num_heads=self.cfg.micro_heads,
            n_h=self.cfg.n_h,
            chunk_size=self.cfg.chunk_size,
            conv_init=self.cfg.micro_conv_init,
        )

        # Stage 1: Fixed-shape patch packer + Macro RDFL v0.5 Core
        self.n_pack = RMSNorm(p_sz * d_b)
        p1 = nn.Linear(p_sz * d_b, self.cfg.patch_bottleneck, bias=False)
        p2 = nn.Linear(self.cfg.patch_bottleneck, d_m, bias=False)
        nn.init.normal_(p1.weight, mean=0.0, std=1.0 / math.sqrt(float(p_sz * d_b)))
        nn.init.normal_(p2.weight, mean=0.0, std=0.5 / math.sqrt(float(self.cfg.patch_bottleneck)))
        self.patch_proj = nn.Sequential(p1, p2)
        self.reason_in_proj: nn.Module = (
            nn.Identity() if d_b == d_m else nn.Linear(d_b, d_m, bias=False)
        )

        self.macro = MacroRDFLv05Core(
            d_model=d_m,
            num_heads=self.cfg.macro_heads,
            ffn_dim=self.cfg.ffn_macro,
            num_loops=self.cfg.num_loops,
            chunk_size=self.cfg.chunk_size,
            c_sn=self.cfg.c_sn,
            c_sn_out=self.cfg.c_sn_out,
            use_bdh=self.cfg.use_bdh,
            boundary_mode=self.cfg.boundary_mode,
            reasoning_mode=self.cfg.reasoning_mode,
            halt_bias_init=self.cfg.halt_bias_init,
        )
        self.h_bos = nn.Parameter(torch.randn(1, 1, d_m) * 0.02)

        # Stage 3: Orthogonal Slow Memory (outside P1 loop)
        self.slow_mem = OrthogonalSlowGDN(self.cfg)

        # Stage 4: Direct single-pass residual byte decoder + untied head + K3 + MTP-3
        self.n_ctx = RMSNorm(d_m)
        self.ctx_proj = nn.Linear(d_m, d_b, bias=False)
        self.ctx_proj_slow = nn.Linear(d_m, d_b, bias=False)
        self.slot_emb = nn.Parameter(torch.randn(p_sz, d_b) * 0.02)
        self.n_dec = RMSNorm(d_b)
        self.dec_ffn = SwiGLU(d_b, self.cfg.dec_ffn)
        self.final_norm = RMSNorm(d_b)
        self.head = nn.Linear(d_b, self.cfg.vocab_size, bias=False)
        self.register_buffer("_slot_eye", torch.eye(p_sz), persistent=False)

        self.k3_adapter = BottleneckAdapter(d_b, divisor=4)
        self.mtp_heads = nn.ModuleList(
            [BottleneckAdapter(d_b, divisor=self.cfg.mtp_divisor) for _ in range(self.cfg.mtp_heads)]
        )

        nn.init.normal_(self.ctx_proj.weight, mean=0.0, std=0.5 / math.sqrt(float(d_m)))
        nn.init.normal_(self.ctx_proj_slow.weight, mean=0.0, std=0.5 / math.sqrt(float(d_m)))
        nn.init.normal_(self.dec_ffn.up.weight, mean=0.0, std=0.5 / math.sqrt(float(d_b)))
        nn.init.normal_(self.dec_ffn.out.weight, mean=0.0, std=0.5 / math.sqrt(float(self.cfg.dec_ffn)))
        nn.init.normal_(self.head.weight, mean=0.0, std=0.08)

    @torch.no_grad()
    def update_spectral_vectors(self) -> None:
        """Update power-iteration vectors on all SoftSpectralLinear layers after optimizer.step()."""
        for mod in self.modules():
            if isinstance(mod, SoftSpectralLinear):
                mod.update_spectral_vec()

    def _init_state(
        self, b: int, device: torch.device, w_dtype: torch.dtype, state: RDFLv05State | None
    ) -> RDFLv05State:
        if state is not None:
            return state
        dt = _calc_dtype(w_dtype)
        d_b = self.cfg.byte_dim
        d_m = self.cfg.macro_dim
        p_sz = self.cfg.patch_size
        pad = self.micro_gdn.short_conv.kernel_size - 1
        s_fast = self.micro_gdn.init_s(b, device, dt, None)
        conv_state = torch.zeros(b, d_b, pad, device=device, dtype=w_dtype)
        s_macro = torch.zeros(
            b, self.cfg.macro_heads, self.macro.head_dim, self.macro.head_dim, device=device, dtype=dt
        )
        conv_macro = torch.zeros(b, d_m, pad, device=device, dtype=w_dtype)
        slot_state = (
            self.macro.boundary.initial_state(b, dtype=dt).to(device=device)
            if self.macro.boundary is not None
            else torch.zeros(b, 1, d_m, device=device, dtype=dt)
        )
        s_slow, c_slow = self.slow_mem.init_state(b, device, w_dtype)
        prev_fast_logits = torch.zeros(b, self.cfg.vocab_size, device=device, dtype=dt)

        prev_bytes = torch.full((b, 5), 256, dtype=torch.long, device=device)
        patch_ring = torch.zeros(b, p_sz, d_b, device=device, dtype=w_dtype)
        h_bos = self.h_bos[:, 0].to(device=device, dtype=w_dtype).expand(b, -1).clone()
        delta_slow_cur = torch.zeros(b, d_m, device=device, dtype=w_dtype)
        patch_surp_acc = torch.zeros(b, 2, device=device, dtype=dt)
        stem_state = (prev_bytes, patch_ring, h_bos, delta_slow_cur, h_bos.clone(), patch_surp_acc)
        return RDFLv05State(
            s_fast=s_fast,
            conv_state=conv_state,
            s_macro=s_macro,
            conv_macro=conv_macro,
            slot_state=slot_state,
            chunk_sum=None,
            s_slow=s_slow,
            c_slow=c_slow,
            prev_fast_logits=prev_fast_logits,
            stem_state=stem_state,
            pos=0,
        )

    def _decode_fast_byte(
        self, c_seq: Tensor, h_fast_ctx: Tensor, slot_indices: Tensor | int
    ) -> Tensor:
        """Compute [B, L, d_b] single-pass fast pre-norm byte representation (1 dec_ffn call)."""
        d_b = self.cfg.byte_dim
        z_ctx = self.ctx_proj(self.n_ctx(h_fast_ctx))
        if isinstance(slot_indices, int):
            slot_v = self.slot_emb[slot_indices].to(c_seq.dtype).view(1, 1, d_b)
        else:
            slot_v = self.slot_emb[slot_indices].to(c_seq.dtype).unsqueeze(0)
        y_base = c_seq + z_ctx + slot_v
        return cast(Tensor, y_base + self.dec_ffn(self.n_dec(y_base)))

    def _decode_k3_detached(
        self, c_seq: Tensor, h_r1_ctx: Tensor, slot_indices: Tensor | int
    ) -> Tensor:
        """Detached K3 (OY-1) early-exit logits with 0.000 gradient tax on core/head."""
        d_b = self.cfg.byte_dim
        c_d = c_seq.detach()
        hr_d = h_r1_ctx.detach()
        n_ctx_w = self.n_ctx.weight.detach()
        hn_ctx = _apply_rmsnorm(hr_d, n_ctx_w, self.n_ctx.eps)
        z_ctx = F.linear(hn_ctx, self.ctx_proj.weight.detach())
        if isinstance(slot_indices, int):
            slot_v = self.slot_emb[slot_indices].detach().to(c_d.dtype).view(1, 1, d_b)
        else:
            slot_v = self.slot_emb[slot_indices].detach().to(c_d.dtype).unsqueeze(0)
        y = c_d + z_ctx + slot_v
        n_dec_hn = _apply_rmsnorm(y, self.n_dec.weight.detach(), self.n_dec.eps)
        a, b = F.linear(n_dec_hn, self.dec_ffn.up.weight.detach()).chunk(2, dim=-1)
        y_dec = y + F.linear(F.silu(a) * b, self.dec_ffn.out.weight.detach())
        z_k3 = self.k3_adapter(y_dec)
        hn_k3 = _apply_rmsnorm(z_k3, self.final_norm.weight.detach(), self.final_norm.eps)
        return F.linear(hn_k3, self.head.weight.detach())

    def _gate_from_patch_stats(
        self,
        u_nll_p: Tensor,
        u_msp_p: Tensor,
        delta_conv_p: Tensor,
        *,
        macro_pos: int,
        update_slow: bool,
        gate_mode: GateMode,
        random_gate_seed: int | None,
        random_gate_prob: float,
    ) -> Tensor:
        if gate_mode not in _VALID_GATE_MODES:
            raise ValueError(f"gate_mode must be in {_VALID_GATE_MODES}")
        dt = u_nll_p.dtype
        u_p1 = torch.tanh(delta_conv_p.detach().to(dt) / self.cfg.tau_delta)
        u_tot = self.cfg.w_nll * u_nll_p + self.cfg.w_msp * u_msp_p + self.cfg.w_p1 * u_p1
        if (not update_slow) or gate_mode == "off":
            return torch.zeros_like(u_tot)
        if gate_mode == "always_one":
            return torch.ones_like(u_tot)
        if gate_mode == "random_matched":
            base_seed = 0 if random_gate_seed is None else int(random_gate_seed)
            n_m = u_tot.shape[1]
            draws: list[Tensor] = []
            gen = torch.Generator(device=u_tot.device)
            for m_off in range(n_m):
                patch_idx = macro_pos + m_off
                seed_m = ((base_seed * 1000003) ^ ((patch_idx + 1) * 2654435761)) & 0x7FFFFFFF
                gen.manual_seed(seed_m)
                draws.append(
                    (
                        torch.rand((u_tot.shape[0], 1), generator=gen, device=u_tot.device, dtype=dt)
                        < random_gate_prob
                    ).to(dt)
                )
            return torch.cat(draws, dim=1) if draws else torch.zeros_like(u_tot)
        ramp = ((u_tot - self.cfg.tau_0) / max(1e-6, 1.0 - self.cfg.tau_0)).clamp(0.0, 1.0)
        return torch.sigmoid(self.cfg.kappa_surp * (u_tot - self.cfg.tau_surp)) * ramp

    def _step_from_c(
        self,
        x_t: Tensor,
        c_t: Tensor,
        st: RDFLv05State,
        new_pb: Tensor,
        new_s_fast: Tensor,
        new_conv_fast: Tensor,
        n_loops: int,
        *,
        eps_conv: float,
        eps_halt: float,
        update_slow: bool,
        gate_mode: GateMode,
        random_gate_seed: int | None,
        random_gate_prob: float,
        gate_override: float | None,
    ) -> tuple[Tensor, Tensor, Tensor, RDFLv05State, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        b = x_t.shape[0]
        p_sz = self.cfg.patch_size
        d_b = self.cfg.byte_dim
        dt = _calc_dtype(c_t.dtype)

        assert st.stem_state is not None
        _pb, patch_ring, h_fast_cur, delta_slow_cur, h_r1_cur, surp_acc = st.stem_state
        slot_idx = st.pos % p_sz
        slot_mask = cast(Tensor, self._slot_eye)[slot_idx].to(c_t.dtype).view(1, p_sz, 1)
        new_ring = patch_ring * (1.0 - slot_mask) + c_t * slot_mask

        s_macro = st.s_macro
        conv_macro: Tensor | None = st.conv_macro
        slot_st = st.slot_state
        s_slow = st.s_slow
        c_slow = st.c_slow
        r_used_t = torch.full((b, 1), n_loops, dtype=torch.long, device=x_t.device)
        delta_conv_t = torch.zeros((b, 1), dtype=dt, device=x_t.device)
        g_step_t = torch.ones((b, 1), dtype=dt, device=x_t.device)

        if slot_idx == p_sz - 1:
            u_m = self.patch_proj(self.n_pack(new_ring.reshape(b, 1, p_sz * d_b)))
            macro_pos = st.pos // p_sz
            h_fm, h_rm, dc_m, s_macro, conv_macro, slot_st, ru_m, gs_m, _ = self.macro.forward_macro(
                u_m,
                s_macro,
                conv_macro,
                slot_st,
                macro_pos,
                n_loops,
                eps_conv=eps_conv,
                eps_halt=eps_halt,
                gate_override=gate_override,
            )
            h_fast_cur = h_fm[:, 0]
            h_r1_cur = h_rm[:, 0]
            delta_conv_t = dc_m.to(dt)
            r_used_t = ru_m
            g_step_t = gs_m.to(dt)

        h_fast_byte_t = self._decode_fast_byte(c_t, h_fast_cur.unsqueeze(1), slot_idx)
        logits_fast_t = self.head(self.final_norm(h_fast_byte_t))
        logits_r1_t = self._decode_k3_detached(c_t, h_r1_cur.unsqueeze(1), slot_idx)

        prev_lg = st.prev_fast_logits
        assert prev_lg is not None
        nll_t = -F.log_softmax(prev_lg.detach().to(dt), dim=-1).gather(-1, x_t.long()).squeeze(-1)
        u_nll_t = (
            (nll_t - self.cfg.nll_low) / max(1e-6, self.cfg.nll_high - self.cfg.nll_low)
        ).clamp(0.0, 1.0)
        u_msp_t = _causal_msp_uncertainty(logits_fast_t[:, 0].detach(), temperature=self.cfg.msp_temp)

        new_acc = surp_acc.to(dt) + torch.stack([u_nll_t, u_msp_t], dim=-1)
        g_surp_t = torch.zeros((b, 1), dtype=dt, device=x_t.device)
        if slot_idx == p_sz - 1:
            macro_pos = st.pos // p_sz
            u_nll_p = new_acc[:, 0:1] / float(p_sz)
            u_msp_p = new_acc[:, 1:2] / float(p_sz)
            g_surp_m = self._gate_from_patch_stats(
                u_nll_p,
                u_msp_p,
                delta_conv_t,
                macro_pos=macro_pos,
                update_slow=update_slow,
                gate_mode=gate_mode,
                random_gate_seed=random_gate_seed,
                random_gate_prob=random_gate_prob,
            )
            g_surp_t = g_surp_m
            delta_slow_m, s_slow, c_slow = self.slow_mem.forward_segment(
                h_fast_cur.unsqueeze(1),
                g_surp_m,
                s_slow,
                c_slow,
                macro_pos,
                update_slow=update_slow,
                return_delta=True,
            )
            delta_slow_cur = delta_slow_m[:, 0]
            new_acc = torch.zeros_like(new_acc)

        h_dec_t = h_fast_byte_t + self.ctx_proj_slow(delta_slow_cur.unsqueeze(1))
        logits_t = self.head(self.final_norm(h_dec_t))
        new_st = RDFLv05State(
            s_fast=new_s_fast,
            conv_state=new_conv_fast,
            s_macro=s_macro,
            conv_macro=conv_macro,
            slot_state=slot_st,
            chunk_sum=None,
            s_slow=s_slow,
            c_slow=c_slow,
            prev_fast_logits=logits_fast_t[:, 0].detach().to(dt),
            stem_state=(new_pb, new_ring, h_fast_cur, delta_slow_cur, h_r1_cur, new_acc),
            pos=st.pos + 1,
        )
        return (
            logits_t,
            logits_r1_t,
            h_dec_t,
            new_st,
            g_surp_t,
            u_msp_t.unsqueeze(1),
            nll_t.unsqueeze(1),
            delta_conv_t,
            r_used_t,
            g_step_t,
            logits_fast_t,
        )

    def _forward_core(
        self,
        x: Tensor,
        state: RDFLv05State | None = None,
        num_loops: int | None = None,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        k_bptt: int | None = None,
        update_slow: bool = False,
        gate_mode: GateMode = "surprise",
        random_gate_seed: int | None = None,
        random_gate_prob: float = 0.25,
        gate_override: float | None = None,
        return_aux: bool = False,
        collect_traj: bool = False,
        collect_snapshots: bool = False,
        need_k3: bool = True,
        need_state: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, RDFLv05State | None, dict[str, Any] | None]:
        b, t = x.shape
        n_loops = self.cfg.num_loops if num_loops is None else int(num_loops)
        if n_loops < 1:
            raise ValueError("num_loops must be >= 1")

        w_dtype = self.byte_emb.weight.dtype
        dt = _calc_dtype(w_dtype)
        st_was_none = state is None
        st = self._init_state(b, x.device, w_dtype, state)
        assert st.stem_state is not None
        prev_bytes, _ring, _hf, _hfin, _hr1, _acc = st.stem_state

        e0 = self.byte_emb(x)
        delta_ng, new_pb, pb_snaps = self.ngram.embed_delta(
            x, prev_bytes, collect_snapshots=collect_snapshots
        )
        h_b0 = e0 + delta_ng
        nm = self.n_micro(h_b0)
        p_sz = self.cfg.patch_size
        d_b = self.cfg.byte_dim
        d_m = self.cfg.macro_dim

        if st.pos == 0 and not collect_snapshots:
            if t == 1:
                yg, (new_sf, new_cf) = self.micro_gdn.step(nm, (st.s_fast, st.conv_state))
            else:
                yg, (new_sf, new_cf) = self.micro_gdn(nm, (st.s_fast, st.conv_state))
            c = h_b0 + yg
            n_full = t // p_sz

            if n_full > 0:
                u_macro = self.patch_proj(
                    self.n_pack(c[:, : p_sz * n_full].reshape(b, n_full, p_sz * d_b))
                )
                h_fast_m, h_r1_m, dc_m, new_sm, new_cm, new_slot, r_used_m, gs_m, h_traj = (
                    self.macro.forward_macro(
                        u_macro,
                        st.s_macro,
                        st.conv_macro,
                        st.slot_state,
                        macro_pos=0,
                        n_loops=n_loops,
                        eps_conv=eps_conv,
                        eps_halt=eps_halt,
                        k_bptt=k_bptt,
                        collect_traj=collect_traj,
                        gate_override=gate_override,
                    )
                )
            else:
                h_fast_m = c.new_zeros(b, 0, d_m)
                h_r1_m = c.new_zeros(b, 0, d_m)
                dc_m = torch.zeros((b, 0), dtype=dt, device=x.device)
                new_sm, new_cm, new_slot = st.s_macro, st.conv_macro, st.slot_state
                r_used_m = torch.zeros((b, 0), dtype=torch.long, device=x.device)
                gs_m = torch.zeros((b, 0), dtype=dt, device=x.device)
                h_traj = []

            def _expand_macro(seq_m: Tensor, bos_vec: Tensor) -> Tensor:
                all_m = torch.cat([bos_vec.expand(b, -1, -1), seq_m], dim=1)
                pre_len = min(t, p_sz - 1)
                pre = all_m[:, 0:1].expand(b, pre_len, d_m)
                if t > p_sz - 1:
                    rem = t - (p_sz - 1)
                    rest = (
                        all_m[:, 1 : 1 + n_full]
                        .unsqueeze(2)
                        .expand(b, n_full, p_sz, d_m)
                        .reshape(b, n_full * p_sz, d_m)[:, :rem]
                    )
                    return torch.cat([pre, rest], dim=1)
                return pre

            h_bos_cast = self.h_bos.to(c.dtype)
            h_fast_ctx = _expand_macro(h_fast_m, h_bos_cast)
            slot_seq = torch.arange(t, device=x.device) % p_sz
            logits_r1 = (
                self._decode_k3_detached(c, _expand_macro(h_r1_m, h_bos_cast), slot_seq)
                if need_k3
                else c.new_zeros(b, 0, self.cfg.vocab_size)
            )

            h_fast_byte = self._decode_fast_byte(c, h_fast_ctx, slot_seq)
            logits_fast = self.head(self.final_norm(h_fast_byte))
            delta_slow_zero = c.new_zeros(b, 1, d_m)

            skip_slow = (
                st_was_none and ((not update_slow) or gate_mode == "off") and (not return_aux)
            )
            t_full = n_full * p_sz
            if skip_slow:
                logits = logits_fast
                h_dec = h_fast_byte
                delta_slow_m = None
                new_s_slow, new_c_slow = st.s_slow, st.c_slow
                if need_state and t_full < t:
                    assert st.prev_fast_logits is not None
                    if t_full > 0:
                        lg_prev_tail = logits_fast[:, t_full - 1 : t - 1].detach().to(dt)
                    elif t > 1:
                        lg_prev_tail = torch.cat(
                            [
                                st.prev_fast_logits.detach().to(dt).unsqueeze(1),
                                logits_fast[:, : t - 1].detach().to(dt),
                            ],
                            dim=1,
                        )
                    else:
                        lg_prev_tail = st.prev_fast_logits.detach().to(dt).unsqueeze(1)
                    nll_tail = (
                        -F.log_softmax(lg_prev_tail, dim=-1)
                        .gather(-1, x[:, t_full:].long().unsqueeze(-1))
                        .squeeze(-1)
                    )
                    u_nll_tail = (
                        (nll_tail - self.cfg.nll_low)
                        / max(1e-6, self.cfg.nll_high - self.cfg.nll_low)
                    ).clamp(0.0, 1.0)
                    u_msp_tail = _causal_msp_uncertainty(
                        logits_fast[:, t_full:].detach(), temperature=self.cfg.msp_temp
                    )
                    end_acc = torch.stack([u_nll_tail.sum(dim=-1), u_msp_tail.sum(dim=-1)], dim=-1)
                else:
                    end_acc = torch.zeros((b, 2), dtype=dt, device=x.device)
                g_surp_m = torch.zeros((b, n_full), dtype=dt, device=x.device)
                u_msp = torch.zeros((b, t), dtype=dt, device=x.device)
                nll_t = torch.zeros((b, t), dtype=dt, device=x.device)
            else:
                assert st.prev_fast_logits is not None
                lg_prev = torch.cat(
                    [
                        st.prev_fast_logits.detach().to(dt).unsqueeze(1),
                        logits_fast[:, :-1].detach().to(dt),
                    ],
                    dim=1,
                )
                nll_t = (
                    -F.log_softmax(lg_prev, dim=-1)
                    .gather(-1, x.long().unsqueeze(-1))
                    .squeeze(-1)
                )
                u_nll = (
                    (nll_t - self.cfg.nll_low) / max(1e-6, self.cfg.nll_high - self.cfg.nll_low)
                ).clamp(0.0, 1.0)
                u_msp = _causal_msp_uncertainty(logits_fast.detach(), temperature=self.cfg.msp_temp)

                if n_full > 0:
                    u_nll_p = u_nll[:, :t_full].reshape(b, n_full, p_sz).mean(dim=-1)
                    u_msp_p = u_msp[:, :t_full].reshape(b, n_full, p_sz).mean(dim=-1)
                    g_surp_m = self._gate_from_patch_stats(
                        u_nll_p,
                        u_msp_p,
                        dc_m,
                        macro_pos=0,
                        update_slow=update_slow,
                        gate_mode=gate_mode,
                        random_gate_seed=random_gate_seed,
                        random_gate_prob=random_gate_prob,
                    )
                else:
                    g_surp_m = torch.zeros((b, 0), dtype=dt, device=x.device)

                delta_slow_m, new_s_slow, new_c_slow = self.slow_mem.forward_segment(
                    h_fast_m,
                    g_surp_m,
                    st.s_slow,
                    st.c_slow,
                    macro_pos=0,
                    update_slow=update_slow,
                    return_delta=True,
                )
                delta_slow_ctx = _expand_macro(delta_slow_m, delta_slow_zero)
                h_dec = h_fast_byte + self.ctx_proj_slow(delta_slow_ctx)
                logits = self.head(self.final_norm(h_dec))

                if t_full < t:
                    end_acc = torch.stack(
                        [u_nll[:, t_full:].sum(dim=-1), u_msp[:, t_full:].sum(dim=-1)], dim=-1
                    )
                else:
                    end_acc = torch.zeros((b, 2), dtype=dt, device=x.device)

            new_state: RDFLv05State | None = None
            if need_state:
                start_ring = t_full if (t % p_sz) != 0 else max(0, t - p_sz)
                tail = c[:, start_ring:]
                tail_len = tail.shape[1]
                if tail_len < p_sz:
                    if t >= p_sz:
                        prev_patch = c[:, (n_full - 1) * p_sz : n_full * p_sz]
                        end_ring = torch.cat([tail, prev_patch[:, tail_len:]], dim=1)
                    else:
                        end_ring = torch.cat([tail, c.new_zeros(b, p_sz - tail_len, d_b)], dim=1)
                else:
                    end_ring = tail

                h_fast_end = h_fast_m[:, -1] if n_full > 0 else h_bos_cast[:, 0].expand(b, -1).clone()
                delta_slow_end = (
                    delta_slow_m[:, -1]
                    if (delta_slow_m is not None and n_full > 0)
                    else delta_slow_zero[:, 0].clone()
                )
                h_r1_end = h_r1_m[:, -1] if n_full > 0 else h_bos_cast[:, 0].expand(b, -1).clone()

                new_state = RDFLv05State(
                    s_fast=new_sf,
                    conv_state=new_cf,
                    s_macro=new_sm,
                    conv_macro=new_cm,
                    slot_state=new_slot,
                    chunk_sum=None,
                    s_slow=new_s_slow,
                    c_slow=new_c_slow,
                    prev_fast_logits=logits_fast[:, -1].detach().to(dt),
                    stem_state=(new_pb, end_ring, h_fast_end, delta_slow_end, h_r1_end, end_acc),
                    pos=t,
                )

            if return_aux:
                aux: dict[str, Any] = {
                    "g_surp": g_surp_m,
                    "u_msp": u_msp,
                    "nll_t": nll_t,
                    "delta_conv": dc_m.to(dt),
                    "r_used": r_used_m,
                    "g_step_last": gs_m.to(dt),
                    "logits_fast": logits_fast,
                }
                if collect_traj:
                    aux["h_traj"] = h_traj
                    aux["c_byte"] = c.detach()
                if collect_snapshots:
                    aux["snapshots"] = None
                return logits, logits_r1, h_dec, new_state, aux
            return logits, logits_r1, h_dec, new_state, None

        # Streaming / arbitrary split resume / snapshot path (symmetric return_aux keys & shapes!)
        yg, (new_sf, new_cf), micro_snaps = self.micro_gdn.forward_impl(
            nm, (st.s_fast, st.conv_state), collect_snapshots=True
        )
        c = h_b0 + yg
        cur = st
        outs: list[Tensor] = []
        r1_outs: list[Tensor] = []
        dec_outs: list[Tensor] = []
        snaps: list[RDFLv05State] = []
        g_surps: list[Tensor] = []
        u_msps: list[Tensor] = []
        nlls: list[Tensor] = []
        dcs: list[Tensor] = []
        rus: list[Tensor] = []
        gss: list[Tensor] = []
        lfasts: list[Tensor] = []

        prev_pos = st.pos
        for idx in range(t):
            pb_i = pb_snaps[idx] if collect_snapshots else new_pb
            sf_i, cf_i = micro_snaps[idx]
            lg_i, r1_i, dec_i, cur, gs_i, um_i, nl_i, dc_i, ru_i, gst_i, lf_i = self._step_from_c(
                x[:, idx : idx + 1],
                c[:, idx : idx + 1],
                cur,
                pb_i,
                sf_i,
                cf_i,
                n_loops,
                eps_conv=eps_conv,
                eps_halt=eps_halt,
                update_slow=update_slow,
                gate_mode=gate_mode,
                random_gate_seed=random_gate_seed,
                random_gate_prob=random_gate_prob,
                gate_override=gate_override,
            )
            outs.append(lg_i)
            r1_outs.append(r1_i)
            dec_outs.append(dec_i)
            if ((prev_pos + idx) % p_sz) == (p_sz - 1):
                g_surps.append(gs_i)
                dcs.append(dc_i)
                rus.append(ru_i)
                gss.append(gst_i)
            u_msps.append(um_i)
            nlls.append(nl_i)
            lfasts.append(lf_i)
            if collect_snapshots:
                snaps.append(cur)

        assert cur.stem_state is not None
        _, end_ring, hf_end, dslow_end, hr1_end, end_acc = cur.stem_state
        final_state = RDFLv05State(
            s_fast=new_sf,
            conv_state=new_cf,
            s_macro=cur.s_macro,
            conv_macro=cur.conv_macro,
            slot_state=cur.slot_state,
            chunk_sum=None,
            s_slow=cur.s_slow,
            c_slow=cur.c_slow,
            prev_fast_logits=cur.prev_fast_logits,
            stem_state=(new_pb, end_ring, hf_end, dslow_end, hr1_end, end_acc),
            pos=cur.pos,
        )
        logits = torch.cat(outs, dim=1)
        logits_r1 = torch.cat(r1_outs, dim=1)
        h_dec = torch.cat(dec_outs, dim=1)
        if return_aux:
            aux = {
                "g_surp": torch.cat(g_surps, dim=1)
                if g_surps
                else torch.zeros((b, 0), dtype=dt, device=x.device),
                "u_msp": torch.cat(u_msps, dim=1),
                "nll_t": torch.cat(nlls, dim=1),
                "delta_conv": torch.cat(dcs, dim=1)
                if dcs
                else torch.zeros((b, 0), dtype=dt, device=x.device),
                "r_used": torch.cat(rus, dim=1)
                if rus
                else torch.zeros((b, 0), dtype=torch.long, device=x.device),
                "g_step_last": torch.cat(gss, dim=1)
                if gss
                else torch.zeros((b, 0), dtype=dt, device=x.device),
                "logits_fast": torch.cat(lfasts, dim=1),
            }
            if collect_snapshots:
                aux["snapshots"] = snaps
            return logits, logits_r1, h_dec, final_state, aux
        return logits, logits_r1, h_dec, final_state, None

    @overload
    def forward_stateful(
        self,
        x: Tensor,
        state: RDFLv05State | None = None,
        num_loops: int | None = None,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        k_bptt: int | None = None,
        update_slow: bool = False,
        gate_mode: GateMode = "surprise",
        random_gate_seed: int | None = None,
        random_gate_prob: float = 0.25,
        gate_override: float | None = None,
        return_aux: Literal[False] = False,
        collect_traj: bool = False,
        collect_snapshots: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RDFLv05State]: ...

    @overload
    def forward_stateful(
        self,
        x: Tensor,
        state: RDFLv05State | None = None,
        num_loops: int | None = None,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        k_bptt: int | None = None,
        update_slow: bool = False,
        gate_mode: GateMode = "surprise",
        random_gate_seed: int | None = None,
        random_gate_prob: float = 0.25,
        gate_override: float | None = None,
        return_aux: Literal[True],
        collect_traj: bool = False,
        collect_snapshots: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RDFLv05State, dict[str, Any]]: ...

    def forward_stateful(
        self,
        x: Tensor,
        state: RDFLv05State | None = None,
        num_loops: int | None = None,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        k_bptt: int | None = None,
        update_slow: bool = False,
        gate_mode: GateMode = "surprise",
        random_gate_seed: int | None = None,
        random_gate_prob: float = 0.25,
        gate_override: float | None = None,
        return_aux: bool = False,
        collect_traj: bool = False,
        collect_snapshots: bool = False,
    ) -> (
        tuple[Tensor, Tensor, Tensor, RDFLv05State]
        | tuple[Tensor, Tensor, Tensor, RDFLv05State, dict[str, Any]]
    ):
        logits, logits_r1, h_dec, new_state, aux = self._forward_core(
            x,
            state=state,
            num_loops=num_loops,
            eps_conv=eps_conv,
            eps_halt=eps_halt,
            k_bptt=k_bptt,
            update_slow=update_slow,
            gate_mode=gate_mode,
            random_gate_seed=random_gate_seed,
            random_gate_prob=random_gate_prob,
            gate_override=gate_override,
            return_aux=return_aux,
            collect_traj=collect_traj,
            collect_snapshots=collect_snapshots,
            need_k3=True,
            need_state=True,
        )
        assert new_state is not None
        if return_aux:
            assert aux is not None
            return logits, logits_r1, h_dec, new_state, aux
        return logits, logits_r1, h_dec, new_state

    def forward_with_snapshots(
        self,
        x: Tensor,
        state: RDFLv05State | None = None,
        num_loops: int | None = None,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        update_slow: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[RDFLv05State]]:
        logits, logits_r1, h_dec, _, aux = self._forward_core(
            x,
            state=state,
            num_loops=num_loops,
            eps_conv=eps_conv,
            eps_halt=eps_halt,
            update_slow=update_slow,
            return_aux=True,
            collect_snapshots=True,
            need_k3=True,
            need_state=True,
        )
        assert aux is not None
        return logits, logits_r1, h_dec, cast(list[RDFLv05State], aux["snapshots"])

    @overload
    def forward(
        self,
        x: Tensor,
        num_loops: int | None = None,
        return_pre_norm: Literal[False] = False,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        update_slow: bool = False,
        gate_mode: GateMode = "surprise",
        gate_override: float | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        num_loops: int | None = None,
        return_pre_norm: Literal[True] = ...,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        update_slow: bool = False,
        gate_mode: GateMode = "surprise",
        gate_override: float | None = None,
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        num_loops: int | None = None,
        return_pre_norm: bool = False,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        update_slow: bool = False,
        gate_mode: GateMode = "surprise",
        gate_override: float | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        logits, _, h_dec, _, _ = self._forward_core(
            x,
            state=None,
            num_loops=num_loops,
            eps_conv=eps_conv,
            eps_halt=eps_halt,
            update_slow=update_slow,
            gate_mode=gate_mode,
            gate_override=gate_override,
            need_k3=False,
            need_state=False,
        )
        if return_pre_norm:
            return logits, h_dec
        return logits

    def step(
        self,
        x_t: Tensor,
        state: RDFLv05State | None = None,
        num_loops: int | None = None,
        *,
        eps_conv: float = 0.0,
        eps_halt: float = 0.0,
        update_slow: bool = False,
        gate_mode: GateMode = "surprise",
        random_gate_seed: int | None = None,
        random_gate_prob: float = 0.25,
        gate_override: float | None = None,
    ) -> tuple[Tensor, RDFLv05State]:
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(1)
        b = x_t.shape[0]
        n_loops = self.cfg.num_loops if num_loops is None else int(num_loops)
        if n_loops < 1:
            raise ValueError("num_loops must be >= 1")
        w_dtype = self.byte_emb.weight.dtype
        st = self._init_state(b, x_t.device, w_dtype, state)
        assert st.stem_state is not None
        prev_bytes = st.stem_state[0]

        e0 = self.byte_emb(x_t)
        delta_ng, new_pb, _ = self.ngram.embed_delta(x_t, prev_bytes, collect_snapshots=False)
        h_b0 = e0 + delta_ng
        yg, (new_sf, new_cf) = self.micro_gdn.step(self.n_micro(h_b0), (st.s_fast, st.conv_state))
        logits_t, _, _, new_st, _, _, _, _, _, _, _ = self._step_from_c(
            x_t,
            h_b0 + yg,
            st,
            new_pb,
            new_sf,
            new_cf,
            n_loops,
            eps_conv=eps_conv,
            eps_halt=eps_halt,
            update_slow=update_slow,
            gate_mode=gate_mode,
            random_gate_seed=random_gate_seed,
            random_gate_prob=random_gate_prob,
            gate_override=gate_override,
        )
        return logits_t, new_st

    def mtp_logits(self, h_fin: Tensor) -> list[Tensor]:
        """Compute t+2..t+4 lookahead logits from pre-norm h_fin."""
        if self.cfg.mtp_stop_grad:
            h_in = h_fin.detach()
            wn = self.final_norm.weight.detach()
            wh = self.head.weight.detach()
            return [
                F.linear(_apply_rmsnorm(adapter(h_in), wn, self.final_norm.eps), wh)
                for adapter in self.mtp_heads
            ]
        return [self.head(self.final_norm(adapter(h_fin))) for adapter in self.mtp_heads]

    # -----------------------------------------------------------------------
    # Unified Single-Token Silent Multi-Step Reasoning (v0.5 StepReasoner API)
    # -----------------------------------------------------------------------

    def encode_for_reasoning(
        self, x: Tensor, state: RDFLv05State | None = None
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """Run Stage 0 causal byte encoder on prompt x and return (h_0, s_fast, c_last, slot_last)."""
        b, t = x.shape
        w_dtype = self.byte_emb.weight.dtype
        dt = _calc_dtype(w_dtype)
        st = self._init_state(b, x.device, w_dtype, state)
        assert st.stem_state is not None
        prev_bytes = st.stem_state[0]

        e0 = self.byte_emb(x)
        delta_ng, _, _ = self.ngram.embed_delta(x, prev_bytes, collect_snapshots=False)
        h_b0 = e0 + delta_ng
        nm = self.n_micro(h_b0)
        half = nm.shape[-1] // 2
        if t > 1 and half > 0:
            nm_prev = torch.cat([torch.zeros_like(nm[:, :1, half:]), nm[:, :-1, half:]], dim=1)
            nm_in = 1.4 * torch.cat([nm[:, :, :half], nm_prev], dim=-1)
        else:
            nm_in = nm
        if t == 1:
            yg, (s_fast, _) = self.micro_gdn.step(nm_in, (st.s_fast, st.conv_state))
        else:
            yg, (s_fast, _) = self.micro_gdn(nm_in, (st.s_fast, st.conv_state))
        c = h_b0 + yg
        c_last = c[:, -1]
        if self.micro_gdn.n_assoc < self.cfg.micro_heads:
            s_fast = torch.cat(
                [
                    s_fast[:, : self.micro_gdn.n_assoc],
                    torch.zeros_like(s_fast[:, self.micro_gdn.n_assoc :]),
                ],
                dim=1,
            )
        h_0 = self.macro.n_in(self.reason_in_proj(c_last)).to(dt)
        slot_last = (st.pos + t - 1) % self.cfg.patch_size
        return h_0, s_fast, c_last, slot_last

    def logits_from_reasoning_h(
        self, h: Tensor, c_last: Tensor, slot_idx: int = 0
    ) -> Tensor:
        """Decode logits [B, 256] directly from reasoning hidden state h [B, d_macro]."""
        _ = slot_idx
        h_byte = (
            h.to(c_last.dtype)
            if self.cfg.byte_dim == self.cfg.macro_dim
            else self.ctx_proj(self.n_ctx(h.to(c_last.dtype)))
        )
        return cast(Tensor, self.head(self.final_norm(h_byte)))

    def reason_step(
        self,
        h_cur: Tensor,
        w_cur: Tensor,
        s_fast: Tensor,
        h_0: Tensor,
        r_idx: int,
        *,
        gate_override: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Execute a single silent reasoning step matching cfg.reasoning_mode."""
        dt = h_cur.dtype
        b = h_cur.shape[0]
        if self.cfg.reasoning_mode == "adfp":
            o_s = self.macro.gdn.read_only(rms_norm_tau(h_cur.unsqueeze(1), tau=1.0), s_fast)
            y_bdh = (
                self.macro.bdh(rms_norm_tau(h_cur.unsqueeze(1), tau=1.0), o_s)
                if self.macro.bdh is not None
                else 0.0
            )
            v_r = rms_norm_tau(h_cur.unsqueeze(1) + o_s + y_bdh, tau=1.0)
            core_sum = o_s + y_bdh + self.macro.ffn(v_r)
            f_val = rms_norm_tau(
                0.5 * rms_norm_tau(h_0.unsqueeze(1), tau=1.0) + 0.85 * core_sum, tau=1.0
            )
            lam = 0.25 + 0.25 * torch.sigmoid(self.macro.lambda_logit.to(dt))
            h_next = ((1.0 - lam) * h_cur.unsqueeze(1) + lam * f_val)[:, 0]
            w_next = w_cur
            g_step = torch.ones((b, 1), device=h_cur.device, dtype=dt)
            return h_next, w_next, g_step
        return self.macro.reason_step_from_memory(
            h_cur, w_cur, s_fast, r_idx, gate_override=gate_override
        )

    def reason_from_state(
        self,
        h_0: Tensor,
        s_fast: Tensor,
        c_last: Tensor,
        num_loops: int,
        *,
        slot_idx: int = 0,
        eps_halt: float = 0.0,
        collect_trace: bool = False,
        collect_w: bool = False,
        gate_override: float | None = None,
    ) -> dict[str, Any]:
        """Execute R silent reasoning steps on O(1) state (h_0, s_fast, W_r) without modifying s_fast."""
        if num_loops < 0:
            raise ValueError("num_loops must be >= 0")
        b = h_0.shape[0]
        dt = h_0.dtype
        h_cur = h_0
        w_cur = torch.zeros(
            b,
            self.macro.num_heads,
            self.macro.head_dim,
            self.macro.head_dim,
            device=h_0.device,
            dtype=dt,
        )
        h_trace: list[Tensor] = [h_cur] if collect_trace else []
        w_trace: list[Tensor] = [w_cur] if (collect_trace and collect_w) else []
        g_trace: list[Tensor] = []
        r_used = torch.full((b,), num_loops, dtype=torch.long, device=h_0.device)
        active = torch.ones((b,), dtype=torch.bool, device=h_0.device)

        for r_idx in range(1, num_loops + 1):
            h_next, w_next, g_step = self.reason_step(
                h_cur, w_cur, s_fast, h_0, r_idx, gate_override=gate_override
            )

            g_trace.append(g_step)
            if eps_halt > 0.0:
                newly_halted = active & (g_step.squeeze(-1) < eps_halt)
                r_used = torch.where(newly_halted, torch.full_like(r_used, r_idx - 1), r_used)
                keep = active & (~newly_halted)
                h_cur = torch.where(keep.unsqueeze(-1), h_next, h_cur)
                w_cur = torch.where(keep[:, None, None, None], w_next, w_cur)
                active = active & (~newly_halted)
                if collect_trace:
                    h_trace.append(h_cur)
                    if collect_w:
                        w_trace.append(w_cur)
                if not bool(active.any().item()):
                    break
            else:
                h_cur = h_next
                w_cur = w_next
                if collect_trace:
                    h_trace.append(h_cur)
                    if collect_w:
                        w_trace.append(w_cur)

        logits = self.logits_from_reasoning_h(h_cur, c_last, slot_idx=slot_idx)
        return {
            "logits": logits,
            "h_final": h_cur,
            "w_final": w_cur,
            "r_used": r_used,
            "g_trace": torch.stack(g_trace, dim=1) if g_trace else h_0.new_zeros(b, 0, 1),
            "h_trace": h_trace,
            "w_trace": w_trace,
        }

    def forward_step_reasoning(
        self,
        x: Tensor,
        num_loops: int,
        *,
        eps_halt: float = 0.0,
        collect_trace: bool = False,
        collect_w: bool = False,
        gate_override: float | None = None,
    ) -> dict[str, Any]:
        """End-to-end single-token silent multi-step reasoning directly on RDFLv05Model."""
        h_0, s_fast, c_last, slot_last = self.encode_for_reasoning(x)
        res = self.reason_from_state(
            h_0,
            s_fast,
            c_last,
            num_loops,
            slot_idx=slot_last,
            eps_halt=eps_halt,
            collect_trace=collect_trace,
            collect_w=collect_w,
            gate_override=gate_override,
        )
        res["s_fast"] = s_fast
        res["c_last"] = c_last
        res["slot_last"] = slot_last
        return res


# ---------------------------------------------------------------------------
# Time-Scale 3: ConsolidationManager (Post-AdamW OGP + Online EWC + OY-7 KL)
# ---------------------------------------------------------------------------


class ConsolidationManager:
    """Zero-replay-buffer Time-Scale 3 consolidation for RDFLv05Model."""

    def __init__(
        self,
        model: RDFLv05Model,
        *,
        mu_ogp: float = 10.0,
        tau_sigma: float = 1.0,
        gamma_sigma: float = 0.90,
        lambda_ewc: float = 8.0,
        gamma_ewc: float = 0.90,
        lambda_kl: float = 0.03,
    ) -> None:
        self.model = model
        self.mu_ogp = mu_ogp
        self.tau_sigma = tau_sigma
        self.gamma_sigma = gamma_sigma
        self.lambda_ewc = lambda_ewc
        self.gamma_ewc = gamma_ewc
        self.lambda_kl = lambda_kl

        self.ogp_modules: dict[str, nn.Linear | SoftSpectralLinear] = {}
        for name, mod in model.named_modules():
            if (
                isinstance(mod, (nn.Linear, SoftSpectralLinear))
                and "k3_adapter" not in name
                and "mtp_heads" not in name
            ):
                self.ogp_modules[name] = mod

        self.cov_acc: dict[str, Tensor] = {
            name: torch.zeros(mod.in_features, mod.in_features, dtype=torch.float32)
            for name, mod in self.ogp_modules.items()
        }
        self.proj_ogp: dict[str, Tensor] = {}
        self.fisher: dict[str, Tensor] = {
            name: torch.zeros_like(p.data)
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.anchor_params: dict[str, Tensor] = {}
        self.anchor_model: RDFLv05Model | None = None
        self.has_consolidated = False

    def penalty_loss(self, probe_x: Tensor | None = None) -> Tensor:
        if not self.has_consolidated:
            return torch.zeros((), dtype=torch.float32)
        ewc = torch.zeros((), dtype=torch.float32)
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            ewc = ewc + 0.5 * (self.fisher[name] * (p - self.anchor_params[name]).pow(2)).sum()
        loss = self.lambda_ewc * ewc
        if self.lambda_kl > 0.0 and self.anchor_model is not None and probe_x is not None:
            logits_pol = cast(Tensor, self.model(probe_x))
            with torch.no_grad():
                logits_ref = cast(Tensor, self.anchor_model(probe_x))
            loss = loss + self.lambda_kl * reverse_kl_per_token(logits_pol, logits_ref)
        return loss

    def snapshot_weights(self) -> dict[int, Tensor]:
        return {id(mod.weight): mod.weight.data.clone() for mod in self.ogp_modules.values()}

    @torch.no_grad()
    def apply_post_step_ogp(self, prev_weights: dict[int, Tensor]) -> None:
        if not self.has_consolidated:
            return
        for name, mod in self.ogp_modules.items():
            p_ogp = self.proj_ogp.get(name)
            if p_ogp is None:
                continue
            w_prev = prev_weights[id(mod.weight)]
            dw = mod.weight.data - w_prev
            mod.weight.data.copy_(w_prev + torch.matmul(dw, p_ogp))

    def consolidate_task(self, task_batches: list[Any], *, update_slow: bool = False) -> None:
        self.model.eval()
        batch_cov: dict[str, Tensor] = {n: torch.zeros_like(c) for n, c in self.cov_acc.items()}
        batch_cnt: dict[str, int] = {n: 0 for n in self.cov_acc}
        handles = []

        def make_hook(layer_name: str) -> Any:
            def _hook(_mod: nn.Module, inp: tuple[Tensor, ...], _out: Tensor) -> None:
                a = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                batch_cov[layer_name].add_(torch.matmul(a.T, a))
                batch_cnt[layer_name] += int(a.shape[0])

            return _hook

        for name, mod in self.ogp_modules.items():
            handles.append(mod.register_forward_hook(make_hook(name)))
        try:
            with torch.no_grad():
                for tb in task_batches:
                    self.model(tb.x, update_slow=update_slow)
        finally:
            for h in handles:
                h.remove()

        for name, mod in self.ogp_modules.items():
            cnt = max(1, batch_cnt[name])
            cov_k = batch_cov[name] / float(cnt)
            self.cov_acc[name] = self.gamma_sigma * self.cov_acc[name] + cov_k
            c_mat = self.cov_acc[name]
            tr = float(c_mat.trace().item())
            c_tilde = (float(mod.in_features) * c_mat) / (tr + self.tau_sigma)
            eye = torch.eye(mod.in_features, dtype=torch.float32)
            p_ogp = torch.linalg.solve(eye + self.mu_ogp * c_tilde, eye)
            self.proj_ogp[name] = 0.5 * (p_ogp + p_ogp.T)

        raw_fisher: dict[str, Tensor] = {
            name: torch.zeros_like(p.data)
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        for tb in task_batches:
            self.model.zero_grad(set_to_none=True)
            logits = cast(Tensor, self.model(tb.x, update_slow=update_slow))
            tgt = tb.masked_targets(-100)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100
            )
            torch.autograd.backward(loss)
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    raw_fisher[name].add_(p.grad.detach().pow(2))

        n_batches = float(max(1, len(task_batches)))
        total_sum = 0.0
        active_elems = 0
        for name, rf_val in raw_fisher.items():
            self.fisher[name] = self.gamma_ewc * self.fisher[name] + (rf_val / n_batches)
            s_val = float(self.fisher[name].sum().item())
            if s_val > 0.0:
                total_sum += s_val
                active_elems += self.fisher[name].numel()
        mean_f = total_sum / float(max(1, active_elems))
        for name in self.fisher:
            self.fisher[name] = self.fisher[name] / (mean_f + 1e-8)

        self.anchor_params = {
            name: p.data.detach().clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.anchor_model = copy.deepcopy(self.model).eval()
        for p in self.anchor_model.parameters():
            p.requires_grad_(False)
        self.has_consolidated = True


# ---------------------------------------------------------------------------
# Analytical FLOP/byte & State Byte Accounting
# ---------------------------------------------------------------------------


def state_nbytes(state: RDFLv05State) -> int:
    """Measure unique tensor bytes inside RDFLv05State."""
    seen: set[tuple[int, int]] = set()
    total = 0

    def _visit(obj: Any) -> None:
        nonlocal total
        if isinstance(obj, Tensor):
            key = (obj.data_ptr(), obj.numel() * obj.element_size())
            if key not in seen:
                seen.add(key)
                total += obj.numel() * obj.element_size()
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _visit(item)

    for fld in (
        state.s_fast,
        state.conv_state,
        state.s_macro,
        state.conv_macro,
        state.slot_state,
        state.chunk_sum,
        state.s_slow,
        state.c_slow,
        state.prev_fast_logits,
        state.stem_state,
    ):
        _visit(fld)
    return total


def analytic_flops_per_byte(cfg: RDFLv05Config, num_loops: int | None = None) -> dict[str, Any]:
    """Compute exact analytical forward FLOPs/byte (2*MACs) for RDFLv05Model."""
    d_b = cfg.byte_dim
    d_m = cfg.macro_dim
    p_sz = cfg.patch_size
    n_h = cfg.n_h
    r_loops = cfg.num_loops if num_loops is None else num_loops
    c_tok = max(1, cfg.chunk_size // n_h) if n_h > 1 else cfg.chunk_size
    dh_b = d_b // cfg.micro_heads
    dh_m = d_m // cfg.macro_heads

    # Stage 0 (per byte)
    ng_flops = 2 * 4 * cfg.ngram_dim * d_b
    micro_proj = 2 * d_b * (d_b + 2 * n_h * d_b + d_b + d_b)
    micro_wy = cfg.micro_heads * (2 * n_h * c_tok * dh_b + 4 * n_h * dh_b * dh_b)
    stage0 = ng_flops + micro_proj + micro_wy

    # Stage 1 (per P=4 macro step -> divide by P)
    pack_flops = 2 * (p_sz * d_b * cfg.patch_bottleneck + cfg.patch_bottleneck * d_m)
    macro_r0 = 2 * (4 * d_m * d_m + 3 * d_m * (2 * d_m) + 3 * d_m * cfg.ffn_macro) + cfg.macro_heads * (
        2 * cfg.chunk_size * dh_m + 4 * dh_m * dh_m
    )
    extra_halt = 2 * (2 * d_m + d_m * d_m) if cfg.reasoning_mode == "step_halting" else 0
    macro_r_read = max(0, r_loops - 1) * (
        2 * (3 * d_m * d_m + 3 * d_m * (2 * d_m) + 3 * d_m * cfg.ffn_macro)
        + extra_halt
        + cfg.macro_heads * (2 * cfg.chunk_size * dh_m + 2 * dh_m * dh_m)
    )
    stage1_per_byte = (pack_flops + macro_r0 + macro_r_read) // p_sz

    # Stage 3 (per P=4 macro step, update_slow=True)
    slow_per_byte = (
        2 * (5 * d_m * d_m) + cfg.macro_heads * (2 * cfg.chunk_size * dh_m + 4 * dh_m * dh_m)
    ) // p_sz

    # Stage 4 (per byte: single-pass dec_ffn; slowON adds only ctx_proj_slow + extra head)
    dec_core_fast = 2 * (d_m * d_b + 3 * d_b * cfg.dec_ffn)
    dec_core_slow_add = 2 * d_m * d_b
    head_flops = 2 * d_b * cfg.vocab_size
    slow_off_core = stage0 + stage1_per_byte + dec_core_fast
    slow_on_core = slow_off_core + slow_per_byte + dec_core_slow_add
    return {
        "stage0_micro_per_byte": stage0,
        "stage1_macro_per_byte": stage1_per_byte,
        "stage3_slow_per_byte": slow_per_byte,
        "stage4_dec_core_fast_per_byte": dec_core_fast,
        "stage4_dec_core_slow_add_per_byte": dec_core_slow_add,
        "head_flops_per_byte": head_flops,
        "slow_off_core_flops_per_byte": slow_off_core,
        "slow_off_total_with_head_flops_per_byte": slow_off_core + head_flops,
        "slow_on_core_flops_per_byte": slow_on_core,
        "slow_on_total_with_head_flops_per_byte": slow_on_core + 2 * head_flops,
        "core_flops_per_byte": slow_off_core,
        "total_with_head_flops_per_byte": slow_off_core + head_flops,
    }


__all__ = [
    "BiasFreeCausalShortConv",
    "BoundedBdhGate",
    "BoundedSwiGLU",
    "ConsolidationManager",
    "GDNReadOnlyLayer",
    "GateMode",
    "MSNgramHashStem",
    "MacroRDFLv05Core",
    "MicroDeltaProductGDNLayer",
    "OrthogonalSlowGDN",
    "RDFLv05Config",
    "RDFLv05Model",
    "RDFLv05State",
    "RMSNormTauLayer",
    "ReasoningMode",
    "SlowMode",
    "SoftRMSNorm",
    "SoftSpectralLinear",
    "analytic_flops_per_byte",
    "bounded_q_norm",
    "rms_norm_tau",
    "state_nbytes",
]
