"""RDFL standalone model and Zero-Diff RDFL Adapter for pretrained backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rdfl.layers import (
    BdhSparseGate,
    BottleneckAdapter,
    Fp32SlotCoreBoundary,
    GatedDeltaNetLayer,
    RMSNorm,
    SwiGLU,
)


@dataclass(frozen=True)
class RDFLConfig:
    """Configuration for a standalone byte-level RDFL model."""

    vocab_size: int = 256
    d_model: int = 192
    num_heads: int = 6
    ffn_dim: int = 512
    num_loops: int = 2
    chunk_size: int = 64
    num_slots: int = 4
    mtp_heads: int = 3
    mtp_divisor: int = 2
    mtp_stop_grad: bool = False


@dataclass
class RDFLState:
    """Streaming state guaranteeing exact forward() == step() equivalence."""

    gdn_states: list[tuple[Tensor, Tensor | None] | None]
    slot_state: Tensor
    pos: int = 0


class RDFLModel(nn.Module):
    """Standalone byte-level Recursive Depth Flash Loop model with K3 and MTP-4."""

    def __init__(self, cfg: RDFLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.n_anchor = RMSNorm(cfg.d_model)
        self.step_emb = nn.Embedding(cfg.num_loops, cfg.d_model)
        self.anchor_scale = nn.Parameter(torch.full((cfg.num_loops,), 0.25))
        self.n_gdn = RMSNorm(cfg.d_model)
        self.gdn = GatedDeltaNetLayer(cfg.d_model, cfg.num_heads, chunk_size=cfg.chunk_size)
        self.bdh = BdhSparseGate(cfg.d_model, expansion=2)
        self.n_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_dim)
        self.boundary = (
            Fp32SlotCoreBoundary(cfg.d_model, num_slots=cfg.num_slots)
            if cfg.num_slots > 0
            else None
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight

        # Decoupled K3 early-exit adapter (0.000 gradient tax on main R-loop)
        self.k3_adapter = BottleneckAdapter(cfg.d_model, divisor=4)
        # MTP-4 lookahead heads (t+2 .. t+4)
        self.mtp_heads = nn.ModuleList(
            [BottleneckAdapter(cfg.d_model, divisor=cfg.mtp_divisor) for _ in range(cfg.mtp_heads)]
        )

    def _run_loop_step(
        self,
        h: Tensor,
        anchor: Tensor,
        r_idx: int,
        gdn_state: tuple[Tensor, Tensor | None] | None,
        single_step: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        inp = h + self.anchor_scale[r_idx] * anchor + self.step_emb.weight[r_idx]
        norm_in = self.n_gdn(inp)
        y_gdn, new_gdn = (
            self.gdn.step(norm_in, gdn_state) if single_step else self.gdn(norm_in, gdn_state)
        )
        h = h + y_gdn + self.bdh(norm_in, y_gdn)
        h = h + self.ffn(self.n_ffn(h))
        return h, new_gdn

    def _init_slot_state(self, batch: int, h0: Tensor, state: RDFLState | None) -> Tensor:
        slot_dt = torch.float64 if h0.dtype == torch.float64 else torch.float32
        if state is not None:
            return state.slot_state.to(device=h0.device, dtype=slot_dt)
        if self.boundary is not None:
            return self.boundary.initial_state(batch, dtype=slot_dt).to(device=h0.device)
        return h0.new_zeros(batch, 1, self.cfg.d_model, dtype=slot_dt)

    def _run_chunk(
        self,
        h_seg: Tensor,
        a_seg: Tensor,
        gdn_in: list[tuple[Tensor, Tensor | None] | None],
        slot_st: Tensor,
        single: bool,
        at_boundary: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.boundary is not None:
            h_seg = self.boundary.apply_feedback(h_seg, slot_st)
        r1_seg = h_seg
        for r_idx in range(self.cfg.num_loops):
            h_seg, gdn_in[r_idx] = self._run_loop_step(h_seg, a_seg, r_idx, gdn_in[r_idx], single)
            if r_idx == 0:
                r1_seg = h_seg
        if self.boundary is not None and at_boundary:
            slot_st = self.boundary.commit_boundary(slot_st, h_seg[:, -1])
        return r1_seg, h_seg, slot_st

    def forward_stateful(
        self, x: Tensor, state: RDFLState | None = None
    ) -> tuple[Tensor, Tensor, Tensor, RDFLState]:
        """Exact chunk-boundary-aligned forward returning (logits, logits_r1, h_final, state)."""
        b, t = x.shape
        single = t == 1
        h0 = self.emb(x)
        anchor = self.n_anchor(h0)
        bas_pos = 0 if state is None else state.pos
        gdn_in: list[tuple[Tensor, Tensor | None] | None] = (
            [None] * self.cfg.num_loops if state is None else list(state.gdn_states)
        )
        slot_st = self._init_slot_state(b, h0, state)

        r1_chunks, final_chunks = [], []
        offset = 0
        while offset < t:
            pos = bas_pos + offset
            step_len = min(t - offset, self.cfg.chunk_size - (pos % self.cfg.chunk_size))
            at_boundary = ((pos + step_len) % self.cfg.chunk_size) == 0
            r1_seg, h_seg, slot_st = self._run_chunk(
                h0[:, offset : offset + step_len],
                anchor[:, offset : offset + step_len],
                gdn_in,
                slot_st,
                single,
                at_boundary,
            )
            r1_chunks.append(r1_seg)
            final_chunks.append(h_seg)
            offset += step_len

        h_r1 = torch.cat(r1_chunks, dim=1)
        h_fin = torch.cat(final_chunks, dim=1)
        hn = self.final_norm(h_fin)
        logits = self.head(hn)
        # Strictly detached K3 projection -> 0.000 gradient tax on core/head
        z_k3 = self.k3_adapter(h_r1.detach())
        wn = self.final_norm.weight.detach()
        hn_k3 = z_k3 * torch.rsqrt(z_k3.pow(2).mean(dim=-1, keepdim=True) + self.final_norm.eps) * wn
        logits_r1 = F.linear(hn_k3, self.head.weight.detach())
        return logits, logits_r1, hn, RDFLState(gdn_in, slot_st, bas_pos + t)

    def forward(self, x: Tensor) -> Tensor:
        logits, _, _, _ = self.forward_stateful(x, state=None)
        return logits

    def step(self, x_t: Tensor, state: RDFLState | None = None) -> tuple[Tensor, RDFLState]:
        logits, _, _, new_state = self.forward_stateful(x_t, state=state)
        return logits, new_state

    def mtp_logits(self, hn: Tensor) -> list[Tensor]:
        """Compute t+2..t+4 lookahead logits (respecting mtp_stop_grad)."""
        if self.cfg.mtp_stop_grad:
            h_in = hn.detach()
            wn = self.final_norm.weight.detach()
            wh = self.head.weight.detach()
            out = []
            for adapter in self.mtp_heads:
                z = adapter(h_in)
                zn = z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + self.final_norm.eps) * wn
                out.append(F.linear(zn, wh))
            return out
        return [self.head(self.final_norm(adapter(hn))) for adapter in self.mtp_heads]


class ZeroDiffRDFLAdapter(nn.Module):
    """L2-resident narrow RDFL + FP32 SlotCore adapter with bit-exact 0.0 step-0 diff.

    Attaches onto any pretrained hidden stream z [B, T, d_base] (e.g., d_base=1536) via
    a narrow d_narrow=384 recursive bottleneck whose output projection starts at 0.0.
    """

    def __init__(
        self,
        d_base: int = 1536,
        d_narrow: int = 384,
        num_heads: int = 6,
        num_loops: int = 2,
        chunk_size: int = 64,
        num_slots: int = 4,
    ) -> None:
        super().__init__()
        self.num_loops = num_loops
        self.chunk_size = chunk_size
        self.n_in = RMSNorm(d_base)
        self.down = nn.Linear(d_base, d_narrow, bias=False)
        self.n_anchor = RMSNorm(d_narrow)
        self.step_emb = nn.Embedding(num_loops, d_narrow)
        self.anchor_scale = nn.Parameter(torch.full((num_loops,), 0.25))
        self.n_gdn = RMSNorm(d_narrow)
        self.gdn = GatedDeltaNetLayer(d_narrow, num_heads, chunk_size=chunk_size)
        self.bdh = BdhSparseGate(d_narrow, expansion=2)
        self.boundary = Fp32SlotCoreBoundary(d_narrow, num_slots=num_slots)
        self.n_out = RMSNorm(d_narrow)
        self.up = nn.Linear(d_narrow, d_base, bias=False)
        # Step-0 exact identity guarantee: delta_z == 0.0!
        nn.init.zeros_(self.up.weight)

    def forward(self, z_base: Tensor) -> Tensor:
        """Returns z_base + delta_z (at step 0: bit-exact delta_z == 0.0)."""
        b, t, _ = z_base.shape
        z0 = self.down(self.n_in(z_base))
        anchor = self.n_anchor(z0)
        slot_dt = torch.float64 if z0.dtype == torch.float64 else torch.float32
        slot_st = self.boundary.initial_state(b, dtype=slot_dt).to(device=z_base.device)
        gdn_states: list[tuple[Tensor, Tensor | None] | None] = [None] * self.num_loops
        chunks: list[Tensor] = []
        for start in range(0, t, self.chunk_size):
            end = min(t, start + self.chunk_size)
            h_seg = self.boundary.apply_feedback(z0[:, start:end], slot_st)
            a_seg = anchor[:, start:end]
            for r_idx in range(self.num_loops):
                inp = h_seg + self.anchor_scale[r_idx] * a_seg + self.step_emb.weight[r_idx]
                norm_in = self.n_gdn(inp)
                y_gdn, gdn_states[r_idx] = self.gdn(norm_in, gdn_states[r_idx])
                h_seg = h_seg + y_gdn + self.bdh(norm_in, y_gdn)
            if end % self.chunk_size == 0:
                slot_st = self.boundary.commit_boundary(slot_st, h_seg[:, -1])
            chunks.append(h_seg)
        delta_z = self.up(self.n_out(torch.cat(chunks, dim=1)))
        return cast(Tensor, z_base + delta_z)
