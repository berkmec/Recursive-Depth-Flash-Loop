"""RDFL standalone model and Zero-Diff RDFL Adapter for pretrained backbones."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rdfl.layers import (
    BdhSparseGate,
    BottleneckAdapter,
    Fp32RLT2Boundary,
    Fp32SlotCoreBoundary,
    GatedDeltaNetLayer,
    RMSNorm,
    SwiGLU,
)

_VALID_BOUNDARY_MODES = {"none", "rlt2_s1", "slotcore_s4"}
_VALID_BOUNDARY_POOLS = {"last", "mean"}
_VALID_STEP_EMB_MODES = {"learned", "clamped", "sinusoidal"}
_VALID_SHORT_CONV_MODES = {"shared", "first_only"}


def _apply_rmsnorm(z: Tensor, weight: Tensor, eps: float) -> Tensor:
    dt = torch.float64 if z.dtype == torch.float64 else torch.float32
    z_f = z.to(dt)
    norm = z_f * torch.rsqrt(z_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (norm * weight.to(dt)).to(z.dtype)


def _sinusoidal_step_emb(
    r_idx: int, d_model: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    half = d_model // 2
    if half == 0:
        return torch.full((d_model,), 0.02 * math.sin(float(r_idx)), device=device, dtype=dtype)
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / float(max(1, half))
    )
    angles = float(r_idx) * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)
    if emb.shape[0] < d_model:
        emb = F.pad(emb, (0, d_model - emb.shape[0]))
    return (emb * 0.02).to(dtype=dtype)


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
    alpha_max: float | None = None
    boundary_mode: str = "rlt2_s1"
    boundary_pool: str = "last"
    step_emb_mode: str = "clamped"
    shared_gdn_state: bool = False
    short_conv_mode: str = "shared"
    use_bdh: bool = True


@dataclass
class RDFLState:
    """Streaming state guaranteeing exact forward() == step() equivalence."""

    gdn_states: list[tuple[Tensor, Tensor | None] | None]
    slot_state: Tensor
    pos: int = 0
    chunk_sum: Tensor | None = None


class RDFLModel(nn.Module):
    """Standalone byte-level Recursive Depth Flash Loop model with K3 and MTP-4."""

    def __init__(self, cfg: RDFLConfig) -> None:
        super().__init__()
        if cfg.num_loops < 1:
            raise ValueError("num_loops must be >= 1")
        if cfg.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if cfg.boundary_mode not in _VALID_BOUNDARY_MODES:
            raise ValueError(f"boundary_mode must be in {_VALID_BOUNDARY_MODES}")
        if cfg.boundary_pool not in _VALID_BOUNDARY_POOLS:
            raise ValueError(f"boundary_pool must be in {_VALID_BOUNDARY_POOLS}")
        if cfg.step_emb_mode not in _VALID_STEP_EMB_MODES:
            raise ValueError(f"step_emb_mode must be in {_VALID_STEP_EMB_MODES}")
        if cfg.short_conv_mode not in _VALID_SHORT_CONV_MODES:
            raise ValueError(f"short_conv_mode must be in {_VALID_SHORT_CONV_MODES}")

        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.n_anchor = RMSNorm(cfg.d_model)
        self.step_emb: nn.Embedding | None = (
            None
            if cfg.step_emb_mode == "sinusoidal"
            else nn.Embedding(cfg.num_loops, cfg.d_model)
        )
        self.anchor_scale = nn.Parameter(torch.full((cfg.num_loops,), 0.25))
        self.n_gdn = RMSNorm(cfg.d_model)
        self.gdn = GatedDeltaNetLayer(
            cfg.d_model, cfg.num_heads, chunk_size=cfg.chunk_size, alpha_max=cfg.alpha_max
        )
        self.bdh: BdhSparseGate | None = (
            BdhSparseGate(cfg.d_model, expansion=2) if cfg.use_bdh else None
        )
        self.n_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_dim)
        self.boundary: Fp32RLT2Boundary | Fp32SlotCoreBoundary | None
        if cfg.boundary_mode == "rlt2_s1":
            self.boundary = Fp32RLT2Boundary(cfg.d_model)
        elif cfg.boundary_mode == "slotcore_s4":
            n_slots = cfg.num_slots if cfg.num_slots > 0 else 4
            self.boundary = Fp32SlotCoreBoundary(cfg.d_model, num_slots=n_slots)
        else:
            self.boundary = None
        self.final_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight

        # Decoupled K3 early-exit adapter (0.000 gradient tax on main R-loop)
        self.k3_adapter = BottleneckAdapter(cfg.d_model, divisor=4)
        # MTP-4 lookahead heads (t+2 .. t+4)
        self.mtp_heads = nn.ModuleList(
            [BottleneckAdapter(cfg.d_model, divisor=cfg.mtp_divisor) for _ in range(cfg.mtp_heads)]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        if self.step_emb is not None:
            nn.init.normal_(self.step_emb.weight, mean=0.0, std=0.02)

        core_blocks: list[nn.Module] = [self.gdn, self.ffn]
        if self.bdh is not None:
            core_blocks.append(self.bdh)
        if self.boundary is not None:
            core_blocks.append(self.boundary)

        for block in core_blocks:
            for mod in block.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)

        # Preserve GDN multiscale slow-decay bias schedule after linear init
        self.gdn._init_multiscale_decay()

        # Scale residual output projections by 1 / sqrt(2 * num_loops)
        res_std = 0.02 / math.sqrt(2.0 * float(self.cfg.num_loops))
        nn.init.normal_(self.gdn.out_proj.weight, mean=0.0, std=res_std)
        nn.init.normal_(self.ffn.out.weight, mean=0.0, std=res_std)
        if self.bdh is not None:
            nn.init.normal_(self.bdh.dec.weight, mean=0.0, std=res_std)

    def _get_step_embedding(
        self, r_idx: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if self.cfg.step_emb_mode == "sinusoidal":
            return _sinusoidal_step_emb(r_idx, self.cfg.d_model, device, dtype)
        assert self.step_emb is not None
        if r_idx >= self.step_emb.num_embeddings:
            if self.cfg.step_emb_mode == "learned":
                raise ValueError(
                    f"r_idx={r_idx} >= num_loops={self.cfg.num_loops} with step_emb_mode='learned'"
                )
            idx = self.step_emb.num_embeddings - 1
        else:
            idx = r_idx
        return self.step_emb.weight[idx].to(device=device, dtype=dtype)

    def _get_anchor_scale(self, r_idx: int) -> Tensor:
        if r_idx >= self.anchor_scale.shape[0]:
            if self.cfg.step_emb_mode == "learned":
                raise ValueError(
                    f"r_idx={r_idx} >= num_loops={self.cfg.num_loops} with step_emb_mode='learned'"
                )
            return self.anchor_scale[-1]
        return self.anchor_scale[r_idx]

    def _prep_loop_input(self, h: Tensor, anchor: Tensor, r_idx: int) -> Tensor:
        step_vec = self._get_step_embedding(r_idx, h.device, h.dtype)
        inp = h + self._get_anchor_scale(r_idx) * anchor + step_vec
        return cast(Tensor, self.n_gdn(inp))

    def _finish_loop_step(self, h: Tensor, norm_in: Tensor, y_gdn: Tensor) -> Tensor:
        h = h + y_gdn + (self.bdh(norm_in, y_gdn) if self.bdh is not None else 0.0)
        return cast(Tensor, h + self.ffn(self.n_ffn(h)))

    def _run_loop_step(
        self,
        h: Tensor,
        anchor: Tensor,
        r_idx: int,
        gdn_state: tuple[Tensor, Tensor | None] | None,
        single_step: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor | None]]:
        norm_in = self._prep_loop_input(h, anchor, r_idx)
        use_conv = (self.cfg.short_conv_mode == "shared") or (r_idx == 0)
        y_gdn, new_gdn = (
            self.gdn.step(norm_in, gdn_state, use_short_conv=use_conv)
            if single_step
            else self.gdn(norm_in, gdn_state, use_short_conv=use_conv)
        )
        return self._finish_loop_step(h, norm_in, y_gdn), new_gdn

    def _run_loop_step_snapshots(
        self,
        h: Tensor,
        anchor: Tensor,
        r_idx: int,
        gdn_state: tuple[Tensor, Tensor | None] | None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor | None]]]:
        norm_in = self._prep_loop_input(h, anchor, r_idx)
        use_conv = (self.cfg.short_conv_mode == "shared") or (r_idx == 0)
        y_gdn, snaps = self.gdn.forward_with_snapshots(
            norm_in, gdn_state, use_short_conv=use_conv
        )
        return self._finish_loop_step(h, norm_in, y_gdn), snaps

    def _init_slot_state(self, batch: int, h0: Tensor, state: RDFLState | None) -> Tensor:
        slot_dt = torch.float64 if h0.dtype == torch.float64 else torch.float32
        if state is not None:
            return state.slot_state.to(device=h0.device, dtype=slot_dt)
        if self.boundary is not None:
            return self.boundary.initial_state(batch, dtype=slot_dt).to(device=h0.device)
        return h0.new_zeros(batch, 1, self.cfg.d_model, dtype=slot_dt)

    def _commit_if_boundary(
        self,
        slot_st: Tensor,
        chunk_sum: Tensor | None,
        h_token_or_seg: Tensor,
        next_pos: int,
        is_full_seg: bool,
    ) -> tuple[Tensor, Tensor | None]:
        if self.boundary is None:
            return slot_st, None
        at_boundary = (next_pos % self.cfg.chunk_size) == 0
        if self.cfg.boundary_pool == "last":
            if at_boundary:
                slot_st = self.boundary.commit_boundary(slot_st, h_token_or_seg[:, -1])
            return slot_st, None
        seg_sum = (
            h_token_or_seg.to(slot_st.dtype).sum(dim=1)
            if is_full_seg
            else h_token_or_seg.to(slot_st.dtype)
        )
        cur_sum = seg_sum if chunk_sum is None else chunk_sum + seg_sum
        if at_boundary:
            slot_st = self.boundary.commit_boundary(
                slot_st, cur_sum / float(self.cfg.chunk_size)
            )
            return slot_st, None
        return slot_st, cur_sum

    def _resolve_num_loops(self, num_loops: int | None) -> int:
        n_loops = self.cfg.num_loops if num_loops is None else num_loops
        if n_loops < 1:
            raise ValueError("num_loops must be >= 1")
        if n_loops > self.cfg.num_loops and self.cfg.step_emb_mode == "learned":
            raise ValueError("R_test > R_train requires step_emb_mode in {'clamped', 'sinusoidal'}")
        return n_loops

    def _forward_shared_gdn(
        self,
        h0: Tensor,
        anchor: Tensor,
        state: RDFLState | None,
        n_loops: int,
        collect_snapshots: bool,
    ) -> tuple[Tensor, Tensor, RDFLState, list[RDFLState]]:
        b, t, _ = h0.shape
        bas_pos = 0 if state is None else state.pos
        slot_st = self._init_slot_state(b, h0, state)
        chunk_sum = None if state is None else state.chunk_sum
        if chunk_sum is not None:
            chunk_sum = chunk_sum.to(device=h0.device, dtype=slot_st.dtype)

        s_shared: Tensor | None = None
        conv_states: list[Tensor | None] = [None] * n_loops
        if state is not None:
            for r_idx in range(min(n_loops, len(state.gdn_states))):
                st_r = state.gdn_states[r_idx]
                if st_r is not None:
                    if s_shared is None:
                        s_shared = st_r[0]
                    conv_states[r_idx] = st_r[1]

        r1_tokens: list[Tensor] = []
        fin_tokens: list[Tensor] = []
        snapshots: list[RDFLState] = []
        for idx in range(t):
            h_t = h0[:, idx : idx + 1]
            a_t = anchor[:, idx : idx + 1]
            if self.boundary is not None:
                h_t = self.boundary.apply_feedback(h_t, slot_st)
            r1_t = h_t
            for r_idx in range(n_loops):
                st_in = (
                    None
                    if s_shared is None and conv_states[r_idx] is None
                    else (
                        s_shared
                        if s_shared is not None
                        else h0.new_zeros(
                            b,
                            self.gdn.num_heads,
                            self.gdn.head_dim,
                            self.gdn.head_dim,
                            dtype=slot_st.dtype,
                        ),
                        conv_states[r_idx],
                    )
                )
                h_t, (s_shared, conv_states[r_idx]) = self._run_loop_step(
                    h_t, a_t, r_idx, st_in, single_step=True
                )
                if r_idx == 0:
                    r1_t = h_t
            next_pos = bas_pos + idx + 1
            slot_st, chunk_sum = self._commit_if_boundary(
                slot_st,
                chunk_sum,
                h_t[:, 0] if self.cfg.boundary_pool == "mean" else h_t,
                next_pos,
                is_full_seg=False,
            )
            r1_tokens.append(r1_t)
            fin_tokens.append(h_t)
            if collect_snapshots:
                assert s_shared is not None
                gdn_snap: list[tuple[Tensor, Tensor | None] | None] = [
                    (s_shared, conv_states[r]) for r in range(n_loops)
                ]
                snapshots.append(RDFLState(gdn_snap, slot_st, next_pos, chunk_sum))

        assert s_shared is not None
        final_gdn: list[tuple[Tensor, Tensor | None] | None] = [
            (s_shared, conv_states[r]) for r in range(n_loops)
        ]
        return (
            torch.cat(r1_tokens, dim=1),
            torch.cat(fin_tokens, dim=1),
            RDFLState(final_gdn, slot_st, bas_pos + t, chunk_sum),
            snapshots,
        )

    def _forward_core(
        self,
        x: Tensor,
        state: RDFLState | None,
        num_loops: int | None,
        collect_snapshots: bool,
    ) -> tuple[Tensor, Tensor, Tensor, RDFLState, list[RDFLState]]:
        n_loops = self._resolve_num_loops(num_loops)
        b, t = x.shape
        h0 = self.emb(x)
        anchor = self.n_anchor(h0)

        if self.cfg.shared_gdn_state:
            h_r1, h_fin, final_st, snapshots = self._forward_shared_gdn(
                h0, anchor, state, n_loops, collect_snapshots
            )
        else:
            bas_pos = 0 if state is None else state.pos
            gdn_in: list[tuple[Tensor, Tensor | None] | None] = [None] * n_loops
            if state is not None:
                for r_idx in range(min(n_loops, len(state.gdn_states))):
                    gdn_in[r_idx] = state.gdn_states[r_idx]
            slot_st = self._init_slot_state(b, h0, state)
            chunk_sum = None if state is None else state.chunk_sum
            if chunk_sum is not None:
                chunk_sum = chunk_sum.to(device=h0.device, dtype=slot_st.dtype)

            r1_chunks: list[Tensor] = []
            final_chunks: list[Tensor] = []
            snapshots = []
            offset = 0
            while offset < t:
                pos = bas_pos + offset
                step_len = min(t - offset, self.cfg.chunk_size - (pos % self.cfg.chunk_size))
                h_seg = h0[:, offset : offset + step_len]
                a_seg = anchor[:, offset : offset + step_len]
                if self.boundary is not None:
                    h_seg = self.boundary.apply_feedback(h_seg, slot_st)
                r1_seg = h_seg

                if collect_snapshots:
                    loop_snaps: list[list[tuple[Tensor, Tensor | None]]] = []
                    for r_idx in range(n_loops):
                        h_seg, snaps_r = self._run_loop_step_snapshots(
                            h_seg, a_seg, r_idx, gdn_in[r_idx]
                        )
                        loop_snaps.append(snaps_r)
                        gdn_in[r_idx] = snaps_r[-1]
                        if r_idx == 0:
                            r1_seg = h_seg
                    h_seg_f = h_seg.to(slot_st.dtype)
                    for p in range(step_len):
                        slot_st, chunk_sum = self._commit_if_boundary(
                            slot_st,
                            chunk_sum,
                            h_seg_f[:, p] if self.cfg.boundary_pool == "mean" else h_seg[:, p : p + 1],
                            pos + p + 1,
                            is_full_seg=False,
                        )
                        gdn_snap: list[tuple[Tensor, Tensor | None] | None] = [
                            loop_snaps[r][p] for r in range(n_loops)
                        ]
                        snapshots.append(RDFLState(gdn_snap, slot_st, pos + p + 1, chunk_sum))
                else:
                    single = step_len == 1
                    for r_idx in range(n_loops):
                        h_seg, gdn_in[r_idx] = self._run_loop_step(
                            h_seg, a_seg, r_idx, gdn_in[r_idx], single_step=single
                        )
                        if r_idx == 0:
                            r1_seg = h_seg
                    slot_st, chunk_sum = self._commit_if_boundary(
                        slot_st, chunk_sum, h_seg, pos + step_len, is_full_seg=True
                    )

                r1_chunks.append(r1_seg)
                final_chunks.append(h_seg)
                offset += step_len

            h_r1 = torch.cat(r1_chunks, dim=1)
            h_fin = torch.cat(final_chunks, dim=1)
            final_st = RDFLState(gdn_in, slot_st, bas_pos + t, chunk_sum)

        hn = self.final_norm(h_fin)
        logits = self.head(hn)
        # Strictly detached K3 projection -> 0.000 gradient tax on core/head
        z_k3 = self.k3_adapter(h_r1.detach())
        hn_k3 = _apply_rmsnorm(z_k3, self.final_norm.weight.detach(), self.final_norm.eps)
        logits_r1 = F.linear(hn_k3, self.head.weight.detach())
        return logits, logits_r1, h_fin, final_st, snapshots

    def forward_stateful(
        self,
        x: Tensor,
        state: RDFLState | None = None,
        num_loops: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, RDFLState]:
        """Exact chunk-boundary-aligned forward returning (logits, logits_r1, h_fin, state)."""
        logits, logits_r1, h_fin, final_st, _ = self._forward_core(
            x, state=state, num_loops=num_loops, collect_snapshots=False
        )
        return logits, logits_r1, h_fin, final_st

    def forward_with_snapshots(
        self,
        x: Tensor,
        state: RDFLState | None = None,
        num_loops: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, list[RDFLState]]:
        """Forward pass returning per-token RDFLState snapshots for 0-extra-pass rollback."""
        logits, logits_r1, h_fin, _, snapshots = self._forward_core(
            x, state=state, num_loops=num_loops, collect_snapshots=True
        )
        return logits, logits_r1, h_fin, snapshots

    @overload
    def forward(
        self,
        x: Tensor,
        num_loops: int | None = None,
        return_pre_norm: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        num_loops: int | None = None,
        return_pre_norm: Literal[True] = ...,
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        num_loops: int | None = None,
        return_pre_norm: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        logits, _, h_fin, _ = self.forward_stateful(x, state=None, num_loops=num_loops)
        if return_pre_norm:
            return logits, h_fin
        return logits

    def step(
        self,
        x_t: Tensor,
        state: RDFLState | None = None,
        num_loops: int | None = None,
    ) -> tuple[Tensor, RDFLState]:
        logits, _, _, new_state = self.forward_stateful(x_t, state=state, num_loops=num_loops)
        return logits, new_state

    def mtp_logits(self, h_fin: Tensor) -> list[Tensor]:
        """Compute t+2..t+4 lookahead logits from pre-norm h_fin with a single final_norm."""
        if self.cfg.mtp_stop_grad:
            h_in = h_fin.detach()
            wn = self.final_norm.weight.detach()
            wh = self.head.weight.detach()
            out = []
            for adapter in self.mtp_heads:
                z = adapter(h_in)
                zn = _apply_rmsnorm(z, wn, self.final_norm.eps)
                out.append(F.linear(zn, wh))
            return out
        return [self.head(self.final_norm(adapter(h_fin))) for adapter in self.mtp_heads]


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
