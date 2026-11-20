"""Lossless (0-byte-mismatch) speculative decoding for RDFLModel (K3 and MTP-4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from rdfl.model import RDFLModel, RDFLState


@dataclass(frozen=True)
class SpeculativeDecodeResult:
    """Output tokens and verification telemetry from lossless speculative decoding."""

    tokens: Tensor
    accepted_drafts: int
    proposed_drafts: int
    verify_calls: int
    rollbacks: int
    extra_passes: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed draft tokens accepted by the target model."""
        return self.accepted_drafts / max(1, self.proposed_drafts)

    @property
    def tokens_per_verify(self) -> float:
        """Average committed tokens produced per parallel target verification call."""
        return float(self.tokens.shape[1]) / max(1, self.verify_calls)


def _validate_single_prompt(prompt: Tensor, max_new_tokens: int) -> None:
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("prompt must have shape [1, T] with T >= 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")


def _k3_draft_step(
    model: RDFLModel,
    tok: Tensor,
    gdn_r0: tuple[Tensor, Tensor | None] | None,
    slot_st: Tensor,
    chunk_sum: Tensor | None,
    pos: int,
) -> tuple[Tensor, tuple[Tensor, Tensor | None], Tensor, Tensor | None]:
    """Run a single cheap R=1 + K3 adapter step to propose one draft byte."""
    h0 = model.emb(tok)
    anchor = model.n_anchor(h0)
    h_seg = model.boundary.apply_feedback(h0, slot_st) if model.boundary is not None else h0
    h_r1, new_gdn_r0 = model._run_loop_step(h_seg, anchor, 0, gdn_r0, single_step=True)
    slot_st, chunk_sum = model._commit_if_boundary(
        slot_st,
        chunk_sum,
        h_r1[:, 0] if model.cfg.boundary_pool == "mean" else h_r1,
        pos + 1,
        is_full_seg=False,
    )
    z_k3 = model.k3_adapter(h_r1)
    hn_k3 = model.final_norm(z_k3)
    draft_tok = F.linear(hn_k3, model.head.weight).argmax(dim=-1)
    return draft_tok, new_gdn_r0, slot_st, chunk_sum


def _draft_k3_sequence(
    model: RDFLModel,
    cur_tok: Tensor,
    state: RDFLState,
    draft_k: int,
) -> list[Tensor]:
    """Propose `draft_k` sequential draft tokens using only R=1 + K3."""
    gdn_r0 = state.gdn_states[0]
    slot_st = state.slot_state
    chunk_sum = state.chunk_sum
    pos = state.pos
    inp = cur_tok
    drafts: list[Tensor] = []
    for _ in range(draft_k):
        nxt, gdn_r0, slot_st, chunk_sum = _k3_draft_step(
            model, inp, gdn_r0, slot_st, chunk_sum, pos
        )
        drafts.append(nxt)
        inp = nxt
        pos += 1
    return drafts


def _count_accepted(drafts: list[Tensor], v_logits: Tensor) -> tuple[int, list[Tensor]]:
    """Compare drafts against target verification logits and collect accepted + corrected tokens."""
    n_accept = 0
    committed: list[Tensor] = []
    for idx, draft_tok in enumerate(drafts):
        target_tok = v_logits[:, idx : idx + 1, :].argmax(dim=-1)
        if int(draft_tok.item()) == int(target_tok.item()):
            committed.append(draft_tok)
            n_accept += 1
        else:
            committed.append(target_tok)
            return n_accept, committed
    bonus_tok = v_logits[:, len(drafts) : len(drafts) + 1, :].argmax(dim=-1)
    committed.append(bonus_tok)
    return n_accept, committed


@torch.no_grad()
def greedy_decode(model: RDFLModel, prompt: Tensor, max_new_tokens: int) -> Tensor:
    """Exact sequential greedy reference decoding using full R=num_loops."""
    _validate_single_prompt(prompt, max_new_tokens)
    logits, _, _, state = model.forward_stateful(prompt, state=None)
    cur = logits[:, -1:, :].argmax(dim=-1)
    out = [cur]
    for _ in range(max_new_tokens - 1):
        logits, _, _, state = model.forward_stateful(cur, state=state)
        cur = logits[:, -1:, :].argmax(dim=-1)
        out.append(cur)
    return torch.cat(out, dim=1)


@torch.no_grad()
def speculative_decode_k3(
    model: RDFLModel,
    prompt: Tensor,
    max_new_tokens: int,
    draft_k: int = 2,
) -> SpeculativeDecodeResult:
    """Lossless K3 (R=1 -> full R) speculative decoding with 0-extra-pass snapshot rollback."""
    _validate_single_prompt(prompt, max_new_tokens)
    if draft_k < 1:
        raise ValueError("draft_k must be >= 1")
    logits, _, _, state = model.forward_stateful(prompt, state=None)
    cur = logits[:, -1:, :].argmax(dim=-1)
    out = [cur]
    accepted = proposed = verify_calls = rollbacks = 0

    while len(out) < max_new_tokens:
        rem = max_new_tokens - len(out)
        k = min(draft_k, rem)
        drafts = _draft_k3_sequence(model, cur, state, k)
        proposed += k
        verify_in = torch.cat([cur, *drafts], dim=1)
        v_logits, _, _, snapshots = model.forward_with_snapshots(verify_in, state=state)
        verify_calls += 1

        n_acc, step_toks = _count_accepted(drafts, v_logits)
        accepted += n_acc
        take = step_toks[:rem]
        out.extend(take)
        if len(out) >= max_new_tokens:
            break
        cur = take[-1]
        if n_acc < k:
            rollbacks += 1
        state = snapshots[n_acc]

    return SpeculativeDecodeResult(
        tokens=torch.cat(out, dim=1),
        accepted_drafts=accepted,
        proposed_drafts=proposed,
        verify_calls=verify_calls,
        rollbacks=rollbacks,
        extra_passes=0,
    )


@torch.no_grad()
def speculative_decode_mtp4(
    model: RDFLModel,
    prompt: Tensor,
    max_new_tokens: int,
) -> SpeculativeDecodeResult:
    """Lossless MTP-4 (t+2..t+4 -> full R) speculative decoding with 0-extra-pass rollback."""
    _validate_single_prompt(prompt, max_new_tokens)
    n_heads = len(model.mtp_heads)
    if n_heads < 1:
        raise ValueError("model.mtp_heads must contain at least 1 lookahead head")
    logits, _, h_fin, state = model.forward_stateful(prompt, state=None)
    cur = logits[:, -1:, :].argmax(dim=-1)
    hn_cur = h_fin[:, -1:, :]
    out = [cur]
    accepted = proposed = verify_calls = rollbacks = 0

    while len(out) < max_new_tokens:
        rem = max_new_tokens - len(out)
        k = min(n_heads, rem)
        drafts = [lg.argmax(dim=-1) for lg in model.mtp_logits(hn_cur)[:k]]
        proposed += k
        verify_in = torch.cat([cur, *drafts], dim=1)
        v_logits, _, v_h_fin, snapshots = model.forward_with_snapshots(verify_in, state=state)
        verify_calls += 1

        n_acc, step_toks = _count_accepted(drafts, v_logits)
        accepted += n_acc
        take = step_toks[:rem]
        out.extend(take)
        if len(out) >= max_new_tokens:
            break
        cur = take[-1]
        hn_cur = v_h_fin[:, n_acc : n_acc + 1, :]
        if n_acc < k:
            rollbacks += 1
        state = snapshots[n_acc]

    return SpeculativeDecodeResult(
        tokens=torch.cat(out, dim=1),
        accepted_drafts=accepted,
        proposed_drafts=proposed,
        verify_calls=verify_calls,
        rollbacks=rollbacks,
        extra_passes=0,
    )
