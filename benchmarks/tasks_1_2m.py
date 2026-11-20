"""Deterministic byte-level synthetic benchmarks for RDFL 1.20M suite."""

from __future__ import annotations

import itertools

import torch
from torch import Tensor

_S5_PERMS: list[tuple[int, ...]] = list(itertools.permutations(range(5)))
_S5_INDEX: dict[tuple[int, ...], int] = {p: i for i, p in enumerate(_S5_PERMS)}
_S5_GENS: list[tuple[int, ...]] = [
    (1, 0, 2, 3, 4),
    (0, 2, 1, 3, 4),
    (0, 1, 3, 2, 4),
    (0, 1, 2, 4, 3),
    (1, 2, 3, 4, 0),
    (4, 0, 1, 2, 3),
    (2, 1, 0, 4, 3),
    (0, 3, 2, 1, 4),
]
_S5_COMPOSE: list[list[int]] = [
    [_S5_INDEX[tuple(p[g[i]] for i in range(5))] for g in _S5_GENS] for p in _S5_PERMS
]


def _sample_parity64(batch: int, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """Running XOR parity over 64 binary inputs encoded as ASCII bytes '0'/'1'."""
    bits = torch.randint(0, 2, (batch, 64), generator=gen)
    parity = bits.cumsum(dim=1) % 2
    x = 48 + bits
    y = 48 + parity
    mask = torch.ones_like(y, dtype=torch.bool)
    return x, y, mask


def _sample_s5_perm32(batch: int, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """32-step non-abelian S_5 group composition (120 states, 8 odd/even generators)."""
    g_idx = torch.randint(0, len(_S5_GENS), (batch, 32), generator=gen)
    x = 120 + g_idx
    y = torch.empty((batch, 32), dtype=torch.long)
    for b in range(batch):
        state = 0
        for t in range(32):
            state = _S5_COMPOSE[state][int(g_idx[b, t].item())]
            y[b, t] = state
    mask = torch.ones_like(y, dtype=torch.bool)
    return x, y, mask


def _sample_mqar_overwrite(batch: int, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """Associative key-value recall with mandatory key overwrites (sep=254)."""
    n_keys, n_over, n_q = 12, 6, 8
    seq_len = 2 * (n_keys + n_over) + 1 + 2 * n_q
    x = torch.zeros((batch, seq_len), dtype=torch.long)
    y = torch.zeros((batch, seq_len), dtype=torch.long)
    mask = torch.zeros((batch, seq_len), dtype=torch.bool)
    for b in range(batch):
        keys = torch.randperm(32, generator=gen)[:n_keys].tolist()
        vals = torch.randint(32, 96, (n_keys,), generator=gen).tolist()
        table = dict(zip(keys, vals, strict=True))
        seq: list[int] = []
        for k, v in zip(keys, vals, strict=True):
            seq.extend((k, v))
        over_keys = keys[:n_over]
        new_vals = torch.randint(32, 96, (n_over,), generator=gen).tolist()
        for k, v in zip(over_keys, new_vals, strict=True):
            table[k] = v
            seq.extend((k, v))
        seq.append(254)
        q_order = torch.randperm(n_keys, generator=gen)[:n_q].tolist()
        for qi in q_order:
            k = keys[qi]
            pos = len(seq)
            seq.extend((k, table[k]))
            mask[b, pos] = True
        t_seq = torch.tensor(seq, dtype=torch.long)
        x[b, :-1] = t_seq[:-1]
        y[b, :-1] = t_seq[1:]
    return x, y, mask


def _sample_graf_izli(batch: int, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """3-hop directed graph traversal over 16 nodes (sep=253, query=254)."""
    n_nodes, n_hops = 16, 3
    seq_len = 2 * n_nodes + 1 + 1 + n_hops
    x = torch.zeros((batch, seq_len), dtype=torch.long)
    y = torch.zeros((batch, seq_len), dtype=torch.long)
    mask = torch.zeros((batch, seq_len), dtype=torch.bool)
    for b in range(batch):
        perm = torch.randperm(n_nodes, generator=gen).tolist()
        nxt = {perm[i]: perm[(i + 1) % n_nodes] for i in range(n_nodes)}
        order = torch.randperm(n_nodes, generator=gen).tolist()
        seq: list[int] = []
        for u in order:
            seq.extend((u, nxt[u]))
        seq.append(253)
        cur = int(torch.randint(0, n_nodes, (1,), generator=gen).item())
        seq.append(cur)
        for _ in range(n_hops):
            pos = len(seq) - 1
            cur = nxt[cur]
            seq.append(cur)
            mask[b, pos] = True
        t_seq = torch.tensor(seq, dtype=torch.long)
        x[b, :-1] = t_seq[:-1]
        y[b, :-1] = t_seq[1:]
    return x, y, mask


TASK_REGISTRY = {
    "parity64": _sample_parity64,
    "s5_perm32": _sample_s5_perm32,
    "mqar_overwrite": _sample_mqar_overwrite,
    "graf_izli": _sample_graf_izli,
}


def sample_task_batch(task: str, batch: int, gen: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """Sample deterministic (inputs, targets, mask) for one of the 4 algorithmic tasks."""
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Valid: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[task](batch, gen)
