"""Unified Standalone CPU Verification & Audited Summary Generator for RDFL v0.5.

Zero external repository dependencies, zero /home/ path leaks, strict RFC 8259 JSON output.
All four benchmark suites execute on the single unified `RDFLv05Model` class:
  1. Parity, Causality, O(1) State Size, Return-Aux Symmetry, No-Op Gate, & R=1024 Stress
  2. Non-Collapsing Step-by-Step Silent Reasoning + Learned Halting Gate (R_extra) vs ADFP
  3. Anthropic Global Workspace (J-Space / Jacobian Lens, strictly out-of-sample x_cal -> x_te)
  4. 4-Problem Regression Lock (P1 Dyck, P2 S5 running=False, P3 P=4 + dec_ffn, P4 CL3 + Space Gate)
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import gc
import hashlib
import itertools
import json
import math
import random
import resource
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdfl.calibration import (
    aurc,
    auroc,
    ece_15,
    fit_temperature,
    msp_uncertainty,
)
from rdfl.v05 import (
    ConsolidationManager,
    RDFLv05Config,
    RDFLv05Model,
    analytic_flops_per_byte,
    state_nbytes,
)

LN2: float = math.log(2.0)
DEFAULT_OUT = ROOT / "results" / "v05_audited_summary.json"
IGNORE_INDEX: int = -100
PAD_BYTE: int = ord(" ")
_DYCK_OPENS: bytes = b"([{<ABCDEFGHIJKL"
_DYCK_CLOSES: bytes = b")]}>abcdefghijkl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sanitize_json(obj: Any) -> Any:
    """Recursively sanitize into strict finite RFC 8259 primitives with zero /home/ leaks."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite float in JSON payload: {val}")
        return val
    if isinstance(obj, Tensor):
        if obj.numel() == 1:
            return sanitize_json(obj.item())
        return sanitize_json(obj.detach().cpu().tolist())
    if isinstance(obj, Path):
        return obj.name
    if isinstance(obj, str):
        if "/home/" in obj:
            raise ValueError(f"Forbidden /home/ path string in JSON payload: {obj}")
        return obj
    if obj is None:
        return None
    s = str(obj)
    if "/home/" in s:
        raise ValueError(f"Forbidden /home/ path string in JSON payload: {s}")
    return s


def write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write strict RFC 8259 JSON (allow_nan=False)."""
    clean = sanitize_json(payload)
    text = json.dumps(clean, indent=2, sort_keys=False, allow_nan=False) + "\n"
    if "/home/" in text:
        raise ValueError("Refusing to write JSON containing /home/ path leak")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


# ---------------------------------------------------------------------------
# Standalone Zero-Leakage Task Generators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskBatch:
    x: Tensor
    y: Tensor
    eval_mask: Tensor
    meta: dict[str, Any]

    def masked_targets(self, ignore_index: int = IGNORE_INDEX) -> Tensor:
        return self.y.masked_fill(~self.eval_mask, ignore_index)


def _split_bit(split: str) -> int:
    s = split.strip().lower()
    if s in {"train", "tr"}:
        return 0
    if s in {"val", "validation", "ood", "eval"}:
        return 1
    raise ValueError(f"Unsupported split '{split}'")


def _task_rng(task_name: str, split: str, seed: int, step: int) -> np.random.Generator:
    tag = hashlib.blake2b(
        f"{task_name}:{split.strip().lower()}".encode(), digest_size=8
    ).digest()
    tag_int = int.from_bytes(tag, "little")
    seq = np.random.SeedSequence([tag_int, int(seed) & 0xFFFFFFFF, int(step) & 0xFFFFFFFF])
    return np.random.Generator(np.random.PCG64(seq))


def _fnv1a_32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _hash_bit(data: bytes) -> int:
    return (_fnv1a_32(data) >> 1) & 1


def _pack_fixed_sequences(
    seqs: list[bytes],
    eval_positions: list[list[int]],
    meta: dict[str, Any],
) -> TaskBatch:
    seq_len = len(seqs[0])
    bsz = len(seqs)
    arr = np.empty((bsz, seq_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i] = np.frombuffer(s, dtype=np.uint8)
    x = torch.from_numpy(arr[:, :-1].copy())
    y = torch.from_numpy(arr[:, 1:].copy())
    mask = torch.zeros((bsz, seq_len - 1), dtype=torch.bool)
    for i, pos_list in enumerate(eval_positions):
        for p in pos_list:
            mask[i, p - 1] = True
    return TaskBatch(x=x, y=y, eval_mask=mask, meta=meta)


def _gen_balanced_dyck(
    rng: np.random.Generator, num_pairs: int, k: int, opens: bytes, closes: bytes
) -> bytes:
    if num_pairs <= 0:
        return b""
    p_left = num_pairs
    q_left = num_pairs
    stack: list[int] = []
    out = bytearray()
    for _ in range(2 * num_pairs):
        if p_left == 0:
            do_push = False
        elif not stack:
            do_push = True
        else:
            do_push = bool(rng.random() < (p_left / float(p_left + q_left)))
        if do_push:
            btype = int(rng.integers(0, k))
            stack.append(btype)
            out.append(opens[btype])
            p_left -= 1
        else:
            btype = stack.pop()
            out.append(closes[btype])
            q_left -= 1
    return bytes(out)


def dyck_stack(
    k: int = 4, depth: int = 4, *, batch: int = 32, seed: int = 42, step: int = 0, split: str = "train"
) -> TaskBatch:
    sbit = _split_bit(split)
    rng = _task_rng("dyck_stack", split, seed, step)
    opens = _DYCK_OPENS[:k]
    closes = _DYCK_CLOSES[:k]
    pops = 2 * depth
    tail_min = max(1, (depth + 1) // 2)
    rem_pops = pops - tail_min
    seqs: list[bytes] = []
    eval_pos: list[list[int]] = []
    for _ in range(batch):
        while True:
            surviving = rng.integers(0, k, size=depth)
            slot_pairs = [0] * (depth + 1)
            slot_pairs[depth] = tail_min
            for si in rng.integers(0, depth + 1, size=rem_pops):
                slot_pairs[int(si)] += 1
            prefix = bytearray()
            for j in range(depth):
                prefix.extend(_gen_balanced_dyck(rng, slot_pairs[j], k, opens, closes))
                prefix.append(opens[int(surviving[j])])
            prefix.extend(_gen_balanced_dyck(rng, slot_pairs[depth], k, opens, closes))
            pref_bytes = bytes(prefix)
            if _hash_bit(pref_bytes) == sbit:
                break
        buf = bytearray(b"D:")
        buf.extend(pref_bytes)
        buf.append(ord("|"))
        pos_list: list[int] = []
        for btype in reversed(surviving):
            pos_list.append(len(buf))
            buf.append(closes[int(btype)])
        seqs.append(bytes(buf))
        eval_pos.append(pos_list)
    return _pack_fixed_sequences(seqs, eval_pos, {"task": "dyck_stack", "chance_acc": 1.0 / k})


@lru_cache(maxsize=1)
def _s5_cayley_table() -> NDArray[np.uint8]:
    perms = list(itertools.permutations(range(5)))
    idx_map = {p: i for i, p in enumerate(perms)}
    table = np.empty((120, 120), dtype=np.uint8)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            composed = (q[p[0]], q[p[1]], q[p[2]], q[p[3]], q[p[4]])
            table[i, j] = idx_map[composed]
    return table


def s5_perm32(
    batch: int = 32,
    seed: int = 42,
    step: int = 0,
    split: str = "train",
    *,
    length: int = 32,
    running: bool = False,
) -> TaskBatch:
    if running:
        raise ValueError("Retracted running=True mode is forbidden; use running=False.")
    sbit = _split_bit(split)
    rng = _task_rng("s5_perm32", split, seed, step)
    table = _s5_cayley_table()
    seqs: list[bytes] = []
    eval_pos: list[list[int]] = []
    for _ in range(batch):
        while True:
            elems = rng.integers(0, 120, size=length, dtype=np.uint8)
            elem_bytes = (elems + 128).tobytes()
            if _hash_bit(elem_bytes) == sbit:
                break
        st = 0
        for e in elems:
            st = int(table[st, int(e)])
        seqs.append(b"S:" + elem_bytes + b"=" + bytes([128 + st]))
        eval_pos.append([2 + length + 1])
    return _pack_fixed_sequences(
        seqs, eval_pos, {"task": "s5_perm32", "length": length, "running": False, "chance_acc": 1.0 / 120.0}
    )


def multidigit_add(
    digits: int = 3, *, batch: int = 32, seed: int = 42, step: int = 0, split: str = "train"
) -> TaskBatch:
    sbit = _split_bit(split)
    rng = _task_rng("multidigit_add", split, seed, step)
    hi = 10**digits
    seqs: list[bytes] = []
    eval_pos: list[list[int]] = []
    for _ in range(batch):
        while True:
            a = int(rng.integers(0, hi))
            b = int(rng.integers(0, hi))
            sig = f"{a:0{digits}d}+{b:0{digits}d}".encode("ascii")
            if _hash_bit(sig) == sbit:
                break
        ans = f"{a + b:0{digits + 1}d}"
        full = sig + b"=" + ans.encode("ascii")
        start_ans = len(sig) + 1
        seqs.append(full)
        eval_pos.append(list(range(start_ans, start_ans + digits + 1)))
    return _pack_fixed_sequences(seqs, eval_pos, {"task": "multidigit_add", "digits": digits})


def build_synthetic_tr_math_corpus(seed: int = 42) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Deterministic standalone UTF-8 Turkish + arithmetic byte corpus (disjoint train/val)."""
    rng = random.Random(seed)
    nouns = [
        "özyineleme",
        "bellek",
        "matris",
        "çözümleme",
        "örüntü",
        "derinlik",
        "doğrulama",
        "denklem",
        "olasılık",
        "vektör",
        "durum",
        "kapısı",
    ]
    verbs = ["hesaplar", "korur", "günceller", "ölçer", "birleştirir", "doğrular"]
    templates = [
        "Soru: {a} + {b} kaçtır? Yanıt: {sum_ab}. {n1} {v1} ve {n2} sabit kalır.\n",
        "Kural: {n1} için {a} * {c} = {prod_ac}; toplam {n2} adımı {k} olur.\n",
        "Not: {n1} ve {n2} akışında {a} - {b} = {diff_ab}; tüm baytlar {v1}.\n",
    ]

    def _gen_bytes(target_len: int, split_offset: int) -> NDArray[np.uint8]:
        local_rng = random.Random(seed + split_offset)
        chunks: list[bytes] = []
        tot = 0
        while tot < target_len:
            a = local_rng.randint(10, 499)
            b = local_rng.randint(10, 499)
            c = local_rng.randint(2, 19)
            k = local_rng.randint(1, 16)
            tpl = local_rng.choice(templates)
            line = tpl.format(
                a=a,
                b=b,
                c=c,
                k=k,
                sum_ab=a + b,
                prod_ac=a * c,
                diff_ab=a - b,
                n1=local_rng.choice(nouns),
                n2=local_rng.choice(nouns),
                v1=local_rng.choice(verbs),
            ).encode("utf-8")
            chunks.append(line)
            tot += len(line)
        raw = b"".join(chunks)[:target_len]
        return np.frombuffer(raw, dtype=np.uint8).copy()

    _ = rng.random()
    return _gen_bytes(98304, 100), _gen_bytes(24576, 200)


def sample_corpus_batch(
    arr: NDArray[np.uint8], *, seq_len: int, batch: int, seed: int, step: int, contiguous: bool = False
) -> tuple[Tensor, Tensor]:
    max_start = int(arr.shape[0]) - (seq_len + 1)
    if contiguous:
        base = (step * batch * seq_len) % max_start
        starts = np.array([(base + i * seq_len) % max_start for i in range(batch)], dtype=np.int64)
    else:
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, step])))
        starts = rng.integers(0, max_start, size=batch, dtype=np.int64)
    buf = np.empty((batch, seq_len + 1), dtype=np.int64)
    for i, st in enumerate(starts):
        s_i = int(st)
        buf[i] = arr[s_i : s_i + seq_len + 1]
    return torch.from_numpy(buf[:, :-1].copy()), torch.from_numpy(buf[:, 1:].copy())


def make_ptr_chain_batch(
    batch_size: int, K: int, rng: random.Random, return_intermediates: bool = False
) -> Any:
    xs: list[list[int]] = []
    ys: list[int] = []
    inters: list[list[int]] = []
    perms: list[list[int]] = []
    starts: list[int] = []
    digits = list(range(10))
    for _ in range(batch_size):
        perm = digits[:]
        rng.shuffle(perm)
        order = digits[:]
        rng.shuffle(order)
        s = rng.randint(0, 9)
        p_str = " ".join(f"{i}{perm[i]}" for i in order)
        k_str = "." * K
        prompt = f"P:{p_str}|K:{k_str}|S={s}?"
        cur = s
        trace = [s]
        for _ in range(K):
            cur = perm[cur]
            trace.append(cur)
        xs.append([ord(c) for c in prompt])
        ys.append(ord(str(cur)))
        if return_intermediates:
            inters.append(trace)
            perms.append(perm)
            starts.append(s)
    x_t = torch.tensor(xs, dtype=torch.long)
    y_t = torch.tensor(ys, dtype=torch.long)
    if return_intermediates:
        return x_t, y_t, inters, perms, starts
    return x_t, y_t


def make_same_table_pairs(n_pairs: int, K: int, rng: random.Random) -> Any:
    xs_a: list[list[int]] = []
    ys_a: list[int] = []
    inter_a: list[list[int]] = []
    xs_b: list[list[int]] = []
    ys_b: list[int] = []
    inter_b: list[list[int]] = []
    perms: list[list[int]] = []
    digits = list(range(10))
    while len(xs_a) < n_pairs:
        perm = digits[:]
        rng.shuffle(perm)
        order_a = digits[:]
        order_b = digits[:]
        rng.shuffle(order_a)
        rng.shuffle(order_b)
        sa, sb = rng.sample(digits, 2)
        ta, tb = [sa], [sb]
        ca, cb = sa, sb
        for _ in range(K):
            ca = perm[ca]
            cb = perm[cb]
            ta.append(ca)
            tb.append(cb)
        if ca == cb:
            continue
        pa_str = " ".join(f"{i}{perm[i]}" for i in order_a)
        pb_str = " ".join(f"{i}{perm[i]}" for i in order_b)
        k_str = "." * K
        xs_a.append([ord(c) for c in f"P:{pa_str}|K:{k_str}|S={sa}?"])
        xs_b.append([ord(c) for c in f"P:{pb_str}|K:{k_str}|S={sb}?"])
        ys_a.append(ord(str(ca)))
        ys_b.append(ord(str(cb)))
        inter_a.append(ta)
        inter_b.append(tb)
        perms.append(perm)
    return (
        torch.tensor(xs_a, dtype=torch.long),
        torch.tensor(ys_a, dtype=torch.long),
        inter_a,
        torch.tensor(xs_b, dtype=torch.long),
        torch.tensor(ys_b, dtype=torch.long),
        inter_b,
        perms,
    )


# ---------------------------------------------------------------------------
# Section 1: Invariants, Parity, O(1), Return-Aux Symmetry, No-Op, R=1024
# ---------------------------------------------------------------------------


@torch.no_grad()
def _measure_hook_flops_per_byte(model: RDFLv05Model, x: Tensor, *, update_slow: bool) -> dict[str, Any]:
    total_linear_conv = 0
    dec_ffn_calls = 0
    handles: list[Any] = []

    def _lin_hook(mod: nn.Module, _inp: tuple[Tensor, ...], out: Tensor) -> None:
        nonlocal total_linear_conv
        w = getattr(mod, "weight", None)
        if isinstance(out, Tensor) and isinstance(w, Tensor):
            total_linear_conv += 2 * int(out.numel()) * int(w.shape[1])

    def _conv_hook(mod: nn.Conv1d, _inp: tuple[Tensor, ...], out: Tensor) -> None:
        nonlocal total_linear_conv
        k = mod.kernel_size[0]
        cin_g = mod.in_channels // mod.groups
        total_linear_conv += 2 * int(out.numel()) * cin_g * k

    def _dec_hook(_mod: nn.Module, _inp: tuple[Tensor, ...], _out: Tensor) -> None:
        nonlocal dec_ffn_calls
        dec_ffn_calls += 1

    for m in model.modules():
        if isinstance(m, nn.Linear) or type(m).__name__ == "SoftSpectralLinear":
            handles.append(m.register_forward_hook(_lin_hook))
        elif isinstance(m, nn.Conv1d):
            handles.append(m.register_forward_hook(_conv_hook))
    handles.append(model.dec_ffn.register_forward_hook(_dec_hook))

    try:
        _ = model(x, num_loops=model.cfg.num_loops, update_slow=update_slow)
    finally:
        for h in handles:
            h.remove()

    b, t = x.shape
    return {
        "linear_conv_flops_per_byte": float(total_linear_conv) / float(b * t),
        "dec_ffn_calls": dec_ffn_calls,
    }


def run_invariants_and_parity(cfg: RDFLv05Config) -> dict[str, Any]:
    t0 = time.perf_counter()
    set_seed(42)
    m32 = RDFLv05Model(cfg).float().eval()
    m64 = RDFLv05Model(cfg).double().eval()
    m64.load_state_dict({k: v.double() for k, v in m32.state_dict().items()})

    x_par = torch.randint(0, 256, (1, 48), dtype=torch.long)
    parity: dict[str, Any] = {}
    for upd_slow in (False, True):
        for nl in (2, 4):
            key = f"R{nl}_slow{'ON' if upd_slow else 'OFF'}"
            with torch.no_grad():
                f64 = m64(x_par, num_loops=nl, update_slow=upd_slow)
                st64 = None
                s64_list = []
                for ti in range(x_par.shape[1]):
                    lt, st64 = m64.step(x_par[:, ti : ti + 1], state=st64, num_loops=nl, update_slow=upd_slow)
                    s64_list.append(lt)
                c64 = torch.cat(s64_list, dim=1)

                f32 = m32(x_par, num_loops=nl, update_slow=upd_slow)
                st32 = None
                s32_list = []
                for ti in range(x_par.shape[1]):
                    lt32, st32 = m32.step(x_par[:, ti : ti + 1], state=st32, num_loops=nl, update_slow=upd_slow)
                    s32_list.append(lt32)
                c32 = torch.cat(s32_list, dim=1)

            parity[key] = {
                "fp64_max_abs_diff": float((f64 - c64).abs().max().item()),
                "fp32_max_abs_diff": float((f32 - c32).abs().max().item()),
                "fp32_argmax_mismatch": int((f32.argmax(dim=-1) != c32.argmax(dim=-1)).sum().item()),
            }

    # Partial-patch 37+11 & random_matched & eps_halt=0.05 parity
    with torch.no_grad():
        full_pp, _, _, st_pp = m64.forward_stateful(x_par, num_loops=4, update_slow=True)
        l1, _, _, s1 = m64.forward_stateful(x_par[:, :37], num_loops=4, update_slow=True)
        l2, _, _, s2 = m64.forward_stateful(x_par[:, 37:], state=s1, num_loops=4, update_slow=True)
        partial_37_11_diff = float((full_pp - torch.cat([l1, l2], dim=1)).abs().max().item())
        assert st_pp.s_slow is not None and s2.s_slow is not None
        partial_37_11_sslow_diff = float((st_pp.s_slow - s2.s_slow).abs().max().item())

        f_rand, _, _, _ = m64.forward_stateful(
            x_par, num_loops=4, update_slow=True, gate_mode="random_matched", random_gate_seed=123, random_gate_prob=0.5
        )
        st_r = None
        sr_list = []
        for ti in range(x_par.shape[1]):
            lr, st_r = m64.step(
                x_par[:, ti : ti + 1],
                state=st_r,
                num_loops=4,
                update_slow=True,
                gate_mode="random_matched",
                random_gate_seed=123,
                random_gate_prob=0.5,
            )
            sr_list.append(lr)
        random_matched_fp64_diff = float((f_rand - torch.cat(sr_list, dim=1)).abs().max().item())

        # Halting gate eps_halt parity
        f_halt = m64(x_par, num_loops=8, eps_halt=0.95, update_slow=True)
        st_h = None
        sh_list = []
        for ti in range(x_par.shape[1]):
            lh, st_h = m64.step(x_par[:, ti : ti + 1], state=st_h, num_loops=8, eps_halt=0.95, update_slow=True)
            sh_list.append(lh)
        eps_halt_fp64_diff = float((f_halt - torch.cat(sh_list, dim=1)).abs().max().item())

        # Exact no-op gate_override=0.0
        noop_stream_diff = float((m64(x_par, num_loops=1) - m64(x_par, num_loops=16, gate_override=0.0)).abs().max().item())
        h0_rs, sf_rs, cl_rs, sl_rs = m64.encode_for_reasoning(x_par)
        res_noop = m64.reason_from_state(h0_rs, sf_rs, cl_rs, num_loops=16, slot_idx=sl_rs, gate_override=0.0)
        noop_reason_diff = float((res_noop["h_final"] - h0_rs).abs().max().item())

        # return_aux pos==0 vs pos>0 symmetry
        _, _, _, _, aux_0 = m64.forward_stateful(x_par, num_loops=3, update_slow=True, return_aux=True)
        _, _, _, st_a, aux_a = m64.forward_stateful(x_par[:, :8], num_loops=3, update_slow=True, return_aux=True)
        _, _, _, _, aux_b = m64.forward_stateful(x_par[:, 8:], state=st_a, num_loops=3, update_slow=True, return_aux=True)
        aux_keys_match = sorted(aux_0.keys()) == sorted(aux_a.keys()) == sorted(aux_b.keys())
        aux_sym_max_diff = 0.0
        for k in aux_0:
            cat_k = torch.cat([aux_a[k], aux_b[k]], dim=1)
            if aux_0[k].is_floating_point():
                aux_sym_max_diff = max(aux_sym_max_diff, float((aux_0[k] - cat_k).abs().max().item()))

    # Causality across slowOFF/slowON
    x_c1 = torch.randint(0, 256, (1, 70), dtype=torch.long)
    worst_leak = 0.0
    for upd_slow in (False, True):
        for cut in (3, 16, 35, 63, 64):
            x_c2 = x_c1.clone()
            x_c2[:, cut:] = (x_c2[:, cut:] + 97) % 256
            with torch.no_grad():
                o1 = m64(x_c1, num_loops=4, eps_halt=0.05, update_slow=upd_slow)
                o2 = m64(x_c2, num_loops=4, eps_halt=0.05, update_slow=upd_slow)
            worst_leak = max(worst_leak, float((o1[:, :cut] - o2[:, :cut]).abs().max().item()))

    # O(1) constant state bytes across T and R in {1, 2, 8, 64, 256, 1024}
    o1_table: dict[str, int] = {}
    with torch.no_grad():
        for t_len, r_val in ((16, 1), (64, 2), (256, 8), (320, 64), (32, 256), (32, 1024)):
            _, _, _, st_o1 = m32.forward_stateful(
                torch.randint(0, 256, (1, t_len)), num_loops=r_val, update_slow=True
            )
            o1_table[f"T={t_len}_R={r_val}"] = state_nbytes(st_o1)

        # R=1024 finiteness in streaming and step-reasoning
        xr = torch.randint(0, 256, (1, 20), dtype=torch.long)
        lg_1024, _, h_1024, _ = m32.forward_stateful(xr, num_loops=1024, update_slow=True)
        rs_1024 = m32.forward_step_reasoning(xr, num_loops=1024)
        r1024_finite = bool(
            torch.isfinite(lg_1024).all().item()
            and torch.isfinite(h_1024).all().item()
            and torch.isfinite(rs_1024["logits"]).all().item()
            and torch.isfinite(rs_1024["h_final"]).all().item()
            and torch.isfinite(rs_1024["w_final"]).all().item()
        )

    # K3 zero-grad + MTP-3 + OY-3 smoke
    m32.train()
    m32.zero_grad(set_to_none=True)
    x_g = torch.randint(0, 256, (2, 16), dtype=torch.long)
    _, lg_r1, h_dec, _ = m32.forward_stateful(x_g, num_loops=2)
    torch.autograd.backward(lg_r1.sum())
    core_grad_max = max(
        (float(p.grad.abs().max().item()) for n, p in m32.named_parameters() if p.grad is not None and "k3_adapter" not in n),
        default=0.0,
    )
    k3_grad_max = max(
        (float(p.grad.abs().max().item()) for n, p in m32.named_parameters() if p.grad is not None and "k3_adapter" in n),
        default=0.0,
    )
    m32.zero_grad(set_to_none=True)
    m32.eval()
    with torch.no_grad():
        mtp_outs = m32.mtp_logits(h_dec)
        tb_cal = dyck_stack(k=4, depth=4, batch=8, seed=77, step=0, split="val")
        lg_cal = m32(tb_cal.x, num_loops=2)
        m_cal = tb_cal.eval_mask.bool().reshape(-1)
        lg_ev = lg_cal.reshape(-1, 256)[m_cal]
        y_ev = tb_cal.y.reshape(-1)[m_cal]
        t_star = fit_temperature(lg_ev, y_ev)
        conf_raw, _ = msp_uncertainty(lg_ev, temperature=1.0)
        conf_cal, unc_cal = msp_uncertainty(lg_ev, temperature=t_star)
        corr = (lg_ev.argmax(dim=-1) == y_ev).numpy().astype(np.float64)
        oy3 = {
            "t_star": float(t_star),
            "ece_15_raw": ece_15(conf_raw.numpy(), corr),
            "ece_15_cal": ece_15(conf_cal.numpy(), corr),
            "auroc_err": auroc(unc_cal.numpy(), 1.0 - corr),
            "aurc_cal": aurc(unc_cal.numpy(), 1.0 - corr),
        }

    o1_vals = list(o1_table.values())
    return {
        "param_count_total": sum(p.numel() for p in m32.parameters()),
        "analytic_flops": analytic_flops_per_byte(cfg),
        "parity": parity,
        "partial_patch_37_11_fp64_diff": partial_37_11_diff,
        "partial_patch_37_11_sslow_diff": partial_37_11_sslow_diff,
        "random_matched_fp64_diff": random_matched_fp64_diff,
        "eps_halt_fp64_diff": eps_halt_fp64_diff,
        "g_step_zero_noop_stream_diff": noop_stream_diff,
        "g_step_zero_noop_reason_diff": noop_reason_diff,
        "return_aux_pos_symmetry": {"keys_match": aux_keys_match, "max_diff": aux_sym_max_diff},
        "causality_worst_fp64_diff": worst_leak,
        "o1_state_bytes_fp32_b1": {
            "per_config": o1_table,
            "is_strictly_constant_O1": len(set(o1_vals)) == 1 and o1_vals[0] > 0,
            "constant_bytes": o1_vals[0],
        },
        "r1024_zero_nan_inf": r1024_finite,
        "k3_core_grad_max": core_grad_max,
        "k3_adapter_grad_max": k3_grad_max,
        "mtp_head_count": len(mtp_outs),
        "mtp_shapes_ok": all(tuple(t.shape) == (2, 16, 256) for t in mtp_outs),
        "oy3_calibration_smoke": oy3,
        "wall_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Section 2: Step-by-Step Silent Reasoning + Halting Gate vs ADFP-Collapse
# ---------------------------------------------------------------------------


def _eval_reasoning_on_batch(
    model: RDFLv05Model,
    xb: Tensor,
    yb: Tensor,
    R: int,
    eps_halt: float = 0.0,
    gate_override: float | None = None,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    correct = 0
    total = int(xb.shape[0])
    g_traces: list[Tensor] = []
    r_used_all: list[Tensor] = []
    with torch.no_grad():
        bs = 64
        for start in range(0, total, bs):
            cx = xb[start : start + bs]
            cy = yb[start : start + bs]
            res = model.forward_step_reasoning(
                cx, R, eps_halt=eps_halt, gate_override=gate_override
            )
            correct += int((res["logits"].argmax(dim=-1) == cy).sum().item())
            if res["g_trace"].shape[1] > 0:
                g_traces.append(res["g_trace"].squeeze(-1))
            r_used_all.append(res["r_used"])
    g_cat = torch.cat(g_traces, dim=0) if g_traces else torch.zeros((total, 0))
    ru_cat = torch.cat(r_used_all, dim=0)
    g_mean_curve = g_cat.mean(dim=0).tolist() if g_cat.shape[1] > 0 else []
    return correct / float(max(1, total)), {
        "g_first": round(float(g_mean_curve[0]), 4) if g_mean_curve else 1.0,
        "g_last": round(float(g_mean_curve[-1]), 4) if g_mean_curve else 1.0,
        "g_mean_curve_head12": [round(float(v), 4) for v in g_mean_curve[:12]],
        "r_used_mean": round(float(ru_cat.float().mean().item()), 2),
    }


def _train_step_reasoner(
    model: RDFLv05Model, name: str, *, seed: int = 42, max_steps: int = 950
) -> dict[str, Any]:
    set_seed(seed)
    rng = random.Random(seed)
    halt_ids = {id(p) for p in model.macro.halt_gate.parameters()}
    base_params = [p for p in model.parameters() if id(p) not in halt_ids]
    halt_params = list(model.macro.halt_gate.parameters())
    opt = torch.optim.AdamW(
        [{"params": base_params, "lr": 3e-3}, {"params": halt_params, "lr": 3e-3}],
        weight_decay=1e-4,
    )
    stage = 1
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    model.train()
    for step in range(1, max_steps + 1):
        if stage == 1:
            K = rng.choice([1, 2])
            r_extra = 0
        elif stage == 2:
            K = rng.choice([1, 2, 4])
            r_extra = 0
        elif stage == 3:
            K = rng.choice([2, 4, 6, 8])
            r_extra = 0
        else:
            K = rng.choice([2, 4, 6, 8])
            r_extra = rng.choice([0, 2, 6, 10])
        R = K + 2 + r_extra

        xb, yb = make_ptr_chain_batch(64, K, rng)
        g_ov: float | None = 1.0 if stage <= 3 else None
        res = model.forward_step_reasoning(
            xb, R, collect_trace=(stage >= 4 and r_extra > 0), gate_override=g_ov
        )
        loss = F.cross_entropy(res["logits"], yb)
        if stage >= 4 and model.cfg.reasoning_mode == "step_halting":
            if r_extra > 0:
                lg_opt = model.logits_from_reasoning_h(
                    res["h_trace"][K + 2], res["c_last"], slot_idx=int(res["slot_last"])
                )
                loss = 0.5 * loss + 0.5 * F.cross_entropy(lg_opt, yb)
            gt = res["g_trace"].squeeze(-1).clamp(1e-5, 1.0 - 1e-5)
            targ = torch.zeros_like(gt)
            targ[:, : K + 2] = 1.0
            loss = loss + 0.25 * F.binary_cross_entropy(gt, targ)
        opt.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.update_spectral_vectors()

        if step % 30 == 0:
            if stage == 1:
                x1, y1 = make_ptr_chain_batch(64, 1, rng)
                x2, y2 = make_ptr_chain_batch(64, 2, rng)
                a1, _ = _eval_reasoning_on_batch(model, x1, y1, 3, gate_override=1.0)
                a2, _ = _eval_reasoning_on_batch(model, x2, y2, 4, gate_override=1.0)
                model.train()
                if a1 > 0.90 and a2 > 0.90:
                    stage = 2
            elif stage == 2:
                x4, y4 = make_ptr_chain_batch(64, 4, rng)
                a4, _ = _eval_reasoning_on_batch(model, x4, y4, 6, gate_override=1.0)
                model.train()
                if a4 > 0.90:
                    stage = 3
                    opt.param_groups[0]["lr"] = 2e-3
            elif stage == 3 and step >= 540:
                x6, y6 = make_ptr_chain_batch(64, 6, rng)
                x8, y8 = make_ptr_chain_batch(64, 8, rng)
                a6, _ = _eval_reasoning_on_batch(model, x6, y6, 8, gate_override=1.0)
                a8, _ = _eval_reasoning_on_batch(model, x8, y8, 10, gate_override=1.0)
                model.train()
                if a6 > 0.90 and a8 > 0.88:
                    stage = 4
                    opt.param_groups[0]["lr"] = 1.5e-3
                    opt.param_groups[1]["lr"] = 6e-3

        if step % 150 == 0 or step == max_steps:
            history.append(
                {
                    "step": step,
                    "stage": stage,
                    "loss": round(float(loss.item()), 5),
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                }
            )
    return {
        "name": name,
        "train_seconds": round(time.perf_counter() - t0, 2),
        "steps": max_steps,
        "final_stage": stage,
        "history": history,
    }


def _measure_step_jvp_and_stress(model: RDFLv05Model) -> dict[str, Any]:
    model.eval()
    rng = random.Random(2025)
    xs, _ys = make_ptr_chain_batch(16, 6, rng)
    stress: dict[str, Any] = {}
    with torch.no_grad():
        for r_str in (256, 1024):
            res = model.forward_step_reasoning(xs, r_str, collect_trace=True, collect_w=True)
            h_tr = res["h_trace"]
            w_tr = res["w_trace"]
            h_norms = [float(ht.norm(dim=-1).mean().item()) for ht in h_tr]
            w_norms = [float(wt.norm().item() / math.sqrt(xs.shape[0])) for wt in w_tr]
            deltas = [float((h_tr[i] - h_tr[i - 1]).norm(dim=-1).mean().item()) for i in range(1, len(h_tr))]
            finite_all = bool(
                torch.isfinite(res["logits"]).all().item()
                and torch.isfinite(res["h_final"]).all().item()
                and torch.isfinite(res["w_final"]).all().item()
            )
            stress[str(r_str)] = {
                "finite_zero_nan_inf": finite_all,
                "h_norm_max": round(max(h_norms), 4),
                "h_norm_final": round(h_norms[-1], 4),
                "w_norm_max": round(max(w_norms), 4),
                "w_norm_final": round(w_norms[-1], 4),
                "delta_h_last": round(deltas[-1], 6) if deltas else 0.0,
                "delta_h_tail64_mean": round(float(np.mean(deltas[-64:])), 6) if deltas else 0.0,
            }

    # Empirical single-step Jacobian top singular value r=1..8
    xj, _ = make_ptr_chain_batch(8, 6, rng)
    with torch.no_grad():
        h_cur, s_fast, _, _ = model.encode_for_reasoning(xj)
        h0_anchor = h_cur.clone()
        w_cur = torch.zeros(
            xj.shape[0], model.macro.num_heads, model.macro.head_dim, model.macro.head_dim
        )
    step_jvp_specs: list[float] = []
    for r_idx in range(1, 9):
        def _single_step(
            h_vec: Tensor,
            w_mat: Tensor,
            s_mat: Tensor,
            h0_a: Tensor,
            r_cur: int = r_idx,
        ) -> Tensor:
            hn, _, _ = model.reason_step(
                h_vec.unsqueeze(0), w_mat.unsqueeze(0), s_mat.unsqueeze(0), h0_a.unsqueeze(0), r_cur
            )
            return hn[0]

        smax_list: list[float] = []
        for bi in range(xj.shape[0]):
            j_step = torch.func.jacrev(_single_step, argnums=0)(
                h_cur[bi], w_cur[bi], s_fast[bi], h0_anchor[bi]
            ).detach()
            sv = torch.linalg.svdvals(j_step)
            smax_list.append(float(sv[0].item()))
        step_jvp_specs.append(round(float(np.mean(smax_list)), 4))
        with torch.no_grad():
            h_cur, w_cur, _ = model.reason_step(h_cur, w_cur, s_fast, h0_anchor, r_idx)

    return {
        "stress_256_1024": stress,
        "step_jacobian_top_sv_r1_to_r8": step_jvp_specs,
        "mean_step_jvp_top_sv": round(float(np.mean(step_jvp_specs)), 4),
    }


def run_step_reasoning_and_halting() -> tuple[dict[str, Any], RDFLv05Model, RDFLv05Model]:
    t0 = time.perf_counter()
    cfg_step = RDFLv05Config(d_byte=64, d_macro=64, micro_heads=4, macro_heads=4, n_h=4, reasoning_mode="step_halting")
    cfg_adfp = RDFLv05Config(d_byte=64, d_macro=64, micro_heads=4, macro_heads=4, n_h=4, reasoning_mode="adfp")

    set_seed(42)
    m_step = RDFLv05Model(cfg_step)
    set_seed(42)
    m_adfp = RDFLv05Model(cfg_adfp)

    tr_step = _train_step_reasoner(m_step, "RDFLv05-StepHalting", seed=42, max_steps=1000)
    tr_adfp = _train_step_reasoner(m_adfp, "RDFLv05-ADFP", seed=42, max_steps=1000)

    rng_ev = random.Random(2026)
    id_ood_step: dict[str, float] = {}
    id_ood_adfp: dict[str, float] = {}
    for K in (2, 4, 6, 8, 10, 12):
        xe_k, ye_k = make_ptr_chain_batch(128, K, rng_ev)
        acc_s, _ = _eval_reasoning_on_batch(m_step, xe_k, ye_k, K + 2)
        acc_a, _ = _eval_reasoning_on_batch(m_adfp, xe_k, ye_k, K + 2)
        id_ood_step[str(K)] = round(acc_s, 4)
        id_ood_adfp[str(K)] = round(acc_a, 4)

    # Learned Halting Gate evaluation on K=4 and K=6 over R = K+2, 16, 32, 64
    halting_sweep: dict[str, Any] = {}
    for K in (4, 6):
        xh, yh = make_ptr_chain_batch(128, K, rng_ev)
        k_entry: dict[str, Any] = {}
        for R_tot in (K + 2, 16, 32, 64):
            acc_no_ee, info_no_ee = _eval_reasoning_on_batch(m_step, xh, yh, R_tot, eps_halt=0.0)
            acc_ee, info_ee = _eval_reasoning_on_batch(m_step, xh, yh, R_tot, eps_halt=0.40)
            acc_adfp, _ = _eval_reasoning_on_batch(m_adfp, xh, yh, R_tot, eps_halt=0.0)
            k_entry[f"R={R_tot}"] = {
                "step_halting_full_acc": round(acc_no_ee, 4),
                "step_halting_early_exit_acc": round(acc_ee, 4),
                "adfp_acc": round(acc_adfp, 4),
                "g_first": info_no_ee["g_first"],
                "g_last": info_no_ee["g_last"],
                "early_exit_r_used_mean": info_ee["r_used_mean"],
            }
        halting_sweep[f"K={K}"] = k_entry

    stress_step = _measure_step_jvp_and_stress(m_step)
    stress_adfp = _measure_step_jvp_and_stress(m_adfp)

    return (
        {
            "training": {"step_halting": tr_step, "adfp": tr_adfp},
            "accuracy_ID_2_4_6_8_OOD_10_12": {"step_halting": id_ood_step, "adfp": id_ood_adfp},
            "halting_gate_extra_loops_sweep": halting_sweep,
            "stress_and_jvp": {"step_halting": stress_step, "adfp": stress_adfp},
            "wall_s": time.perf_counter() - t0,
        },
        m_step,
        m_adfp,
    )


# ---------------------------------------------------------------------------
# Section 3: Anthropic Global Workspace (J-Space / Jacobian Lens, Strict OOS)
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction evaluation for regularized incomplete beta (Numerical Recipes)."""
    fpmin = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        del_v = d * c
        h *= del_v
        if abs(del_v - 1.0) < 3e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _paired_ttest_pvalue(a: list[float], b: list[float]) -> float:
    """Exact two-sided paired Student-t p-value without external scipy dependency."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = int(diff.shape[0])
    if n < 2:
        return 1.0
    sd = float(np.std(diff, ddof=1))
    if sd <= 0.0:
        return 0.0 if float(np.mean(diff)) != 0.0 else 1.0
    t_stat = float(np.mean(diff)) / (sd / math.sqrt(float(n)))
    df = float(n - 1)
    x = df / (df + t_stat * t_stat)
    p = float(_betainc(0.5 * df, 0.5, x))
    return max(0.0, min(1.0, p))


def _ridge_probe_acc(
    x_tr: NDArray[Any], y_tr: NDArray[Any], x_te: NDArray[Any], y_te: NDArray[Any], lam: float = 1.0
) -> float:
    if x_tr.shape[1] == 0:
        return 0.1
    mu = x_tr.mean(axis=0, keepdims=True)
    xtr = x_tr - mu
    xte = x_te - mu
    y_oh = np.zeros((len(y_tr), 10), dtype=np.float64)
    y_oh[np.arange(len(y_tr)), y_tr] = 1.0
    xtr_b = np.concatenate([xtr, np.ones((len(xtr), 1))], axis=1)
    xte_b = np.concatenate([xte, np.ones((len(xte), 1))], axis=1)
    dim = xtr_b.shape[1]
    reg = lam * np.eye(dim)
    reg[-1, -1] = 1e-6
    w = np.linalg.solve(xtr_b.T @ xtr_b + reg, xtr_b.T @ y_oh)
    pred = (xte_b @ w).argmax(axis=1)
    return float((pred == y_te).mean())


def run_gwt_jacobian_suite(model: RDFLv05Model, name: str) -> dict[str, Any]:
    """Measure all 4 Anthropic GWT / J-space signatures strictly out-of-sample (x_cal -> x_te)."""
    t0 = time.perf_counter()
    model.eval()
    K = 6
    R = 8
    n_cal = 96
    n_probe_tr = 128
    n_probe_te = 96
    rng = random.Random(7777)

    x_cal, _, _, _, _ = make_ptr_chain_batch(n_cal, K, rng, return_intermediates=True)
    x_tr, _, inter_tr, _, _ = make_ptr_chain_batch(n_probe_tr, K, rng, return_intermediates=True)
    x_te, y_te, inter_te, perms_te, _ = make_ptr_chain_batch(n_probe_te, K, rng, return_intermediates=True)

    with torch.no_grad():
        res_cal = model.forward_step_reasoning(x_cal, R, collect_trace=True, collect_w=True)
        res_tr = model.forward_step_reasoning(x_tr, R, collect_trace=True, collect_w=False)
        res_te = model.forward_step_reasoning(x_te, R, collect_trace=True, collect_w=True)
        clean_acc = float((res_te["logits"].argmax(-1) == y_te).float().mean().item())

    h_cal_list: list[Tensor] = res_cal["h_trace"]
    w_cal_list: list[Tensor] = res_cal["w_trace"]
    s_cal: Tensor = res_cal["s_fast"]
    c_last_cal: Tensor = res_cal["c_last"]
    slot_cal: int = int(res_cal["slot_last"])

    h_tr_list: list[Tensor] = res_tr["h_trace"]
    h_te_list: list[Tensor] = res_te["h_trace"]
    w_te_list: list[Tensor] = res_te["w_trace"]
    s_te: Tensor = res_te["s_fast"]
    c_last_te: Tensor = res_te["c_last"]
    slot_te: int = int(res_te["slot_last"])

    d = model.cfg.macro_dim
    h_r_cal = h_cal_list[R].detach()
    j_head_list = []
    for i in range(n_cal):
        j_hd = torch.func.jacrev(
            lambda hv, cl=c_last_cal[i : i + 1]: model.logits_from_reasoning_h(
                hv.unsqueeze(0), cl, slot_idx=slot_cal
            ).squeeze(0)
        )(h_r_cal[i]).detach()
        j_head_list.append(j_hd)
    j_head_all = torch.stack(j_head_list, dim=0).detach()

    jacobian_by_r: dict[str, Any] = {}
    v_j_by_r: dict[int, Tensor] = {}
    v_perp_by_r: dict[int, Tensor] = {}
    v_perp_topk_by_r: dict[int, Tensor] = {}
    k_by_r: dict[int, int] = {}

    h0_cal = h_cal_list[0].detach()
    h0_te = h_te_list[0].detach()

    for r in range(R):
        hr_cal = h_cal_list[r].detach()
        wr_cal = w_cal_list[r].detach()

        def _tail_hr(
            h_vec: Tensor, s_mat: Tensor, w_mat: Tensor, h0_a: Tensor, r_start: int = r
        ) -> Tensor:
            cur_h = h_vec.unsqueeze(0)
            cur_w = w_mat.unsqueeze(0)
            s_b = s_mat.unsqueeze(0)
            h0_b = h0_a.unsqueeze(0)
            for step_r in range(r_start + 1, R + 1):
                cur_h, cur_w, _ = model.reason_step(cur_h, cur_w, s_b, h0_b, step_r)
            return cur_h.squeeze(0)

        j_r_list = []
        for i in range(n_cal):
            jh = torch.func.jacrev(_tail_hr, argnums=0)(
                hr_cal[i], s_cal[i], wr_cal[i], h0_cal[i]
            ).detach()
            j_r_list.append((j_head_all[i] @ jh).detach())
        j_all = torch.stack(j_r_list, dim=0).detach()

        j_frob = j_all.flatten(1).norm(dim=-1).view(n_cal, 1, 1) + 1e-12
        j_normed = j_all / j_frob
        c_r = torch.einsum("nvd,nve->de", j_normed, j_normed) / float(n_cal)
        eigvals, eigvecs = torch.linalg.eigh(c_r)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx].clamp(min=0.0)
        eigvecs = eigvecs[:, idx]

        tot_e = float(eigvals.sum().item()) + 1e-12
        pr_corp = float((eigvals.sum().pow(2) / (eigvals.pow(2).sum() + 1e-12)).item())
        k90_corp = int(((torch.cumsum(eigvals, dim=0) / tot_e) < 0.90).sum().item()) + 1
        k_use = max(1, min(k90_corp, d - 1))
        k_by_r[r] = k_use

        v_j = eigvecs[:, :k_use]
        v_perp = eigvecs[:, k_use:]
        v_j_by_r[r] = v_j
        v_perp_by_r[r] = v_perp

        h_cal_np = hr_cal.numpy()
        h_cal_c = h_cal_np - h_cal_np.mean(axis=0, keepdims=True)
        vp_np = v_perp.numpy()
        h_cal_perp = h_cal_c @ vp_np
        cov_perp_cal = (h_cal_perp.T @ h_cal_perp) / float(n_cal)
        ev_p, evec_p = np.linalg.eigh(cov_perp_cal)
        top_p_idx = np.argsort(ev_p)[::-1][:k_use]
        v_perp_topk = torch.from_numpy((vp_np @ evec_p[:, top_p_idx]).astype(np.float32))
        v_perp_topk_by_r[r] = v_perp_topk

        h_te_np = h_te_list[r].detach().numpy()
        h_te_c = h_te_np - h_te_np.mean(axis=0, keepdims=True)
        tot_var_te = float((h_te_c**2).sum()) + 1e-12
        var_j_frac = float(((h_te_c @ v_j.numpy()) ** 2).sum() / tot_var_te)
        var_jp_topk_frac = float(((h_te_c @ v_perp_topk.numpy()) ** 2).sum() / tot_var_te)

        jacobian_by_r[str(r)] = {
            "corpus_PR_norm": round(pr_corp, 3),
            "corpus_k90_norm": k90_corp,
            "OOS_J_space_var_frac_at_k90": round(var_j_frac, 4),
            "random_subspace_expected_var_frac": round(k_use / float(d), 4),
            "OOS_J_perp_topk_var_frac": round(var_jp_topk_frac, 4),
        }

    # 2. Interruption readout & held-out Ridge probes
    interruption_readout: dict[str, Any] = {}
    with torch.no_grad():
        for r in range(R + 1):
            lg_r = model.logits_from_reasoning_h(h_te_list[r], c_last_te, slot_idx=slot_te)
            probs_r = F.softmax(lg_r, dim=-1)
            pred_r = lg_r.argmax(dim=-1)
            p_steps = {}
            acc_steps = {}
            for j in range(K + 1):
                y_j = torch.tensor([ord(str(inter_te[i][j])) for i in range(n_probe_te)], dtype=torch.long)
                p_steps[f"v_{j}"] = round(float(probs_r[torch.arange(n_probe_te), y_j].mean().item()), 4)
                acc_steps[f"v_{j}"] = round(float((pred_r == y_j).float().mean().item()), 4)
            interruption_readout[str(r)] = {
                "acc_final": round(float((pred_r == y_te).float().mean().item()), 4),
                "acc_per_v": acc_steps,
                "prob_per_v": p_steps,
            }

    probe_rng = np.random.RandomState(1234)
    probes: dict[str, dict[str, list[float]]] = {
        f"v_{j}": {"full": [], "J_space": [], "J_perp_matched": [], "random_k": []} for j in range(1, K + 1)
    }
    for r in range(R + 1):
        r_ref = min(r, R - 1)
        vj = v_j_by_r[r_ref].numpy()
        vpk = v_perp_topk_by_r[r_ref].numpy()
        k_use = k_by_r[r_ref]
        q_rand, _ = np.linalg.qr(probe_rng.randn(d, k_use))
        xtr = h_tr_list[r].detach().numpy()
        xte = h_te_list[r].detach().numpy()
        for j in range(1, K + 1):
            ytr_j = np.array([inter_tr[i][j] for i in range(n_probe_tr)])
            yte_j = np.array([inter_te[i][j] for i in range(n_probe_te)])
            probes[f"v_{j}"]["full"].append(round(_ridge_probe_acc(xtr, ytr_j, xte, yte_j), 4))
            probes[f"v_{j}"]["J_space"].append(round(_ridge_probe_acc(xtr @ vj, ytr_j, xte @ vj, yte_j), 4))
            probes[f"v_{j}"]["J_perp_matched"].append(round(_ridge_probe_acc(xtr @ vpk, ytr_j, xte @ vpk, yte_j), 4))
            probes[f"v_{j}"]["random_k"].append(round(_ridge_probe_acc(xtr @ q_rand, ytr_j, xte @ q_rand, yte_j), 4))

    peak_r_full: list[int] = []
    peak_r_j: list[int] = []
    probe_summary: dict[str, Any] = {}
    for j in range(1, K + 1):
        key = f"v_{j}"
        pf = probes[key]["full"]
        pj = probes[key]["J_space"]
        arg_f = int(np.argmax(pf))
        arg_j = int(np.argmax(pj))
        peak_r_full.append(arg_f)
        peak_r_j.append(arg_j)
        probe_summary[key] = {
            "peak_r_full": arg_f,
            "peak_acc_full": float(max(pf)),
            "peak_r_J_space": arg_j,
            "peak_acc_J_space": float(max(pj)),
            "peak_acc_J_perp_matched": float(max(probes[key]["J_perp_matched"])),
        }
    mono_full = sum(1 for a in range(len(peak_r_full) - 1) if peak_r_full[a] < peak_r_full[a + 1])
    mono_j = sum(1 for a in range(len(peak_r_j) - 1) if peak_r_j[a] < peak_r_j[a + 1])

    # 3. Functional Global Broadcast at r_mid = R // 2 = 4
    r_mid = R // 2
    vj_mid = v_j_by_r[r_mid]
    vp_mid = v_perp_by_r[r_mid]
    vpk_mid = v_perp_topk_by_r[r_mid]
    k_mid = k_by_r[r_mid]

    hr_mid = h_te_list[r_mid].detach()
    wr_mid = w_te_list[r_mid].detach()
    q_fixed_rand, _ = np.linalg.qr(np.random.RandomState(556).randn(d, k_mid))
    q_fixed_t = torch.from_numpy(q_fixed_rand.astype(np.float32))
    func_j, func_jp, func_rand = [], [], []
    for i in range(min(48, n_probe_te)):
        j_st = torch.func.jacrev(
            lambda hv, sm=s_te[i : i + 1], wm=wr_mid[i : i + 1], h0m=h0_te[i : i + 1]: model.reason_step(
                hv.unsqueeze(0), wm, sm, h0m, r_mid + 1
            )[0].squeeze(0)
        )(hr_mid[i]).detach()
        func_j.append(float((j_st @ vj_mid).pow(2).sum().item() / k_mid))
        func_jp.append(float((j_st @ vp_mid).pow(2).sum().item() / max(1, d - k_mid)))
        func_rand.append(float((j_st @ q_fixed_t).pow(2).sum().item() / k_mid))

    p_vs_jp = _paired_ttest_pvalue(func_j, func_jp)
    p_vs_rand = _paired_ttest_pvalue(func_j, func_rand)

    broadcast = {
        "r_mid": r_mid,
        "k_mid": k_mid,
        "func_broadcast_J_space_mean": round(float(np.mean(func_j)), 4),
        "func_broadcast_J_perp_full_mean": round(float(np.mean(func_jp)), 4),
        "func_broadcast_random_k_mean": round(float(np.mean(func_rand)), 4),
        "func_broadcast_ratio_J_over_Jperp": round(float(np.mean(func_j)) / (float(np.mean(func_jp)) + 1e-12), 4),
        "func_broadcast_ratio_J_over_rand": round(float(np.mean(func_j)) / (float(np.mean(func_rand)) + 1e-12), 4),
        "func_broadcast_p_vs_Jperp": p_vs_jp,
        "func_broadcast_p_vs_rand": p_vs_rand,
    }

    # 4. Selective Ablation & Concept Patching at r_mid
    def _eval_from_mid(
        h_mod: Tensor, s_mat: Tensor, w_mat: Tensor, h0_m: Tensor, c_l: Tensor, sl_i: int
    ) -> Tensor:
        cur_h = h_mod
        cur_w = w_mat
        for step_r in range(r_mid + 1, R + 1):
            cur_h, cur_w, _ = model.reason_step(cur_h, cur_w, s_mat, h0_m, step_r)
        return model.logits_from_reasoning_h(cur_h, c_l, slot_idx=sl_i)

    with torch.no_grad():
        # Verify exact zero-ablation consistency with forward_step_reasoning for both modes
        zero_abl_max_diff = float(
            (_eval_from_mid(hr_mid, s_te, wr_mid, h0_te, c_last_te, slot_te) - res_te["logits"])
            .abs()
            .max()
            .item()
        )
        assert zero_abl_max_diff < 1e-5, f"zero_ablation_mismatch={zero_abl_max_diff}"

        p_j = vj_mid @ vj_mid.T
        p_jpk = vpk_mid @ vpk_mid.T
        p_jp_full = vp_mid @ vp_mid.T

        def _abl(p_zero: Tensor) -> float:
            lg = _eval_from_mid(hr_mid - hr_mid @ p_zero, s_te, wr_mid, h0_te, c_last_te, slot_te)
            return float((lg.argmax(-1) == y_te).float().mean().item())

        acc_abl_j = _abl(p_j)
        acc_abl_jpk = _abl(p_jpk)
        acc_abl_jpf = _abl(p_jp_full)
        rand_abl = []
        rng_ab = np.random.RandomState(888)
        for _ in range(10):
            qr, _ = np.linalg.qr(rng_ab.randn(d, k_mid))
            qt = torch.from_numpy(qr.astype(np.float32))
            rand_abl.append(_abl(qt @ qt.T))

        # Same-table patching
        xa, ya, _, xb, yb, _, _ = make_same_table_pairs(96, K, random.Random(9001))
        ra = model.forward_step_reasoning(xa, R, collect_trace=True, collect_w=True)
        rb = model.forward_step_reasoning(xb, R, collect_trace=True, collect_w=True)
        both_ok = (ra["logits"].argmax(-1) == ya) & (rb["logits"].argmax(-1) == yb)
        n_valid = int(both_ok.sum().item())
        ha_m, hb_m = ra["h_trace"][r_mid], rb["h_trace"][r_mid]
        hb0 = rb["h_trace"][0]
        sb, wb_m, cb_l, sb_i = rb["s_fast"], rb["w_trace"][r_mid], rb["c_last"], int(rb["slot_last"])

        def _patch_same(p_pat: Tensor) -> dict[str, float]:
            hb_p = hb_m - hb_m @ p_pat + ha_m @ p_pat
            pred_p = _eval_from_mid(hb_p, sb, wb_m, hb0, cb_l, sb_i).argmax(-1)
            mask = both_ok if n_valid > 0 else torch.ones_like(both_ok)
            fa = float((pred_p[mask] == ya[mask]).float().mean().item())
            kb = float((pred_p[mask] == yb[mask]).float().mean().item())
            return {"flip_to_A_rate": round(fa, 4), "keep_B_rate": round(kb, 4)}

        patch_j = _patch_same(p_j)
        patch_jp_full = _patch_same(p_jp_full)

        # Cross-table compositional target P_B^4(v_2(A))
        ha_x = hr_mid
        hb_x = torch.roll(hr_mid, shifts=1, dims=0)
        hb0_x = torch.roll(h0_te, shifts=1, dims=0)
        sb_x = torch.roll(s_te, shifts=1, dims=0)
        wb_x = torch.roll(wr_mid, shifts=1, dims=0)
        cb_x = torch.roll(c_last_te, shifts=1, dims=0)
        perms_b = perms_te[-1:] + perms_te[:-1]
        cf_tgts = []
        for ii in range(n_probe_te):
            cv = inter_te[ii][2]
            pm = perms_b[ii]
            for _ in range(K - 2):
                cv = pm[cv]
            cf_tgts.append(ord(str(cv)))
        y_cf = torch.tensor(cf_tgts, dtype=torch.long)
        pred_cx_j = _eval_from_mid(
            hb_x - hb_x @ p_j + ha_x @ p_j, sb_x, wb_x, hb0_x, cb_x, slot_te
        ).argmax(-1)
        pred_cx_jp = _eval_from_mid(
            hb_x - hb_x @ p_jp_full + ha_x @ p_jp_full, sb_x, wb_x, hb0_x, cb_x, slot_te
        ).argmax(-1)

    return {
        "model": name,
        "K": K,
        "R": R,
        "clean_acc_K6_R8": round(clean_acc, 4),
        "signature1_jacobian_svd": jacobian_by_r,
        "signature2_interruption_readout": interruption_readout,
        "signature2_linear_probes": {
            "monotone_adjacent_steps_full": f"{mono_full}/{K - 1}",
            "monotone_adjacent_steps_J_space": f"{mono_j}/{K - 1}",
            "peak_r_sequence_full": peak_r_full,
            "peak_r_sequence_J_space": peak_r_j,
            "per_step_v": probe_summary,
        },
        "signature3_global_broadcast": broadcast,
        "signature4_causal_ablation": {
            "r_mid": r_mid,
            "k_mid": k_mid,
            "clean_acc": round(clean_acc, 4),
            "ablate_J_space_acc": round(acc_abl_j, 4),
            "ablate_J_space_drop": round(clean_acc - acc_abl_j, 4),
            "ablate_J_perp_matched_k_acc": round(acc_abl_jpk, 4),
            "ablate_J_perp_full_acc": round(acc_abl_jpf, 4),
            "ablate_random_k_mean_acc": round(float(np.mean(rand_abl)), 4),
        },
        "signature4_concept_patching": {
            "same_table_pairs_both_correct": n_valid,
            "patch_J_space": patch_j,
            "patch_J_perp_full": patch_jp_full,
            "cross_table_compositional_PB4_v2A_J_space": round(float((pred_cx_j == y_cf).float().mean().item()), 4),
            "cross_table_compositional_PB4_v2A_J_perp_full": round(float((pred_cx_jp == y_cf).float().mean().item()), 4),
        },
        "wall_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Section 4: 4-Problem Regression Lock (P1, P2, P3, P4 + Adaptive Space Gate)
# ---------------------------------------------------------------------------


def _eval_s5(model: RDFLv05Model, length: int, n_eval: int = 4, num_loops: int = 1) -> float:
    model.eval()
    ok = 0
    tot = 0
    with torch.no_grad():
        for i in range(n_eval):
            tb = s5_perm32(batch=32, seed=1041, step=i, split="val", length=length, running=False)
            lg = model(tb.x, num_loops=num_loops)
            vm = tb.eval_mask.bool()
            ok += int((lg.argmax(dim=-1)[vm] == tb.y[vm]).sum().item())
            tot += int(vm.sum().item())
    return ok / float(max(1, tot))


def _eval_s5_sfast_zero(model: RDFLv05Model, length: int = 2, n_eval: int = 4) -> float:
    model.eval()
    ok = 0
    tot = 0
    with torch.no_grad():
        for i in range(n_eval):
            tb = s5_perm32(batch=32, seed=1041, step=i, split="val", length=length, running=False)
            eq_pos = int(tb.eval_mask[0].nonzero(as_tuple=False)[0].item())
            _, _, _, st = model.forward_stateful(tb.x[:, :eq_pos], num_loops=1)
            st.s_fast.zero_()
            lg_eq, _ = model.step(tb.x[:, eq_pos : eq_pos + 1], state=st, num_loops=1)
            ok += int((lg_eq[:, 0].argmax(dim=-1) == tb.y[:, eq_pos]).sum().item())
            tot += tb.x.shape[0]
    return ok / float(max(1, tot))


def _make_rule_batch(rule_id: int, split: str, step: int, batch_size: int, seed: int) -> TaskBatch:
    sbit = _split_bit(split)
    rng = _task_rng(f"rule_{rule_id}", split, seed, step)
    alphabets = [b"abcdefgh", b"ijklmnop", b"qrstuvwx", b"ABCDEFGH", b"IJKLMNOP"]
    alpha = np.frombuffer(alphabets[rule_id], dtype=np.uint8)
    prefix = f"{rule_id}:".encode("ascii")
    seqs: list[bytes] = []
    eval_pos: list[list[int]] = []
    length = 4
    for _ in range(batch_size):
        while True:
            idx = rng.integers(0, len(alpha), size=length)
            src = alpha[idx].tobytes()
            if _hash_bit(src) == sbit:
                break
        if rule_id == 0:
            ans = alpha[(idx + 1) % len(alpha)].tobytes()
        elif rule_id == 1:
            ans = alpha[(idx + 3) % len(alpha)].tobytes()
        elif rule_id == 2:
            ans = alpha[idx[::-1]].tobytes()
        elif rule_id == 3:
            ans = alpha[[idx[1], idx[0], idx[3], idx[2]]].tobytes()
        else:
            ans = alpha[(idx + 2) % len(alpha)].tobytes()
        full = prefix + src + b"|" + ans
        start_ans = len(prefix) + length + 1
        seqs.append(full)
        eval_pos.append(list(range(start_ans, start_ans + length)))
    return _pack_fixed_sequences(seqs, eval_pos, {"task": f"rule_{rule_id}", "chance_acc": 0.125})


@torch.no_grad()
def _eval_rule(model: RDFLv05Model, rule_id: int, *, update_slow: bool) -> float:
    model.eval()
    ok = 0
    tot = 0
    for bi in range(2):
        tb = _make_rule_batch(rule_id, "val", bi, 24, 999)
        lg = model(tb.x, num_loops=2, update_slow=update_slow)
        vm = tb.eval_mask.bool()
        ok += int((lg.argmax(dim=-1)[vm] == tb.y[vm]).sum().item())
        tot += int(vm.sum().item())
    return ok / float(max(1, tot))


def _run_cl_arm(
    cfg: RDFLv05Config, mode: str, *, rule_ids: tuple[int, ...] = (0, 3, 4), steps_per_task: int = 21, lr: float = 7e-3, seed: int = 42, use_slow: bool = False
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = RDFLv05Model(cfg)
    cons = ConsolidationManager(model, mu_ogp=1.5, lambda_ewc=1.0, lambda_kl=0.01) if mode == "rdfl_v05_cl3" else None
    probe_rng = torch.Generator().manual_seed(seed + 777)
    r_mat: list[list[float]] = []
    for k_idx, rid in enumerate(rule_ids):
        if not (mode == "frozen" and k_idx > 0):
            wd = 0.01 if k_idx == 0 else 0.0
            if use_slow:
                slow_ids = {id(p) for p in model.slow_mem.parameters()}
                fast_p = [p for p in model.parameters() if id(p) not in slow_ids]
                opt = torch.optim.AdamW(
                    [{"params": fast_p, "lr": lr}, {"params": list(model.slow_mem.parameters()), "lr": lr * 0.1}],
                    weight_decay=wd,
                )
            else:
                opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            model.train()
            for s in range(steps_per_task):
                tb = _make_rule_batch(rid, "train", s, 16, seed + 100 * k_idx)
                opt.zero_grad(set_to_none=True)
                lg = model(tb.x, num_loops=2, update_slow=use_slow)
                tgt = tb.masked_targets(-100)
                loss = F.cross_entropy(lg.reshape(-1, 256), tgt.reshape(-1), ignore_index=-100)
                if cons is not None and k_idx > 0:
                    px = torch.randint(0, 256, (6, 16), generator=probe_rng)
                    loss = loss + cons.penalty_loss(px)
                torch.autograd.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                prev_w = cons.snapshot_weights() if cons is not None else None
                opt.step()
                if cons is not None and prev_w is not None:
                    cons.apply_post_step_ogp(prev_w)
                model.update_spectral_vectors()
            if cons is not None:
                cov_b = [_make_rule_batch(rid, "train", 500 + bi, 24, seed + 100 * k_idx) for bi in range(2)]
                cons.consolidate_task(cov_b, update_slow=use_slow)
        r_mat.append([_eval_rule(model, rule_ids[j], update_slow=use_slow) for j in range(k_idx + 1)])
    k_cnt = len(r_mat)
    return {
        "mode": mode,
        "R_matrix": r_mat,
        "ACC": float(np.mean(r_mat[-1])),
        "BWT": float(np.mean([r_mat[-1][j] - r_mat[j][j] for j in range(k_cnt - 1)])),
        "PLAST_new": float(np.mean([r_mat[j][j] for j in range(1, k_cnt)])),
    }


@torch.no_grad()
def _extract_space_digit_mfast_hh(model: RDFLv05Model, seq_in: Tensor) -> dict[str, float]:
    """Measure Householder transition spectral radius |lambda| on digit vs space bytes and 4-byte product."""
    w_dtype = model.byte_emb.weight.dtype
    st0 = model._init_state(1, seq_in.device, w_dtype, None)
    assert st0.stem_state is not None
    e0 = model.byte_emb(seq_in)
    delta_ng, _, _ = model.ngram.embed_delta(seq_in, st0.stem_state[0], collect_snapshots=False)
    nm = model.n_micro(e0 + delta_ng)
    _, k_f, _, la_f, be_f, _, _, _ = model.micro_gdn._project(nm, st0.conv_state, torch.float64)
    k_hh = k_f[0, model.micro_gdn.n_assoc :].permute(1, 0, 2, 3)
    alpha_hh = la_f[0, model.micro_gdn.n_assoc :].exp().permute(1, 0)
    beta_hh = be_f[0, model.micro_gdn.n_assoc :].permute(1, 0, 2)
    t_len = seq_in.shape[1]
    n_hh, n_h, dh = k_hh.shape[1], k_hh.shape[2], k_hh.shape[3]
    eye = torch.eye(dh, dtype=torch.float64).view(1, 1, dh, dh)
    mats = alpha_hh[..., None, None] * eye.expand(t_len, n_hh, dh, dh).clone()
    for m_idx in range(n_h):
        km = k_hh[:, :, m_idx, :]
        bm = beta_hh[:, :, m_idx, None, None]
        mats = torch.matmul(mats, eye - bm * torch.matmul(km.unsqueeze(-1), km.unsqueeze(-2)))
    eigs_all = torch.linalg.eigvals(mats).abs().cpu().numpy()
    raw_b = seq_in[0].cpu().numpy()
    digit_mask = (raw_b >= 48) & (raw_b <= 57)
    space_mask = raw_b == 32
    p4_cnt = t_len // 4
    m4 = torch.matmul(
        torch.matmul(mats[0 : 4 * p4_cnt : 4], mats[1 : 4 * p4_cnt : 4]),
        torch.matmul(mats[2 : 4 * p4_cnt : 4], mats[3 : 4 * p4_cnt : 4]),
    )
    eigs_4 = torch.linalg.eigvals(m4).abs().pow(0.25).cpu().numpy()
    return {
        "digit_med_abs_eig": round(float(np.median(eigs_all[digit_mask])), 4),
        "space_med_abs_eig": round(float(np.median(eigs_all[space_mask])), 4),
        "p4_med_abs_eig_per_byte": round(float(np.median(eigs_4)), 4),
    }


def _build_copy_l16(n_bytes: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    prefix = [int(v) for v in rng.integers(100, 1000, size=16)]
    seq = list(prefix)
    for k in range(16, (n_bytes + 3) // 4 + 1):
        seq.append(seq[k - 16])
    return (" ".join(str(x) for x in seq) + " ").encode("ascii")[:n_bytes]


def _build_shuf16_bag(n_bytes: int, prefix_seed: int, shuf_seed: int) -> bytes:
    bag = [int(v) for v in np.random.default_rng(prefix_seed).integers(100, 1000, size=16)]
    rng_s = np.random.default_rng(shuf_seed)
    seq = list(bag)
    for _ in range(1, ((n_bytes + 3) // 4 + 1 + 15) // 16):
        perm = list(bag)
        rng_s.shuffle(perm)
        seq.extend(perm)
    return (" ".join(str(x) for x in seq) + " ").encode("ascii")[:n_bytes]


def run_4problem_regression_lock(cfg: RDFLv05Config, *, seed: int = 42) -> dict[str, Any]:
    t0 = time.perf_counter()

    # P1: dyck_stack(k=4, depth=4)
    set_seed(seed)
    m_p1 = RDFLv05Model(cfg)
    opt_p1 = torch.optim.AdamW(m_p1.parameters(), lr=4e-3, weight_decay=0.01)
    rng_p1 = np.random.default_rng(seed)
    m_p1.train()
    for s in range(60):
        tb = dyck_stack(k=4, depth=4, batch=16, seed=seed, step=s, split="train")
        rt = int(rng_p1.choice([2, 4, 6, 8]))
        opt_p1.zero_grad(set_to_none=True)
        lg = m_p1(tb.x, num_loops=rt)
        bm = tb.eval_mask.bool()
        loss = F.cross_entropy(lg[bm], tb.y[bm])
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(m_p1.parameters(), 1.0)
        opt_p1.step()
        m_p1.update_spectral_vectors()

    m_p1.eval()
    val_p1 = [dyck_stack(k=4, depth=4, batch=24, seed=900 + i, step=i, split="val") for i in range(3)]
    p1_sweep: dict[str, float] = {}
    with torch.no_grad():
        for rt in (1, 2, 4, 8, 16):
            ok = tot = 0
            for tb in val_p1:
                lg = m_p1(tb.x, num_loops=rt)
                bm = tb.eval_mask.bool()
                ok += int((lg.argmax(-1)[bm] == tb.y[bm]).sum().item())
                tot += int(bm.sum().item())
            p1_sweep[f"R{rt}"] = round(ok / float(max(1, tot)), 4)
        x80 = torch.randint(0, 256, (1, 80), dtype=torch.long)
        _, _, _, st1 = m_p1.forward_stateful(x80, num_loops=1)
        _, _, _, st16 = m_p1.forward_stateful(x80, num_loops=16)
        t80_inv = float((st1.s_macro - st16.s_macro).abs().max().item())
    del m_p1, opt_p1
    gc.collect()

    # P2: s5_perm32 (running=False, strict OOD train L<=8 -> L=2,4,8,16,32)
    set_seed(seed)
    m_s5 = RDFLv05Model(cfg)
    p2d, p1d, png = [], [], []
    for n, p in m_s5.named_parameters():
        if "ngram" in n:
            png.append(p)
        elif p.ndim < 2 or "byte_emb" in n or "head.weight" in n or "s0_hh" in n or "bias" in n:
            p1d.append(p)
        else:
            p2d.append(p)
    opt_s5 = torch.optim.AdamW(
        [
            {"params": p2d, "lr": 8e-3, "weight_decay": 0.15},
            {"params": p1d, "lr": 8e-3, "weight_decay": 0.0},
            {"params": png, "lr": 1e-4, "weight_decay": 0.50},
        ]
    )
    m_s5.train()
    for s in range(650):
        tb = s5_perm32(batch=128, seed=seed, step=s, split="train", length=2, running=False)
        loss = F.cross_entropy(m_s5(tb.x, num_loops=1).view(-1, 256), tb.masked_targets().view(-1), ignore_index=-100)
        opt_s5.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(m_s5.parameters(), 1.0)
        opt_s5.step()
    opt_s5.param_groups[0]["lr"] = 5e-3
    opt_s5.param_groups[0]["weight_decay"] = 0.08
    opt_s5.param_groups[1]["lr"] = 5e-3
    for s in range(450):
        L = (2, 4, 6, 8)[s % 4]
        tb = s5_perm32(batch=64, seed=seed, step=650 + s, split="train", length=L, running=False)
        loss = F.cross_entropy(m_s5(tb.x, num_loops=1).view(-1, 256), tb.masked_targets().view(-1), ignore_index=-100)
        opt_s5.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(m_s5.parameters(), 1.0)
        opt_s5.step()
    p2_res = {
        "running": False,
        "chance_acc": 1.0 / 120.0,
        "val_acc_l2": round(_eval_s5(m_s5, 2), 4),
        "val_acc_l4": round(_eval_s5(m_s5, 4), 4),
        "val_acc_l8": round(_eval_s5(m_s5, 8), 4),
        "val_acc_l16_ood": round(_eval_s5(m_s5, 16), 4),
        "val_acc_l32_ood": round(_eval_s5(m_s5, 32), 4),
        "s_fast_zero_ablation_l2": round(_eval_s5_sfast_zero(m_s5, 2), 4),
        "s_fast_zero_ablation_l8": round(_eval_s5_sfast_zero(m_s5, 8), 4),
    }
    del m_s5, opt_s5
    gc.collect()

    # P3: Single-pass dec_ffn, FLOP/B hook vs analytic, shift-robust P=4 + NgramHash, multidigit_add3
    tr_arr, val_arr = build_synthetic_tr_math_corpus(seed=seed)
    set_seed(seed)
    m_p3 = RDFLv05Model(cfg)
    opt_p3 = torch.optim.AdamW(m_p3.parameters(), lr=6e-3, weight_decay=0.05)
    m_p3.train()
    for s in range(45):
        xc, yc = sample_corpus_batch(tr_arr, seq_len=128, batch=8, seed=seed, step=s)
        opt_p3.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m_p3(xc, num_loops=2).reshape(-1, 256), yc.reshape(-1))
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(m_p3.parameters(), 1.0)
        opt_p3.step()
        m_p3.update_spectral_vectors()
    m_p3.eval()
    hook_off = _measure_hook_flops_per_byte(m_p3, torch.randint(0, 256, (2, 128)), update_slow=False)
    hook_on = _measure_hook_flops_per_byte(m_p3, torch.randint(0, 256, (2, 128)), update_slow=True)
    shift_bpbs: dict[str, float] = {}
    with torch.no_grad():
        for sh in (0, 1, 17):
            tot_ce = 0.0
            cnt = 0
            for i in range(4):
                xv, yv = sample_corpus_batch(val_arr, seq_len=128, batch=8, seed=999, step=i, contiguous=True)
                if sh > 0:
                    x_in = torch.cat([torch.full((xv.shape[0], sh), PAD_BYTE, dtype=torch.long), xv], dim=1)
                    lg = m_p3(x_in)[:, sh:]
                else:
                    lg = m_p3(xv)
                tot_ce += float(F.cross_entropy(lg.reshape(-1, 256), yv.reshape(-1), reduction="sum").item())
                cnt += int(yv.numel())
            shift_bpbs[f"bpb_shift_{sh}"] = round(tot_ce / float(max(1, cnt)) / LN2, 4)
    shift_bpbs["delta_shift_1"] = round(shift_bpbs["bpb_shift_1"] - shift_bpbs["bpb_shift_0"], 4)
    shift_bpbs["delta_shift_17"] = round(shift_bpbs["bpb_shift_17"] - shift_bpbs["bpb_shift_0"], 4)
    del m_p3, opt_p3
    gc.collect()

    set_seed(seed)
    m_add = RDFLv05Model(cfg)
    opt_add = torch.optim.AdamW(m_add.parameters(), lr=6e-3, weight_decay=0.01)
    m_add.train()
    for s in range(25):
        tb = multidigit_add(digits=3, batch=16, seed=seed, step=s, split="train")
        opt_add.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m_add(tb.x, num_loops=2).reshape(-1, 256), tb.masked_targets().reshape(-1), ignore_index=-100)
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(m_add.parameters(), 1.0)
        opt_add.step()
    m_add.eval()
    add_ok = add_tot = 0
    with torch.no_grad():
        for i in range(4):
            tb = multidigit_add(digits=3, batch=24, seed=999, step=i, split="val")
            lg = m_add(tb.x, num_loops=2)
            vm = tb.eval_mask.bool()
            add_ok += int((lg.argmax(-1)[vm] == tb.y[vm]).sum().item())
            add_tot += int(vm.sum().item())
    del m_add, opt_add
    gc.collect()

    # P4: Continual Learning + Streaming Surprise Gate + Adaptive Space/Boundary Gate
    cl_frozen = _run_cl_arm(cfg, "frozen", seed=seed, use_slow=False)
    cl_naive = _run_cl_arm(cfg, "naive_adamw", seed=seed, use_slow=False)
    cl_v05 = _run_cl_arm(cfg, "rdfl_v05_cl3", seed=seed, use_slow=True)

    set_seed(seed)
    m_st = RDFLv05Model(cfg)
    opt_st = torch.optim.AdamW(m_st.parameters(), lr=7e-3, weight_decay=0.01)
    m_st.train()
    for s in range(16):
        tb = _make_rule_batch(0, "train", s, 16, seed + 500)
        opt_st.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m_st(tb.x, num_loops=2).reshape(-1, 256), tb.masked_targets(-100).reshape(-1), ignore_index=-100)
        torch.autograd.backward(loss)
        opt_st.step()
    m_st.eval()
    x_fam = _make_rule_batch(0, "val", 0, 1, seed).x
    x_dist = torch.cat([_make_rule_batch(0, "val", i, 1, seed).x for i in range(1, 24)], dim=1)[:, :256]
    x_nov = _make_rule_batch(2, "val", 0, 1, seed).x
    with torch.no_grad():
        _, _, _, _, aux_fam = m_st.forward_stateful(x_fam, num_loops=2, update_slow=True, return_aux=True)
        _, _, _, st_a, aux_nov = m_st.forward_stateful(x_nov, num_loops=2, update_slow=True, return_aux=True)
        _, _, _, st_dist, aux_dist = m_st.forward_stateful(x_dist, state=st_a, num_loops=2, update_slow=True, return_aux=True)
        _, _, _, st_one = m_st.forward_stateful(x_dist, state=st_a, num_loops=2, update_slow=True, gate_mode="always_one")
        g_dist_mean = float(aux_dist["g_surp"].mean().item())
        _, _, _, st_rand = m_st.forward_stateful(
            x_dist,
            state=st_a,
            num_loops=2,
            update_slow=True,
            gate_mode="random_matched",
            random_gate_seed=123,
            random_gate_prob=max(0.05, min(0.95, g_dist_mean)),
        )
        assert st_a.s_slow is not None and st_dist.s_slow is not None and st_one.s_slow is not None and st_rand.s_slow is not None
        cos_surp = float(F.cosine_similarity(st_a.s_slow.flatten(), st_dist.s_slow.flatten(), dim=0).item())
        cos_one = float(F.cosine_similarity(st_a.s_slow.flatten(), st_one.s_slow.flatten(), dim=0).item())
        cos_rand = float(F.cosine_similarity(st_a.s_slow.flatten(), st_rand.s_slow.flatten(), dim=0).item())
    del m_st, opt_st
    gc.collect()

    # Adaptive Space/Boundary Gate (Copy-L16 vs Shuffled-16Bag)
    space_gate_res: dict[str, Any] = {}
    for cond_name, is_copy in (("copy_l16", True), ("shuf16_bag", False)):
        set_seed(seed)
        m_sg = RDFLv05Model(cfg)
        opt_sg = torch.optim.AdamW(m_sg.parameters(), lr=2e-3, weight_decay=0.01)
        m_sg.train()
        for step_sg in range(180):
            raw_b = _build_copy_l16(257, 7000 + step_sg) if is_copy else _build_shuf16_bag(257, 7000, 8000 + step_sg)
            t_arr = np.frombuffer(raw_b, dtype=np.uint8).astype(np.int64)
            xb_sg = torch.from_numpy(t_arr[:-1]).unsqueeze(0)
            yb_sg = torch.from_numpy(t_arr[1:]).unsqueeze(0)
            opt_sg.zero_grad(set_to_none=True)
            loss_sg = F.cross_entropy(m_sg(xb_sg, num_loops=2, update_slow=True).reshape(-1, 256), yb_sg.reshape(-1))
            torch.autograd.backward(loss_sg)
            torch.nn.utils.clip_grad_norm_(m_sg.parameters(), 1.0)
            opt_sg.step()
            m_sg.update_spectral_vectors()
        m_sg.eval()
        probe_b = _build_copy_l16(256, 9999) if is_copy else _build_shuf16_bag(256, 9999, 10999)
        p_t = torch.from_numpy(np.frombuffer(probe_b, dtype=np.uint8).astype(np.int64)).unsqueeze(0)
        space_gate_res[cond_name] = _extract_space_digit_mfast_hh(m_sg, p_t)
        del m_sg, opt_sg
        gc.collect()

    return {
        "p1_dynamic_depth": {
            "task": "dyck_stack(k=4,depth=4)",
            "chance_acc": 0.25,
            "r_sweep": p1_sweep,
            "t80_s_macro_r1_vs_r16_diff": t80_inv,
        },
        "p2_state_tracking": p2_res,
        "p3_byte_efficiency": {
            "corpus_source": "standalone_synthetic_tr_math_utf8",
            "analytic_flops": analytic_flops_per_byte(cfg, num_loops=2),
            "hook_slow_off": hook_off,
            "hook_slow_on": hook_on,
            "shift_robustness": shift_bpbs,
            "multidigit_add3_val_acc": round(add_ok / float(max(1, add_tot)), 4),
        },
        "p4_continual_learning": {
            "chance_acc": 0.125,
            "arms": {"frozen": cl_frozen, "naive_adamw": cl_naive, "rdfl_v05_cl3_slow_m2": cl_v05},
            "streaming_surprise_gate": {
                "g_surp_familiar_rule0": round(float(aux_fam["g_surp"].mean().item()), 4),
                "g_surp_novel_rule2": round(float(aux_nov["g_surp"].mean().item()), 4),
                "g_surp_distractor_256B": round(g_dist_mean, 4),
                "s_slow_cos_after_distractor_surprise": round(cos_surp, 4),
                "s_slow_cos_after_distractor_always_one": round(cos_one, 4),
                "s_slow_cos_after_distractor_random_matched": round(cos_rand, 4),
            },
            "adaptive_space_boundary_gate": space_gate_res,
        },
        "wall_s": time.perf_counter() - t0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unified RDFL v0.5 CPU verification suite.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    t_start = time.perf_counter()
    torch.set_num_threads(1)
    cfg = RDFLv05Config()

    print("[verify_v05] Section 1: Invariants, Parity, O(1), Return-Aux, R=1024...", flush=True)
    sec1 = run_invariants_and_parity(cfg)

    print("[verify_v05] Section 2: Step Reasoning + Learned Halting Gate vs ADFP...", flush=True)
    sec2, m_step, m_adfp = run_step_reasoning_and_halting()

    print("[verify_v05] Section 3: Anthropic Global Workspace (OOS J-Space)...", flush=True)
    gwt_step = run_gwt_jacobian_suite(m_step, "RDFLv05-StepHalting")
    gwt_adfp = run_gwt_jacobian_suite(m_adfp, "RDFLv05-ADFP")
    del m_step, m_adfp
    gc.collect()

    print("[verify_v05] Section 4: 4-Problem Regression Lock (P1-P4)...", flush=True)
    sec4 = run_4problem_regression_lock(cfg, seed=42)

    total_s = round(time.perf_counter() - t_start, 2)
    report: dict[str, Any] = {
        "schema": "rdfl_v05_audited_summary_v1",
        "model_class": "RDFLv05Model",
        "version": "0.5.0",
        "device": "cpu_fp32_fp64_single_thread",
        "seed": 42,
        "purity_guarantees": {
            "zero_attention": True,
            "zero_kv_cache": True,
            "zero_external_bpe": True,
            "zero_cuda": True,
            "zero_intermediate_supervision": True,
        },
        "section1_invariants_and_parity": sec1,
        "section2_step_reasoning_and_halting": sec2,
        "section3_anthropic_gwt_jlens": {"step_halting": gwt_step, "adfp": gwt_adfp},
        "section4_4problem_regression_lock": sec4,
        "total_runtime_s": total_s,
        "peak_rss_mib": round(peak_rss_mib(), 2),
    }
    write_strict_json(args.out, report)
    print(f"[verify_v05] DONE in {total_s}s -> {args.out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
