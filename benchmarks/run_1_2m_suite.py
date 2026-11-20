"""Single-command reproducible 1.20M suite (4 tasks x 4 arms) + FP32 SlotCore cosine diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.tasks_1_2m import TASK_REGISTRY, sample_task_batch
from rdfl.layers import GatedDeltaNetLayer, RMSNorm, SwiGLU
from rdfl.model import RDFLConfig, RDFLModel

ARM_NAMES = ("rlt0", "gdn", "rdfl", "rdfl_slotcore")


class _RLT0Block(nn.Module):
    """Causal self-attention + SwiGLU block without recurrent feedback (RLT-0 baseline)."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.n_attn = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.n_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = (
            p.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            for p in self.qkv(self.n_attn(x)).chunk(3, dim=-1)
        )
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out_proj(y.transpose(1, 2).reshape(b, t, d))
        return cast(Tensor, x + self.ffn(self.n_ffn(x)))


class _RLT0Model(nn.Module):
    """Feedback-off causal Transformer baseline (RLT-0)."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(256, d_model)
        self.blocks = nn.ModuleList(
            [_RLT0Block(d_model, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x: Tensor) -> Tensor:
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return cast(Tensor, self.head(self.final_norm(h)))


class _PureGDNModel(nn.Module):
    """Single-pass GatedDeltaNet baseline without BDH gate or SlotCore."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(256, d_model)
        self.n_gdn = RMSNorm(d_model)
        self.gdn = GatedDeltaNetLayer(d_model, num_heads, chunk_size=chunk_size)
        self.n_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim)
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x: Tensor) -> Tensor:
        h = self.emb(x)
        y, _ = self.gdn(self.n_gdn(h))
        h = h + y
        h = h + self.ffn(self.n_ffn(h))
        return cast(Tensor, self.head(self.final_norm(h)))


def build_arm(arm: str, d_model: int, num_heads: int, ffn_dim: int, chunk_size: int) -> nn.Module:
    """Instantiate one of the 4 benchmark arms."""
    if arm == "rlt0":
        return _RLT0Model(d_model, num_heads, ffn_dim, num_layers=2)
    if arm == "gdn":
        return _PureGDNModel(d_model, num_heads, ffn_dim, chunk_size)
    if arm == "rdfl":
        cfg = RDFLConfig(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_loops=2,
            chunk_size=chunk_size,
            num_slots=0,
            boundary_mode="none",
        )
        return RDFLModel(cfg)
    if arm == "rdfl_slotcore":
        cfg = RDFLConfig(
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_loops=2,
            chunk_size=chunk_size,
            num_slots=4,
            boundary_mode="slotcore_s4",
        )
        return RDFLModel(cfg)
    raise ValueError(f"Unknown arm '{arm}'. Valid: {ARM_NAMES}")


def _pairwise_slot_stats(slots: Tensor) -> dict[str, float]:
    """Compute off-diagonal pairwise cosine similarity and L2 norm for slot bank [S, D]."""
    s_mat = slots.detach().float()
    norms = s_mat.norm(dim=-1)
    unit = F.normalize(s_mat, p=2, dim=-1, eps=1e-8)
    sim = torch.matmul(unit, unit.transpose(0, 1))
    s_cnt = sim.shape[0]
    if s_cnt < 2:
        return {
            "cos_mean": 1.0,
            "cos_min": 1.0,
            "cos_max": 1.0,
            "norm_mean": round(float(norms.mean().item()), 4),
        }
    mask = ~torch.eye(s_cnt, dtype=torch.bool, device=sim.device)
    off = sim[mask]
    return {
        "cos_mean": round(float(off.mean().item()), 6),
        "cos_min": round(float(off.min().item()), 6),
        "cos_max": round(float(off.max().item()), 6),
        "norm_mean": round(float(norms.mean().item()), 4),
    }


@torch.no_grad()
def measure_slot_cosine(
    model: nn.Module,
    chunk_size: int,
    num_chunks: int,
    seed: int,
    device: torch.device,
) -> dict[str, object] | None:
    """Measure init vs. post-chunk pairwise slot cosine similarity for FP32 SlotCore."""
    if not isinstance(model, RDFLModel) or model.boundary is None:
        return None
    model.eval()
    init_stats = _pairwise_slot_stats(model.boundary.init_slots[0])
    gen = torch.Generator().manual_seed(seed + 9001)
    seq = torch.randint(0, 256, (1, chunk_size * num_chunks), generator=gen).to(device)
    _, _, _, state = model.forward_stateful(seq, state=None)
    post_stats = _pairwise_slot_stats(state.slot_state[0])
    return {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "init_slots": init_stats,
        "post_chunks_slots": post_stats,
    }


def _masked_loss_and_acc(logits: Tensor, targets: Tensor, mask: Tensor) -> tuple[Tensor, float]:
    sel_logits = logits[mask]
    sel_targets = targets[mask]
    loss = F.cross_entropy(sel_logits, sel_targets)
    acc = float((sel_logits.argmax(dim=-1) == sel_targets).float().mean().item() * 100.0)
    return loss, acc


def run_single_experiment(
    arm: str,
    task: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Train and evaluate one (arm, task) pair deterministically."""
    torch.manual_seed(args.seed)
    model = build_arm(arm, args.d_model, args.num_heads, args.ffn_dim, args.chunk_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_gen = torch.Generator().manual_seed(args.seed + 101)
    eval_gen = torch.Generator().manual_seed(args.seed + 202)

    t0 = time.perf_counter()
    model.train()
    last_loss = 0.0
    for _ in range(args.steps):
        x, y, mask = sample_task_batch(task, args.batch_size, train_gen)
        logits = model(x.to(device))
        loss, _ = _masked_loss_and_acc(logits, y.to(device), mask.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())

    model.eval()
    accs: list[float] = []
    with torch.no_grad():
        for _ in range(args.eval_batches):
            x, y, mask = sample_task_batch(task, args.batch_size, eval_gen)
            _, acc = _masked_loss_and_acc(model(x.to(device)), y.to(device), mask.to(device))
            accs.append(acc)

    rec: dict[str, object] = {
        "arm": arm,
        "task": task,
        "params": int(sum(p.numel() for p in model.parameters())),
        "steps": args.steps,
        "final_loss": round(last_loss, 4),
        "eval_acc_pct": round(sum(accs) / max(1, len(accs)), 2),
        "elapsed_sec": round(time.perf_counter() - t0, 3),
    }
    if args.measure_slot_cosine:
        slot_diag = measure_slot_cosine(model, args.chunk_size, args.slot_chunks, args.seed, device)
        if slot_diag is not None:
            rec["slot_cosine"] = slot_diag
    return rec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RDFL 1.20M algorithmic benchmark suite.")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_REGISTRY.keys()))
    parser.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--slot-chunks", type=int, default=16)
    parser.add_argument("--measure-slot-cosine", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Fast CPU smoke test (<10s).")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.smoke:
        args.d_model = 48
        args.num_heads = 4
        args.ffn_dim = 96
        args.chunk_size = 16
        args.steps = 2
        args.batch_size = 4
        args.eval_batches = 2
        args.slot_chunks = 8
        args.measure_slot_cosine = True
        args.device = "cpu"
    return args


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    t_start = time.perf_counter()
    runs = [
        run_single_experiment(arm, task, args, device)
        for task in args.tasks
        for arm in args.arms
    ]
    payload = {
        "mode": "smoke" if args.smoke else "full",
        "device": str(device),
        "seed": args.seed,
        "total_elapsed_sec": round(time.perf_counter() - t_start, 3),
        "runs": runs,
    }
    encoded = json.dumps(payload, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
