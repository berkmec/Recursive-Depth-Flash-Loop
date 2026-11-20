"""Profile O(1) decode state memory vs. O(T) full-KV cache directly from PyTorch tensors on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdfl import GatedDeltaNetLayer, RDFLConfig, RDFLModel

MIB = float(1024 * 1024)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def measure_full_kv_fp16(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
) -> dict[str, float | int]:
    """Measure FP16 full-KV cache bytes directly from CPU PyTorch tensors."""
    total_bytes = 0
    for _ in range(num_layers):
        k_cache = torch.empty((1, num_heads, seq_len, head_dim), dtype=torch.float16, device="cpu")
        v_cache = torch.empty((1, num_heads, seq_len, head_dim), dtype=torch.float16, device="cpu")
        total_bytes += _tensor_nbytes(k_cache) + _tensor_nbytes(v_cache)
    return {
        "seq_len": seq_len,
        "bytes": total_bytes,
        "mib": round(total_bytes / MIB, 4),
    }


@torch.no_grad()
def measure_gdn_6l_o1(
    num_layers: int = 6,
    d_model: int = 768,
    num_heads: int = 8,
) -> dict[str, float | int | str]:
    """Measure 6-layer GatedDeltaNet O(1) step() state (FP32 S + FP16 short-conv) on CPU."""
    layer = GatedDeltaNetLayer(d_model=d_model, num_heads=num_heads).eval().to(dtype=torch.float16)
    x_step = torch.zeros((1, 1, d_model), dtype=torch.float16, device="cpu")
    _, (s_state, conv_state) = layer.step(x_step)
    s_bytes = num_layers * _tensor_nbytes(s_state)
    conv_bytes = num_layers * _tensor_nbytes(conv_state)
    total_bytes = s_bytes + conv_bytes
    return {
        "num_layers": num_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "head_dim": d_model // num_heads,
        "s_dtype": str(s_state.dtype).replace("torch.", ""),
        "conv_dtype": str(conv_state.dtype).replace("torch.", ""),
        "matrix_state_bytes": s_bytes,
        "conv_state_bytes": conv_bytes,
        "total_bytes": total_bytes,
        "mib": round(total_bytes / MIB, 4),
    }


@torch.no_grad()
def measure_default_rdfl_state(dtype: torch.dtype) -> dict[str, float | int | str]:
    """Measure default 1.2M RDFLConfig streaming state directly from RDFLModel.step()."""
    cfg = RDFLConfig()
    model = RDFLModel(cfg).eval().to(dtype=dtype)
    x_step = torch.zeros((1, 1), dtype=torch.long, device="cpu")
    _, state = model.step(x_step)
    gdn_s_bytes = sum(
        _tensor_nbytes(item[0]) for item in state.gdn_states if item is not None
    )
    conv_bytes = sum(
        _tensor_nbytes(item[1])
        for item in state.gdn_states
        if item is not None and item[1] is not None
    )
    slot_bytes = _tensor_nbytes(state.slot_state)
    total_bytes = gdn_s_bytes + conv_bytes + slot_bytes
    return {
        "activation_dtype": str(dtype).replace("torch.", ""),
        "slot_state_dtype": str(state.slot_state.dtype).replace("torch.", ""),
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "num_loops": cfg.num_loops,
        "num_slots": cfg.num_slots,
        "d_model": cfg.d_model,
        "num_heads": cfg.num_heads,
        "gdn_matrix_bytes": gdn_s_bytes,
        "short_conv_bytes": conv_bytes,
        "slot_state_bytes": slot_bytes,
        "total_bytes": total_bytes,
        "mib": round(total_bytes / MIB, 6),
    }


def run_profile() -> dict[str, object]:
    """Run complete CPU-only decode memory profile and return structured dictionary."""
    gdn_6l = measure_gdn_6l_o1(num_layers=6, d_model=768, num_heads=8)
    gdn_bytes = int(gdn_6l["total_bytes"])
    kv_points: list[dict[str, float | int]] = []
    for seq_len in (512, 4096, 32768):
        kv = measure_full_kv_fp16(num_layers=6, num_heads=8, head_dim=96, seq_len=seq_len)
        kv["gdn_o1_mib"] = float(gdn_6l["mib"])
        kv["reduction_factor_x"] = round(int(kv["bytes"]) / gdn_bytes, 1)
        kv_points.append(kv)

    return {
        "device": "cpu",
        "backbone_6l_d768_h8": {
            "gdn_o1_state": gdn_6l,
            "full_kv_fp16_scaling": kv_points,
        },
        "default_1_2m_rdfl": {
            "fp32_activations": measure_default_rdfl_state(torch.float32),
            "fp16_activations": measure_default_rdfl_state(torch.float16),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile RDFL O(1) decode state memory on CPU.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()
    payload = run_profile()
    encoded = json.dumps(payload, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
