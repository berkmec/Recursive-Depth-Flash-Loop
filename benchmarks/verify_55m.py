"""Shared cell checker and 55M gate / live snapshot / SlotCore probe verifiers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast


class CellChecker:
    """Deterministic cell-by-cell comparator tracking checked count and mismatches."""

    def __init__(self) -> None:
        self.checked = 0
        self.errors: list[str] = []

    def eq(self, key: str, summary_val: Any, raw_val: Any) -> None:
        self.checked += 1
        if summary_val != raw_val:
            self.errors.append(f"{key}: summary={summary_val!r} != raw={raw_val!r}")

    def close(self, key: str, summary_val: float, raw_val: float, tol: float = 1e-4) -> None:
        self.checked += 1
        if not math.isclose(float(summary_val), float(raw_val), rel_tol=0.0, abs_tol=tol):
            self.errors.append(f"{key}: summary={summary_val} != raw={raw_val} (tol={tol})")


def load_json(path: Path) -> dict[str, Any]:
    """Read and parse a UTF-8 JSON object."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _verify_55m_params_and_gate1(
    ck: CellChecker, g55: dict[str, Any], s55: dict[str, Any]
) -> None:
    po = g55["matematik_ve_akis"]["param_ozeti"]
    for sk, rk in (
        ("base_55m", "taban_55m"),
        ("rdfl_narrow", "rdfl_dar_gdn_bdh"),
        ("fp32_slotcore", "oy6_fp32_slotcore"),
        ("k3_decoupled", "oy1_k3_kopuk"),
        ("mtp4_heads", "oy4_mtp4"),
        ("new_total", "yeni_toplam"),
        ("grand_total", "genel_toplam"),
    ):
        ck.eq(f"55m.param.{sk}", s55["parameters"][sk], po[rk])

    g1, m_ak, fp_d = s55["gate1_pass"], g55["matematik_ve_akis"], g55["fp16_ve_vram_duman"]
    ck.eq("55m.g1.logits", g1["step0_logits_diff"], m_ak["adim0_logits_maks_fark"])
    ck.eq("55m.g1.hn", g1["step0_hn_diff"], m_ak["adim0_hn_maks_fark"])
    ck.eq("55m.g1.cmem", g1["step0_cmem_diff"], m_ak["adim0_cmem_maks_fark"])
    ck.eq("55m.g1.causal", g1["causality_diff"], m_ak["nedensel_maks_fark"])
    ck.close("55m.g1.fwd_step", g1["forward_eq_step_fp32"], m_ak["forward_step_maks_fark"], 1e-10)
    ck.eq("55m.g1.nan", g1["smoke_12steps_nan"], fp_d["nan_adim"])
    ck.close("55m.g1.vram", g1["peak_vram_mib"], fp_d["tepe_vram_mib"], 0.1)
    ck.eq("55m.g1.passed", g1["passed"], g55["kapi1_gecti"])


def _verify_55m_ab_and_battery(
    ck: CellChecker,
    mtp: dict[str, Any],
    base: dict[str, Any],
    snap: dict[str, Any],
    s55: dict[str, Any],
) -> None:
    st0 = s55["pretrained_base_step0_fast_val"]
    raw_st0 = snap["pretrained_base_step0_fast_val"]
    for fld in ("lm_bits_512", "r1_lm_bits_512", "lm_bits_2048", "sft_val_bits", "mtp_t2_acc", "mtp_t3_acc", "mtp_t4_acc"):
        ck.close(f"55m.step0.{fld}", st0[fld], raw_st0[fld], 1e-4)

    ab_m = s55["hour1_mtp4_ab_1500steps"]
    for arm_s, arm_r in (("G0_control", "G0_kontrol"), ("G1_mtp4_full_55m_prop", "G1_mtp4_tam_yayilim")):
        for fld in ("lm_bits_512", "r1_lm_bits_512", "lm_bits_2048", "sft_val_bits", "mtp_t2_acc", "mtp_t3_acc", "mtp_t4_acc"):
            ck.close(f"55m.ab.{arm_s}.{fld}", ab_m[arm_s][fld], mtp[arm_r]["son_val"][fld], 1e-4)
    ck.close("55m.ab.d_lm512", ab_m["delta_lm_512"], mtp["delta_lm_bits_512"], 1e-4)
    ck.close("55m.ab.d_sft", ab_m["delta_sft_val"], mtp["delta_sft_val_bits"], 1e-4)
    ck.close("55m.ab.d_lm2048", ab_m["delta_lm_2048"], mtp["delta_lm_bits_2048"], 1e-4)
    ck.eq("55m.ab.winner", ab_m["winner"], mtp["secilen_kol"])

    b_ev = s55["baseline_55m_full_battery"]
    ck.close("55m.base.lm512", b_ev["mix_val_512_r2_bpb"], base["dil"]["mix_val_512_r2_bpb"], 1e-4)
    ck.close("55m.base.lm2048_full", b_ev["mix_val_2048_r2_bpb"], base["dil"]["mix_val_2048_r2_bpb"], 1e-4)
    ck.close("55m.base.sft_full_bpb", b_ev["sft_val_bits"], base["sft"]["sft_val_bits"], 1e-4)
    ck.close("55m.base.mtp_wall", b_ev["mtp4_step0_wall_speedup"], base["oy1b_ve_oy4_spekulatif"]["oy4_mtp4_duvar_hizlanma"], 1e-4)
    ck.close("55m.base.msp_mix", b_ev["msp_mix_val_auroc"], base["oy3_kalibrasyon"]["mix_val"]["auroc"], 1e-4)
    ck.close("55m.base.msp_sft", b_ev["msp_sft_val_auroc"], base["oy3_kalibrasyon"]["sft_val"]["auroc"], 1e-4)
    ck.close("55m.base.sft_acc", b_ev["sft_acc_full"], base["oy3_kalibrasyon"]["sft_val"]["dogruluk_tam"], 1e-4)
    ck.close("55m.base.sft_abs40", b_ev["sft_acc_abstain40"], base["oy3_kalibrasyon"]["sft_val"]["dogruluk_cekin40"], 1e-4)
    ck.close("55m.base.slot_lm2048", b_ev["slotcore_fast_val_lm_2048"], base["oy6_slotcore_ablasyon"]["slot_acik"]["mix_val_bpb_2048"], 1e-4)


def verify_55m_and_live(
    ck: CellChecker,
    raw_dir: Path,
    s55: dict[str, Any],
    live: dict[str, Any],
) -> None:
    """Verify 55M gates, full battery, live trajectory snapshot, and SlotCore cosine probe."""
    g55 = load_json(raw_dir / "dogrulama_kapisi_55m.json")
    mtp = load_json(raw_dir / "saat1_mtp_kapisi.json")
    base = load_json(raw_dir / "taban_55m_degerlendirme.json")
    snap = load_json(raw_dir / "diogenes_55m_live_val_snapshot.json")
    probe = load_json(raw_dir / "slotcore_55m_cosine_probe.json")

    ck.eq("55m.hash", s55["frozen_code_hash"], g55["kod_sha256"])
    ck.eq("55m.hash_mtp", s55["frozen_code_hash"], mtp["kod_sha256"])
    ck.eq("55m.hash_snap", s55["frozen_code_hash"], snap["meta"]["kod_sha256"])
    _verify_55m_params_and_gate1(ck, g55, s55)
    _verify_55m_ab_and_battery(ck, mtp, base, snap, s55)

    by_step = {int(r["step"]): r for r in snap["live_g1_trajectory_fast_val"]}
    for key, step_num in (
        ("preliminary_step_7000", 7000),
        ("preliminary_step_17500", 17500),
        ("preliminary_step_21000", 21000),
    ):
        row_s, row_r = live[key], by_step[step_num]
        ck.close(f"live.{step_num}.lm512", row_s["lm_bits_512"], row_r["lm_bits_512"], 1e-4)
        ck.close(f"live.{step_num}.r1", row_s["k3_r1_bits_512"], row_r["r1_lm_bits_512"], 1e-4)
        ck.close(f"live.{step_num}.lm2048", row_s["lm_bits_2048"], row_r["lm_bits_2048"], 1e-4)
        ck.close(f"live.{step_num}.sft", row_s["sft_val_bits"], row_r["sft_val_bits"], 1e-4)
        ck.close(f"live.{step_num}.t2", row_s["mtp_t2_acc"], row_r["mtp_t2_acc"], 1e-4)
        ck.close(f"live.{step_num}.t3", row_s["mtp_t3_acc"], row_r["mtp_t3_acc"], 1e-4)
        ck.close(f"live.{step_num}.t4", row_s["mtp_t4_acc"], row_r["mtp_t4_acc"], 1e-4)
        ck.close(f"live.{step_num}.vram", row_s["peak_vram_mib"], row_r["tepe_vram_mib"], 0.1)

    sc = live["mechanistic_finding_slotcore_symmetry_collapse"]
    ck.eq("probe.step", sc["checkpoint_step"], probe["meta"]["checkpoint_step"])
    ck.eq("probe.chunks", sc["chunks_evaluated"], probe["meta"]["chunks_evaluated"])
    ck.eq("probe.bytes", sc["bytes_evaluated"], probe["meta"]["bytes_evaluated"])
    ck.close("probe.init_min", sc["init_slots_cos_min"], probe["init_slots"]["cos_min"], 1e-6)
    ck.close("probe.init_max", sc["init_slots_cos_max"], probe["init_slots"]["cos_max"], 1e-6)
    ck.close("probe.init_mean", sc["init_slots_cos_mean"], probe["init_slots"]["cos_mean"], 1e-6)
    ck.close("probe.init_norm", sc["init_slots_norm_mean"], probe["init_slots"]["norm_mean"], 1e-4)
    ck.close("probe.post_mean", sc["post_16_chunks_cos_mean"], probe["post_16_chunks_slots"]["cos_mean"], 1e-6)
    ck.close("probe.post_min", sc["post_16_chunks_cos_min"], probe["post_16_chunks_slots"]["cos_min"], 1e-6)
    ck.close("probe.post_max", sc["post_16_chunks_cos_max"], probe["post_16_chunks_slots"]["cos_max"], 1e-6)
    ck.close("probe.post_norm", sc["post_16_chunks_norm_mean"], probe["post_16_chunks_slots"]["norm_mean"], 1e-4)
