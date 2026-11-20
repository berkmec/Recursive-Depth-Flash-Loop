"""Shared cell checker and 55M gate / final eval / RSI / G_BASE_ONLY / SlotCore verifiers."""

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


def _verify_tree(
    ck: CellChecker, prefix: str, sum_node: Any, raw_node: Any, tol: float = 1e-4
) -> None:
    """Recursively verify every leaf scalar in a summary subtree against its raw JSON subtree."""
    if isinstance(sum_node, dict):
        for k, v in sum_node.items():
            _verify_tree(ck, f"{prefix}.{k}", v, raw_node[k], tol)
    elif isinstance(sum_node, list):
        ck.eq(f"{prefix}.len", len(sum_node), len(raw_node))
        for i, (sv, rv) in enumerate(zip(sum_node, raw_node, strict=True)):
            _verify_tree(ck, f"{prefix}[{i}]", sv, rv, tol)
    elif isinstance(sum_node, float):
        ck.close(prefix, sum_node, float(raw_node), tol)
    else:
        ck.eq(prefix, sum_node, raw_node)


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
    flds = (
        "lm_bits_512", "r1_lm_bits_512", "lm_bits_2048",
        "sft_val_bits", "mtp_t2_acc", "mtp_t3_acc", "mtp_t4_acc",
    )
    st0, raw_st0 = s55["pretrained_base_step0_fast_val"], snap["pretrained_base_step0_fast_val"]
    for fld in flds:
        ck.close(f"55m.step0.{fld}", st0[fld], raw_st0[fld], 1e-4)

    ab_m = s55["hour1_mtp4_ab_1500steps"]
    for arm_s, arm_r in (("G0_control", "G0_kontrol"), ("G1_mtp4_full_55m_prop", "G1_mtp4_tam_yayilim")):
        for fld in flds:
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


def _verify_55m_trajectory_and_probe(
    ck: CellChecker,
    snap: dict[str, Any],
    probe: dict[str, Any],
    ana: dict[str, Any],
    rsi: dict[str, Any],
    fin: dict[str, Any],
) -> None:
    by_step = {int(r["step"]): r for r in snap["live_g1_trajectory_fast_val"]}
    traj = fin["trajectory_fast_val"]
    for step_num in (7000, 17500, 21000, 42000):
        rs, rr = traj[f"step_{step_num}"], by_step[step_num]
        ck.close(f"traj.{step_num}.lm512", rs["lm_bits_512"], rr["lm_bits_512"], 1e-4)
        ck.close(f"traj.{step_num}.r1", rs["k3_r1_bits_512"], rr["r1_lm_bits_512"], 1e-4)
        ck.close(f"traj.{step_num}.lm2048", rs["lm_bits_2048"], rr["lm_bits_2048"], 1e-4)
        ck.close(f"traj.{step_num}.sft", rs["sft_val_bits"], rr["sft_val_bits"], 1e-4)
        ck.close(f"traj.{step_num}.bil", rs["bilesik_skor"], rr["bilesik_skor"], 1e-4)
        ck.close(f"traj.{step_num}.t2", rs["mtp_t2_acc"], rr["mtp_t2_acc"], 1e-4)
        ck.close(f"traj.{step_num}.t3", rs["mtp_t3_acc"], rr["mtp_t3_acc"], 1e-4)
        ck.close(f"traj.{step_num}.t4", rs["mtp_t4_acc"], rr["mtp_t4_acc"], 1e-4)
        ck.close(f"traj.{step_num}.vram", rs["peak_vram_mib"], rr["tepe_vram_mib"], 0.1)

    sc = fin["mechanistic_finding_slotcore_symmetry_collapse"]
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
    ck.close("probe.ana_dbpb", sc["ana_55m_delta_bpb_2048"], ana["oy6_slotcore_ablasyon"]["delta_bpb_2048"], 1e-4)
    ck.close("probe.ana_ddur", sc["ana_55m_delta_durum_puan"], ana["oy6_slotcore_ablasyon"]["delta_durum_puan"], 1e-4)
    ck.close("probe.rsi_dbpb", sc["rsi_55m_delta_bpb_2048"], rsi["oy6_slotcore_ablasyon"]["delta_bpb_2048"], 1e-4)
    ck.close("probe.rsi_ddur", sc["rsi_55m_delta_durum_puan"], rsi["oy6_slotcore_ablasyon"]["delta_durum_puan"], 1e-4)


def _verify_55m_base_only_and_ablation(
    ck: CellChecker, gbo: dict[str, Any], fin: dict[str, Any]
) -> None:
    s_bo, ba, ks1 = fin["g_base_only_kontrol"], gbo["bolum_a_g_base_only"], gbo["karsilastirma_saat1"]
    ck.eq("gbo.steps", s_bo["toplam_adim"], ba["ayar"]["toplam_adim"])
    ck.eq("gbo.seed", s_bo["tohum"], ba["ayar"]["tohum"])
    ck.close("gbo.sec", s_bo["sure_sn"], ba["sure_sn"], 0.1)
    ck.close("gbo.vram", s_bo["tepe_vram_mib"], ba["tepe_vram_mib"], 0.1)
    ck.eq("gbo.skip", s_bo["atlanan_adim"], ba["atlanan_adim"])
    ck.eq("gbo.nan", s_bo["nan_sayisi"], ba["nan_sayisi"])
    ck.eq("gbo.dz0", s_bo["adim0_maks_abs_delta_r2"], ba["adim0_delta_z"]["maks_abs_delta_r2"])
    ck.eq("gbo.dz1", s_bo["son_maks_abs_delta_r2"], ba["son_delta_z"]["maks_abs_delta_r2"])
    _verify_tree(ck, "gbo.params", s_bo["param_sayim"], ba["param_sayim"])
    _verify_tree(ck, "gbo.son_val", s_bo["son_val"], ba["son_val"])
    _verify_tree(ck, "gbo.dg0", s_bo["delta_g0_eksi_base_only"], ks1["delta_g0_eksi_base_only"])
    _verify_tree(ck, "gbo.dg1", s_bo["delta_g1_eksi_base_only"], ks1["delta_g1_eksi_base_only"])
    td = ba["tam_degerlendirme"]
    _verify_tree(ck, "gbo.dil", s_bo["dil"], td["dil"])
    _verify_tree(ck, "gbo.sft", s_bo["sft"], td["sft"])
    _verify_tree(ck, "gbo.qa", s_bo["real_qa_d4000_qa"], td["real_qa_d4000"]["qa"])
    ck.close("gbo.durum", s_bo["durum_izleme_dogruluk"], td["durum_izleme_dogruluk"], 1e-4)
    _verify_tree(ck, "gbo.oy10", s_bo["oy10_toolrunner"], td["oy10_toolrunner"])

    s_ab, bb = fin["rdfl_on_off_21k"], gbo["bolum_b_en_iyi_rdfl_ablasyon"]
    ck.eq("ab21k.step", s_ab["ckpt_adim"], bb["ckpt_adim"])
    _verify_tree(ck, "ab21k.dz_on", s_ab["rdfl_acik_delta_z_tani"], bb["rdfl_acik"]["delta_z_tani"], 1e-6)
    _verify_tree(ck, "ab21k.dz_off", s_ab["rdfl_kapali_delta_z_tani"], bb["rdfl_kapali"]["delta_z_tani"], 1e-6)
    for branch in ("rdfl_acik", "rdfl_kapali"):
        sb, rb = s_ab[branch], bb[branch]
        _verify_tree(ck, f"ab21k.{branch}.hizli", sb["hizli_val"], rb["hizli_val"])
        _verify_tree(ck, f"ab21k.{branch}.dil", sb["dil"], rb["dil"])
        _verify_tree(ck, f"ab21k.{branch}.sft", sb["sft"], rb["sft"])
        _verify_tree(ck, f"ab21k.{branch}.qa", sb["real_qa_d4000_qa"], rb["real_qa_d4000"]["qa"])
        ck.close(f"ab21k.{branch}.plain_bits", sb["real_qa_d4000_plain_on_bits"], rb["real_qa_d4000"]["plain"]["on_bits"], 1e-4)
        ck.close(f"ab21k.{branch}.durum", sb["durum_izleme_dogruluk"], rb["durum_izleme_dogruluk"], 1e-4)
        _verify_tree(ck, f"ab21k.{branch}.oy10", sb["oy10_toolrunner"], rb["oy10_toolrunner"])
    _verify_tree(ck, "ab21k.delta", s_ab["delta_acik_eksi_kapali"], bb["delta_acik_eksi_kapali"])


def _verify_55m_final_evals(
    ck: CellChecker,
    ozet: dict[str, Any],
    ana: dict[str, Any],
    rsi_log: dict[str, Any],
    rsi: dict[str, Any],
    fin: dict[str, Any],
) -> None:
    tr = fin["training_42k_summary"]
    ck.eq("55m.tr.steps", tr["toplam_adim"], ozet["toplam_adim"])
    ck.eq("55m.tr.skip", tr["atlanan_adim"], ozet["atlanan_adim"])
    ck.close("55m.tr.sec", tr["sure_sn"], ozet["sure_sn"], 0.1)
    ck.close("55m.tr.vram", tr["tepe_vram_mib"], ozet["tepe_vram_mib"], 0.1)
    ck.close("55m.tr.best", tr["en_iyi_bilesik"], ozet["en_iyi_bilesik"], 1e-4)
    _verify_tree(ck, "55m.tr.son42k", tr["son_val_42k"], ozet["son_val"])

    _verify_tree(ck, "55m.ana", fin["ana_55m"], ana)
    _verify_tree(ck, "55m.rsi", fin["rsi_55m"], rsi)

    s_r = fin["oy7_rsi_55m"]
    for fld in ("nesil_sayisi", "nesil_adim", "rollback_sayisi", "toplam_atlanan"):
        ck.eq(f"55m.oy7.{fld}", s_r[fld], rsi_log[fld])
    ck.close("55m.oy7.sec", s_r["sure_sn"], rsi_log["sure_sn"], 0.1)
    ck.close("55m.oy7.vram", s_r["tepe_vram_mib"], rsi_log["tepe_vram_mib"], 0.1)
    _verify_tree(ck, "55m.oy7.tur0", s_r["tur0"], rsi_log["tur0"])
    _verify_tree(ck, "55m.oy7.son", s_r["son_nesil"], rsi_log["son_nesil"])
    l4_acc = [g["hakem_kabul_orani"] for g in rsi_log["nesiller"] if g["l_cur"] == 4]
    l5_acc = [g["hakem_kabul_orani"] for g in rsi_log["nesiller"] if g["l_cur"] == 5]
    ck.close("55m.oy7.l4_acc", s_r["l4_kabul_orani"], sum(l4_acc) / len(l4_acc), 1e-4)
    ck.close("55m.oy7.l5_acc", s_r["l5_kabul_orani"], sum(l5_acc) / len(l5_acc), 1e-4)


def verify_55m_and_final(
    ck: CellChecker,
    raw_dir: Path,
    s55: dict[str, Any],
    fin: dict[str, Any],
    oy10_sum: dict[str, Any],
) -> None:
    """Verify all 55M gates, 42K final eval, 8-gen RSI, G_BASE_ONLY, RDFL ON/OFF, and SlotCore."""
    g55 = load_json(raw_dir / "dogrulama_kapisi_55m.json")
    mtp = load_json(raw_dir / "saat1_mtp_kapisi.json")
    base = load_json(raw_dir / "taban_55m_degerlendirme.json")
    snap = load_json(raw_dir / "diogenes_55m_live_val_snapshot.json")
    probe = load_json(raw_dir / "slotcore_55m_cosine_probe.json")
    ozet = load_json(raw_dir / "ana_egitim_ozet.json")
    ana = load_json(raw_dir / "ana_55m_degerlendirme.json")
    rsi_log = load_json(raw_dir / "oy7_rsi_55m.json")
    rsi = load_json(raw_dir / "rsi_55m_degerlendirme.json")
    gbo = load_json(raw_dir / "g_base_only_kontrol.json")

    for lbl, h in (
        ("g55", g55["kod_sha256"]), ("mtp", mtp["kod_sha256"]), ("snap", snap["meta"]["kod_sha256"]),
        ("ozet", ozet["kod_sha256"]), ("ana", ana["kod_sha256"]), ("rsi_log", rsi_log["kod_sha256"]),
        ("rsi", rsi["kod_sha256"]), ("gbo", gbo["kod_sha256"]), ("fin", fin["frozen_code_hash"]),
    ):
        ck.eq(f"55m.hash.{lbl}", s55["frozen_code_hash"], h)

    _verify_55m_params_and_gate1(ck, g55, s55)
    _verify_55m_ab_and_battery(ck, mtp, base, snap, s55)
    _verify_55m_trajectory_and_probe(ck, snap, probe, ana, rsi, fin)
    _verify_55m_final_evals(ck, ozet, ana, rsi_log, rsi, fin)
    _verify_55m_base_only_and_ablation(ck, gbo, fin)

    ck.close("oy10.post_ana_call", oy10_sum["posttraining_55m_ana_call_rate"], ana["oy10_toolrunner"]["toolrunner_cagri_orani"], 1e-4)
    ck.close("oy10.post_ana_acc", oy10_sum["posttraining_55m_ana_tool_acc"], ana["oy10_toolrunner"]["toolrunner_dogruluk"], 1e-4)
    ck.close("oy10.post_ana_notool", oy10_sum["posttraining_55m_ana_no_tool_acc"], ana["oy10_toolrunner"]["aracsiz_dogruluk"], 1e-4)
    ck.close("oy10.post_rsi_acc", oy10_sum["posttraining_55m_rsi_tool_acc"], rsi["oy10_toolrunner"]["toolrunner_dogruluk"], 1e-4)
    ck.close("oy10.post_gbo_acc", oy10_sum["posttraining_55m_base_only_1500_tool_acc"], gbo["bolum_a_g_base_only"]["tam_degerlendirme"]["oy10_toolrunner"]["toolrunner_dogruluk"], 1e-4)
    ck.close("oy10.post_off21k_acc", oy10_sum["posttraining_55m_rdfl_off_21k_tool_acc"], gbo["bolum_b_en_iyi_rdfl_ablasyon"]["rdfl_kapali"]["oy10_toolrunner"]["toolrunner_dogruluk"], 1e-4)
