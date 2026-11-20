"""Programmatically verify every audited cell in results/audited_summary.json against results/raw."""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.verify_55m import CellChecker, load_json, verify_55m_and_final

RAW_DIR = ROOT / "results" / "raw"
SUMMARY_PATH = ROOT / "results" / "audited_summary.json"


def _mean_std(vals: list[float], ndigits: int) -> tuple[float, float]:
    mean_v = round(statistics.mean(vals), ndigits)
    std_v = round(statistics.stdev(vals), ndigits) if len(vals) > 1 else 0.0
    return mean_v, std_v


def _filter_rows(rows: list[dict[str, Any]], kol: str, gorev: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["kol"] == kol and r["gorev"] == gorev]


def _verify_arch_arms(ck: CellChecker, ab: dict[str, Any], arch: dict[str, Any]) -> None:
    by_arm: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in ab["kosular"]:
        by_arm.setdefault(row["kol"], {}).setdefault(row["gorev"], []).append(row)

    spec = (
        ("parity64", "parity64", "alt_metrik", 2),
        ("s5_perm32", "s5_perm32", "alt_metrik", 2),
        ("mqar_overwrite", "mqar_overwrite", "alt_metrik", 2),
        ("multihop_icl", "multihop_icl", "alt_metrik", 2),
        ("byte_lm", "byte_lm_bpb", "birincil_metrik", 4),
    )
    arms = arch["all_8_arms_1_20m"]
    ck.eq("arch.arm_count", len(arms), 8)
    for arm_name, entry in arms.items():
        tmap = by_arm[arm_name]
        ck.eq(f"arch.{arm_name}.params", entry["params"], tmap["parity64"][0]["parametre"])
        for raw_task, sum_key, field, nd in spec:
            vals = [float(r[field]) for r in tmap[raw_task]]
            m_val, s_val = _mean_std(vals, nd)
            ck.close(f"arch.{arm_name}.{sum_key}.mean", entry[sum_key]["mean"], m_val, 10**-nd)
            ck.close(f"arch.{arm_name}.{sum_key}.std", entry[sum_key]["std"], s_val, 10**-nd)


def _verify_arch_ablations(ck: CellChecker, ab: dict[str, Any], arch: dict[str, Any]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ab["dongu_cipa_rlt2_ablasyonu"]:
        grouped.setdefault((row["etiket"], row["gorev"]), []).append(row)

    for tag, entry in arch["structural_ablations"].items():
        if not isinstance(entry, dict):
            continue
        mh_rows = grouped[(tag, "multihop_icl")]
        lm_rows = grouped[(tag, "byte_lm")]
        mh_m, mh_s = _mean_std([float(r["alt_metrik"]) for r in mh_rows], 2)
        lm_m, lm_s = _mean_std([float(r["birincil_metrik"]) for r in lm_rows], 4)
        kb_s = round(statistics.mean(float(r["hiz_kb_sn"]) for r in lm_rows), 2)
        ck.close(f"arch.abl.{tag}.mh_mean", entry["multihop_3hop_mean"], mh_m, 0.01)
        ck.close(f"arch.abl.{tag}.mh_std", entry["multihop_3hop_std"], mh_s, 0.01)
        ck.close(f"arch.abl.{tag}.lm_bpb", entry["byte_lm_bpb_mean"], lm_m, 1e-4)
        ck.close(f"arch.abl.{tag}.lm_std", entry["byte_lm_bpb_std"], lm_s, 1e-4)
        ck.close(f"arch.abl.{tag}.kb_s", entry["byte_lm_kb_s"], kb_s, 0.01)


def _verify_arch_memory(ck: CellChecker, gp: dict[str, Any], arch: dict[str, Any]) -> None:
    mem = arch["memory_scaling"]
    dec = gp["uzun_baglam_decode_32k"]
    kv_rows = [r for r in dec if r["mimari"] == "Transformer_6L_TamKV"]
    gdn_rows = [r for r in dec if r["mimari"] == "GatedDeltaNet_6L_O1"]
    ck.eq("arch.mem.seq_lens", mem["measured_seq_lens"], [r["baglam_t"] for r in kv_rows])
    ck.eq("arch.mem.kv_mib", mem["decode_transformer_6l_kv_mib"], [r["durum_vram_mib"] for r in kv_rows])
    ck.eq("arch.mem.gdn_mib", mem["decode_gdn_6l_o1_measured_mib"], [r["durum_vram_mib"] for r in gdn_rows])
    exact_gdn_mib = (6 * 8 * 96 * 96 * 4 + 6 * 768 * 3 * 2) / (1024 * 1024)
    ck.close("arch.mem.gdn_exact_mib", mem["decode_gdn_6l_o1_exact_mib"], round(exact_gdn_mib, 4), 1e-4)
    ck.close("arch.mem.ratio_32k", mem["decode_32k_reduction_factor_exact"], round(576.0 / exact_gdn_mib, 1), 0.1)
    ck.close("arch.mem.vram_ceil", mem["vram_ceiling_mib"], gp["meta"]["vram_tavan_mib"], 0.1)
    b_rows = {(r["etiket"], r["baglam_t"]): r for r in gp["baglam_olceklenme_55m"]}
    ck.eq("arch.mem.rlt2_t8192", mem["train_55m_t8192_rlt2_status"], b_rows[("RLT2_55M_d768_L3_B64", 8192)]["durum"])
    ck.close("arch.mem.rdfl_t8192", mem["train_55m_t8192_rdfl_r2_mib"], b_rows[("DeltaBDHRLT_55M_d768_L3_R2", 8192)]["tepe_vram_mib"], 0.1)


def _verify_arch_study(ck: CellChecker, arch: dict[str, Any]) -> None:
    gate = load_json(RAW_DIR / "dogrulama_kapisi.json")
    ab = load_json(RAW_DIR / "ablasyon_sonuclari.json")
    gp = load_json(RAW_DIR / "gpu_profil_55m.json")

    ck.eq("arch.hash", arch["frozen_code_hash"], gate["meta"]["kod_sha256"])
    ck.eq("arch.hash_ab", arch["frozen_code_hash"], ab["meta"]["kod_sha256"])
    ck.eq("arch.hash_gp", arch["frozen_code_hash"], gp["meta"]["kod_sha256"])
    total_runs = sum(len(v) for v in ab.values() if isinstance(v, list))
    ck.eq("arch.total_runs", arch["total_gpu_runs"], total_runs)
    core_secs = ("kosular", "h2_yogun_vs_cok_basli", "dongu_cipa_rlt2_ablasyonu", "oz_damitma_ablasyonu")
    core_nans = sum(1 for sec in core_secs for r in ab[sec] if r.get("nan_kayip_adim", 0) > 0)
    slot_nans = sum(1 for r in ab["slotcore_paralel_yuva_ablasyonu"] if r.get("nan_kayip_adim", 0) > 0)
    ck.eq("arch.core_nans", arch["core_186_runs_fp16_nans"], core_nans)
    ck.eq("arch.legacy_slot_nans", arch["legacy_fp16_slot4_nan_runs"], slot_nans)
    ck.close("arch.eq_fp64", arch["forward_eq_step_fp64_max_diff"], gate["parca_ozyineleme_esitligi"]["a6_tam_model_step_maks_fark"], 1e-18)
    max_causal = max(v["nedensel_maks_fark"] for v in gate["kol_dogrulamalari"].values())
    ck.eq("arch.causality", arch["causality_max_diff"], max_causal)

    _verify_arch_arms(ck, ab, arch)
    _verify_arch_ablations(ck, ab, arch)
    _verify_arch_memory(ck, gp, arch)


def _verify_oy1_k3(ck: CellChecker, oy14: dict[str, Any], oy_gate: dict[str, Any], oy: dict[str, Any]) -> None:
    ck.eq("oy.hash", oy["frozen_code_hash"], oy_gate["meta"]["kod_sha256"])
    ck.eq("oy.hash_14", oy["frozen_code_hash"], oy14["meta"]["kod_sha256"])
    lm = oy14["ozet"]["oy1_3_ozet"]["byte_lm"]
    k3 = oy["oy1_k3_early_exit"]
    ck.close("oy1.r1_before", k3["r1_bpb_before"], lm["OY1_K0_STANDART"]["R1"]["ort"])
    ck.close("oy1.r1_after", k3["r1_bpb_after"], lm["OY1_K3_KOPUK_BASLIK"]["R1"]["ort"])
    ck.close("oy1.r1_gain", k3["r1_gain_bpb"], lm["OY1_K3_KOPUK_BASLIK"]["R1"]["ort"] - lm["OY1_K0_STANDART"]["R1"]["ort"])
    ck.close("oy1.r4_before", k3["r4_bpb_before"], lm["OY1_K0_STANDART"]["R4"]["ort"])
    ck.close("oy1.r4_after", k3["r4_bpb_after"], lm["OY1_K3_KOPUK_BASLIK"]["R4"]["ort"])
    ck.close("oy1.r4_tax", k3["r4_tax_bpb"], lm["OY1_K3_KOPUK_BASLIK"]["R4"]["ort"] - lm["OY1_K0_STANDART"]["R4"]["ort"])

    prompts = diff_bytes = 0
    for row in oy14["oy1_3_kosular"]:
        for kd in (row.get("spekulatif_oy1b") or {}).get("k_tarama", {}).values():
            prompts += int(kd.get("duvar_saati_n_istem", 0))
            diff_bytes += int(kd.get("kayipsiz_fark_bayt", 0))
    ck.eq("oy1.spec_prompts", k3["speculative_prompts_total"], prompts)
    ck.eq("oy1.spec_diff", k3["speculative_byte_diff"], diff_bytes)

    k3_lm = _filter_rows(oy14["oy1_3_kosular"], "OY1_K3_KOPUK_BASLIK", "byte_lm")
    k1_lm = _filter_rows(oy14["oy1_3_kosular"], "OY1_K1_KL050_SABIT", "byte_lm")
    ck.close("oy1.k3_k2_flop", k3["speculative_flop_speedup_k2_k4"][0], statistics.mean(r["spekulatif_oy1b"]["k_tarama"]["K2"]["rollback_dahil_flop_hizlanma"] for r in k3_lm))
    ck.close("oy1.k3_k4_flop", k3["speculative_flop_speedup_k2_k4"][1], statistics.mean(r["spekulatif_oy1b"]["k_tarama"]["K4"]["rollback_dahil_flop_hizlanma"] for r in k3_lm))
    ck.close("oy1.k3_k2_wall", k3["wall_speedup_k3_k2"], statistics.mean(r["spekulatif_oy1b"]["k_tarama"]["K2"]["duvar_saati_hizlanma"] for r in k3_lm))
    ck.close("oy1.k3_k4_wall", k3["wall_speedup_k3_k4"], statistics.mean(r["spekulatif_oy1b"]["k_tarama"]["K4"]["duvar_saati_hizlanma"] for r in k3_lm))
    ck.close("oy1.k1_k4_wall", k3["wall_speedup_k1_k4"], statistics.mean(r["spekulatif_oy1b"]["k_tarama"]["K4"]["duvar_saati_hizlanma"] for r in k1_lm))


def _verify_oy2(ck: CellChecker, oy14: dict[str, Any], oy: dict[str, Any]) -> None:
    o2 = oy["oy2_adaptive_depth"]
    k3_lm = _filter_rows(oy14["oy1_3_kosular"], "OY1_K3_KOPUK_BASLIK", "byte_lm")
    k3_gr = _filter_rows(oy14["oy1_3_kosular"], "OY1_K3_KOPUK_BASLIK", "graf_izli")
    ck.close("oy2.lm_b32_flop", o2["byte_lm_chunk_b32_flop_save_pct"], statistics.mean(r["oy2_durumlu_parca_b32"]["flop_tasarruf_pct"] for r in k3_lm), 0.01)
    ck.close("oy2.lm_b32_bpb", o2["byte_lm_chunk_b32_delta_bpb"], statistics.mean(r["oy2_durumlu_parca_b32"]["delta_bpb_vs_r4"] for r in k3_lm), 1e-4)
    ck.close("oy2.lm_b32_acc", o2["byte_lm_chunk_b32_delta_acc_pp"], statistics.mean(r["oy2_durumlu_parca_b32"]["delta_acc_vs_r4"] for r in k3_lm), 0.01)
    ck.close("oy2.lm_st_flop", o2["byte_lm_step_flop_save_pct"], statistics.mean(r["oy2_durumlu_bayt_step"]["flop_tasarruf_pct"] for r in k3_lm), 0.01)
    ck.close("oy2.lm_st_bpb", o2["byte_lm_step_delta_bpb"], statistics.mean(r["oy2_durumlu_bayt_step"]["delta_bpb_vs_r4"] for r in k3_lm), 1e-4)
    ck.close("oy2.gr_b32_flop", o2["graf_chunk_b32_flop_save_pct"], statistics.mean(r["oy2_durumlu_parca_b32"]["flop_tasarruf_pct"] for r in k3_gr), 0.01)
    ck.close("oy2.gr_b32_bpb", o2["graf_chunk_b32_delta_bpb"], statistics.mean(r["oy2_durumlu_parca_b32"]["delta_bpb_vs_r4"] for r in k3_gr), 1e-4)
    ck.close("oy2.gr_b32_acc", o2["graf_chunk_b32_delta_acc_pp"], statistics.mean(r["oy2_durumlu_parca_b32"]["delta_acc_vs_r4"] for r in k3_gr), 0.01)


def _verify_oy3(ck: CellChecker, oy14: dict[str, Any], oy: dict[str, Any]) -> None:
    cal = oy["oy3_msp_calibration"]
    k3_lm = _filter_rows(oy14["oy1_3_kosular"], "OY1_K3_KOPUK_BASLIK", "byte_lm")
    k3_gr = _filter_rows(oy14["oy1_3_kosular"], "OY1_K3_KOPUK_BASLIK", "graf_izli")
    ck.close("oy3.auroc", cal["byte_lm_auroc"], statistics.mean(r["oy3_kalibrasyon"]["sinyaller"]["sicaklik_kalibre_msp"]["auroc_hata"] for r in k3_lm))
    ck.close("oy3.ece15", cal["byte_lm_ece15"], statistics.mean(r["oy3_kalibrasyon"]["sinyaller"]["sicaklik_kalibre_msp"]["ece_15"] for r in k3_lm))
    b_acc = statistics.mean(r["oy3_kalibrasyon"]["taban_dogruluk_pct"] for r in k3_lm)
    s60 = statistics.mean(r["oy3_kalibrasyon"]["sinyaller"]["sicaklik_kalibre_msp"]["secici_acc_60pct"] for r in k3_lm)
    ck.close("oy3.abstain_before", cal["abstain40_acc_before"], b_acc, 0.01)
    ck.close("oy3.abstain_after", cal["abstain40_acc_after"], s60, 0.01)
    ck.close("oy3.abstain_gain", cal["abstain40_gain_pp"], s60 - b_acc, 0.01)
    g_raw = statistics.mean(r["oy3_kalibrasyon"]["sinyaller"]["msp_1_eksi_maxp"]["ece_15"] for r in k3_gr)
    g_cal = statistics.mean(r["oy3_kalibrasyon"]["sinyaller"]["sicaklik_kalibre_msp"]["ece_15"] for r in k3_gr)
    ck.close("oy3.graf_ece_raw", cal["graf_ece_raw"], g_raw, 1e-4)
    ck.close("oy3.graf_ece_cal", cal["graf_ece_cal"], g_cal, 1e-4)
    ck.close("oy3.graf_ece_ratio", cal["graf_ece_reduction_x"], g_raw / g_cal, 0.02)


def _verify_oy4_and_oy5(ck: CellChecker, oy14: dict[str, Any], oy56: dict[str, Any], oy: dict[str, Any]) -> None:
    o4, m4 = oy14["ozet"]["oy4_ozet"], oy["oy4_mtp4_at_1_2m"]
    ck.close("oy4.lm_ntp", m4["byte_lm_ntp_bpb"], o4["byte_lm"]["OY4_NTP"]["t1_bpb"]["ort"])
    ck.close("oy4.lm_mtp4", m4["byte_lm_mtp4_bpb"], o4["byte_lm"]["OY4_MTP4"]["t1_bpb"]["ort"])
    ck.close("oy4.lm_tax", m4["byte_lm_tax_bpb"], o4["byte_lm"]["OY4_MTP4"]["t1_bpb"]["ort"] - o4["byte_lm"]["OY4_NTP"]["t1_bpb"]["ort"])
    ck.close("oy4.rsi_ntp", m4["rsi_l34_ntp_bpb"], o4["rsi_zincir_L34"]["OY4_NTP"]["t1_bpb"]["ort"])
    ck.close("oy4.rsi_mtp4", m4["rsi_l34_mtp4_bpb"], o4["rsi_zincir_L34"]["OY4_MTP4"]["t1_bpb"]["ort"])
    ck.close("oy4.rsi_tax", m4["rsi_l34_tax_bpb"], o4["rsi_zincir_L34"]["OY4_MTP4"]["t1_bpb"]["ort"] - o4["rsi_zincir_L34"]["OY4_NTP"]["t1_bpb"]["ort"])
    s_rows = _filter_rows(oy14["oy4_kosular"], "OY4_MTP4", "rsi_zincir_L34")
    for idx, fk in enumerate(("t2_acc", "t3_acc", "t4_acc")):
        ck.close(f"oy4.struct_{fk}", m4["structural_acc_t2_t3_t4"][idx], statistics.mean(r[fk] for r in s_rows), 0.01)

    o5, s5 = oy56["ozet"]["oy5_ozet"], oy["oy5_prompt_self_correction"]
    ck.close("oy5.d0", s5["d0_draft_acc"], o5["D0_pass1"]["ort"], 0.01)
    ck.close("oy5.d1_rev", s5["d1_d2_after_revision_acc"][0], o5["D1_d0_taslak_revize"]["ort"], 0.01)
    ck.close("oy5.d2_rev", s5["d1_d2_after_revision_acc"][1], o5["D2_d0_taslak_revize"]["ort"], 0.01)
    ck.close("oy5.logp_auroc", s5["zero_param_logp_auroc"], o5["logp_auroc"]["ort"], 1e-4)
    ck.close("oy5.mlp_auroc", s5["mlp_verifier_auroc"], o5["oz_dog_auroc"]["ort"], 1e-4)


def _verify_oy6(ck: CellChecker, oy56: dict[str, Any], oy: dict[str, Any]) -> None:
    s6, rows6, o6 = oy["oy6_fp32_slotcore"], oy56["oy6_kosular"], oy56["ozet"]["oy6_ozet"]
    ck.eq("oy6.total_runs", s6["total_runs"], len(rows6))
    ck.eq("oy6.zero_nan_runs", s6["runs_zero_nan"], sum(1 for r in rows6 if r["nan_kayip_adim"] == 0))
    gr, lm = o6["graf_izli"], o6["byte_lm"]
    ck.close("oy6.mqar_mean", s6["mqar_acc_mean"], o6["mqar_overwrite"]["OY6_A6_SLOT4_FP32"]["genel_acc"]["ort"], 0.01)
    ck.close("oy6.mqar_std", s6["mqar_acc_std"], o6["mqar_overwrite"]["OY6_A6_SLOT4_FP32"]["genel_acc"]["std"], 0.01)
    ck.close("oy6.slot_graf_m", s6["graf_acc_mean"], gr["OY6_A6_SLOT4_FP32"]["genel_acc"]["ort"], 0.01)
    ck.close("oy6.slot_graf_s", s6["graf_acc_std"], gr["OY6_A6_SLOT4_FP32"]["genel_acc"]["std"], 0.01)
    ck.close("oy6.base_graf_m", s6["baseline_graf_mean"], gr["OY6_A6_TABAN"]["genel_acc"]["ort"], 0.01)
    ck.close("oy6.base_graf_s", s6["baseline_graf_std"], gr["OY6_A6_TABAN"]["genel_acc"]["std"], 0.01)
    ck.close("oy6.bdhl_graf_m", s6["bdhless_graf_mean"], gr["OY6_A6_BDHSIZ"]["genel_acc"]["ort"], 0.01)
    ck.close("oy6.bdhl_graf_s", s6["bdhless_graf_std"], gr["OY6_A6_BDHSIZ"]["genel_acc"]["std"], 0.01)
    ck.close("oy6.titans_graf_m", s6["titans_a_graf_mean"], gr["OY6_A6_SASKINLIK_GDN"]["genel_acc"]["ort"], 0.01)
    ck.close("oy6.titans_graf_s", s6["titans_a_graf_std"], gr["OY6_A6_SASKINLIK_GDN"]["genel_acc"]["std"], 0.01)
    ck.close("oy6.std_red", s6["std_reduction_x"], gr["OY6_A6_TABAN"]["genel_acc"]["std"] / gr["OY6_A6_SLOT4_FP32"]["genel_acc"]["std"], 0.02)
    ck.close("oy6.bdh_lm_delta", s6["bdh_vs_bdhless_byte_lm_delta_bpb"], lm["OY6_A6_TABAN"]["bpb"]["ort"] - lm["OY6_A6_BDHSIZ"]["bpb"]["ort"], 1e-4)
    ck.close("oy6.slot_lm_m", s6["slot_byte_lm_bpb_mean"], lm["OY6_A6_SLOT4_FP32"]["bpb"]["ort"], 1e-4)
    ck.close("oy6.slot_lm_s", s6["slot_byte_lm_bpb_std"], lm["OY6_A6_SLOT4_FP32"]["bpb"]["std"], 1e-4)
    ck.close("oy6.base_lm_m", s6["baseline_byte_lm_bpb_mean"], lm["OY6_A6_TABAN"]["bpb"]["ort"], 1e-4)
    ck.close("oy6.base_lm_s", s6["baseline_byte_lm_bpb_std"], lm["OY6_A6_TABAN"]["bpb"]["std"], 1e-4)
    ck.close("oy6.slot_lm_delta", s6["slot_vs_baseline_byte_lm_delta_bpb"], lm["OY6_A6_SLOT4_FP32"]["bpb"]["ort"] - lm["OY6_A6_TABAN"]["bpb"]["ort"], 1e-4)

    seeds = s6["graf_izli_per_seed"]["seeds"]
    by_ks = {(r["kol"], r["tohum"]): r["genel_acc"] for r in rows6 if r["gorev"] == "graf_izli"}
    for arm in ("OY6_A6_TABAN", "OY6_A6_BDHSIZ", "OY6_A6_SLOT4_FP32", "OY6_A6_SASKINLIK_GDN"):
        ck.eq(f"oy6.seed.{arm}", s6["graf_izli_per_seed"][arm], [by_ks[(arm, sd)] for sd in seeds])


def _verify_oy7_8_10(
    ck: CellChecker,
    oy14: dict[str, Any],
    oy56: dict[str, Any],
    oy7: dict[str, Any],
    oy8: dict[str, Any],
    base55: dict[str, Any],
    oy: dict[str, Any],
) -> None:
    vrams = [r["tepe_vram_mib"] for r in oy14["oy1_3_kosular"] + oy14["oy4_kosular"] + oy56["oy6_kosular"]]
    vrams.extend(v["tepe_vram_mib_maks"] for v in oy7["ozet"].values())
    ck.close("oy.peak_vram", oy["peak_vram_mib"], max(vrams), 0.1)
    total_nans = sum(r["nan_kayip_adim"] for r in oy14["oy1_3_kosular"] + oy14["oy4_kosular"] + oy56["oy6_kosular"])
    ck.eq("oy.fp16_nans", oy["fp16_nans"], int(total_nans))

    s7, o7 = oy["oy7_bounded_rsi_75gens"], oy7["ozet"]
    ck.eq("oy7.total_gens", s7["total_generations"], oy7["meta"]["toplam_nesil"])
    ck.close("oy7.rsi1_lm", s7["blind_rsi1_lm_bpb"], o7["RSI1_KOR_COKUS"]["tur5_lm_bpb"]["ort"], 1e-4)
    ck.close("oy7.rsi1_id", s7["blind_rsi1_id_acc"], o7["RSI1_KOR_COKUS"]["tur5_id"]["ort"], 0.01)
    ck.close("oy7.rsi1_ipt", s7["blind_rsi1_ipt_acc"], o7["RSI1_KOR_COKUS"]["tur5_ipt"]["ort"], 0.01)
    ck.close("oy7.rsi0_lm", s7["control_rsi0_lm_bpb"], o7["RSI0_KONTROL"]["tur5_lm_bpb"]["ort"], 1e-4)
    ck.close("oy7.rsi0_ipt", s7["control_rsi0_ipt_acc"], o7["RSI0_KONTROL"]["tur5_ipt"]["ort"], 0.01)
    ck.close("oy7.rsi3_lm", s7["azr_rsi3_lm_bpb"], o7["RSI3_AZR_MUFREDAT"]["tur5_lm_bpb"]["ort"], 1e-4)
    ck.close("oy7.rsi3_id", s7["azr_rsi3_id_acc"], o7["RSI3_AZR_MUFREDAT"]["tur5_id"]["ort"], 0.01)
    ck.eq("oy7.rsi3_rb", s7["azr_rsi3_rollbacks"], o7["RSI3_AZR_MUFREDAT"]["toplam_rollback"])
    ck.close("oy7.rsi3_ipt", s7["azr_rsi3_ipt_acc"], o7["RSI3_AZR_MUFREDAT"]["tur5_ipt"]["ort"], 0.01)
    ck.close("oy7.rsi4_lm", s7["oracle_rsi4_lm_bpb"], o7["RSI4_ORACLE_USTSINIR"]["tur5_lm_bpb"]["ort"], 1e-4)
    ck.close("oy7.rsi4_ipt", s7["oracle_rsi4_ipt_acc"], o7["RSI4_ORACLE_USTSINIR"]["tur5_ipt"]["ort"], 0.01)
    ck.close("oy7.ood_far", s7["ood_far_l78_acc"], max(v["tur5_ood_uzak"]["ort"] for v in o7.values()), 1e-4)

    s8 = oy["oy8_zero_retention_ring"]
    ck.eq("oy8.vram", s8["vram_mib"], oy8["veri_motoru_envanter"]["gpu_vram_mib"])
    ck.eq("oy8.stream", s8["stream_mib"], oy8["halka_tampon_olcumu"]["islenen_toplam_mib"])
    ck.eq("oy8.peak_disk", s8["peak_disk_mib"], oy8["halka_tampon_olcumu"]["tepe_aktif_disk_mib"])
    ck.eq("oy8.final_disk", s8["final_disk_mib"], oy8["halka_tampon_olcumu"]["son_aktif_disk_mib"])
    ck.eq("oy8.receipt", s8["receipt_lock"], oy8["halka_tampon_olcumu"]["makbuzsuz_silme_reddedildi"])

    s10, r10 = oy["oy10_toolrunner"], base55["oy10_toolrunner"]
    ck.eq("oy10.n_q", s10["pretraining_55m_n_questions"], r10["n_soru"])
    ck.close("oy10.no_tool", s10["pretraining_55m_no_tool_acc"], r10["aracsiz_dogruluk"], 1e-4)
    ck.close("oy10.call_rate", s10["pretraining_55m_call_rate"], r10["toolrunner_cagri_orani"], 1e-4)
    ck.close("oy10.tool_acc", s10["pretraining_55m_tool_acc"], r10["toolrunner_dogruluk"], 1e-4)
    ck.close("oy10.gain", s10["pretraining_55m_gain_pp"], r10["kazanc_puan"], 1e-4)


def _paired_p_df2(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    t_val = statistics.mean(diffs) / (statistics.stdev(diffs) / math.sqrt(len(diffs)))
    return float(1.0 - abs(t_val) / math.sqrt(t_val * t_val + 2.0))


def _paired_p_df4(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    t_val = statistics.mean(diffs) / (statistics.stdev(diffs) / math.sqrt(len(diffs)))
    return float(1.0 - (abs(t_val) * (t_val * t_val + 6.0)) / ((t_val * t_val + 4.0) ** 1.5))


def _verify_v03_study(ck: CellChecker, v03: dict[str, Any]) -> None:
    deney_a = load_json(RAW_DIR / "deney_a_k1_k5.json")
    h5 = load_json(RAW_DIR / "h5_memory_latency_and_serving.json")
    mx = load_json(RAW_DIR / "math_xray_master_synthesis.json")
    p3 = load_json(RAW_DIR / "p3_long_v03.json")
    p2 = load_json(RAW_DIR / "p2_long_v03.json")

    ck.eq(
        "v03.audit_status",
        v03["audit_status"],
        "PASS (Independent Zero-Context Carbon Closure Audit; 0 KRITIK, 0 YUKSEK, 0 ORTA)",
    )

    # K1-K5 ablation (deney_a_k1_k5.json)
    ab = v03["k1_k5_ablation"]
    k3 = ab["k3_std002_init"]
    s_leg = deney_a["summary"]["byte_lm_k3::K3_legacy_std1"]
    s_v03 = deney_a["summary"]["byte_lm_k3::K3_v03_std002"]
    ck.eq("v03.k3.n_seeds", ab["n_seeds"], s_v03["n_seeds"])
    ck.close("v03.k3.leg_bpb", k3["legacy_std1_val_bpb_mean"], round(s_leg["val_bpb_mean"], 4), 1e-4)
    ck.close("v03.k3.leg_std", k3["legacy_std1_val_bpb_std"], round(s_leg["val_bpb_std"], 4), 1e-4)
    ck.close("v03.k3.leg_acc", k3["legacy_std1_val_acc_mean"], round(s_leg["val_acc_mean"], 4), 1e-4)
    ck.close("v03.k3.v03_bpb", k3["v03_std002_val_bpb_mean"], round(s_v03["val_bpb_mean"], 4), 1e-4)
    ck.close("v03.k3.v03_std", k3["v03_std002_val_bpb_std"], round(s_v03["val_bpb_std"], 4), 1e-4)
    ck.close("v03.k3.v03_acc", k3["v03_std002_val_acc_mean"], round(s_v03["val_acc_mean"], 4), 1e-4)
    ck.close("v03.k3.delta_bpb", k3["delta_bpb"], round(s_v03["val_bpb_mean"] - s_leg["val_bpb_mean"], 4), 1e-4)
    k3_p = _paired_p_df2(
        [float(r["val_bpb"]) for r in s_v03["val_bpbs_by_seed"]],
        [float(r["val_bpb"]) for r in s_leg["val_bpbs_by_seed"]],
    )
    ck.close("v03.k3.p_val", k3["paired_p_value"], k3_p, 1e-5)
    ovf_before = sum(int(r["grad_overflows"]) for r in deney_a["runs"] if r["model"] == "K3_legacy_std1")
    ovf_after = sum(int(r["grad_overflows"]) for r in deney_a["runs"] if r["model"] == "K3_v03_std002")
    ck.eq("v03.k3.ovf_before", k3["fp16_overflow_count_before"], ovf_before)
    ck.eq("v03.k3.ovf_after", k3["fp16_overflow_count_after"], ovf_after)

    dim_probe = {(r["d_model"], r["init"]): r for r in deney_a["k3_dim_probe"]}
    ck.close("v03.k3.ce_192_leg", k3["step0_ce_d192_legacy"], round(dim_probe[(192, "legacy_std1")]["step0_ce"], 4), 1e-4)
    ck.close("v03.k3.ce_192_v03", k3["step0_ce_d192_v03"], round(dim_probe[(192, "v03_std002")]["step0_ce"], 4), 1e-4)
    ck.close("v03.k3.ce_768_leg", k3["step0_ce_d768_legacy"], round(dim_probe[(768, "legacy_std1")]["step0_ce"], 4), 1e-4)
    ck.close("v03.k3.ce_768_v03", k3["step0_ce_d768_v03"], round(dim_probe[(768, "v03_std002")]["step0_ce"], 4), 1e-4)

    k5 = ab["k5_snapshot_speculative"]
    rk5 = deney_a["k5_speculative"]
    ck.eq("v03.k5.n_prompts", k5["n_prompts"], rk5["n_prompts"])
    ck.eq("v03.k5.gen_toks", k5["gen_tokens_per_prompt"], rk5["gen_tokens_per_prompt"])
    ck.close("v03.k5.greedy_sec", k5["greedy_sec"], round(rk5["greedy_sec"], 4), 1e-4)
    ck.close("v03.k5.k3_sec", k5["k3_sec"], round(rk5["k3_sec"], 4), 1e-4)
    ck.close("v03.k5.k3_spd", k5["k3_speedup"], round(rk5["k3_speedup"], 4), 1e-4)
    ck.close("v03.k5.k3_acc", k5["k3_acceptance_rate"], round(rk5["k3_acceptance_rate"], 4), 1e-4)
    ck.eq("v03.k5.k3_mis", k5["k3_mismatch_bytes"], rk5["k3_mismatch_bytes"])
    ck.eq("v03.k5.k3_ext", k5["k3_extra_passes"], rk5["k3_extra_passes"])
    ck.eq("v03.k5.k3_rb", k5["k3_rollbacks"], rk5["k3_rollbacks"])
    ck.close("v03.k5.mtp4_sec", k5["mtp4_sec"], round(rk5["mtp4_sec"], 4), 1e-4)
    ck.close("v03.k5.mtp4_spd", k5["mtp4_speedup"], round(rk5["mtp4_speedup"], 4), 1e-4)
    ck.close("v03.k5.mtp4_acc", k5["mtp4_acceptance_rate"], round(rk5["mtp4_acceptance_rate"], 4), 1e-4)
    ck.eq("v03.k5.mtp4_mis", k5["mtp4_mismatch_bytes"], rk5["mtp4_mismatch_bytes"])
    ck.eq("v03.k5.mtp4_ext", k5["mtp4_extra_passes"], rk5["mtp4_extra_passes"])
    ck.eq("v03.k5.mtp4_rb", k5["mtp4_rollbacks"], rk5["mtp4_rollbacks"])

    # H5 memory latency and serving
    sh5 = v03["h5_memory_latency_and_serving"]
    gw = {(r["batch_size"], r["seq_len"]): r for r in h5["gpu_wall"]}
    ck.close("v03.h5.b8_32k_rdfl", sh5["gpu_wall_b8_t32768_pure_rdfl_mib"], round(gw[(8, 32768)]["pure_rdfl"]["peak_vram_mib"], 4), 1e-4)
    ck.close("v03.h5.b8_32k_tx", sh5["gpu_wall_b8_t32768_rope_tx_mib"], round(gw[(8, 32768)]["rope_transformer"]["peak_vram_mib"], 4), 1e-4)
    ck.close("v03.h5.b8_64k_rdfl", sh5["gpu_wall_b8_t65536_pure_rdfl_mib"], round(gw[(8, 65536)]["pure_rdfl"]["peak_vram_mib"], 4), 1e-4)
    ck.eq("v03.h5.b8_64k_rdfl_oom", sh5["gpu_wall_b8_t65536_pure_rdfl_oom"], gw[(8, 65536)]["pure_rdfl"]["oom"])
    ck.eq("v03.h5.b8_64k_tx_oom", sh5["gpu_wall_b8_t65536_rope_tx_oom"], gw[(8, 65536)]["rope_transformer"]["oom"])

    tbl = {(r["scale"], r["tx_variant"], r["text_bytes"]): r for r in h5["text_normalized_serving_table"]}
    r1b = tbl[("1B_class", "GQA8_FP8", 1048576)]
    r55 = tbl[("55M_class", "GQA8_FP8", 1048576)]
    for tag, src_r, dst_r in (("1b", r1b, sh5["iso_1b_gqa8_fp8_1m_bytes"]), ("55m", r55, sh5["iso_55m_gqa8_fp8_1m_bytes"])):
        ck.close(f"v03.h5.{tag}.rdfl", dst_r["rdfl_r2_state_mib"], src_r["rdfl_r2_state_mib"], 1e-4)
        ck.close(f"v03.h5.{tag}.tx", dst_r["tx_kv_cache_mib"], src_r["tx_kv_cache_mib"], 1e-4)
        ck.close(f"v03.h5.{tag}.ratio", dst_r["kv_to_rdfl_state_ratio"], src_r["kv_to_rdfl_state_ratio"], 1e-3)
        ck.close(f"v03.h5.{tag}.cross", dst_r["crossover_text_bytes"], src_r["crossover_text_bytes"], 1e-1)

    # Math X-ray master synthesis
    smx = v03["math_xray_master_synthesis"]
    rmq = mx["verified_solved_xrays"]["mqar_k8"]
    rbl = mx["verified_solved_xrays"]["byte_lm"]
    for k in ("eval_acc_R1", "eval_acc_R2", "eval_acc_R8", "cross_loop_vec_cos", "r0_hl_median_bytes", "r1_hl_median_bytes", "r0_p_beta_gt_1_0", "r1_p_beta_gt_1_0", "r0_max_head_p_beta_gt_1_0", "r1_max_head_p_beta_gt_1_0"):
        ck.close(f"v03.mx.mqar.{k}", smx["mqar_k8"][k], round(rmq[k], 4), 1e-4)
    for k in ("eval_bpb_R1", "eval_bpb_R2", "eval_bpb_R8", "r0_bdh_l0", "r1_bdh_l0", "r0_bdh_byte_selectivity", "r0_p_beta_gt_1_0", "r1_p_beta_gt_1_0", "cross_loop_vec_cos"):
        ck.close(f"v03.mx.blm.{k}", smx["byte_lm"][k], round(rbl[k], 4), 1e-4)
    ck.eq("v03.mx.retracted", smx["retracted_shortcut_xrays"], list(mx["retracted_shortcut_xrays"].keys()))

    # P3_long and P2_long
    sp3 = v03["p3_long_v03_bdh_sparsity_tax"]
    vp3 = p3["result"]["variants"]
    ck.eq("v03.p3.n_seeds", sp3["n_seeds"], len(p3["meta"]["seeds"]))
    ck.eq("v03.p3.steps", sp3["steps"], p3["meta"]["steps"])
    ck.close("v03.p3.relu_mix", sp3["relu_relu_mix_val_bpb_mean"], round(vp3["relu_relu"]["mix_val_bpb_by_step"]["1000"]["mean"], 4), 1e-4)
    ck.close("v03.p3.nobdh_mix", sp3["no_bdh_iso_param_mix_val_bpb_mean"], round(vp3["no_bdh_iso_param"]["mix_val_bpb_by_step"]["1000"]["mean"], 4), 1e-4)
    ck.close("v03.p3.relu_web", sp3["relu_relu_web_val_bpb_mean"], round(vp3["relu_relu"]["web_val_bpb_by_step"]["1000"]["mean"], 4), 1e-4)
    ck.close("v03.p3.nobdh_web", sp3["no_bdh_iso_param_web_val_bpb_mean"], round(vp3["no_bdh_iso_param"]["web_val_bpb_by_step"]["1000"]["mean"], 4), 1e-4)
    p3_relu_mix = [float(s["lm_trajectory"]["1000"]["mix_val_bpb"]) for s in vp3["relu_relu"]["seeds"]]
    p3_nob_mix = [float(s["lm_trajectory"]["1000"]["mix_val_bpb"]) for s in vp3["no_bdh_iso_param"]["seeds"]]
    p3_relu_web = [float(s["lm_trajectory"]["1000"]["web_val_bpb"]) for s in vp3["relu_relu"]["seeds"]]
    p3_nob_web = [float(s["lm_trajectory"]["1000"]["web_val_bpb"]) for s in vp3["no_bdh_iso_param"]["seeds"]]
    ck.close("v03.p3.mix_delta", sp3["mix_val_delta_bpb"], round(statistics.mean(p3_nob_mix) - statistics.mean(p3_relu_mix), 4), 1e-4)
    ck.close("v03.p3.mix_p", sp3["mix_val_p_value"], round(_paired_p_df4(p3_nob_mix, p3_relu_mix), 4), 1e-4)
    ck.close("v03.p3.web_delta", sp3["web_val_delta_bpb"], round(statistics.mean(p3_nob_web) - statistics.mean(p3_relu_web), 4), 1e-4)
    ck.close("v03.p3.web_p", sp3["web_val_p_value"], round(_paired_p_df4(p3_nob_web, p3_relu_web), 4), 1e-4)

    sp2 = v03["p2_long_v03_conv_bias_sink"]
    vp2 = p2["result"]["variants"]
    ck.eq("v03.p2.n_seeds", sp2["n_seeds"], len(p2["meta"]["seeds"]))
    ck.eq("v03.p2.steps", sp2["steps"], p2["meta"]["steps"])
    ck.close("v03.p2.def_mix", sp2["v03_default_mix_val_bpb_mean"], round(vp2["v03_default"]["mix_val_bpb_by_step"]["1000"]["mean"], 4), 1e-4)
    ck.close("v03.p2.nob_mix", sp2["no_conv_bias_mix_val_bpb_mean"], round(vp2["no_conv_bias"]["mix_val_bpb_by_step"]["1000"]["mean"], 4), 1e-4)
    p2_def_mix = [float(s["lm_trajectory"]["1000"]["mix_val_bpb"]) for s in vp2["v03_default"]["seeds"]]
    p2_nob_mix = [float(s["lm_trajectory"]["1000"]["mix_val_bpb"]) for s in vp2["no_conv_bias"]["seeds"]]
    ck.close("v03.p2.mix_delta", sp2["mix_val_delta_bpb"], round(statistics.mean(p2_nob_mix) - statistics.mean(p2_def_mix), 4), 1e-4)
    ck.close("v03.p2.mix_p", sp2["mix_val_p_value"], round(_paired_p_df4(p2_nob_mix, p2_def_mix), 4), 1e-4)
    ck.close("v03.p2.def_web", sp2["v03_default_web_val_bpb_mean"], round(vp2["v03_default"]["web_val_bpb_by_step"]["1000"]["mean"], 4), 1e-4)
    ck.close("v03.p2.nob_web", sp2["no_conv_bias_web_val_bpb_mean"], round(vp2["no_conv_bias"]["web_val_bpb_by_step"]["1000"]["mean"], 4), 1e-4)
    ck.close("v03.p2.def_erank", sp2["v03_default_post_erank"], round(vp2["v03_default"]["post_erank"]["mean"], 4), 1e-4)
    ck.close("v03.p2.nob_erank", sp2["no_conv_bias_post_erank"], round(vp2["no_conv_bias"]["post_erank"]["mean"], 4), 1e-4)


def main() -> int:
    summary = load_json(SUMMARY_PATH)
    studies = summary["studies"]
    ck = CellChecker()

    _verify_arch_study(ck, studies["architecture_v2_1_2m"])
    oy14 = load_json(RAW_DIR / "oy1_4_sonuclari.json")
    oy56 = load_json(RAW_DIR / "oy5_6_sonuclari.json")
    oy7 = load_json(RAW_DIR / "oy7_rsi_sonuclari.json")
    oy8 = load_json(RAW_DIR / "oy8_halka_tampon.json")
    oy_gate = load_json(RAW_DIR / "dogrulama_kapisi_oy.json")
    base55 = load_json(RAW_DIR / "taban_55m_degerlendirme.json")
    oy = studies["self_capabilities_v2_1_2m"]

    _verify_oy1_k3(ck, oy14, oy_gate, oy)
    _verify_oy2(ck, oy14, oy)
    _verify_oy3(ck, oy14, oy)
    _verify_oy4_and_oy5(ck, oy14, oy56, oy)
    _verify_oy6(ck, oy56, oy)
    _verify_oy7_8_10(ck, oy14, oy56, oy7, oy8, base55, oy)
    verify_55m_and_final(
        ck,
        RAW_DIR,
        studies["diogenes_rdfl_55m_gates"],
        studies["diogenes_rdfl_55m_final"],
        oy["oy10_toolrunner"],
    )
    _verify_v03_study(ck, studies["rdfl_v0_3_hardening_and_xray"])

    if ck.errors:
        for err in ck.errors:
            sys.stderr.write(f"[MISMATCH] {err}\n")
        sys.stderr.write(f"FAILED: {len(ck.errors)} mismatches out of {ck.checked} verified cells.\n")
        return 1
    print(f"VERIFIED: {ck.checked}/{ck.checked} audited cells match results/raw/*.json (0 mismatches).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
