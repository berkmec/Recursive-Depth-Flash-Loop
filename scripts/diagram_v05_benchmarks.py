"""Data-driven academic SVG chart builder for RDFL v0.5.0 audited benchmarks."""

from __future__ import annotations

from typing import Any

from scripts.svg_primitives import (
    AMBER_STROKE,
    AMBER_TEXT,
    BLUE_FILL,
    BLUE_STROKE,
    BLUE_TEXT,
    INK,
    MUTED,
    SANS,
    SERIF,
    TEAL_FILL,
    TEAL_STROKE,
    TEAL_TEXT,
    box,
    svg_open,
    txt,
)


def _draw_grid(
    s: list[str], x0: float, y0: float, pw: float, ph: float, ticks: tuple[int, ...] = (0, 25, 50, 75, 100)
) -> None:
    for pct in ticks:
        yy = y0 - ph * (pct / 100.0)
        sw, col = ("1.1", INK) if pct == 0 else ("0.7", "#E5E7EB")
        s.append(f'  <line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="{col}" stroke-width="{sw}"/>\n')
        s.append(txt(x0 - 8, yy + 4, f"{pct}%", 11, MUTED, anchor="end"))
    s.append(f'  <line x1="{x0}" y1="{y0 - ph}" x2="{x0}" y2="{y0}" stroke="{INK}" stroke-width="1.1"/>\n')


def _draw_panel_a_reasoning(s: list[str], sec2: dict[str, Any]) -> None:
    s.append(txt(36, 36, "(a) Non-collapsing step reasoning (K=2..12 ID/OOD, 1K budget)", 15, INK, anchor="start", weight="700", font=SANS))
    acc_map = sec2["accuracy_ID_2_4_6_8_OOD_10_12"]
    sh_map = acc_map["step_halting"]
    adfp_map = acc_map["adfp"]
    sweep_k4_r32 = sec2["halting_gate_extra_loops_sweep"]["K=4"]["R=32"]
    stress_sh = sec2["stress_and_jvp"]["step_halting"]
    stress_ad = sec2["stress_and_jvp"]["adfp"]

    # Legend
    s.append(f'  <rect x="36" y="50" width="11" height="11" rx="2" fill="{BLUE_TEXT}" stroke="#374151" stroke-width="0.6"/>\n')
    s.append(txt(52, 60, "RDFLv05 step_halting (default)", 11, INK, anchor="start"))
    s.append(f'  <rect x="242" y="50" width="11" height="11" rx="2" fill="{AMBER_STROKE}" stroke="#374151" stroke-width="0.6"/>\n')
    s.append(txt(258, 60, "RDFLv05 adfp (contractive)", 11, INK, anchor="start"))

    x0, y0, pw, ph = 66, 310, 560, 224
    _draw_grid(s, x0, y0, pw, ph)
    s.append(f'  <text x="20" y="{y0 - ph / 2:.0f}" font-family="{SERIF}" font-size="12" fill="{INK}" text-anchor="middle" transform="rotate(-90 20 {y0 - ph / 2:.0f})">Accuracy (%)</text>\n')

    ks = ["2", "4", "6", "8", "10", "12"]
    grp_w = pw / len(ks)
    bar_w = 28
    # OOD shaded separator between K=8 and K=10
    ood_x = x0 + 4 * grp_w
    s.append(f'  <rect x="{ood_x:.1f}" y="{y0 - ph:.1f}" width="{2 * grp_w:.1f}" height="{ph:.1f}" fill="#FEF3C7" fill-opacity="0.25"/>\n')
    s.append(f'  <line x1="{ood_x:.1f}" y1="{y0 - ph:.1f}" x2="{ood_x:.1f}" y2="{y0:.1f}" stroke="{AMBER_STROKE}" stroke-width="1.0" stroke-dasharray="4,3"/>\n')
    s.append(txt(ood_x + grp_w, y0 - ph + 14, "OOD Depth", 10, AMBER_TEXT, weight="700"))

    for i, k in enumerate(ks):
        gx = x0 + i * grp_w + 18
        v_sh = float(sh_map[k]) * 100.0
        v_ad = float(adfp_map[k]) * 100.0
        bh_sh = max(2.0, ph * (v_sh / 100.0))
        bh_ad = max(2.0, ph * (v_ad / 100.0))

        s.append(f'  <rect x="{gx:.1f}" y="{y0 - bh_sh:.1f}" width="{bar_w}" height="{bh_sh:.1f}" fill="{BLUE_TEXT}" stroke="#374151" stroke-width="0.6"/>\n')
        s.append(txt(gx + bar_w / 2, y0 - bh_sh - 5, f"{v_sh:.1f}", 10, BLUE_TEXT, weight="700"))

        bx_ad = gx + bar_w + 4
        s.append(f'  <rect x="{bx_ad:.1f}" y="{y0 - bh_ad:.1f}" width="{bar_w}" height="{bh_ad:.1f}" fill="{AMBER_STROKE}" stroke="#374151" stroke-width="0.6"/>\n')
        s.append(txt(bx_ad + bar_w / 2, y0 - bh_ad - 5, f"{v_ad:.1f}", 10, AMBER_TEXT))

        lbl = f"K={k}" + ("*" if int(k) >= 10 else "")
        s.append(txt(gx + bar_w + 2, y0 + 18, lbl, 11, INK, weight="600"))

    jvp_sh = float(stress_sh["mean_step_jvp_top_sv"])
    jvp_ad = float(stress_ad["mean_step_jvp_top_sv"])
    r32_sh = float(sweep_k4_r32["step_halting_full_acc"]) * 100.0
    r32_ad = float(sweep_k4_r32["adfp_acc"]) * 100.0
    g_first = float(sweep_k4_r32["g_first"])
    g_last = float(sweep_k4_r32["g_last"])
    s.append(
        txt(
            346,
            352,
            f"JVP σ₁: {jvp_sh:.2f} vs {jvp_ad:.2f} · R=1024 finite (0 NaN) · R=32 (K=4): {r32_sh:.1f}% vs {r32_ad:.1f}% (g_step {g_first:.3f}→{g_last:.3f})",
            11,
            MUTED,
        )
    )


def _draw_panel_b_gwt(s: list[str], sec3: dict[str, Any]) -> None:
    s.append(txt(710, 36, "(b) Strict OOS J-Space global workspace (K=6, R=8)", 15, INK, anchor="start", weight="700", font=SANS))
    sh = sec3["step_halting"]
    ad = sec3["adfp"]
    ab = sh["signature4_causal_ablation"]
    pt = sh["signature4_concept_patching"]
    br = sh["signature3_global_broadcast"]
    ad_pt = ad["signature4_concept_patching"]

    s.append(
        txt(
            710,
            58,
            f"OOS broadcast (r=4, k=14): J/J⊥ = {float(br['func_broadcast_ratio_J_over_Jperp']):.2f}× (p={float(br['func_broadcast_p_vs_Jperp']):.2e}) · ADFP null: {float(ad['clean_acc_K6_R8'])*100:.1f}%",
            11,
            TEAL_TEXT,
            anchor="start",
        )
    )

    x0, y0, pw, ph = 740, 310, 580, 224
    _draw_grid(s, x0, y0, pw, ph)

    items = [
        ("Clean\nAcc", float(ab["clean_acc"]) * 100.0, BLUE_TEXT),
        ("Ablate\nJ⊥ full", float(ab["ablate_J_perp_full_acc"]) * 100.0, TEAL_STROKE),
        ("Ablate\nRand-k", float(ab["ablate_random_k_mean_acc"]) * 100.0, "#64748B"),
        ("Ablate\nJ-Space", float(ab["ablate_J_space_acc"]) * 100.0, "#DC2626"),
        ("Same\nPatch J", float(pt["patch_J_space"]["flip_to_A_rate"]) * 100.0, BLUE_TEXT),
        ("Same\nPatch J⊥", float(pt["patch_J_perp_full"]["flip_to_A_rate"]) * 100.0, "#94A3B8"),
        ("Comp\nPatch J", float(pt["cross_table_compositional_PB4_v2A_J_space"]) * 100.0, TEAL_STROKE),
        ("Comp\nPatch J⊥", float(pt["cross_table_compositional_PB4_v2A_J_perp_full"]) * 100.0, "#94A3B8"),
    ]
    slot_w = pw / len(items)
    bar_w = 40
    for i, (lbl, val, col) in enumerate(items):
        bx = x0 + i * slot_w + (slot_w - bar_w) / 2
        bh = max(2.0, ph * (val / 100.0))
        by = y0 - bh
        s.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.6"/>\n')
        s.append(txt(bx + bar_w / 2, by - 5, f"{val:.1f}%", 10, col if col != "#94A3B8" else MUTED, weight="700"))
        l1, l2 = lbl.split("\n")
        s.append(txt(bx + bar_w / 2, y0 + 16, l1, 10, INK))
        s.append(txt(bx + bar_w / 2, y0 + 29, l2, 10, INK))

    drop_pp = float(ab["ablate_J_space_drop"]) * 100.0
    ad_comp = float(ad_pt["cross_table_compositional_PB4_v2A_J_space"]) * 100.0
    s.append(
        txt(
            1030,
            352,
            f"J-Space ablation drop: −{drop_pp:.2f} pp · ADFP null compositional: {ad_comp:.2f}% (same-table 0.0%)",
            11,
            MUTED,
        )
    )


def _draw_panel_c_s5(s: list[str], p2: dict[str, Any]) -> None:
    s.append(txt(36, 404, "(c) P2 non-Abelian S5-Perm32 (running=False, OOD L=16,32)", 15, INK, anchor="start", weight="700", font=SANS))
    chance = float(p2["chance_acc"]) * 100.0
    sf0 = float(p2["s_fast_zero_ablation_l2"]) * 100.0
    s.append(
        txt(
            36,
            426,
            f"Unified RDFLv05Model (n_h=4) · chance={chance:.2f}% · S_fast=0 ablation collapses to {sf0:.2f}%",
            11,
            MUTED,
            anchor="start",
        )
    )

    x0, y0, pw, ph = 66, 676, 560, 224
    _draw_grid(s, x0, y0, pw, ph)
    s.append(f'  <text x="20" y="{y0 - ph / 2:.0f}" font-family="{SERIF}" font-size="12" fill="{INK}" text-anchor="middle" transform="rotate(-90 20 {y0 - ph / 2:.0f})">Exact Accuracy (%)</text>\n')

    entries = [
        ("L=2 (ID)", float(p2["val_acc_l2"]) * 100.0, BLUE_TEXT),
        ("L=4 (ID)", float(p2["val_acc_l4"]) * 100.0, BLUE_TEXT),
        ("L=8 (ID)", float(p2["val_acc_l8"]) * 100.0, BLUE_TEXT),
        ("L=16 (OOD)", float(p2["val_acc_l16_ood"]) * 100.0, TEAL_STROKE),
        ("L=32 (OOD)", float(p2["val_acc_l32_ood"]) * 100.0, TEAL_STROKE),
        ("S_fast=0", float(p2["s_fast_zero_ablation_l8"]) * 100.0, "#DC2626"),
    ]
    slot_w = pw / len(entries)
    bar_w = 50
    for i, (lbl, val, col) in enumerate(entries):
        bx = x0 + i * slot_w + (slot_w - bar_w) / 2
        bh = max(2.0, ph * (val / 100.0))
        by = y0 - bh
        s.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.6"/>\n')
        mult = val / chance
        lbl_str = f"{val:.1f}% ({mult:.0f}×)" if val >= 5.0 else f"{val:.2f}%"
        s.append(txt(bx + bar_w / 2, by - 6, lbl_str, 10, col, weight="700"))
        s.append(txt(bx + bar_w / 2, y0 + 18, lbl, 11, INK))

    # Chance line
    cy = y0 - ph * (chance / 100.0)
    s.append(f'  <line x1="{x0}" y1="{cy:.1f}" x2="{x0 + pw}" y2="{cy:.1f}" stroke="#DC2626" stroke-width="1.0" stroke-dasharray="4,3"/>\n')


def _draw_panel_d_continual_and_o1(s: list[str], sec1: dict[str, Any], p4: dict[str, Any]) -> None:
    s.append(txt(710, 404, "(d) P4 Continual learning (0 replay) &amp; O(1) parity contract", 15, INK, anchor="start", weight="700", font=SANS))
    arms = p4["arms"]
    sg = p4["streaming_surprise_gate"]
    chance = float(p4["chance_acc"]) * 100.0

    s.append(
        txt(
            710,
            426,
            f"3-task sequential split (chance={chance:.1f}%) · Post-distractor S_slow cos: Surp {float(sg['s_slow_cos_after_distractor_surprise']):.4f} > Rand {float(sg['s_slow_cos_after_distractor_random_matched']):.4f} > All1 {float(sg['s_slow_cos_after_distractor_always_one']):.4f}",
            10,
            TEAL_TEXT,
            anchor="start",
        )
    )

    x0, y0, pw, ph = 740, 684, 360, 212
    # Y-axis from -50% to +50% to show both ACC (+) and BWT (-) cleanly!
    zy = y0 - ph * 0.5  # 0% line at mid-height
    for pct in (-50, -25, 0, 25, 50):
        yy = zy - (pct / 50.0) * (ph * 0.5)
        sw, col = ("1.1", INK) if pct == 0 else ("0.7", "#E5E7EB")
        s.append(f'  <line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="{col}" stroke-width="{sw}"/>\n')
        s.append(txt(x0 - 8, yy + 4, f"{pct:+d}%" if pct != 0 else "0%", 10, MUTED, anchor="end"))
    s.append(f'  <line x1="{x0}" y1="{y0 - ph}" x2="{x0}" y2="{y0}" stroke="{INK}" stroke-width="1.1"/>\n')

    # Legend for ACC vs BWT
    s.append(txt(740, 448, "Left bar (arm color): ACC (↑)   ·   Right bar (light): BWT forgetting (closer to 0 ↑)", 10, INK, anchor="start"))

    arm_list = [
        ("Frozen", arms["frozen"], "#94A3B8"),
        ("Naive AdamW", arms["naive_adamw"], "#DC2626"),
        ("RDFLv05 M2+OGP", arms["rdfl_v05_cl3_slow_m2"], TEAL_STROKE),
    ]
    grp_w = pw / len(arm_list)
    bar_w = 34
    for i, (nm, d, col) in enumerate(arm_list):
        gx = x0 + i * grp_w + 18
        acc = float(d["ACC"]) * 100.0
        bwt = float(d["BWT"]) * 100.0

        # ACC bar (positive)
        bh_acc = max(2.0, (acc / 50.0) * (ph * 0.5))
        s.append(f'  <rect x="{gx:.1f}" y="{zy - bh_acc:.1f}" width="{bar_w}" height="{bh_acc:.1f}" fill="{col}" stroke="#374151" stroke-width="0.6"/>\n')
        s.append(txt(gx + bar_w / 2, zy - bh_acc - 5, f"{acc:.1f}%", 10, col, weight="700"))

        # BWT bar (negative or 0)
        bx_bwt = gx + bar_w + 6
        if abs(bwt) < 0.05:
            s.append(f'  <rect x="{bx_bwt:.1f}" y="{zy - 1.5:.1f}" width="{bar_w}" height="3" fill="#CBD5E1" stroke="#374151" stroke-width="0.6"/>\n')
            s.append(txt(bx_bwt + bar_w / 2, zy - 6, "0.0%", 10, MUTED))
        else:
            bh_bwt = max(2.0, (abs(bwt) / 50.0) * (ph * 0.5))
            b_col = "#FCA5A5" if bwt < -20.0 else "#CBD5E1"
            s.append(f'  <rect x="{bx_bwt:.1f}" y="{zy:.1f}" width="{bar_w}" height="{bh_bwt:.1f}" fill="{b_col}" stroke="#374151" stroke-width="0.6"/>\n')
            s.append(txt(bx_bwt + bar_w / 2, zy + bh_bwt + 13, f"{bwt:+.1f}%", 10, "#DC2626" if bwt < -20.0 else MUTED, weight="600"))

        s.append(txt(gx + bar_w + 3, y0 + 18, nm, 10, INK, weight="600"))

    # O(1) State & Parity Badge Box on the right side of Panel (d)
    const_bytes = int(sec1["o1_state_bytes_fp32_b1"]["constant_bytes"])
    p_r4_on = float(sec1["parity"]["R4_slowON"]["fp64_max_abs_diff"])
    p_halt = float(sec1["eps_halt_fp64_diff"])
    causality = float(sec1["causality_worst_fp64_diff"])
    params = int(sec1["param_count_total"])
    flops_off = int(sec1["analytic_flops"]["slow_off_core_flops_per_byte"])

    s.append(box(1118, 456, 212, 224, BLUE_FILL, BLUE_STROKE, sw=1.2, rx=8))
    s.append(txt(1224, 480, "v0.5.0 Verified Badge", 13, BLUE_TEXT, weight="700", font=SANS))
    s.append('  <line x1="1132" y1="492" x2="1316" y2="492" stroke="#C7DCF2" stroke-width="1.0"/>\n')
    badge_lines = [
        ("Parameters:", f"{params:,}"),
        ("O(1) State (B=1):", f"{const_bytes:,} B"),
        ("T=16..320, R=1..1024:", "Constant"),
        ("FP64 R4 slowON:", f"{p_r4_on:.2e}"),
        ("FP64 eps_halt>0:", f"{p_halt:.2e}"),
        ("Causality Leak:", f"{causality:.1f}"),
        ("Core FLOP/B:", f"{flops_off:,}"),
    ]
    by = 516
    for k_lbl, v_lbl in badge_lines:
        s.append(txt(1130, by, k_lbl, 10, INK, anchor="start"))
        s.append(txt(1318, by, v_lbl, 10, BLUE_TEXT, anchor="end", weight="700"))
        by += 23

    s.append(box(1118, 688, 212, 24, TEAL_FILL, TEAL_STROKE, sw=0.9, rx=5))
    s.append(txt(1224, 704, "0 Attention · 0 KV · 0 BPE", 10, TEAL_TEXT, weight="700"))


def build_v05_benchmarks_svg(v05_summary: dict[str, Any]) -> str:
    """Build 4-panel audited RDFL v0.5.0 benchmark SVG directly from v05_audited_summary.json."""
    s = [svg_open(1360, 740, "RDFL v0.5.0 Audited Reasoning, J-Space Global Workspace, S5, and Continual Learning")]
    _draw_panel_a_reasoning(s, v05_summary["section2_step_reasoning_and_halting"])
    _draw_panel_b_gwt(s, v05_summary["section3_anthropic_gwt_jlens"])
    _draw_panel_c_s5(s, v05_summary["section4_4problem_regression_lock"]["p2_state_tracking"])
    _draw_panel_d_continual_and_o1(
        s,
        v05_summary["section1_invariants_and_parity"],
        v05_summary["section4_4problem_regression_lock"]["p4_continual_learning"],
    )
    s.append("</svg>\n")
    return "".join(s)
