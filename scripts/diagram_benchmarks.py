"""Data-driven academic benchmark and self-capability SVG chart builders."""

from __future__ import annotations

import math
from typing import Any

from scripts.svg_primitives import (
    AMBER_STROKE,
    AMBER_TEXT,
    BLUE_TEXT,
    INK,
    MUTED,
    SERIF,
    TEAL_STROKE,
    TEAL_TEXT,
    svg_open,
    txt,
)


def _draw_panel_a_bars(s: list[str], arms_data: dict[str, Any]) -> None:
    s.append(txt(36, 36, "(a) Algorithmic state &amp; memory accuracy (1.20M; 3 seeds)", 15, INK, anchor="start", weight="700"))
    arm_specs = (
        ("A0_RLT0", "RLT-0", "#CBD5E1"),
        ("A1_RLT1", "RLT-1", "#93C5FD"),
        ("A2_RLT2", "RLT-2", "#E6A259"),
        ("A4_GDN_DELTA", "GDN 1.2M", "#9CA3AF"),
        ("A4B_GDN_ISOFLOP", "GDN 2.6M", "#4B5563"),
        ("A6_DELTA_BDH_RLT", "RDFL 1.2M", BLUE_TEXT),
    )
    lx = 36
    for _, label, col in arm_specs:
        s.append(f'  <rect x="{lx}" y="52" width="10" height="10" rx="2" fill="{col}" stroke="#64748B" stroke-width="0.6"/>\n')
        s.append(txt(lx + 14, 61, label, 11, INK, anchor="start"))
        lx += 94 if len(label) > 6 else 78

    x0, y0, pw, ph = 64, 386, 580, 286
    for pct in (0, 25, 50, 75, 100):
        yy = y0 - ph * (pct / 100.0)
        sw, col = ("1.1", INK) if pct == 0 else ("0.7", "#E5E7EB")
        s.append(f'  <line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="{col}" stroke-width="{sw}"/>\n')
        s.append(txt(x0 - 10, yy + 4, f"{pct}", 12, MUTED, anchor="end"))
    s.append(f'  <line x1="{x0}" y1="{y0 - ph}" x2="{x0}" y2="{y0}" stroke="{INK}" stroke-width="1.1"/>\n')
    s.append(f'  <text x="22" y="{y0 - ph / 2:.0f}" font-family="{SERIF}" font-size="13" fill="{INK}" text-anchor="middle" transform="rotate(-90 22 {y0 - ph / 2:.0f})">Accuracy (%)</text>\n')

    tasks = (
        ("parity64", "Parity-64"),
        ("s5_perm32", "S5-Perm32"),
        ("mqar_overwrite", "MQAR Overwrite"),
        ("multihop_icl", "Multi-Hop ICL"),
    )
    grp_w, bar_w, gap = pw / len(tasks), 16, 3
    for ti, (tkey, tlabel) in enumerate(tasks):
        gx = x0 + ti * grp_w + 17
        s.append(txt(gx + 3 * (bar_w + gap) - 2, y0 + 22, tlabel, 13, INK))
        for bi, (akey, _, col) in enumerate(arm_specs):
            val = float(arms_data[akey][tkey]["mean"])
            bx = gx + bi * (bar_w + gap)
            bh = max(1.5, ph * (val / 100.0))
            by = y0 - bh
            s.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.6"/>\n')
            if bi == len(arm_specs) - 1:
                lbl = "100" if val >= 99.9 else f"{val:.1f}"
                s.append(txt(bx + bar_w / 2, by - 6, lbl, 11, BLUE_TEXT, weight="700"))

    lm_rlt2 = float(arms_data["A2_RLT2"]["byte_lm_bpb"]["mean"])
    lm_rlt0 = float(arms_data["A0_RLT0"]["byte_lm_bpb"]["mean"])
    lm_rlt1 = float(arms_data["A1_RLT1"]["byte_lm_bpb"]["mean"])
    lm_rdfl = float(arms_data["A6_DELTA_BDH_RLT"]["byte_lm_bpb"]["mean"])
    delta = lm_rdfl - lm_rlt0
    s.append(
        txt(
            354,
            442,
            f"Byte LM (bpb ↓):  RLT-2 {lm_rlt2:.4f} · RLT-0 {lm_rlt0:.4f} · RLT-1 {lm_rlt1:.4f} · RDFL {lm_rdfl:.4f} ({delta:+.4f} vs RLT-0)",
            12,
            MUTED,
        )
    )


def _extract_decode_series(
    gpu_profile: dict[str, Any],
) -> tuple[list[int], list[float], list[float]]:
    dec = gpu_profile["uzun_baglam_decode_32k"]
    kv_rows = [r for r in dec if r["mimari"] == "Transformer_6L_TamKV"]
    gdn_rows = [r for r in dec if r["mimari"] == "GatedDeltaNet_6L_O1"]
    seq_lens = [int(r["baglam_t"]) for r in kv_rows]
    kv_mibs = [float(r["durum_vram_mib"]) for r in kv_rows]
    gdn_mibs = [float(r["durum_vram_mib"]) for r in gdn_rows]
    return seq_lens, kv_mibs, gdn_mibs


def _log_y(val: float, my0: float, mph: float) -> float:
    return my0 - mph * (math.log10(max(1.0, val)) / 3.0)


def _draw_panel_b_memory(
    s: list[str],
    gpu_profile: dict[str, Any],
    mem_summary: dict[str, Any],
) -> None:
    s.append(txt(690, 36, "(b) Decode state memory vs. T (L=6, d=768, H=8)", 15, INK, anchor="start", weight="700"))
    mx0, my0, mpw, mph = 758, 386, 344, 286
    for lv in (1.0, 10.0, 100.0, 1000.0):
        yy = _log_y(lv, my0, mph)
        sw, col = ("1.1", INK) if lv == 1.0 else ("0.7", "#E5E7EB")
        s.append(f'  <line x1="{mx0}" y1="{yy:.1f}" x2="{mx0 + mpw}" y2="{yy:.1f}" stroke="{col}" stroke-width="{sw}"/>\n')
        s.append(txt(mx0 - 10, yy + 4, f"{int(lv)}", 12, MUTED, anchor="end"))
    s.append(f'  <line x1="{mx0}" y1="{my0 - mph}" x2="{mx0}" y2="{my0}" stroke="{INK}" stroke-width="1.1"/>\n')
    s.append(f'  <text x="710" y="{my0 - mph / 2:.0f}" font-family="{SERIF}" font-size="13" fill="{INK}" text-anchor="middle" transform="rotate(-90 710 {my0 - mph / 2:.0f})">State Memory (MiB, log scale)</text>\n')

    seq_lens, kv_mibs, gdn_mibs = _extract_decode_series(gpu_profile)
    xs = [mx0 + mpw * (i / max(1, len(seq_lens) - 1)) for i in range(len(seq_lens))]
    kv_pts = " ".join(f"{x:.1f},{_log_y(v, my0, mph):.1f}" for x, v in zip(xs, kv_mibs, strict=True))
    gdn_pts = " ".join(f"{x:.1f},{_log_y(v, my0, mph):.1f}" for x, v in zip(xs, gdn_mibs, strict=True))
    s.append(f'  <polyline points="{kv_pts}" fill="none" stroke="{AMBER_STROKE}" stroke-width="2.4"/>\n')
    s.append(f'  <polyline points="{gdn_pts}" fill="none" stroke="{TEAL_STROKE}" stroke-width="2.4"/>\n')
    for idx, (x, sl, kv, gd) in enumerate(zip(xs, seq_lens, kv_mibs, gdn_mibs, strict=True)):
        s.append(f'  <circle cx="{x:.1f}" cy="{_log_y(kv, my0, mph):.1f}" r="4.4" fill="{AMBER_STROKE}"/>\n')
        s.append(f'  <circle cx="{x:.1f}" cy="{_log_y(gd, my0, mph):.1f}" r="4.4" fill="{TEAL_STROKE}"/>\n')
        lbl = f"{sl // 1024}K" if sl >= 1024 else str(sl)
        s.append(txt(x, my0 + 22, lbl, 12, INK))
        if idx < len(seq_lens) - 1:
            s.append(txt(x, _log_y(kv, my0, mph) - 10, f"{kv:.0f} MiB", 11, AMBER_TEXT))

    ratio_x = float(mem_summary["decode_32k_reduction_factor_exact"])
    s.append(txt(mx0 + mpw / 2, my0 + 46, "Measured Sequence Length T (bytes; log axis)", 13, MUTED))
    s.append(txt(xs[-1] - 4, _log_y(kv_mibs[-1], my0, mph) - 12, f"Full-KV Transformer (L=6, d=768, H=8): {kv_mibs[-1]:.0f} MiB", 11, AMBER_TEXT, anchor="end", weight="700"))
    s.append(txt(xs[-1] - 4, _log_y(gdn_mibs[-1], my0, mph) - 12, f"RDFL GDN O(1) State: {gdn_mibs[-1]:.2f} MiB ({ratio_x:.1f}x smaller)", 12, TEAL_TEXT, anchor="end", weight="700"))


def build_benchmarks_1_2m_svg(summary: dict[str, Any], gpu_profile: dict[str, Any]) -> str:
    """Build 2-panel 1.20M accuracy and O(1) decode memory figure directly from audited JSONs."""
    arch = summary["studies"]["architecture_v2_1_2m"]
    s = [svg_open(1140, 470, "Iso-Parameter 1.20M Benchmarks and O(1) Decode Memory Scaling")]
    _draw_panel_a_bars(s, arch["all_8_arms_1_20m"])
    _draw_panel_b_memory(s, gpu_profile, arch["memory_scaling"])
    s.append("</svg>\n")
    return "".join(s)


def _draw_cap_panel_a(s: list[str], oy6: dict[str, Any]) -> None:
    s.append(txt(190, 34, "(a) FP32 SlotCore (3-hop graph)", 14, INK, weight="700"))
    ax0, ay0, aw, ah = 54, 340, 276, 250
    for pct in (0, 50, 100):
        yy = ay0 - ah * (pct / 100.0)
        sw, col = ("1.1", INK) if pct == 0 else ("0.7", "#E5E7EB")
        s.append(f'  <line x1="{ax0}" y1="{yy:.1f}" x2="{ax0 + aw}" y2="{yy:.1f}" stroke="{col}" stroke-width="{sw}"/>\n')
        s.append(txt(ax0 - 8, yy + 4, f"{pct}%", 11, MUTED, anchor="end"))
    s.append(f'  <line x1="{ax0}" y1="{ay0 - ah}" x2="{ax0}" y2="{ay0}" stroke="{INK}" stroke-width="1.1"/>\n')

    bars = (
        ("Base", float(oy6["baseline_graf_mean"]), f"±{float(oy6['baseline_graf_std']):.1f}", "#CBD5E1"),
        ("Titans-A", float(oy6["titans_a_graf_mean"]), f"±{float(oy6['titans_a_graf_std']):.1f}", "#6BA3D6"),
        ("FP32 Slot", float(oy6["graf_acc_mean"]), f"±{float(oy6['graf_acc_std']):.2f}", TEAL_STROKE),
    )
    for i, (nm, val, sub, col) in enumerate(bars):
        bx = ax0 + 26 + i * 84
        bh = ah * (val / 100.0)
        s.append(f'  <rect x="{bx}" y="{ay0 - bh:.1f}" width="54" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.7"/>\n')
        s.append(txt(bx + 27, ay0 - bh - 7, f"{val:.1f}%", 12, TEAL_TEXT if i == 2 else INK, weight="700" if i == 2 else "400"))
        s.append(txt(bx + 27, ay0 + 18, nm, 12, INK))
        s.append(txt(bx + 27, ay0 + 34, sub, 11, MUTED))
    std_red = float(oy6["std_reduction_x"])
    s.append(txt(190, 405, f"0 FP16 NaN ({oy6['runs_zero_nan']}/{oy6['total_runs']} runs) · {std_red:.1f}x lower std", 12, MUTED))


def _draw_cap_panel_b(s: list[str], oy7: dict[str, Any]) -> None:
    s.append(txt(570, 34, "(b) Bounded RSI: IPT vs. LM drift", 14, INK, weight="700"))
    bx0, by0, bw, bh_max = 426, 340, 290, 250
    for pct in (0, 50, 100):
        yy = by0 - bh_max * (pct / 100.0)
        sw, col = ("1.1", INK) if pct == 0 else ("0.7", "#E5E7EB")
        s.append(f'  <line x1="{bx0}" y1="{yy:.1f}" x2="{bx0 + bw}" y2="{yy:.1f}" stroke="{col}" stroke-width="{sw}"/>\n')
        s.append(txt(bx0 - 8, yy + 4, f"{pct}%", 11, MUTED, anchor="end"))
    s.append(f'  <line x1="{bx0}" y1="{by0 - bh_max}" x2="{bx0}" y2="{by0}" stroke="{INK}" stroke-width="1.1"/>\n')

    bars = (
        ("Blind RSI1", float(oy7["blind_rsi1_ipt_acc"]), f"LM {float(oy7['blind_rsi1_lm_bpb']):.2f}!", "#DC2626"),
        ("Control", float(oy7["control_rsi0_ipt_acc"]), f"LM {float(oy7['control_rsi0_lm_bpb']):.2f}", "#CBD5E1"),
        ("Oracle", float(oy7["oracle_rsi4_ipt_acc"]), f"LM {float(oy7['oracle_rsi4_lm_bpb']):.2f}", "#6BA3D6"),
        ("AZR RSI3", float(oy7["azr_rsi3_ipt_acc"]), f"LM {float(oy7['azr_rsi3_lm_bpb']):.2f}", BLUE_TEXT),
    )
    for i, (nm, val, sub, col) in enumerate(bars):
        bx = bx0 + 16 + i * 68
        bh = bh_max * (val / 100.0)
        s.append(f'  <rect x="{bx}" y="{by0 - bh:.1f}" width="46" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.7"/>\n')
        s.append(txt(bx + 23, by0 - bh - 7, f"{val:.1f}%", 11, col if i in (0, 3) else INK, weight="700" if i in (0, 3) else "400"))
        s.append(txt(bx + 23, by0 + 18, nm, 11, INK))
        s.append(txt(bx + 23, by0 + 34, sub, 11, "#DC2626" if i == 0 else MUTED))
    s.append(txt(570, 405, f"{oy7['azr_rsi3_rollbacks']} rollbacks · permanent ID anchor + exact 3D KL", 12, MUTED))


def _draw_cap_panel_c(s: list[str], oy4: dict[str, Any], ab_55m: dict[str, Any], step7k: dict[str, Any]) -> None:
    s.append(txt(945, 34, "(c) MTP-4 scale crossover (1.2M &#x2192; 55M)", 14, INK, weight="700"))
    cx0, cy0, cw, ch, zy = 804, 340, 294, 250, 290.0
    s.append(f'  <line x1="{cx0}" y1="{zy}" x2="{cx0 + cw}" y2="{zy}" stroke="{INK}" stroke-width="1.1"/>\n')
    s.append(f'  <line x1="{cx0}" y1="{cy0 - ch}" x2="{cx0}" y2="{cy0}" stroke="{INK}" stroke-width="1.1"/>\n')
    for dv in (0.0, 0.01, 0.02, 0.03):
        yy = zy - (dv / 0.035) * 185.0
        if dv > 0:
            s.append(f'  <line x1="{cx0}" y1="{yy:.1f}" x2="{cx0 + cw}" y2="{yy:.1f}" stroke="#E5E7EB" stroke-width="0.7"/>\n')
        s.append(txt(cx0 - 8, yy + 4, f"+{dv:.2f}" if dv > 0 else "0.00", 11, MUTED, anchor="end"))
    s.append(txt(cx0 - 8, 92, "&#x394;bpb", 11, MUTED, anchor="end"))

    tax_1_2m = float(oy4["byte_lm_tax_bpb"])
    tax_55m_lm = float(ab_55m["delta_lm_512"])
    tax_55m_sft = float(ab_55m["delta_sft_val"])
    bars = (
        ("1.2M LM", tax_1_2m, "#DC2626"),
        ("55M LM", tax_55m_lm, AMBER_STROKE),
        ("55M SFT", tax_55m_sft, TEAL_STROKE),
    )
    for i, (nm, dv, col) in enumerate(bars):
        bx = cx0 + 28 + i * 88
        bh = max(4.0, abs(dv) / 0.035 * 185.0)
        by = zy - bh if dv >= 0 else zy
        s.append(f'  <rect x="{bx}" y="{by:.1f}" width="54" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.7"/>\n')
        lbl_y = by - 7 if dv >= 0 else by + bh + 15
        sign = "+" if dv >= 0 else ""
        s.append(txt(bx + 27, lbl_y, f"{sign}{dv:.4f}", 11, col, weight="700"))
        s.append(txt(bx + 27, cy0 + 18, nm, 12, INK))
    ratio = round(tax_1_2m / max(1e-6, abs(tax_55m_lm)))
    t2_7k = float(step7k["mtp_t2_acc"]) * 100.0
    s.append(txt(945, 405, f"{ratio}x lower t+1 tax at 55M · t+2 = {t2_7k:.1f}% at 7K steps", 12, MUTED))


def build_self_capabilities_svg(summary: dict[str, Any]) -> str:
    """Build 3-panel audited self-capabilities & 55M scale crossover SVG from audited JSON."""
    oy = summary["studies"]["self_capabilities_v2_1_2m"]
    ab_55m = summary["studies"]["diogenes_rdfl_55m_gates"]["hour1_mtp4_ab_1500steps"]
    step7k = summary["in_progress_live_observations"]["preliminary_step_7000"]
    s = [svg_open(1140, 430, "Audited Self-Capabilities V2 and 46x MTP-4 Scale Crossover")]
    _draw_cap_panel_a(s, oy["oy6_fp32_slotcore"])
    _draw_cap_panel_b(s, oy["oy7_bounded_rsi_75gens"])
    _draw_cap_panel_c(s, oy["oy4_mtp4_at_1_2m"], ab_55m, step7k)
    s.append("</svg>\n")
    return "".join(s)
