"""Generate crisp, publication-grade vector SVG diagrams for the RDFL repository."""

from __future__ import annotations

import math
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagrams"

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, monospace"


def _svg_head(w: int, h: int, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#F8FAFC"/>
      <stop offset="100%" stop-color="#F1F5F9"/>
    </linearGradient>
    <linearGradient id="heroGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#EEF2FF"/>
      <stop offset="100%" stop-color="#E0E7FF"/>
    </linearGradient>
    <linearGradient id="darkHeader" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="100%" stop-color="#1E293B"/>
    </linearGradient>
    <filter id="shadow" x="-4%" y="-4%" width="108%" height="112%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0F172A" flood-opacity="0.08"/>
    </filter>
    <marker id="arrowSlate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#475569"/>
    </marker>
    <marker id="arrowIndigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#4F46E5"/>
    </marker>
    <marker id="arrowEmerald" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669"/>
    </marker>
    <marker id="arrowAmber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#D97706"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="{w}" height="{h}" rx="16" fill="url(#bgGrad)" stroke="#CBD5E1" stroke-width="1.5"/>
"""


def _card(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str = "#FFFFFF",
    stroke: str = "#CBD5E1",
    sw: float = 1.5,
    rx: float = 12,
    dash: str = "",
) -> str:
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d_attr} filter="url(#shadow)"/>\n'
    )


def _text(
    x: float,
    y: float,
    txt: str,
    size: int = 13,
    weight: str = "600",
    fill: str = "#0F172A",
    anchor: str = "start",
    mono: bool = False,
) -> str:
    fam = MONO if mono else FONT
    safe = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'  <text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{safe}</text>\n'
    )


def build_architecture_svg() -> str:
    w, h = 1240, 770
    s = [_svg_head(w, h, "Recursive Depth Flash Loop (RDFL) Architecture")]

    # Header bar
    s.append('  <rect x="20" y="18" width="1200" height="68" rx="12" fill="url(#darkHeader)"/>\n')
    s.append(_text(44, 48, "Recursive Depth Flash Loop (RDFL) — Unified Architecture", 22, "700", "#FFFFFF"))
    s.append(
        _text(
            44,
            71,
            "Byte-Level O(1)-Memory Hybrid · L2-Resident Recursive Bottleneck (<=4 MiB) · FP32 SlotCore · K3 & MTP-4 · Verifier-Gated RSI",
            12,
            "400",
            "#CBD5E1",
        )
    )
    s.append('  <rect x="1032" y="34" width="166" height="34" rx="8" fill="#4F46E5"/>\n')
    s.append(_text(1115, 56, "Step-0 Diff = 0.0", 13, "700", "#FFFFFF", anchor="middle", mono=True))

    # LEFT COLUMN: Input + Base Trunk + cmem
    s.append(_card(24, 106, 252, 92, "#FFFFFF", "#94A3B8"))
    s.append('  <rect x="24" y="106" width="252" height="26" rx="10" fill="#1E293B"/>\n')
    s.append(_text(150, 124, "1. BYTE STREAM INPUT", 11, "700", "#FFFFFF", anchor="middle"))
    s.append(_text(150, 152, "Raw uint8 bytes (0..255)", 13, "700", "#0F172A", anchor="middle", mono=True))
    s.append(_text(150, 172, "Tokenizer-free · Text, Code, Telemetry", 11, "400", "#475569", anchor="middle"))
    s.append(_text(150, 189, "EOT = 0x04 · Arbitrary Horizon", 11, "500", "#64748B", anchor="middle", mono=True))

    s.append(_card(24, 224, 252, 146, "#FFFFFF", "#3B82F6", sw=1.8))
    s.append('  <rect x="24" y="224" width="252" height="26" rx="10" fill="#2563EB"/>\n')
    s.append(_text(150, 242, "2. BASE TRUNK (PATH 1 OR STANDALONE)", 10, "700", "#FFFFFF", anchor="middle"))
    s.append(_text(150, 271, "Diogenes-0.2-55M Trunk", 14, "700", "#1E3A8A", anchor="middle"))
    s.append(_text(150, 292, "d=1536 · H=8 · ffn=3072 (unshrunk)", 11, "500", "#334155", anchor="middle", mono=True))
    s.append(_text(150, 312, "Banded HoldCore (M=512, SWA=4)", 11, "400", "#475569", anchor="middle"))
    s.append(_text(150, 333, "Blind beyond 512 B alone -> RDFL", 11, "600", "#2563EB", anchor="middle"))
    s.append(_text(150, 353, "carries O(1) state past 512..32K+", 11, "600", "#2563EB", anchor="middle"))

    s.append(_card(24, 396, 252, 134, "#F0FDFA", "#0D9488", sw=1.6))
    s.append('  <rect x="24" y="396" width="252" height="26" rx="10" fill="#0F766E"/>\n')
    s.append(_text(150, 414, "COMPLEMENTARY EPISODIC MEMORY", 10, "700", "#FFFFFF", anchor="middle"))
    s.append(_text(150, 443, "cmem Content Reader (0.86M)", 13, "700", "#115E59", anchor="middle"))
    s.append(_text(150, 464, "Exact-match n-gram & passage slots", 11, "400", "#134E4A", anchor="middle"))
    s.append(_text(150, 484, "Infinite-horizon verbatim lookup", 11, "400", "#134E4A", anchor="middle"))
    s.append(_text(150, 507, "Interface: (logits, hn) bit-exact", 11, "600", "#0F766E", anchor="middle", mono=True))

    # Vertical arrows in left column
    s.append('  <line x1="150" y1="198" x2="150" y2="221" stroke="#475569" stroke-width="2" marker-end="url(#arrowSlate)"/>\n')
    s.append('  <line x1="276" y1="296" x2="314" y2="296" stroke="#4F46E5" stroke-width="2.4" marker-end="url(#arrowIndigo)"/>\n')
    s.append('  <path d="M 150 370 L 150 393" stroke="#0D9488" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrowEmerald)"/>\n')

    # CENTER HERO CONTAINER: L2-Resident RDFL + FP32 SlotCore
    s.append(_card(318, 106, 566, 424, "url(#heroGrad)", "#4F46E5", sw=2.2, rx=14, dash="8,4"))
    s.append('  <rect x="332" y="116" width="538" height="30" rx="8" fill="#4338CA"/>\n')
    s.append(
        _text(
            601,
            136,
            "3. L2-RESIDENT RECURSIVE CORE (d_dar = 384 · FP16 3.11 MiB <= 4 MiB L2 SRAM)",
            12,
            "700",
            "#FFFFFF",
            anchor="middle",
        )
    )

    # Top bar inside hero: Down-projection + Anchor
    s.append(_card(336, 158, 530, 50, "#FFFFFF", "#818CF8", sw=1.4, rx=10))
    s.append(_text(601, 179, "Down-Projection & Input Anchor (Prevents Unanchored Depth Collapse)", 12, "700", "#312E81", anchor="middle"))
    s.append(_text(601, 198, "z0 = W_down(RMSNorm(z_base))  [1536 -> 384]   |   e_anchor = RMSNorm(z0)", 11, "500", "#4338CA", anchor="middle", mono=True))

    # Left-inside hero: FP32 SlotCore Boundary (OY-6)
    s.append(_card(336, 224, 194, 214, "#FFFBEB", "#D97706", sw=1.8, rx=10))
    s.append('  <rect x="336" y="224" width="194" height="26" rx="8" fill="#D97706"/>\n')
    s.append(_text(433, 242, "OY-6: FP32 SLOTCORE (S=4)", 10, "700", "#FFFFFF", anchor="middle"))
    s.append(_text(433, 270, "Chunk Boundary (B = 64)", 12, "700", "#92400E", anchor="middle"))
    s.append(_text(433, 290, "4 persistent working slots", 11, "400", "#78350F", anchor="middle"))
    s.append(_text(433, 309, "Slot-to-slot self-attention", 11, "400", "#78350F", anchor="middle"))
    s.append(_text(433, 328, "strictly outside FP16 autocast", 11, "600", "#B45309", anchor="middle"))
    s.append('  <rect x="350" y="344" width="166" height="40" rx="7" fill="#FEF3C7" stroke="#F59E0B"/>\n')
    s.append(_text(433, 362, "0 FP16 NaN (36/36 runs)", 11, "700", "#92400E", anchor="middle", mono=True))
    s.append(_text(433, 377, "graf_izli: 96.89% (+28.8pp)", 10, "600", "#B45309", anchor="middle", mono=True))
    s.append(_text(433, 405, "Injects before chunk c;", 10, "500", "#78350F", anchor="middle"))
    s.append(_text(433, 421, "commits at chunk end (causal)", 10, "500", "#78350F", anchor="middle"))

    # Right-inside hero: R=2..4 Recursive Flash Loop Block
    s.append(_card(548, 224, 318, 214, "#FFFFFF", "#4F46E5", sw=2.0, rx=10))
    s.append('  <rect x="548" y="224" width="318" height="26" rx="8" fill="#4F46E5"/>\n')
    s.append(_text(707, 242, "RECURSIVE LOOP BLOCK  (r = 1 .. R, weight-tied)", 10, "700", "#FFFFFF", anchor="middle"))

    s.append('  <rect x="562" y="260" width="290" height="32" rx="6" fill="#EEF2FF" stroke="#A5B4FC"/>\n')
    s.append(_text(707, 280, "h_in = h + alpha_r * e_anchor + e_r", 11, "600", "#312E81", anchor="middle", mono=True))

    s.append('  <rect x="562" y="302" width="290" height="72" rx="7" fill="#ECFEFF" stroke="#0891B2" stroke-width="1.5"/>\n')
    s.append(_text(707, 320, "Chunk-Parallel Gated DeltaNet WY (B = 64)", 12, "700", "#155E75", anchor="middle"))
    s.append(_text(707, 338, "k=4 causal conv · beta = 2*sigmoid(.) in (0, 2)", 10, "500", "#164E63", anchor="middle", mono=True))
    s.append(_text(707, 354, "Slow decay alpha_0 in [0.953, 0.999] · O(1) S_t", 10, "500", "#164E63", anchor="middle", mono=True))
    s.append(_text(707, 368, "Decode State @ 32K = 1.71 MiB (337x < Full KV)", 10, "700", "#0E7490", anchor="middle"))

    s.append('  <rect x="562" y="384" width="290" height="44" rx="7" fill="#F0FDF4" stroke="#16A34A" stroke-width="1.4"/>\n')
    s.append(_text(707, 403, "BDH Sparse Interaction Gate", 12, "700", "#166534", anchor="middle"))
    s.append(_text(707, 420, "h <- h + y_GDN + W_dec(ReLU(W_x x) * ReLU(W_y y))", 10, "500", "#15803D", anchor="middle", mono=True))

    # Arrows between SlotCore and Recursive Block
    s.append('  <line x1="530" y1="305" x2="545" y2="305" stroke="#D97706" stroke-width="2" marker-end="url(#arrowAmber)"/>\n')
    s.append('  <line x1="548" y1="360" x2="533" y2="360" stroke="#D97706" stroke-width="2" stroke-dasharray="3,2" marker-end="url(#arrowAmber)"/>\n')

    # Bottom bar inside hero: Zero-Init Residual Merge
    s.append(_card(336, 454, 530, 62, "#FFFFFF", "#059669", sw=1.8, rx=10))
    s.append(_text(601, 475, "Zero-Init Up-Projection & Residual Fusion (Step-0 Difference = 0.0)", 12, "700", "#065F46", anchor="middle"))
    s.append(_text(601, 494, "W_up initialized to 0.0  =>  hn = outn(z_base + W_up(RMSNorm(z_R)))", 11, "500", "#047857", anchor="middle", mono=True))
    s.append(_text(601, 510, "Preserves 100% of pretrained 55M LM + cmem weights at initialization", 10, "500", "#64748B", anchor="middle"))

    # Arrows inside hero
    s.append('  <line x1="601" y1="208" x2="601" y2="221" stroke="#4F46E5" stroke-width="2" marker-end="url(#arrowIndigo)"/>\n')
    s.append('  <line x1="707" y1="438" x2="707" y2="451" stroke="#059669" stroke-width="2" marker-end="url(#arrowEmerald)"/>\n')

    # RIGHT COLUMN: 4 Audited Self-Capabilities
    cards_r = [
        (106, "#ECFDF5", "#059669", "OY-1 / 1B · K3 DECOUPLED EARLY-EXIT", "R=1 Detached Speculative Draft", "0.000 grad tax on R=2 · 0 byte diff", "1.48x-1.54x FLOP · 1.09x-1.28x wall"),
        (216, "#F5F3FF", "#7C3AED", "OY-4 · MTP-4 MULTI-BYTE LOOKAHEAD", "3 Medusa Bottlenecks (t+2..t+4)", "46x Scale Crossover (1.2M -> 55M):", "+0.032 -> +0.001 bpb · t+2=51.6% live"),
        (326, "#F0F9FF", "#0284C7", "OY-3 · MSP + T* CALIBRATION", "Zero-Param Selective Abstention", "AUROC 0.84-0.90 · ECE-15 ~0.01", "+20.1 to +20.8 pp accuracy @ 40% abstain"),
        (436, "#F0FDFA", "#0D9488", "OY-10 · DETERMINISTIC TOOLRUNNER", "Live <calc> / <check> / <note>", "Sandbox exact arithmetic & FTS5", "EOT-weighted clean termination"),
    ]
    for cy, bg, brd, head, t1, t2, t3 in cards_r:
        s.append(_card(918, cy, 298, 94, bg, brd, sw=1.7, rx=10))
        s.append(f'  <rect x="918" y="{cy}" width="298" height="24" rx="8" fill="{brd}"/>\n')
        s.append(_text(1067, cy + 16, head, 10, "700", "#FFFFFF", anchor="middle"))
        s.append(_text(1067, cy + 44, t1, 12, "700", "#0F172A", anchor="middle"))
        s.append(_text(1067, cy + 63, t2, 11, "500", "#334155", anchor="middle", mono=True))
        s.append(_text(1067, cy + 81, t3, 11, "600", brd, anchor="middle"))
        s.append(f'  <line x1="884" y1="{cy + 47}" x2="915" y2="{cy + 47}" stroke="{brd}" stroke-width="2" marker-end="url(#arrowSlate)"/>\n')

    # BOTTOM BANNER: Bounded Verifier-Gated RSI (OY-7) + Zero-Retention Stream (OY-8)
    s.append(_card(24, 554, 1192, 194, "#FFFFFF", "#334155", sw=1.8, rx=12))
    s.append('  <rect x="24" y="554" width="1192" height="28" rx="10" fill="#1E293B"/>\n')
    s.append(
        _text(
            620,
            573,
            "4. BOUNDED VERIFIER-GATED RSI (OY-7) & ZERO-RETENTION CPU DATA RING BUFFER (OY-8)",
            11,
            "700",
            "#FFFFFF",
            anchor="middle",
        )
    )

    steps_b = [
        (42, 196, "#F8FAFC", "#475569", "OY-8 Data Engine", "100% CPU · 0 MiB VRAM", "SHA-256 Receipt Ring", "0.0 MiB Final Disk"),
        (264, 200, "#EEF2FF", "#4F46E5", "AZR Curriculum", "Adaptive Difficulty l_cur", "Rollout Candidates", "log-p Best-of-4"),
        (490, 208, "#ECFDF5", "#059669", "External Verifier", "Exact Deterministic Check", "Rejects Reward Hacking", "Separates Pos / Neg"),
        (724, 224, "#FFFBEB", "#D97706", "Safe Policy Update", "Permanent Tur-0 ID Anchor", "+ 30% Natural LM Anchor", "Exact 3D Reverse KL"),
        (974, 224, "#FEF2F2", "#DC2626", "Validation Rollback Gate", "Checks LM bpb & ID Acc", "0 Rollbacks in AZR V2", "Blind RSI -> 7.52 bpb!"),
    ]
    for idx, (bx, bw, bg, brd, t0, l1, l2, l3) in enumerate(steps_b):
        s.append(_card(bx, 598, bw, 118, bg, brd, sw=1.6, rx=9))
        s.append(_text(bx + bw / 2, 622, t0, 12, "700", brd, anchor="middle"))
        s.append(_text(bx + bw / 2, 646, l1, 11, "500", "#1E293B", anchor="middle"))
        s.append(_text(bx + bw / 2, 668, l2, 11, "500", "#334155", anchor="middle"))
        s.append(_text(bx + bw / 2, 692, l3, 11, "700", brd, anchor="middle", mono=True))
        if idx < len(steps_b) - 1:
            nx = steps_b[idx + 1][0]
            s.append(
                f'  <line x1="{bx + bw}" y1="657" x2="{nx - 3}" y2="657" '
                f'stroke="#475569" stroke-width="2" marker-end="url(#arrowSlate)"/>\n'
            )

    s.append(
        _text(
            620,
            737,
            "Honest Kernel Note: Current benchmarks run pure-PyTorch chunk-parallel WY + SDPA; fused Triton/FLA kernels accelerate from-scratch 200M+ training.",
            11,
            "500",
            "#64748B",
            anchor="middle",
        )
    )
    s.append("</svg>\n")
    return "".join(s)


def build_benchmarks_1_2m_svg() -> str:
    w, h = 1240, 560
    s = [_svg_head(w, h, "RDFL Audited 1.20M Benchmarks and O(1) Memory Scaling")]

    s.append('  <rect x="20" y="16" width="1200" height="58" rx="12" fill="url(#darkHeader)"/>\n')
    s.append(_text(42, 43, "Audited 1.20M Iso-Parameter Benchmarks (204 GPU Runs, 3 Seeds, 0 NaN)", 20, "700", "#FFFFFF"))
    s.append(
        _text(
            42,
            63,
            "Independent Zero-Context Audit: PASS (411/411 cells verified) · Single Frozen Hash ab9e7db6220038a6",
            12,
            "400",
            "#CBD5E1",
        )
    )

    # LEFT PANEL: Grouped Bar Chart
    s.append(_card(20, 90, 710, 450, "#FFFFFF", "#CBD5E1"))
    s.append(_text(44, 120, "A. Algorithmic State & Memory Accuracy (%) — Same 1.20M Budget (Except A4B: 2.57M)", 14, "700", "#0F172A"))

    # Legend
    arms = [
        ("RLT-0 (1.20M)", "#94A3B8"),
        ("RLT-1 (1.20M)", "#60A5FA"),
        ("RLT-2 (1.20M)", "#F59E0B"),
        ("GDN IsoFLOP (2.57M)", "#A78BFA"),
        ("RDFL (1.20M)", "#4F46E5"),
    ]
    lx = 44
    for name, col in arms:
        s.append(f'  <rect x="{lx}" y="134" width="12" height="12" rx="3" fill="{col}"/>\n')
        s.append(_text(lx + 17, 145, name, 11, "600", "#334155"))
        lx += 132

    # Axes
    x0, y0, pw, ph = 72, 466, 636, 285
    for pct in (0, 25, 50, 75, 100):
        yy = y0 - ph * (pct / 100.0)
        s.append(f'  <line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="#E2E8F0" stroke-width="1"/>\n')
        s.append(_text(x0 - 10, yy + 4, f"{pct}%", 11, "500", "#64748B", anchor="end", mono=True))

    tasks = [
        ("Parity-64", [50.18, 79.78, 50.15, 100.0, 100.0]),
        ("S5-Perm32", [19.45, 99.93, 20.27, 100.0, 100.0]),
        ("MQAR Overwrite", [22.40, 21.96, 20.92, 23.63, 100.0]),
        ("Multi-Hop ICL", [7.13, 7.32, 7.03, 3.97, 22.72]),
    ]
    grp_w = pw / len(tasks)
    bar_w = 22
    gap = 4
    for ti, (tname, vals) in enumerate(tasks):
        gx = x0 + ti * grp_w + 18
        s.append(_text(gx + 2.5 * (bar_w + gap) - 2, y0 + 24, tname, 12, "700", "#1E293B", anchor="middle"))
        for bi, val in enumerate(vals):
            bx = gx + bi * (bar_w + gap)
            bh = max(2.0, ph * (val / 100.0))
            by = y0 - bh
            col = arms[bi][1]
            s.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" rx="4" fill="{col}"/>\n')
            if bi == 4 or val >= 99.0 or (ti == 3 and bi == 4):
                lbl = f"{val:.1f}" if val < 99.9 else "100"
                s.append(_text(bx + bar_w / 2, by - 6, lbl, 10, "700", col, anchor="middle", mono=True))

    s.append(_text(375, 522, "RDFL achieves 100.0% on Parity-64, S5-Perm32 & MQAR Overwrite (4.5x over RLT), and 5.7x over 2.57M GDN on Multi-Hop ICL.", 11, "500", "#475569", anchor="middle"))

    # RIGHT TOP PANEL: Log-Scale Decode Memory T=512..32768
    s.append(_card(748, 90, 472, 250, "#FFFFFF", "#CBD5E1"))
    s.append(_text(770, 118, "B. Decode State Memory vs Sequence Length (Log Scale)", 14, "700", "#0F172A"))

    mx0, my0, mpw, mph = 812, 294, 384, 146
    log_vals = [1.0, 10.0, 100.0, 1000.0]
    for lv in log_vals:
        frac = math.log10(lv) / 3.0
        yy = my0 - mph * frac
        s.append(f'  <line x1="{mx0}" y1="{yy:.1f}" x2="{mx0 + mpw}" y2="{yy:.1f}" stroke="#E2E8F0" stroke-width="1"/>\n')
        s.append(_text(mx0 - 8, yy + 4, f"{int(lv)}", 10, "500", "#64748B", anchor="end", mono=True))
    s.append(_text(764, 224, "MiB", 10, "600", "#64748B", anchor="middle"))

    seq_lens = [512, 2048, 8192, 32768]
    kv_mibs = [9.0, 36.0, 144.0, 576.0]
    gdn_mibs = [1.71, 1.71, 1.71, 1.71]
    xs = [mx0 + mpw * (i / 3.0) for i in range(4)]

    def _ly(v: float) -> float:
        return my0 - mph * (math.log10(max(1.0, v)) / 3.0)

    kv_pts = " ".join(f"{x:.1f},{_ly(v):.1f}" for x, v in zip(xs, kv_mibs, strict=True))
    gdn_pts = " ".join(f"{x:.1f},{_ly(v):.1f}" for x, v in zip(xs, gdn_mibs, strict=True))
    s.append(f'  <polyline points="{kv_pts}" fill="none" stroke="#EF4444" stroke-width="3"/>\n')
    s.append(f'  <polyline points="{gdn_pts}" fill="none" stroke="#059669" stroke-width="3"/>\n')
    for x, sl, kv, gd in zip(xs, seq_lens, kv_mibs, gdn_mibs, strict=True):
        s.append(f'  <circle cx="{x:.1f}" cy="{_ly(kv):.1f}" r="4.5" fill="#EF4444"/>\n')
        s.append(f'  <circle cx="{x:.1f}" cy="{_ly(gd):.1f}" r="4.5" fill="#059669"/>\n')
        lbl = f"{sl//1024}K" if sl >= 1024 else str(sl)
        s.append(_text(x, my0 + 18, lbl, 10, "600", "#475569", anchor="middle", mono=True))

    s.append(_text(xs[-1] - 6, _ly(576.0) - 10, "Transformer 6L KV: 576 MiB", 11, "700", "#DC2626", anchor="end"))
    s.append(_text(xs[-1] - 6, _ly(1.71) - 10, "RDFL / GDN O(1): 1.71 MiB (337x smaller!)", 11, "700", "#059669", anchor="end"))
    s.append(_text(984, 330, "55M Training @ T=8192: RLT-2 OOM (>3 GiB) vs RDFL 1,877 MiB (1.83 GiB)", 11, "600", "#1E293B", anchor="middle"))

    # RIGHT BOTTOM PANEL: Natural Language byte_lm bpb
    s.append(_card(748, 356, 472, 184, "#FFFFFF", "#CBD5E1"))
    s.append(_text(770, 384, "C. Byte-Level Language Modeling (bpb, Lower is Better)", 14, "700", "#0F172A"))
    lm_rows = [
        ("RLT-2 (1.20M)", 2.4207, "#F59E0B"),
        ("RLT-0 (1.20M)", 2.4019, "#94A3B8"),
        ("RLT-1 (1.20M)", 2.3648, "#60A5FA"),
        ("RDFL (1.20M)", 2.2896, "#4F46E5"),
    ]
    for idx, (lbl, bpb, col) in enumerate(lm_rows):
        ry = 404 + idx * 30
        s.append(_text(770, ry + 15, lbl, 11, "700" if idx == 3 else "500", "#0F172A"))
        # Scale bpb [2.20 .. 2.45]
        bw = 90.0 + ((bpb - 2.20) / 0.25) * 190.0
        s.append(f'  <rect x="884" y="{ry + 2}" width="{bw:.1f}" height="18" rx="4" fill="{col}"/>\n')
        extra = "  (-0.1123 vs RLT-0)" if idx == 3 else ""
        s.append(_text(892 + bw, ry + 15, f"{bpb:.4f}{extra}", 11, "700" if idx == 3 else "500", col, mono=True))

    s.append("</svg>\n")
    return "".join(s)


def build_self_capabilities_svg() -> str:
    w, h = 1240, 560
    s = [_svg_head(w, h, "RDFL Audited Self-Capabilities V2 and 46x MTP-4 Scale Crossover")]

    s.append('  <rect x="20" y="16" width="1200" height="58" rx="12" fill="url(#darkHeader)"/>\n')
    s.append(_text(42, 43, "Audited Self-Capabilities V2 & Live 46x Scale Crossover (1.2M -> 55M)", 20, "700", "#FFFFFF"))
    s.append(
        _text(
            42,
            63,
            "V2 Self-Capabilities Audit: PASS (662/662 cells, unseen seeds 42/137/2026, 0 NaN) · Live 55M Gate-1 & Hour-1 A/B Verified",
            12,
            "400",
            "#CBD5E1",
        )
    )

    # PANEL A: OY-6 FP32 SlotCore
    s.append(_card(20, 90, 386, 450, "#FFFFFF", "#D97706", sw=1.6))
    s.append('  <rect x="20" y="90" width="386" height="30" rx="10" fill="#D97706"/>\n')
    s.append(_text(213, 110, "A. OY-6 · FP32 SlotCore Stability (36/36 = 0 NaN)", 12, "700", "#FFFFFF", anchor="middle"))
    s.append(_text(213, 142, "3-Hop Graph Reasoning Accuracy (graf_izli)", 12, "600", "#334155", anchor="middle"))

    p1_bars = [
        ("RDFL Base", 68.09, "±54.7", "#94A3B8"),
        ("Titans-A", 81.46, "2x State", "#0EA5E9"),
        ("FP32 SlotCore", 96.89, "±1.47", "#D97706"),
    ]
    bx0, by0, bph = 64, 372, 195
    for idx, (nm, val, sub, col) in enumerate(p1_bars):
        bx = bx0 + idx * 112
        bh = bph * (val / 100.0)
        s.append(f'  <rect x="{bx}" y="{by0 - bh:.1f}" width="76" height="{bh:.1f}" rx="6" fill="{col}"/>\n')
        s.append(_text(bx + 38, by0 - bh - 8, f"{val:.2f}%", 12, "700", col, anchor="middle", mono=True))
        s.append(_text(bx + 38, by0 + 18, nm, 11, "700", "#0F172A", anchor="middle"))
        s.append(_text(bx + 38, by0 + 34, sub, 10, "500", "#64748B", anchor="middle", mono=True))

    s.append('  <rect x="40" y="428" width="346" height="92" rx="8" fill="#FFFBEB" stroke="#F59E0B"/>\n')
    s.append(_text(213, 452, "Key Audited Gains:", 12, "700", "#92400E", anchor="middle"))
    s.append(_text(213, 474, "• +28.80 pp mean · 37.2x lower std (1,386x lower var)", 11, "600", "#78350F", anchor="middle"))
    s.append(_text(213, 494, "• MQAR Overwrite #1: 99.88% (0.0068 bpb)", 11, "600", "#78350F", anchor="middle"))
    s.append(_text(213, 512, "• At 55M: base ffn=3072 kept unshrunk (zero LM tax)", 10, "500", "#B45309", anchor="middle"))

    # PANEL B: OY-7 Blind Collapse vs Verifier-Gated AZR
    s.append(_card(426, 90, 388, 450, "#FFFFFF", "#059669", sw=1.6))
    s.append('  <rect x="426" y="90" width="388" height="30" rx="10" fill="#059669"/>\n')
    s.append(_text(620, 110, "B. OY-7 · Blind RSI Collapse vs Verifier-Gated AZR", 12, "700", "#FFFFFF", anchor="middle"))

    s.append(_text(620, 142, "Natural LM Preservation (bpb, lower = healthier)", 12, "600", "#334155", anchor="middle"))
    lm_rsi = [
        ("RSI1 Blind (No Verifier)", 7.52, "#DC2626"),
        ("RSI0 Control", 2.97, "#64748B"),
        ("RSI4 Oracle", 2.95, "#0284C7"),
        ("RSI3 AZR Verifier", 2.93, "#059669"),
    ]
    for i, (nm, val, col) in enumerate(lm_rsi):
        ry = 158 + i * 32
        s.append(_text(444, ry + 15, nm, 11, "700" if i in (0, 3) else "500", "#0F172A"))
        bw = (val / 8.0) * 162.0
        s.append(f'  <rect x="598" y="{ry + 2}" width="{bw:.1f}" height="18" rx="4" fill="{col}"/>\n')
        s.append(_text(605 + bw, ry + 15, f"{val:.2f}", 11, "700", col, mono=True))

    s.append(_text(620, 306, "Invariant Algebraic Generalization (IPT Accuracy %)", 12, "600", "#334155", anchor="middle"))
    ipt_rsi = [
        ("RSI0 Control", 66.59, "#64748B"),
        ("RSI4 Oracle", 71.26, "#0284C7"),
        ("RSI3 AZR (0 rollbacks)", 80.45, "#059669"),
    ]
    for i, (nm, val, col) in enumerate(ipt_rsi):
        ry = 322 + i * 34
        s.append(_text(444, ry + 16, nm, 11, "700" if i == 2 else "500", "#0F172A"))
        bw = (val / 100.0) * 165.0
        s.append(f'  <rect x="598" y="{ry + 3}" width="{bw:.1f}" height="18" rx="4" fill="{col}"/>\n')
        s.append(_text(605 + bw, ry + 16, f"{val:.2f}%", 11, "700", col, mono=True))

    s.append('  <rect x="444" y="434" width="352" height="86" rx="8" fill="#ECFDF5" stroke="#10B981"/>\n')
    s.append(_text(620, 456, "5 Audited Safeguards (75 Live Gens, 0 NaN):", 11, "700", "#065F46", anchor="middle"))
    s.append(_text(620, 476, "• External verifier + 30% LM + Exact 3D Reverse KL", 10, "500", "#047857", anchor="middle"))
    s.append(_text(620, 494, "• Permanent Tur-0 ID anchor (rollbacks 6 -> 0)", 10, "500", "#047857", anchor="middle"))
    s.append(_text(620, 512, "• Honest limit @ 1.2M: OOD-Far L=7,8 = 0% -> testing 55M", 10, "600", "#B45309", anchor="middle"))

    # PANEL C: OY-4 MTP-4 46x Scale Crossover (1.2M -> 55M)
    s.append(_card(834, 90, 386, 450, "#FFFFFF", "#7C3AED", sw=1.6))
    s.append('  <rect x="834" y="90" width="386" height="30" rx="10" fill="#7C3AED"/>\n')
    s.append(_text(1027, 110, "C. OY-4 · MTP-4 46x Scale Crossover (1.2M -> 55M)", 12, "700", "#FFFFFF", anchor="middle"))

    s.append(_text(1027, 142, "t+1 Capacity Tax vs Matched Control (bpb)", 12, "600", "#334155", anchor="middle"))
    # Visual bar comparison of +0.0323 vs +0.0010 vs -0.0005
    s.append(_text(856, 178, "1.20M (LM)", 11, "700", "#DC2626"))
    s.append('  <rect x="970" y="164" width="180" height="20" rx="4" fill="#EF4444"/>\n')
    s.append(_text(1156, 178, "+0.0323", 11, "700", "#DC2626", mono=True))

    s.append(_text(856, 214, "55M A/B (LM)", 11, "700", "#7C3AED"))
    s.append('  <rect x="970" y="200" width="8" height="20" rx="3" fill="#7C3AED"/>\n')
    s.append(_text(986, 214, "+0.0010 (32x smaller!)", 11, "700", "#7C3AED", mono=True))

    s.append(_text(856, 250, "55M A/B (SFT)", 11, "700", "#059669"))
    s.append('  <rect x="962" y="236" width="8" height="20" rx="3" fill="#059669"/>\n')
    s.append(_text(986, 250, "-0.0005 (Beats Control!)", 11, "700", "#059669", mono=True))

    s.append('  <rect x="854" y="276" width="346" height="136" rx="8" fill="#F5F3FF" stroke="#8B5CF6"/>\n')
    s.append(_text(1027, 298, "Live Diogenes-RDFL-55M Trajectory (Step 0 -> 7K)", 11, "700", "#5B21B6", anchor="middle"))
    s.append(_text(1027, 322, "MTP t+2 Acc:  10.1% -> 34.1% (1.5K) -> 51.6% (7K)", 11, "600", "#4C1D95", anchor="middle", mono=True))
    s.append(_text(1027, 344, "MTP t+3 / t+4: 9.2/10.7% -> 39.6% / 32.4% (7K)", 11, "500", "#5B21B6", anchor="middle", mono=True))
    s.append(_text(1027, 366, "Long-Context T=2048: 1.0419 -> 1.0277 (-0.0142)", 11, "700", "#059669", anchor="middle", mono=True))
    s.append(_text(1027, 388, "Peak VRAM: 3.36 GiB (RTX 2060S 8 GB) · 0 NaN", 11, "600", "#475569", anchor="middle", mono=True))

    s.append('  <rect x="854" y="428" width="346" height="92" rx="8" fill="#F0F9FF" stroke="#0284C7"/>\n')
    s.append(_text(1027, 450, "OY-3 Calibrated Uncertainty (Zero Extra Params):", 11, "700", "#0369A1", anchor="middle"))
    s.append(_text(1027, 472, "• 55M Mix Val: AUROC 0.8994 · +20.79 pp @ 40% abstain", 10, "600", "#0C4A6E", anchor="middle"))
    s.append(_text(1027, 492, "• 55M SFT Val: AUROC 0.9041 · 74.7% -> 95.4% acc", 10, "600", "#0C4A6E", anchor="middle"))
    s.append(_text(1027, 510, "• OY-5 Prompt Self-Correction: REJECTED -> log-p 0.9513", 10, "500", "#64748B", anchor="middle"))

    s.append("</svg>\n")
    return "".join(s)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "rdfl_architecture.svg": build_architecture_svg(),
        "benchmarks_1_2m.svg": build_benchmarks_1_2m_svg(),
        "self_capabilities_v2.svg": build_self_capabilities_svg(),
    }
    for name, content in files.items():
        path = OUT_DIR / name
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
