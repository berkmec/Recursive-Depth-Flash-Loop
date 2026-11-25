"""Build the academic TikZ/LaTeX-style RDFL v0.5.0 pure-recurrent architecture SVG."""

from __future__ import annotations

from scripts.svg_primitives import (
    AMBER_CONTAINER,
    AMBER_FILL,
    AMBER_STROKE,
    AMBER_TEXT,
    BLUE_CONTAINER,
    BLUE_FILL,
    BLUE_STROKE,
    BLUE_TEXT,
    INK,
    MUTED,
    SANS,
    TEAL_FILL,
    TEAL_STROKE,
    TEAL_TEXT,
    WIRE,
    box,
    math_var,
    plus_node,
    svg_open,
    txt,
)


def _draw_header(s: list[str]) -> None:
    s.append(
        txt(
            770,
            34,
            "RDFL v0.5.0 — Pure-Recurrent Unified Architecture (RDFLv05Model)",
            22,
            INK,
            weight="700",
            font=SANS,
        )
    )
    s.append(box(130, 48, 1280, 30, TEAL_FILL, TEAL_STROKE, sw=1.0, rx=15))
    s.append(
        txt(
            770,
            68,
            "0 Attention · 0 KV-Cache · 0 SlotCore · 0 External BPE · Strict O(1) State (25,136 B @ 565K ref) · FP64 Parity ≤ 8.10e-15 · Causality Leak = 0.0",
            12,
            TEAL_TEXT,
            weight="700",
            font=SANS,
        )
    )


def _draw_stage0_micro(s: list[str]) -> None:
    # Column 1: x = 24..344
    cx = 184
    s.append(txt(cx, 110, "Stage 0 · Byte Stem &amp; Micro-DeltaProduct", 15, BLUE_TEXT, weight="700", font=SANS))
    s.append(txt(cx, 130, f"Byte-level P=1 · {math_var('d', 'byte')}=64 · {math_var('n', 'h')}=4 Householder", 12, MUTED))

    # Raw byte input at bottom
    s.append(txt(cx, 1022, f"Raw byte stream {math_var('x', '1:t')} &#x2208; &#x7B;0, &#x2026;, 255&#x7D;", 13, INK, weight="600"))
    s.append(f'  <line x1="{cx}" y1="1002" x2="{cx}" y2="972" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')

    # Byte embedding + MS-Ngram
    s.append(box(38, 882, 292, 86, BLUE_FILL, BLUE_STROKE))
    s.append(txt(cx, 906, "Byte Embedding + MSNgramHashStem", 13, INK, weight="600"))
    s.append(txt(cx, 927, "2/3/4/6-gram Count-Sketch &#xB1;1 + digit gate", 12, MUTED))
    s.append(txt(cx, 949, f"{math_var('h', 'b0')} = {math_var('E', 'byte')}({math_var('x', 't')}) + &#x3B1;<tspan dy='3' font-size='70%'>ng</tspan><tspan dy='-3'> </tspan>&#x3B4;<tspan dy='3' font-size='70%'>ng</tspan><tspan dy='-3'> &#x2208; &#x211D;</tspan><tspan dy='-5' font-size='70%'>64</tspan>", 12, BLUE_TEXT))

    s.append(f'  <line x1="{cx}" y1="882" x2="{cx}" y2="838" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    # Residual skip around micro_gdn
    s.append(f'  <path d="M {cx} 860 L 42 860 L 42 544 L {cx - 14} 544" fill="none" stroke="{WIRE}" stroke-width="1.25" marker-end="url(#arrGray)"/>\n')

    # MicroDeltaProduct container
    s.append(box(58, 578, 272, 256, BLUE_CONTAINER, "#B6D3EE", sw=1.15, rx=8))
    s.append(txt(194, 601, "MicroDeltaProductGDNLayer (Mod-1 WY)", 13, BLUE_TEXT, weight="700"))
    s.append(txt(194, 620, f"RMSNorm &#x2192; BiasFreeCausalShortConv ({math_var('k')}=4)", 12, MUTED))

    # Dual regime boxes inside Stage 0
    s.append(box(72, 634, 244, 58, BLUE_FILL, BLUE_STROKE, sw=1.1))
    s.append(txt(194, 656, f"Associative Heads ({math_var('n', 'assoc')}=2)", 12, INK, weight="600"))
    s.append(txt(194, 676, "Recall &amp; sparse overwrite · SoftRMSNorm", 11, MUTED))

    s.append(box(72, 702, 244, 72, BLUE_FILL, BLUE_STROKE, sw=1.1))
    s.append(txt(194, 724, f"Householder Heads ({math_var('n', 'hh')}=2, {math_var('n', 'h')}=4)", 12, INK, weight="600"))
    s.append(txt(194, 743, f"&#x3A0;<tspan dy='3' font-size='70%'>j=1..4</tspan><tspan dy='-3'> (</tspan>{math_var('I')} &#x2212; &#x3B2;<tspan dy='3' font-size='70%'>j</tspan><tspan dy='-3'> </tspan>{math_var('k', 'j')}{math_var('k', 'j', '&#x22A4;')}) · S5 perm", 11, INK))
    s.append(txt(194, 762, "Adaptive spacebar-reset gate (conv_gate_hh)", 11, BLUE_TEXT))

    s.append(txt(194, 804, f"State {math_var('S', 'fast')} &#x2208; &#x211D;<tspan dy='-4' font-size='70%'>4&#xD7;16&#xD7;16</tspan><tspan dy='4'> + conv cache (4,864 B)</tspan>", 11, TEAL_TEXT, weight="600"))

    s.append(f'  <line x1="{cx}" y1="578" x2="{cx}" y2="558" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(plus_node(cx, 544))
    s.append(f'  <line x1="{cx}" y1="532" x2="{cx}" y2="492" stroke="{BLUE_STROKE}" stroke-width="1.45" marker-end="url(#arrBlue)"/>\n')

    # c_t node
    s.append(box(58, 434, 252, 54, BLUE_FILL, BLUE_STROKE))
    s.append(txt(cx, 456, f"Micro Byte Stream {math_var('c', 't')}", 13, INK, weight="600"))
    s.append(txt(cx, 476, f"{math_var('c', 't')} = {math_var('h', 'b0')} + {math_var('y', 'g')} &#x2208; &#x211D;<tspan dy='-4' font-size='70%'>64</tspan>", 12, BLUE_TEXT))

    # Arrow from c_t to Stage 1 Patch Packer & upward to Stage 4 Decoder
    s.append(f'  <path d="M 310 461 L 342 461 L 342 925 L 386 925" fill="none" stroke="{BLUE_STROKE}" stroke-width="1.5" marker-end="url(#arrBlue)"/>\n')
    s.append(box(312, 496, 60, 20, "#FFFFFF", BLUE_STROKE, sw=0.8, rx=4))
    s.append(txt(342, 510, "4 B / patch", 10, BLUE_TEXT, weight="600"))

    # Skip c_t directly to Stage 4 Byte Decoder at top
    s.append(f'  <path d="M {cx} 434 L {cx} 178 L 844 178" fill="none" stroke="{BLUE_STROKE}" stroke-width="1.45" stroke-dasharray="5,3" marker-end="url(#arrBlue)"/>\n')
    s.append(txt(510, 168, f"Byte stream skip {math_var('c', 't')} &#x2192; single-pass decoder (residual byte resolution)", 12, BLUE_TEXT))


def _draw_stage1_macro(s: list[str]) -> None:
    # Column 2: x = 374..814
    cx = 594
    s.append(txt(cx, 110, "Stage 1 · P=4 Macro Patch &amp; Non-Collapsing StepReasoner", 15, AMBER_TEXT, weight="700", font=SANS))
    s.append(txt(cx, 130, f"Macro {math_var('d', 'macro')}=64 · R weight-tied steps (R &#x2208; 1..1024)", 12, MUTED))

    # Patch Packer + rlt2_s1 boundary
    s.append(box(390, 882, 408, 86, AMBER_FILL, AMBER_STROKE))
    s.append(txt(cx, 905, "P=4 Causal Patch Packer + rlt2_s1 Boundary", 13, INK, weight="600"))
    s.append(txt(cx, 926, f"RMSNorm({math_var('c', '4m:4m+3')}) &#x2192; 256 &#x2192; 48 &#x2192; 64 (4&#xD7; fewer macro steps)", 11, MUTED))
    s.append(txt(cx, 948, f"Fp32RLT2Boundary (single damped slot {math_var('s', 'slot')} &#x2208; &#x211D;<tspan dy='-4' font-size='70%'>1&#xD7;64</tspan><tspan dy='4'>, 0 SlotCore)</tspan>", 11, AMBER_TEXT))

    s.append(f'  <line x1="{cx}" y1="882" x2="{cx}" y2="844" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')

    # Macro core container
    s.append(box(378, 366, 432, 474, AMBER_CONTAINER, "#EDC28E", sw=1.2, rx=10))
    s.append(txt(394, 390, "MacroRDFLv05Core (weight-tied)", 13, AMBER_TEXT, anchor="start", weight="700"))

    # r = 0 write_and_read
    s.append(box(396, 738, 396, 86, AMBER_FILL, AMBER_STROKE))
    s.append(txt(cx, 760, f"Step {math_var('r')}=0 · Single Memory Write + FFN &#x2192; {math_var('h', '1')}", 13, INK, weight="600"))
    s.append(txt(cx, 780, "GDNReadOnlyLayer.write_and_read (K<tspan dy='3' font-size='70%'>mem</tspan><tspan dy='-3'>=1, WY solver) + BDH + SwiGLU</tspan>", 11, MUTED))
    s.append(txt(cx, 801, f"rlt2_s1 commits ONLY from {math_var('h', '1')} &#x21D2; {math_var('S', 'macro')}(R=1) == {math_var('S', 'macro')}(R=16)", 11, TEAL_TEXT, weight="600"))

    s.append(f'  <line x1="{cx}" y1="738" x2="{cx}" y2="704" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')

    # r >= 1 step_halting & adfp
    s.append(box(396, 542, 396, 158, AMBER_FILL, AMBER_STROKE))
    s.append(txt(cx, 565, f"Steps {math_var('r')} &#x2265; 1 · Zero-Solver WY Read-Only + StepReasoner", 13, INK, weight="600"))
    s.append(txt(cx, 586, "GDNReadOnlyLayer.read_only() · phase_proj(r) + BDH + BoundedSwiGLU", 11, MUTED))
    s.append(box(410, 598, 368, 52, "#FFFFFF", AMBER_STROKE, sw=0.9, rx=4))
    s.append(txt(594, 617, "Default step_halting: h<tspan dy='3' font-size='70%'>r+1</tspan><tspan dy='-3'> = h</tspan><tspan dy='3' font-size='70%'>r</tspan><tspan dy='-3'> + g</tspan><tspan dy='3' font-size='70%'>step</tspan><tspan dy='-3'>&#xB7;(norm(h</tspan><tspan dy='3' font-size='70%'>r</tspan><tspan dy='-3'>+&#x3B4;</tspan><tspan dy='3' font-size='70%'>r</tspan><tspan dy='-3'>)&#x2212;h</tspan><tspan dy='3' font-size='70%'>r</tspan><tspan dy='-3'>), g</tspan><tspan dy='3' font-size='70%'>step</tspan><tspan dy='-3'>&#x2208;(0,1)</tspan>", 11, INK, weight="600"))
    s.append(txt(594, 636, "g<tspan dy='3' font-size='70%'>step</tspan><tspan dy='-3'>=0 exact 0.0 no-op · R=1024 finite · reason_from_state adds W</tspan><tspan dy='3' font-size='70%'>r</tspan><tspan dy='-3'> O(1)</tspan>", 10, TEAL_TEXT))
    s.append(txt(cx, 676, "Optional adfp (v0.4.1): contractive anchor-gated loop (1&#x2212;&#x3BB;)h + &#x3BB;f(h<tspan dy='3' font-size='70%'>0</tspan><tspan dy='-3'>,h)</tspan>", 11, MUTED))

    # Loop-back arrow for r < R
    s.append(f'  <path d="M 792 604 L 804 604 L 804 676 L 793 676" fill="none" stroke="{AMBER_STROKE}" stroke-width="1.5" stroke-dasharray="4,2" marker-end="url(#arrAmber)"/>\n')

    # Global Workspace J-Space Callout Box inside Stage 1
    s.append(box(396, 404, 396, 118, TEAL_FILL, TEAL_STROKE, sw=1.15, rx=6, dash="4,3"))
    s.append(txt(cx, 425, "Emergent Global Workspace (J-Space Jacobian Lens · OOS K=6, R=8)", 12, TEAL_TEXT, weight="700"))
    s.append(txt(cx, 446, "Shared low-rank subspace across silent steps (r<tspan dy='3' font-size='70%'>mid</tspan><tspan dy='-3'>=4, k=14) · 5/5 monotone probes</tspan>", 11, INK))
    s.append(txt(cx, 466, "Causal ablation: J-space &#x2212;80.21 pp (98.96% &#x2192; 18.75%) vs J&#x22A5; full 96.88%", 11, INK))
    s.append(txt(cx, 486, "Functional broadcast: 2.63&#xD7; vs J&#x22A5; (p=5.27e-15) · Patching: 100.0% same-table", 11, TEAL_TEXT))
    s.append(txt(cx, 506, "Compositional P<tspan dy='-4' font-size='70%'>4</tspan><tspan dy='7' font-size='70%'>B</tspan><tspan dy='-3'>(v</tspan><tspan dy='3' font-size='70%'>2</tspan><tspan dy='-3'>(A)): J 92.71% vs J&#x22A5; 10.42% (ADFP null: 0.0% / 5.21%)</tspan>", 11, TEAL_TEXT, weight="600"))

    s.append(f'  <line x1="{cx}" y1="542" x2="{cx}" y2="526" stroke="{TEAL_STROKE}" stroke-width="1.2" marker-end="url(#arrTeal)"/>\n')

    # Output h_fast arrow out of Stage 1 -> Stage 2-3 & Stage 4
    s.append(box(454, 276, 280, 56, AMBER_FILL, AMBER_STROKE))
    s.append(txt(cx, 299, f"Macro Fast State {math_var('h', 'fast')} &#x2208; &#x211D;<tspan dy='-4' font-size='70%'>64</tspan>", 13, INK, weight="600"))
    s.append(txt(cx, 319, f"(plus detached {math_var('h', 'r1')} for K3 early-exit)", 11, MUTED))

    s.append(f'  <line x1="{cx}" y1="366" x2="{cx}" y2="336" stroke="{AMBER_STROKE}" stroke-width="1.5" marker-end="url(#arrAmber)"/>\n')
    # h_fast -> Stage 4 Byte Decoder (up-right into y=206)
    s.append(f'  <path d="M 734 298 L 818 298 L 818 206 L 844 206" fill="none" stroke="{AMBER_STROKE}" stroke-width="1.5" marker-end="url(#arrAmber)"/>\n')
    # h_fast -> Stage 2/3 Slow Memory (down-right into Stage 3 SlowGDN)
    s.append(f'  <path d="M 734 316 L 824 316 L 824 720 L 844 720" fill="none" stroke="{TEAL_STROKE}" stroke-width="1.45" marker-end="url(#arrTeal)"/>\n')


def _draw_stage234_right(s: list[str]) -> None:
    # Column 3: x = 848..1198
    cx = 1023
    s.append(txt(cx, 110, "Stage 4 · Single-Pass Decoder &amp; Self-Capabilities", 15, BLUE_TEXT, weight="700", font=SANS))
    s.append(txt(cx, 130, "1&#xD7; dec_ffn · Untied Head (W<tspan dy='3' font-size='70%'>out</tspan><tspan dy='-3'> &#x2260; W</tspan><tspan dy='3' font-size='70%'>emb</tspan><tspan dy='-3'>)</tspan>", 12, MUTED))

    # Fast Byte Decoder
    s.append(box(848, 152, 350, 90, BLUE_FILL, BLUE_STROKE))
    s.append(txt(cx, 174, "Single-Pass Causal Byte Decoder", 13, INK, weight="600"))
    s.append(txt(cx, 194, f"{math_var('y', 'base')} = {math_var('c', 't')} + ctx_proj(RMSNorm({math_var('h', 'fast')})) + slot_emb[t mod 4]", 11, INK))
    s.append(txt(cx, 214, f"{math_var('h', 'fast_byte')} = {math_var('y', 'base')} + dec_ffn(RMSNorm({math_var('y', 'base')}))", 11, BLUE_TEXT))
    s.append(txt(cx, 232, "dec_ffn_calls == 1 (slowOFF and slowON)", 10, MUTED))

    # Fast logits & K3/MTP/OY-3
    s.append(f'  <line x1="{cx}" y1="242" x2="{cx}" y2="274" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(plus_node(cx, 288))
    s.append(txt(cx + 18, 293, "ctx_proj_slow(&#x3B4;<tspan dy='3' font-size='70%'>slow</tspan><tspan dy='-3'>)</tspan>", 11, TEAL_TEXT, anchor="start"))

    s.append(f'  <line x1="{cx}" y1="300" x2="{cx}" y2="334" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(box(848, 338, 350, 58, BLUE_FILL, BLUE_STROKE, sw=1.35))
    s.append(txt(cx, 361, "Untied Readout Head + Softmax", 13, INK, weight="700"))
    s.append(txt(cx, 382, f"logits = {math_var('W', 'out')} RMSNorm({math_var('h', 'dec')}) &#x2192; {math_var('p', '&#x3B8;')}({math_var('x', 't+1')} | {math_var('x', '1:t')})", 12, BLUE_TEXT))

    # Auxiliary Capabilities box
    s.append(box(848, 412, 350, 96, TEAL_FILL, TEAL_STROKE))
    s.append(txt(cx, 434, "Decoupled Ecosystem Heads (Zero-Tax)", 13, TEAL_TEXT, weight="700"))
    s.append(txt(cx, 455, "&#x2022; K3 (&#xD6;Y-1): detached r=1 early-exit (0.0 core grad)", 11, INK))
    s.append(txt(cx, 474, "&#x2022; MTP-3 (&#xD6;Y-4): t+2, t+3, t+4 lookahead adapters", 11, INK))
    s.append(txt(cx, 493, "&#x2022; &#xD6;Y-3: calibrated uncertainty u<tspan dy='3' font-size='70%'>t</tspan><tspan dy='-3'> = 1 &#x2212; max p(&#xB7;/T*)</tspan>", 11, INK))

    # Stage 2-3 Continual Memory
    s.append(txt(cx, 540, "Stage 2–3 · Causal Surprise &amp; M2 OWM-GDN", 15, TEAL_TEXT, weight="700", font=SANS))
    s.append(txt(cx, 560, "Outside Stage-1 loop · Zero future leakage", 12, MUTED))

    # Stage 2 Surprise Gate
    s.append(box(848, 574, 350, 92, AMBER_FILL, AMBER_STROKE))
    s.append(txt(cx, 597, "Stage 2 · Zero-Loop Causal Surprise Gate", 13, INK, weight="600"))
    s.append(txt(cx, 617, "From shifted logits_fast (no S<tspan dy='3' font-size='70%'>slow</tspan><tspan dy='-3'>): 0.72 u</tspan><tspan dy='3' font-size='70%'>NLL</tspan><tspan dy='-3'> + 0.18 u</tspan><tspan dy='3' font-size='70%'>MSP</tspan><tspan dy='-3'> + 0.10 u</tspan><tspan dy='3' font-size='70%'>P1</tspan>", 11, INK))
    s.append(txt(cx, 637, "g<tspan dy='3' font-size='70%'>surp</tspan><tspan dy='-3'> = &#x3C3;(&#x3BA;(u</tspan><tspan dy='3' font-size='70%'>tot</tspan><tspan dy='-3'>&#x2212;&#x3C4;))&#xB7;ramp (novel 0.752 vs distractor 0.134)</tspan>", 11, AMBER_TEXT))
    s.append(txt(cx, 655, "Symmetric return_aux pos==0 vs pos&gt;0 (diff 4.72e-15)", 10, MUTED))

    # Input arrows into Stage 2: (1) delta_conv from Stage 1 MacroCore, (2) shifted logits_fast from Stage 4
    s.append(f'  <line x1="810" y1="600" x2="844" y2="600" stroke="{AMBER_STROKE}" stroke-width="1.35" marker-end="url(#arrAmber)"/>\n')
    s.append(f'  <path d="M 1198 214 L 1210 214 L 1210 618 L 1201 618" fill="none" stroke="{AMBER_STROKE}" stroke-width="1.35" stroke-dasharray="4,2" marker-end="url(#arrAmber)"/>\n')

    s.append(f'  <line x1="{cx}" y1="666" x2="{cx}" y2="696" stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n')

    # Stage 3 M2 OWM-GDN
    s.append(box(848, 700, 350, 126, TEAL_FILL, TEAL_STROKE, sw=1.2))
    s.append(txt(cx, 723, "Stage 3 · OrthogonalSlowGDN (M2 OWM-GDN)", 13, TEAL_TEXT, weight="700"))
    s.append(txt(cx, 744, "P<tspan dy='3' font-size='70%'>&#x22A5;</tspan><tspan dy='-3'> = (I + &#x3BC; C&#x303;</tspan><tspan dy='3' font-size='70%'>commit</tspan><tspan dy='-3'>)</tspan><tspan dy='-4' font-size='70%'>&#x2212;1</tspan><tspan dy='4'> · p = P</tspan><tspan dy='3' font-size='70%'>&#x22A5;</tspan><tspan dy='-3'> k (zero-drift write)</tspan>", 11, INK))
    s.append(txt(cx, 765, "S<tspan dy='3' font-size='70%'>slow</tspan><tspan dy='-3'> updated only on g</tspan><tspan dy='3' font-size='70%'>surp</tspan><tspan dy='-3'> &gt; 0 (&#x3B1;</tspan><tspan dy='3' font-size='70%'>eff</tspan><tspan dy='-3'>, &#x3B2;</tspan><tspan dy='3' font-size='70%'>eff</tspan><tspan dy='-3'> scaled by g</tspan><tspan dy='3' font-size='70%'>surp</tspan><tspan dy='-3'>)</tspan>", 11, INK))
    s.append(txt(cx, 785, "Output &#x3B4;<tspan dy='3' font-size='70%'>slow</tspan><tspan dy='-3'> = tanh(&#x3B3;</tspan><tspan dy='3' font-size='70%'>slow</tspan><tspan dy='-3'>)&#xB7;y</tspan><tspan dy='3' font-size='70%'>slow</tspan><tspan dy='-3'> &#x2192; feeds Stage 4 (+)</tspan>", 11, TEAL_TEXT, weight="600"))
    s.append(txt(cx, 807, "State: S<tspan dy='3' font-size='70%'>slow</tspan><tspan dy='-3'>, C</tspan><tspan dy='3' font-size='70%'>commit</tspan><tspan dy='-3'>, C</tspan><tspan dy='3' font-size='70%'>pending</tspan><tspan dy='-3'> (12,288 B constant)</tspan>", 11, MUTED))

    # Arrow from SlowGDN back up to (+) in Stage 4
    s.append(f'  <path d="M 848 762 L 832 762 L 832 288 L 1011 288" fill="none" stroke="{TEAL_STROKE}" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#arrTeal)"/>\n')

    # ConsolidationManager box
    s.append(box(848, 848, 350, 120, BLUE_CONTAINER, BLUE_STROKE, sw=1.15))
    s.append(txt(cx, 871, "Time-Scale 3 · ConsolidationManager (0 Replay)", 13, BLUE_TEXT, weight="700"))
    s.append(txt(cx, 892, "1. Post-AdamW OGP: &#x394;W &#x2190; &#x394;W (I + &#x3BC; C&#x303;)<tspan dy='-4' font-size='70%'>&#x2212;1</tspan>", 11, INK))
    s.append(txt(cx, 912, "2. Scale-Normalized Online EWC (Fisher diag)", 11, INK))
    s.append(txt(cx, 932, "3. Reverse KL Anchor: D<tspan dy='3' font-size='70%'>KL</tspan><tspan dy='-3'>(&#x3C0;</tspan><tspan dy='3' font-size='70%'>g</tspan><tspan dy='-3'> &#x2016; &#x3C0;</tspan><tspan dy='3' font-size='70%'>0</tspan><tspan dy='-3'>)</tspan>", 11, INK))
    s.append(txt(cx, 954, "P4 CL3: ACC 0.2517 vs Naive 0.0851 (BWT -0.06 vs -0.42)", 11, TEAL_TEXT, weight="600"))


def _draw_state_sidebar(s: list[str]) -> None:
    # Column 4: x = 1222..1516
    x0, w = 1222, 294
    cx = x0 + w / 2
    s.append(txt(cx, 110, "Exact O(1) Constant State", 15, TEAL_TEXT, weight="700", font=SANS))
    s.append(txt(cx, 130, "Independent of T &amp; R (B=1 FP32)", 12, MUTED))

    s.append(box(x0, 152, w, 536, TEAL_FILL, TEAL_STROKE, sw=1.35, rx=8))
    s.append(txt(cx, 180, "RDFLv05State · 25,136 Bytes", 15, TEAL_TEXT, weight="700", font=SANS))
    s.append(txt(cx, 200, "565,216-param reference config", 11, MUTED))
    s.append(f'  <line x1="{x0 + 16}" y1="214" x2="{x0 + w - 16}" y2="214" stroke="#B7DCD4" stroke-width="1.0"/>\n')

    rows = [
        ("s_fast", "(1, 4, 16, 16)", "4,096 B"),
        ("conv_state", "(1, 64, 3)", "768 B"),
        ("s_macro", "(1, 4, 16, 16)", "4,096 B"),
        ("conv_macro", "(1, 64, 3)", "768 B"),
        ("slot_state (rlt2_s1)", "(1, 1, 64)", "256 B"),
        ("s_slow", "(1, 4, 16, 16)", "4,096 B"),
        ("c_slow (2 tensors)", "2&#xD7;(1, 4, 16, 16)", "8,192 B"),
        ("prev_fast_logits", "(1, 256)", "1,024 B"),
        ("stem_state (6 tensors)", "pb,ring,h,&#x3B4;,acc", "1,840 B"),
    ]
    y = 242
    for name, shp, b_str in rows:
        s.append(txt(x0 + 16, y, name, 11, INK, anchor="start", weight="600", font=SANS))
        s.append(txt(x0 + 184, y, shp, 10, MUTED, anchor="middle"))
        s.append(txt(x0 + w - 16, y, b_str, 11, TEAL_TEXT, anchor="end", weight="700"))
        s.append(f'  <line x1="{x0 + 16}" y1="{y + 10}" x2="{x0 + w - 16}" y2="{y + 10}" stroke="#DCEEE9" stroke-width="0.8"/>\n')
        y += 38

    s.append(box(x0 + 14, 594, w - 28, 76, "#FFFFFF", TEAL_STROKE, sw=1.1, rx=6))
    s.append(txt(cx, 616, "Total Constant Memory: 25,136 B", 13, TEAL_TEXT, weight="700"))
    s.append(txt(cx, 636, "Identical across T &#x2208; {16..320}", 11, INK))
    s.append(txt(cx, 654, "and R &#x2208; {1..1024} (0 KV-cache)", 11, INK))

    # Parity & Purity Contract Card
    s.append(box(x0, 710, w, 258, BLUE_CONTAINER, BLUE_STROKE, sw=1.25, rx=8))
    s.append(txt(cx, 738, "Verified Mathematical Contract", 14, BLUE_TEXT, weight="700", font=SANS))
    s.append(f'  <line x1="{x0 + 16}" y1="752" x2="{x0 + w - 16}" y2="752" stroke="#C7DCF2" stroke-width="1.0"/>\n')

    checks = [
        ("FP64 forward == step:", "&#x2264; 8.10e-15"),
        ("Partial patch (37+11):", "3.75e-15"),
        ("FP32 argmax mismatch:", "0 / 48 steps"),
        ("Future-byte leakage:", "0.0 exact"),
        ("g_step == 0 no-op:", "0.0 exact"),
        ("R=1024 stability:", "0 NaN / Inf"),
        ("Single-pass dec_ffn:", "1 call"),
    ]
    cy = 780
    for lbl, val in checks:
        s.append(txt(x0 + 16, cy, lbl, 11, INK, anchor="start"))
        s.append(txt(x0 + w - 16, cy, val, 11, BLUE_TEXT, anchor="end", weight="700"))
        cy += 26


def build_v05_architecture_svg() -> str:
    """Generate the RDFL v0.5.0 pure-recurrent academic architecture SVG."""
    s = [svg_open(1540, 1050, "RDFL v0.5.0 Pure-Recurrent Unified Architecture Diagram")]
    _draw_header(s)
    _draw_stage0_micro(s)
    _draw_stage1_macro(s)
    _draw_stage234_right(s)
    _draw_state_sidebar(s)
    s.append("</svg>\n")
    return "".join(s)
