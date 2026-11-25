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
    SERIF,
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


def _arch_containers(s: list[str]) -> None:
    """Draw translucent stage containers before wires so arrowheads remain unoccluded."""
    s.append(box(48, 360, 264, 506, BLUE_CONTAINER, "#C7DCF2", sw=1.1, rx=10))
    s.append(txt(62, 384, f"Stage 0 ({math_var('P')} = 1)", 14, INK, anchor="start", weight="600"))
    s.append(txt(62, 402, "Micro-DeltaProduct", 12, BLUE_TEXT, anchor="start", weight="600"))

    s.append(box(508, 374, 266, 448, AMBER_CONTAINER, "#F3CFA4", sw=1.15, rx=10))
    s.append(
        txt(520, 396, f"{math_var('R')} weight-tied", 13, INK, anchor="start", weight="600")
    )
    s.append(txt(520, 413, "macro iterations", 13, INK, anchor="start", weight="600"))


def _arch_trunk_column(s: list[str]) -> None:
    s.append(txt(180, 44, "Pure-recurrent byte trunk", 21, BLUE_TEXT, weight="700", font=SANS))
    s.append(
        txt(180, 70, f"Causal byte prefix {math_var('x', '1:t')} (0 Attention)", 14, MUTED)
    )
    s.append(
        txt(
            180,
            1020,
            f"Observed byte {math_var('x', 't')} &#x2208; &#x7B;0, &#x2026;, 255&#x7D;",
            14,
            INK,
        )
    )
    s.append(
        f'  <line x1="180" y1="998" x2="180" y2="970" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )

    s.append(box(64, 898, 232, 68, BLUE_FILL, BLUE_STROKE))
    s.append(txt(180, 921, "Byte embedding + N-gram stem", 14, INK, weight="600"))
    s.append(
        txt(
            180,
            942,
            f"{math_var('h', 'b0')} = {math_var('E', 'byte')}({math_var('x', 't')}) + &#x3B1;"
            "<tspan dy='3' font-size='72%' font-style='italic'>ng</tspan><tspan dy='-3'> &#x3B4;</tspan>"
            "<tspan dy='3' font-size='72%' font-style='italic'>ng</tspan>",
            13,
            INK,
        )
    )
    s.append(txt(180, 959, "2/3/4/6-gram Count-Sketch", 11, MUTED))

    s.append(
        f'  <line x1="180" y1="898" x2="180" y2="782" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )
    s.append(
        f'  <path d="M 180 832 L 72 832 L 72 456 L 164 456" fill="none" '
        f'stroke="{WIRE}" stroke-width="1.3" marker-end="url(#arrGray)"/>\n'
    )

    s.append(box(94, 688, 200, 90, BLUE_FILL, BLUE_STROKE))
    s.append(txt(194, 711, f"BiasFree causal conv ({math_var('k')}=4)", 14, INK, weight="600"))
    s.append(
        txt(
            194,
            733,
            f"RMSNorm &#x2192; {math_var('x', 'sc')} (&#x2192; {math_var('n', 'assoc')}=2)",
            12,
            MUTED,
        )
    )
    s.append(
        txt(
            194,
            754,
            f"{math_var('x', 'hh')} = {math_var('x')} + {math_var('g', 'conv')} &#x2299; {math_var('x', 'sc')} (&#x2192; {math_var('n', 'hh')}=2)",
            12,
            BLUE_TEXT,
        )
    )
    s.append(txt(194, 771, "bias=0 depthwise conv", 11, MUTED))

    s.append(
        f'  <line x1="180" y1="688" x2="180" y2="622" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )

    s.append(box(94, 524, 200, 94, BLUE_FILL, BLUE_STROKE))
    s.append(txt(194, 549, "Dual-regime DeltaProduct", 14, INK, weight="600"))
    s.append(
        txt(
            194,
            571,
            f"WY solver · {math_var('n', 'assoc')}=2, {math_var('n', 'hh')}=2",
            12,
            MUTED,
        )
    )
    s.append(
        txt(
            194,
            593,
            f"{math_var('n', 'h')}=4 Householder · SoftRMSNorm",
            12,
            BLUE_TEXT,
        )
    )

    s.append(
        f'  <line x1="180" y1="524" x2="180" y2="470" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )
    s.append(plus_node(180, 456))

    s.append(
        f'  <line x1="180" y1="444" x2="180" y2="310" stroke="{BLUE_STROKE}" '
        f'stroke-width="1.5" marker-end="url(#arrBlue)"/>\n'
    )
    s.append(box(74, 240, 212, 66, BLUE_FILL, BLUE_STROKE))
    s.append(txt(180, 265, "Micro byte stream", 14, INK, weight="600"))
    s.append(
        txt(
            180,
            289,
            f"{math_var('c', 't')} = {math_var('h', 'b0')} + {math_var('y', 'g')} &#x2208; &#x211D;"
            "<tspan dy='-5' font-size='72%' font-style='italic'>d</tspan>",
            14,
            INK,
        )
    )

    s.append(txt(180, 152, "Pure recurrence · 0 Attention · 0 BPE.", 13, MUTED))
    s.append(txt(180, 171, "Exact chunk / step equivalence.", 13, MUTED))
    s.append(
        txt(180, 190, "Data-dependent decay &#x3B1;<tspan dy='3' font-size='72%' font-style='italic'>t</tspan><tspan dy='-3'> &#x2208; (0, 1).</tspan>", 13, MUTED)
    )


def _arch_bridge(s: list[str]) -> None:
    s.append(
        f'  <line x1="286" y1="273" x2="322" y2="273" stroke="{BLUE_STROKE}" '
        f'stroke-width="1.5" marker-end="url(#arrBlue)"/>\n'
    )
    s.append(
        f'  <path d="M 180 240 L 180 218 L 552 218 L 552 238" fill="none" '
        f'stroke="{BLUE_STROKE}" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#arrBlue)"/>\n'
    )
    s.append(
        txt(
            366,
            208,
            f"byte-resolution residual skip {math_var('c', 't')}",
            13,
            BLUE_TEXT,
        )
    )

    s.append(box(326, 240, 174, 72, BLUE_FILL, BLUE_STROKE))
    s.append(txt(413, 264, f"Causal {math_var('P')}=4 patch packer", 14, INK, weight="600"))
    s.append(
        txt(
            413,
            285,
            f"{math_var('p', 'm')} = {math_var('W', 'pack')} RMSNorm({math_var('c', '4m:4m+3')})",
            11,
            INK,
        )
    )
    s.append(txt(413, 303, "4&#xD7; fewer macro steps", 12, BLUE_TEXT))

    s.append(
        f'  <path d="M 358 312 L 358 946 L 514 946" fill="none" '
        f'stroke="{BLUE_STROKE}" stroke-width="1.45" marker-end="url(#arrBlue)"/>\n'
    )
    s.append(txt(346, 868, f"{math_var('p', 'm')}", 14, BLUE_TEXT, anchor="end"))

    s.append(txt(424, 672, "+ denotes residual", 11, MUTED))
    s.append(txt(424, 689, "addition. Dashed arrows", 11, MUTED))
    s.append(txt(424, 706, "carry stop-grad or skip.", 11, MUTED))


def _arch_recursive_loop(s: list[str]) -> None:
    s.append(txt(635, 44, "Recursive macro loop", 21, AMBER_TEXT, weight="700", font=SANS))
    s.append(
        txt(
            635,
            70,
            f"Patch {math_var('m')} · {math_var('R')} weight-tied steps ({math_var('r')} = 0, &#x2026;, {math_var('R')}&#x2212;1)",
            14,
            MUTED,
        )
    )

    s.append(box(518, 906, 236, 76, AMBER_FILL, AMBER_STROKE))
    s.append(txt(636, 930, "Damped slot boundary merge", 14, INK, weight="600"))
    s.append(
        txt(
            636,
            952,
            f"{math_var('u', 'm')} = {math_var('p', 'm')} + &#x3B1; {math_var('g', 's')} &#x2299; {math_var('W', 'm')} {math_var('r', 'c&#x2212;1')}",
            13,
            INK,
        )
    )
    s.append(txt(636, 972, "rlt2_s1 single FP32 slot · 0 SlotCore", 12, AMBER_TEXT))
    s.append(
        txt(
            636,
            1004,
            f"{math_var('g', 's')} = &#x3C3;({math_var('W', 'g')}[{math_var('p', 'm')}; {math_var('r', 'c&#x2212;1')}] + {math_var('b', 'g')}),   {math_var('r', 'c&#x2212;1')} = RMSNorm({math_var('s', 'c&#x2212;1')})",
            12,
            MUTED,
        )
    )

    s.append(
        f'  <line x1="636" y1="906" x2="636" y2="794" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )
    s.append(
        txt(
            648,
            862,
            f"{math_var('h', 'm', '(0)')} = {math_var('u', 'm')}",
            14,
            INK,
            anchor="start",
        )
    )

    s.append(box(538, 708, 224, 82, AMBER_FILL, AMBER_STROKE))
    s.append(
        txt(
            650,
            731,
            f"Step {math_var('r')} = 0: GDN write + FFN",
            14,
            INK,
            weight="600",
        )
    )
    s.append(
        txt(
            650,
            753,
            f"Write &amp; read ({math_var('K', 'mem')} = 1) · WY solver",
            12,
            MUTED,
        )
    )
    s.append(
        txt(
            650,
            775,
            f"Causal conv ({math_var('k')}=4) · BDH + SwiGLU &#x2192; {math_var('h', '1')}",
            11,
            INK,
        )
    )

    s.append(
        f'  <line x1="636" y1="708" x2="636" y2="664" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )

    s.append(box(538, 574, 224, 86, AMBER_FILL, AMBER_STROKE))
    s.append(
        txt(
            650,
            598,
            f"Steps {math_var('r')} &#x2265; 1: Read-only StepReasoner",
            13,
            INK,
            weight="600",
        )
    )
    s.append(
        txt(
            650,
            620,
            f"Zero-solver WY read_only() · phase_proj({math_var('r')})",
            11,
            MUTED,
        )
    )
    s.append(
        txt(
            650,
            642,
            "BDH + BoundedSwiGLU &#x2192; &#x3B4;<tspan dy='3' font-size='72%' font-style='italic'>r</tspan>",
            12,
            INK,
        )
    )

    s.append(
        f'  <line x1="636" y1="574" x2="636" y2="532" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )

    s.append(box(538, 444, 224, 84, AMBER_FILL, AMBER_STROKE))
    s.append(txt(650, 468, "Learned halting gate (step_halting)", 13, INK, weight="600"))
    s.append(
        txt(
            650,
            491,
            f"{math_var('h', 'r+1')} = {math_var('h', 'r')} + {math_var('g', 'step')} · (norm({math_var('h', 'r')} + &#x3B4;<tspan dy='3' font-size='72%' font-style='italic'>r</tspan><tspan dy='-3'>) &#x2212; </tspan>{math_var('h', 'r')})",
            11,
            INK,
        )
    )
    s.append(
        txt(
            650,
            513,
            f"{math_var('g', 'step')} &#x2208; (0, 1) · {math_var('g', 'step')} = 0 exact no-op",
            11,
            AMBER_TEXT,
        )
    )

    s.append(
        f'  <path d="M 636 428 L 520 428 L 520 684 L 632 684" fill="none" '
        f'stroke="{AMBER_STROKE}" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#arrAmber)"/>\n'
    )
    s.append(
        f'  <text x="498" y="556" font-family="{SERIF}" font-size="12" fill="{AMBER_TEXT}" '
        f'text-anchor="middle" transform="rotate(-90 498 556)">repeat for 1 &#x2264; r &lt; R (weight-tied · J-Space)</text>\n'
    )


def _arch_readout_and_heads(s: list[str]) -> None:
    s.append(
        f'  <line x1="636" y1="444" x2="636" y2="314" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )
    s.append(
        txt(
            646,
            332,
            f"{math_var('h', 'fast')} ({math_var('h', 'bos')} if {math_var('t')}&lt;3,",
            11,
            INK,
            anchor="start",
        )
    )
    s.append(
        txt(
            646,
            350,
            f"else {math_var('m')} = &#x230A;({math_var('t')}+1)/4&#x230B;&#x2212;1)",
            11,
            MUTED,
            anchor="start",
        )
    )

    s.append(box(518, 240, 236, 70, AMBER_FILL, AMBER_STROKE))
    s.append(txt(636, 264, "Single-pass causal decoder", 14, INK, weight="600"))
    s.append(
        txt(
            636,
            285,
            f"{math_var('y', 't')} = {math_var('c', 't')} + ctx_proj(RMSNorm({math_var('h', 'fast')})) + {math_var('e', 't mod 4')}",
            11,
            INK,
        )
    )
    s.append(txt(636, 303, "1&#xD7; dec_ffn (SwiGLU)", 11, MUTED))

    s.append(
        f'  <line x1="636" y1="240" x2="636" y2="212" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )
    s.append(plus_node(636, 198))
    s.append(
        f'  <line x1="636" y1="186" x2="636" y2="160" stroke="{WIRE}" '
        f'stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )
    s.append(txt(624, 178, f"{math_var('h', 'dec')}", 13, INK, anchor="end"))

    s.append(box(534, 94, 204, 62, AMBER_FILL, AMBER_STROKE))
    s.append(txt(636, 119, "Untied readout + softmax", 14, INK, weight="600"))
    s.append(
        txt(
            636,
            142,
            f"{math_var('p', '&#x3B8;')}({math_var('x', 't+1')} | {math_var('x', '1:t')})  ({math_var('W', 'out')} &#x2260; {math_var('W', 'emb')})",
            13,
            INK,
        )
    )

    s.append(txt(520, 118, "Calibrated uncertainty:", 12, MUTED, anchor="end"))
    s.append(
        txt(
            520,
            136,
            f"{math_var('u', 't')} = 1 &#x2212; max {math_var('p', '&#x3B8;')}(&#xB7; / {math_var('T', '*')})",
            12,
            TEAL_TEXT,
            anchor="end",
        )
    )

    s.append(box(856, 94, 234, 62, TEAL_FILL, TEAL_STROKE))
    s.append(txt(973, 119, "Multi-byte heads (MTP-3)", 14, INK, weight="600"))
    s.append(
        txt(
            973,
            142,
            f"{math_var('p', '&#x3B8;', '(k)')}({math_var('x', 't+k')} | {math_var('x', '1:t')}),  {math_var('k')} &#x2208; &#x7B;2, 3, 4&#x7D;",
            13,
            TEAL_TEXT,
        )
    )
    s.append(
        f'  <path d="M 636 166 L 832 166 L 832 125 L 852 125" fill="none" '
        f'stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n'
    )

    s.append(box(856, 182, 234, 68, TEAL_FILL, TEAL_STROKE))
    s.append(txt(973, 204, "Decoupled early-exit (K3)", 14, INK, weight="600"))
    s.append(
        txt(
            973,
            224,
            f"sg({math_var('h', 'm', '(1)')}, {math_var('c', 't')}) &#x2192; dec_ffn + k3 &#x2192; {math_var('p', '&#x3B8;', '(1)')}",
            11,
            TEAL_TEXT,
        )
    )
    s.append(txt(973, 242, "0.0 core / head gradient", 11, MUTED))
    s.append(
        f'  <path d="M 822 320 L 822 214 L 852 214" fill="none" '
        f'stroke="{TEAL_STROKE}" stroke-width="1.3" stroke-dasharray="4,3" marker-end="url(#arrTeal)"/>\n'
    )
    s.append(txt(827, 268, "stop-grad", 11, TEAL_TEXT, anchor="start", style="italic"))


def _arch_o1_states(s: list[str]) -> None:
    s.append(box(856, 280, 234, 82, AMBER_FILL, AMBER_STROKE))
    s.append(txt(973, 304, "rlt2_s1 single-slot commit", 14, INK, weight="600"))
    s.append(
        txt(
            973,
            326,
            f"{math_var('s', 'c')} = &#x3B3;&#x2299;{math_var('s', 'c&#x2212;1')} + (1&#x2212;&#x3B3;)&#x2299;tanh({math_var('W', 'u')}[{math_var('s', 'c&#x2212;1')}; {math_var('h&#x302;', '1')}])",
            11,
            INK,
        )
    )
    s.append(
        txt(
            973,
            348,
            f"strictly from {math_var('r')}=0 chunk-end {math_var('h', '1')} · 0 SlotCore",
            11,
            MUTED,
        )
    )
    s.append(
        f'  <path d="M 762 762 L 822 762 L 822 320 L 852 320" fill="none" '
        f'stroke="{AMBER_STROKE}" stroke-width="1.45" marker-end="url(#arrAmber)"/>\n'
    )

    s.append(txt(973, 434, "Constant O(1) state (25,136 B ref);", 13, TEAL_TEXT))
    s.append(txt(973, 452, "exact chunk / step equivalence", 13, TEAL_TEXT))

    s.append(box(856, 468, 234, 88, TEAL_FILL, TEAL_STROKE))
    s.append(txt(973, 493, "Causal surprise &#x2192; M2 OWM-GDN", 13, INK, weight="600"))
    s.append(
        txt(
            973,
            515,
            f"Zero-loop · {math_var('P', '&#x22A5;')} orthogonal {math_var('S', 'slow')} write",
            12,
            INK,
        )
    )
    s.append(
        txt(
            973,
            537,
            f"{math_var('g', 'surp')}(logits_fast, &#x3B4;<tspan dy='3' font-size='72%' font-style='italic'>conv</tspan><tspan dy='-3'>) &#x2192; &#x3B4;</tspan><tspan dy='3' font-size='72%' font-style='italic'>slow</tspan>",
            12,
            TEAL_TEXT,
        )
    )

    s.append(
        f'  <path d="M 636 364 L 838 364 L 838 488 L 852 488" fill="none" '
        f'stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n'
    )
    s.append(
        txt(
            843,
            406,
            f"{math_var('h', 'fast')}, &#x3B4;<tspan dy='3' font-size='72%' font-style='italic'>conv</tspan>",
            11,
            MUTED,
            anchor="start",
        )
    )

    s.append(
        f'  <path d="M 856 520 L 806 520 L 806 198 L 650 198" fill="none" '
        f'stroke="{TEAL_STROKE}" stroke-width="1.4" marker-end="url(#arrTeal)"/>\n'
    )
    s.append(
        txt(
            732,
            188,
            "ctx_proj_slow(&#x3B4;<tspan dy='3' font-size='72%' font-style='italic'>slow</tspan><tspan dy='-3'>)</tspan>",
            11,
            TEAL_TEXT,
        )
    )

    s.append(box(856, 616, 234, 124, TEAL_FILL, TEAL_STROKE))
    s.append(txt(973, 646, "Recurrent matrix states", 14, INK, weight="600"))
    s.append(
        txt(
            973,
            672,
            f"{math_var('S', 'fast')}, {math_var('S', 'macro')}, {math_var('S', 'slow')}, {math_var('C', 'slow')}",
            12,
            INK,
        )
    )
    s.append(txt(973, 695, "+ causal conv caches", 12, MUTED))
    s.append(
        txt(
            973,
            718,
            f"independent of {math_var('T')} and {math_var('R')} · O(1)",
            12,
            TEAL_TEXT,
            style="italic",
        )
    )

    s.append(
        f'  <path d="M 856 632 L 766 632" fill="none" stroke="{TEAL_STROKE}" '
        f'stroke-width="1.35" marker-end="url(#arrTeal)"/>\n'
    )
    s.append(txt(778, 624, "read_only", 10, TEAL_TEXT, anchor="start"))

    s.append(
        f'  <path d="M 762 716 L 852 716" fill="none" stroke="{TEAL_STROKE}" '
        f'stroke-width="1.35" marker-end="url(#arrTeal)"/>\n'
    )
    s.append(
        f'  <path d="M 856 732 L 766 732" fill="none" stroke="{TEAL_STROKE}" '
        f'stroke-width="1.35" marker-end="url(#arrTeal)"/>\n'
    )
    s.append(
        f'  <line x1="956" y1="616" x2="956" y2="560" stroke="{TEAL_STROKE}" '
        f'stroke-width="1.35" marker-end="url(#arrTeal)"/>\n'
    )
    s.append(
        f'  <line x1="990" y1="556" x2="990" y2="612" stroke="{TEAL_STROKE}" '
        f'stroke-width="1.35" marker-end="url(#arrTeal)"/>\n'
    )

    s.append(box(856, 902, 234, 84, AMBER_FILL, AMBER_STROKE))
    s.append(txt(973, 927, "Previous chunk slot", 14, INK, weight="600"))
    s.append(
        txt(
            973,
            949,
            f"{math_var('s', 'c&#x2212;1')} &#x2208; &#x211D;"
            "<tspan dy='-5' font-size='72%' font-style='italic'>d</tspan><tspan dy='5'>  (1 FP32 vector)</tspan>",
            13,
            INK,
        )
    )
    s.append(
        txt(
            973,
            972,
            f"{math_var('r', 'c&#x2212;1')} = RMSNorm({math_var('s', 'c&#x2212;1')})",
            13,
            MUTED,
        )
    )
    s.append(
        f'  <path d="M 1090 320 L 1114 320 L 1114 944 L 1094 944" fill="none" '
        f'stroke="{AMBER_STROKE}" stroke-width="1.65" marker-end="url(#arrAmber)"/>\n'
    )
    s.append(
        f'  <text x="1128" y="635" font-family="{SERIF}" font-size="13" fill="{AMBER_TEXT}" '
        f'text-anchor="middle" transform="rotate(-90 1128 635)">chunk-boundary delay: c &#x2190; c + 1 (64 patches)</text>\n'
    )
    s.append(
        f'  <line x1="856" y1="944" x2="758" y2="944" stroke="{AMBER_STROKE}" '
        f'stroke-width="1.65" marker-end="url(#arrAmber)"/>\n'
    )


def build_architecture_svg() -> str:
    """Generate the bottom-to-top RDFL v0.5.0 academic computational graph SVG."""
    s = [svg_open(1140, 1060, "Recursive Depth Flash Loop (RDFL) v0.5.0 Pure-Recurrent Architecture Diagram")]
    _arch_containers(s)
    _arch_trunk_column(s)
    _arch_bridge(s)
    _arch_recursive_loop(s)
    _arch_readout_and_heads(s)
    _arch_o1_states(s)
    s.append("</svg>\n")
    return "".join(s)
