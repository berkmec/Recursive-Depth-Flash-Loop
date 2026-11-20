"""Build the academic TikZ/LaTeX-style RDFL computational graph SVG."""

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


def _arch_trunk_column(s: list[str]) -> None:
    s.append(txt(180, 44, "Causal input trunk", 21, BLUE_TEXT, weight="700", font=SANS))
    s.append(txt(180, 70, f"Causal byte prefix {math_var('x', '1:t')}", 14, MUTED))
    s.append(txt(180, 1020, f"Observed byte {math_var('x', 't')} &#x2208; &#x7B;0, &#x2026;, 255&#x7D;", 14, INK))
    s.append(f'  <line x1="180" y1="998" x2="180" y2="968" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(box(82, 920, 196, 44, BLUE_FILL, BLUE_STROKE))
    s.append(txt(180, 947, "Byte embedding", 15, INK))
    s.append(f'  <line x1="180" y1="920" x2="180" y2="854" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')

    s.append(box(48, 360, 264, 514, BLUE_CONTAINER, "#C7DCF2", sw=1.1, rx=10))
    s.append(txt(66, 390, f"{math_var('L', 'B')} trunk layers", 15, INK, anchor="start"))
    s.append(f'  <line x1="180" y1="852" x2="180" y2="812" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(f'  <path d="M 180 834 L 78 834 L 78 684 L 166 684" fill="none" stroke="{WIRE}" stroke-width="1.3" marker-end="url(#arrGray)"/>\n')

    s.append(box(102, 742, 178, 66, BLUE_FILL, BLUE_STROKE))
    s.append(txt(191, 769, "Causal self-attention", 14, INK))
    s.append(txt(191, 790, f"window {math_var('M')} (RoPE)", 13, MUTED))
    s.append(f'  <line x1="180" y1="742" x2="180" y2="698" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(plus_node(180, 684))

    s.append(f'  <line x1="180" y1="672" x2="180" y2="594" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(f'  <path d="M 180 634 L 78 634 L 78 468 L 166 468" fill="none" stroke="{WIRE}" stroke-width="1.3" marker-end="url(#arrGray)"/>\n')
    s.append(box(102, 528, 178, 62, BLUE_FILL, BLUE_STROKE))
    s.append(txt(191, 555, "RMSNorm + FFN", 14, INK))
    s.append(txt(191, 575, "unshrunk base width", 13, MUTED))
    s.append(f'  <line x1="180" y1="528" x2="180" y2="482" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(plus_node(180, 468))

    s.append(f'  <line x1="180" y1="456" x2="180" y2="306" stroke="{BLUE_STROKE}" stroke-width="1.5" marker-end="url(#arrBlue)"/>\n')
    s.append(box(82, 240, 196, 62, BLUE_FILL, BLUE_STROKE))
    s.append(txt(180, 265, "Trunk representation", 14, INK))
    s.append(txt(180, 287, f"{math_var('z', 't', 'base')} = {math_var('E', '&#x3B8;')}({math_var('x', '1:t')})", 14, INK))
    s.append(txt(180, 162, "Causal masking enables parallel", 13, MUTED))
    s.append(txt(180, 181, "prefill across all positions.", 13, MUTED))
    s.append(txt(180, 200, "Optional frozen or low-LR base.", 13, MUTED))


def _arch_bridge(s: list[str]) -> None:
    s.append(f'  <line x1="278" y1="271" x2="342" y2="271" stroke="{BLUE_STROKE}" stroke-width="1.5" marker-end="url(#arrBlue)"/>\n')
    s.append(f'  <path d="M 180 240 L 180 218 L 621 218" fill="none" stroke="{BLUE_STROKE}" stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#arrBlue)"/>\n')
    s.append(txt(405, 208, f"identity skip {math_var('z', 't', 'base')} (step-0 exact preservation)", 13, BLUE_TEXT))

    s.append(box(346, 240, 186, 70, BLUE_FILL, BLUE_STROKE))
    s.append(txt(439, 264, "Narrow projection", 14, INK))
    s.append(txt(439, 284, f"{math_var('z', 't', '0')} = {math_var('W', '&#x2193;')} RMSNorm({math_var('z', 't', 'base')})", 13, INK))
    s.append(txt(439, 302, f"{math_var('e', 't')} = RMSNorm({math_var('z', 't', '0')})", 13, BLUE_TEXT))

    s.append(f'  <path d="M 456 310 L 456 758 L 538 758" fill="none" stroke="{BLUE_STROKE}" stroke-width="1.45" marker-end="url(#arrBlue)"/>\n')
    s.append(txt(446, 515, f"Input anchor {math_var('e', 't')}", 12, BLUE_TEXT, anchor="end"))
    s.append(txt(446, 533, "re-injected at", 12, BLUE_TEXT, anchor="end"))
    s.append(txt(446, 550, f"every step {math_var('r')}", 12, BLUE_TEXT, anchor="end"))

    s.append(f'  <path d="M 358 310 L 358 946 L 516 946" fill="none" stroke="{BLUE_STROKE}" stroke-width="1.45" marker-end="url(#arrBlue)"/>\n')
    s.append(txt(346, 868, f"{math_var('z', 't', '0')}", 14, BLUE_TEXT, anchor="end"))
    s.append(txt(407, 648, "+ denotes residual", 10, MUTED))
    s.append(txt(407, 663, "addition. Dashed", 10, MUTED))
    s.append(txt(407, 678, "arrows carry stop-grad", 10, MUTED))
    s.append(txt(407, 693, "or identity skip.", 10, MUTED))


def _arch_recursive_loop(s: list[str]) -> None:
    s.append(txt(635, 44, "Recursive depth loop", 21, AMBER_TEXT, weight="700", font=SANS))
    s.append(txt(635, 70, f"Chunk {math_var('c')} · {math_var('R')} weight-tied iterations ({math_var('r')} = 1, &#x2026;, {math_var('R')})", 14, MUTED))

    s.append(box(520, 910, 232, 70, AMBER_FILL, AMBER_STROKE))
    s.append(txt(636, 935, "Chunk-boundary gated merge", 14, INK, weight="600"))
    s.append(txt(636, 960, f"{math_var('u', 't')} = {math_var('z', 't', '0')} + &#x3B1; {math_var('g', 't')} &#x2299; {math_var('W', 'm')} {math_var('r', 'c&#x2212;1')}", 14, INK))
    s.append(txt(636, 1004, f"{math_var('g', 't')} = &#x3C3;({math_var('W', 'g')}[{math_var('z', 't', '0')}; {math_var('r', 'c&#x2212;1')}] + {math_var('b', 'g')}),   {math_var('r', 'c&#x2212;1')} = RMSNorm({math_var('M&#x304;', 'c&#x2212;1')})", 13, MUTED))

    s.append(f'  <line x1="636" y1="910" x2="636" y2="796" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(txt(670, 862, f"{math_var('h', 't', '(0)')} = {math_var('u', 't')}", 14, INK))

    s.append(box(508, 366, 266, 456, AMBER_CONTAINER, "#F3CFA4", sw=1.15, rx=10))
    s.append(txt(524, 392, f"{math_var('R')} weight-tied iterations", 15, INK, anchor="start"))

    s.append(box(542, 724, 212, 68, AMBER_FILL, AMBER_STROKE))
    s.append(txt(648, 749, "Anchor &amp; step injection", 14, INK))
    s.append(txt(648, 773, f"{math_var('h&#x303;', 't', '(r)')} = RMSNorm({math_var('h', 't', '(r&#x2212;1)')} + &#x3B1;<tspan dy='3' font-size='72%' font-style='italic'>r</tspan><tspan dy='-3'> </tspan>{math_var('e', 't')} + {math_var('e', 'r')})", 13, INK))
    s.append(f'  <path d="M 636 806 L 522 806 L 522 440 L 622 440" fill="none" stroke="{WIRE}" stroke-width="1.3" marker-end="url(#arrGray)"/>\n')
    s.append(f'  <line x1="636" y1="724" x2="636" y2="682" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')

    s.append(box(542, 594, 212, 84, AMBER_FILL, AMBER_STROKE))
    s.append(txt(648, 619, "Chunk-parallel Gated DeltaNet", 14, INK, weight="600"))
    s.append(txt(648, 641, f"Causal conv ({math_var('k')}=4) · WY solver", 13, MUTED))
    s.append(txt(648, 663, "&#x3B2;<tspan dy='3' font-size='72%' font-style='italic'>t</tspan><tspan dy='-3'> = 2&#x3C3;(&#xB7;) &#x2208; (0, 2) · slow decay &#x3B1;</tspan><tspan dy='3' font-size='72%' font-style='italic'>t</tspan>", 13, INK))
    s.append(f'  <line x1="636" y1="594" x2="636" y2="554" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(txt(618, 578, f"{math_var('y', 't', '(r)')}", 13, INK, anchor="end"))

    s.append(box(542, 480, 212, 70, AMBER_FILL, AMBER_STROKE))
    s.append(txt(648, 505, "BDH sparse interaction gate", 14, INK))
    s.append(txt(648, 529, f"{math_var('W', 'dec')}(ReLU({math_var('W', 'x')} {math_var('h&#x303;', 't', '(r)')}) &#x2299; ReLU({math_var('W', 'y')} {math_var('y', 't', '(r)')}))", 12, INK))
    s.append(f'  <path d="M 636 574 L 764 574 L 764 440 L 650 440" fill="none" stroke="{WIRE}" stroke-width="1.2" marker-end="url(#arrGray)"/>\n')
    s.append(f'  <line x1="636" y1="480" x2="636" y2="454" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(plus_node(636, 440))
    s.append(f'  <path d="M 636 428 L 636 408 L 792 408 L 792 806 L 642 806" fill="none" stroke="{AMBER_STROKE}" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#arrAmber)"/>\n')
    s.append(f'  <text x="807" y="518" font-family="{SERIF}" font-size="12" fill="{AMBER_TEXT}" text-anchor="middle" transform="rotate(-90 807 518)">repeat for r &lt; R (weight-tied)</text>\n')


def _arch_readout_and_heads(s: list[str]) -> None:
    s.append(f'  <line x1="636" y1="408" x2="636" y2="334" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(txt(660, 355, f"{math_var('h', 't', '(R)')}", 14, INK))
    s.append(box(534, 266, 204, 64, AMBER_FILL, AMBER_STROKE))
    s.append(txt(636, 291, "Zero-init residual projection", 14, INK))
    s.append(txt(636, 314, f"&#x394;{math_var('z', 't')} = {math_var('W', '&#x2191;')} RMSNorm({math_var('h', 't', '(R)')}),  {math_var('W', '&#x2191;')}|<tspan dy='3' font-size='72%'>0</tspan><tspan dy='-3'> = 0</tspan>", 13, INK))

    s.append(f'  <line x1="636" y1="266" x2="636" y2="232" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(plus_node(636, 218))
    s.append(f'  <line x1="636" y1="206" x2="636" y2="162" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(txt(648, 176, f"{math_var('z&#x302;', 't')} = RMSNorm({math_var('z', 't', 'base')} + &#x394;{math_var('z', 't')})", 13, INK, anchor="start"))

    s.append(box(534, 96, 204, 62, AMBER_FILL, AMBER_STROKE))
    s.append(txt(636, 121, "Readout + softmax", 15, INK, weight="600"))
    s.append(txt(636, 144, f"{math_var('p', '&#x398;')}({math_var('x', 't+1')} | {math_var('x', '1:t')})", 15, INK))
    s.append(txt(520, 120, "Calibrated uncertainty:", 12, MUTED, anchor="end"))
    s.append(txt(520, 138, f"{math_var('u', 't')} = 1 &#x2212; max {math_var('p', '&#x398;')}(&#xB7; / {math_var('T', '*')})", 12, TEAL_TEXT, anchor="end"))

    s.append(box(856, 96, 234, 62, TEAL_FILL, TEAL_STROKE))
    s.append(txt(973, 121, "Multi-byte heads (MTP-4)", 14, INK, weight="600"))
    s.append(txt(973, 144, f"{math_var('p', '&#x398;', '(k)')}({math_var('x', 't+k')} | {math_var('x', '1:t')}),  {math_var('k')} &#x2208; &#x7B;2, 3, 4&#x7D;", 13, TEAL_TEXT))
    s.append(f'  <path d="M 636 188 L 832 188 L 832 127 L 852 127" fill="none" stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n')

    s.append(box(856, 186, 234, 64, TEAL_FILL, TEAL_STROKE))
    s.append(txt(973, 211, "Decoupled early-exit (K3)", 14, INK, weight="600"))
    s.append(txt(973, 234, f"sg({math_var('h', 't', '(1)')}) &#x2192; {math_var('p', '&#x398;', '(1)')} (speculative draft)", 13, TEAL_TEXT))
    s.append(f'  <path d="M 774 418 L 832 418 L 832 218 L 852 218" fill="none" stroke="{TEAL_STROKE}" stroke-width="1.3" stroke-dasharray="4,3" marker-end="url(#arrTeal)"/>\n')
    s.append(txt(824, 286, "stop-grad", 11, TEAL_TEXT, anchor="end", style="italic"))


def _arch_o1_states(s: list[str]) -> None:
    s.append(txt(973, 466, "Constant O(1) state per step;", 13, TEAL_TEXT))
    s.append(txt(973, 484, "exact chunk / step equivalence", 13, TEAL_TEXT))
    s.append(box(856, 498, 234, 72, TEAL_FILL, TEAL_STROKE))
    s.append(txt(973, 523, f"Next recurrent state {math_var('S', 't', '(r)')}", 14, INK, weight="600"))
    s.append(txt(973, 548, f"&#x3B1;<tspan dy='3' font-size='72%' font-style='italic'>t</tspan><tspan dy='-3'> </tspan>{math_var('S', 't&#x2212;1', '(r)')} + &#x3B2;<tspan dy='3' font-size='72%' font-style='italic'>t</tspan><tspan dy='-3'>(</tspan>{math_var('v', 't')} &#x2212; &#x3B1;<tspan dy='3' font-size='72%' font-style='italic'>t</tspan><tspan dy='-3'> </tspan>{math_var('S', 't&#x2212;1', '(r)')}{math_var('k', 't')}){math_var('k', 't', '&#x22A4;')}", 12, INK))

    s.append(box(856, 614, 234, 76, TEAL_FILL, TEAL_STROKE))
    s.append(txt(973, 639, f"Recurrent matrix state {math_var('S', 't&#x2212;1', '(r)')}", 14, INK))
    s.append(txt(973, 661, f"{math_var('S', '', '(r)')} &#x2208; &#x211D;<tspan dy='-5' font-size='72%' font-style='italic'>H&#xD7;d&#xD7;d</tspan><tspan dy='5'> + causal conv cache</tspan>", 12, MUTED))
    s.append(txt(973, 679, "independent of sequence length T", 12, TEAL_TEXT, style="italic"))
    s.append(f'  <path d="M 754 648 L 852 648" fill="none" stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n')
    s.append(f'  <path d="M 856 628 L 758 628" fill="none" stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n')
    s.append(f'  <line x1="973" y1="614" x2="973" y2="574" stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n')

    s.append(box(856, 326, 234, 74, AMBER_FILL, AMBER_STROKE))
    s.append(txt(973, 350, "FP32 SlotCore commit", 14, INK, weight="600"))
    s.append(txt(973, 372, f"{math_var('M', 'c')} = SlotAttn<tspan dy='3' font-size='70%'>FP32</tspan><tspan dy='-3'>(</tspan>{math_var('M', 'c&#x2212;1')}, {math_var('h', 'end', '(R)')})", 13, INK))
    s.append(txt(973, 391, f"updated at boundary {math_var('t')} &#x2261; 0 (mod {math_var('B')})", 12, MUTED))
    s.append(f'  <path d="M 636 362 L 852 362" fill="none" stroke="{AMBER_STROKE}" stroke-width="1.5" marker-end="url(#arrAmber)"/>\n')

    s.append(box(856, 902, 234, 86, AMBER_FILL, AMBER_STROKE))
    s.append(txt(973, 926, "Previous chunk slots", 14, INK, weight="600"))
    s.append(txt(973, 949, f"{math_var('M', 'c&#x2212;1')} &#x2208; &#x211D;<tspan dy='-5' font-size='72%' font-style='italic'>S&#xD7;d</tspan><tspan dy='5'>  (</tspan>{math_var('S')} = 4 slots)", 13, INK))
    s.append(txt(973, 973, f"{math_var('M&#x304;', 'c&#x2212;1')} = (1/{math_var('S')}) &#x2211;<tspan dy='3' font-size='70%' font-style='italic'>s</tspan><tspan dy='-3'> </tspan>{math_var('M', 'c&#x2212;1,s')}", 13, MUTED))
    s.append(f'  <path d="M 1090 363 L 1114 363 L 1114 945 L 1094 945" fill="none" stroke="{AMBER_STROKE}" stroke-width="1.65" marker-end="url(#arrAmber)"/>\n')
    s.append(f'  <text x="1128" y="655" font-family="{SERIF}" font-size="13" fill="{AMBER_TEXT}" text-anchor="middle" transform="rotate(-90 1128 655)">chunk-boundary delay: c &#x2190; c + 1</text>\n')
    s.append(f'  <line x1="856" y1="945" x2="756" y2="945" stroke="{AMBER_STROKE}" stroke-width="1.65" marker-end="url(#arrAmber)"/>\n')


def build_architecture_svg() -> str:
    """Generate the bottom-to-top RDFL academic computational graph SVG."""
    s = [svg_open(1140, 1060, "Recursive Depth Flash Loop (RDFL) Academic Architecture Diagram")]
    _arch_trunk_column(s)
    _arch_bridge(s)
    _arch_recursive_loop(s)
    _arch_readout_and_heads(s)
    _arch_o1_states(s)
    s.append("</svg>\n")
    return "".join(s)
