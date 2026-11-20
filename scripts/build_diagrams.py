"""Generate publication-grade academic LaTeX/TikZ-style vector SVG diagrams for RDFL."""

from __future__ import annotations

import math
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "diagrams"

SERIF = "'CMU Serif', 'STIX Two Text', 'Nimbus Roman', 'Liberation Serif', Georgia, serif"
SANS = "Inter, 'Liberation Sans', -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif"

# Minimalist academic 3-color palette (Blue: causal trunk, Amber: recursive loop, Teal: O(1) states)
BLUE_STROKE = "#3B82C4"
BLUE_FILL = "#F2F7FD"
BLUE_TEXT = "#1E5B9A"
BLUE_CONTAINER = "#F8FBFF"

AMBER_STROKE = "#D9822B"
AMBER_FILL = "#FDF6EC"
AMBER_TEXT = "#B8620B"
AMBER_CONTAINER = "#FFFBF5"

TEAL_STROKE = "#2C8C7C"
TEAL_FILL = "#EFF8F6"
TEAL_TEXT = "#1D6B5E"

INK = "#1F2937"
MUTED = "#4B5563"
WIRE = "#4B5563"


def _svg_open(w: int, h: int, label: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{label}">
  <defs>
    <marker id="arrGray" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="{WIRE}"/>
    </marker>
    <marker id="arrBlue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="{BLUE_STROKE}"/>
    </marker>
    <marker id="arrAmber" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="{AMBER_STROKE}"/>
    </marker>
    <marker id="arrTeal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="{TEAL_STROKE}"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="{w}" height="{h}" fill="#FFFFFF"/>
"""


def _box(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str,
    sw: float = 1.25,
    rx: float = 5,
    dash: str = "",
) -> str:
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d_attr}/>\n'
    )


def _plus_node(cx: float, cy: float, r: float = 11.5) -> str:
    return (
        f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="#FFFFFF" stroke="{WIRE}" stroke-width="1.2"/>\n'
        f'  <line x1="{cx - 5.5}" y1="{cy}" x2="{cx + 5.5}" y2="{cy}" stroke="{INK}" stroke-width="1.2"/>\n'
        f'  <line x1="{cx}" y1="{cy - 5.5}" x2="{cx}" y2="{cy + 5.5}" stroke="{INK}" stroke-width="1.2"/>\n'
    )


def _t(
    x: float,
    y: float,
    markup: str,
    size: int = 14,
    fill: str = INK,
    anchor: str = "middle",
    weight: str = "400",
    font: str = SERIF,
    style: str = "normal",
) -> str:
    st = f' font-style="{style}"' if style != "normal" else ""
    return (
        f'  <text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{st}>{markup}</text>\n'
    )


def _m(var: str, sub: str = "", sup: str = "") -> str:
    """Academic italic math variable with crisp subscript and superscript tspans."""
    out = f'<tspan font-style="italic">{var}</tspan>'
    if sub:
        out += f'<tspan dy="3.5" font-size="72%" font-style="italic">{sub}</tspan><tspan dy="-3.5" font-size="100%">&#8203;</tspan>'
    if sup:
        out += f'<tspan dy="-5.5" font-size="72%" font-style="italic">{sup}</tspan><tspan dy="5.5" font-size="100%">&#8203;</tspan>'
    return out


def build_architecture_svg() -> str:
    """TikZ/LaTeX-style academic computational graph of RDFL (bottom-to-top)."""
    w, h = 1140, 1060
    s = [_svg_open(w, h, "Recursive Depth Flash Loop (RDFL) Academic Architecture Diagram")]

    # ================= COLUMN TITLES =================
    s.append(_t(180, 44, "Causal input trunk", 21, BLUE_TEXT, weight="700", font=SANS))
    s.append(_t(180, 70, f"Causal byte prefix {_m('x', '1:t')}", 14, MUTED))

    s.append(_t(635, 44, "Recursive depth loop", 21, AMBER_TEXT, weight="700", font=SANS))
    s.append(
        _t(
            635,
            70,
            f"Chunk {_m('c')} · {_m('R')} weight-tied iterations ({_m('r')} = 1, &#x2026;, {_m('R')})",
            14,
            MUTED,
        )
    )

    # ================= LEFT COLUMN: CAUSAL INPUT TRUNK =================
    # Bottom input label
    s.append(_t(180, 1020, f"Observed byte {_m('x', 't')} &#x2208; &#x7B;0, &#x2026;, 255&#x7D;", 14, INK))
    s.append(f'  <line x1="180" y1="998" x2="180" y2="968" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')

    # Byte embedding
    s.append(_box(82, 920, 196, 44, BLUE_FILL, BLUE_STROKE))
    s.append(_t(180, 947, "Byte embedding", 15, INK))
    s.append(f'  <line x1="180" y1="920" x2="180" y2="854" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')

    # Outer L_B trunk box
    s.append(_box(48, 360, 264, 514, BLUE_CONTAINER, "#C7DCF2", sw=1.1, rx=10))
    s.append(_t(66, 390, f"{_m('L', 'B')} trunk layers", 15, INK, anchor="start"))

    # Residual 1 tap below Causal Attention
    s.append(f'  <line x1="180" y1="852" x2="180" y2="812" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(f'  <path d="M 180 834 L 78 834 L 78 684 L 166 684" fill="none" stroke="{WIRE}" stroke-width="1.3" marker-end="url(#arrGray)"/>\n')

    # Causal windowed self-attention
    s.append(_box(102, 742, 178, 66, BLUE_FILL, BLUE_STROKE))
    s.append(_t(191, 769, "Causal self-attention", 14, INK))
    s.append(_t(191, 790, f"window {_m('M')} (RoPE)", 13, MUTED))

    s.append(f'  <line x1="180" y1="742" x2="180" y2="698" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(_plus_node(180, 684))

    # Residual 2 tap below FFN
    s.append(f'  <line x1="180" y1="672" x2="180" y2="594" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(f'  <path d="M 180 634 L 78 634 L 78 468 L 166 468" fill="none" stroke="{WIRE}" stroke-width="1.3" marker-end="url(#arrGray)"/>\n')

    # Feed-forward network
    s.append(_box(102, 528, 178, 62, BLUE_FILL, BLUE_STROKE))
    s.append(_t(191, 555, "RMSNorm + FFN", 14, INK))
    s.append(_t(191, 575, "unshrunk base width", 13, MUTED))

    s.append(f'  <line x1="180" y1="528" x2="180" y2="482" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(_plus_node(180, 468))

    # Arrow out of trunk to Final trunk representation
    s.append(f'  <line x1="180" y1="456" x2="180" y2="306" stroke="{BLUE_STROKE}" stroke-width="1.5" marker-end="url(#arrBlue)"/>\n')

    # Trunk representation box
    s.append(_box(82, 240, 196, 62, BLUE_FILL, BLUE_STROKE))
    s.append(_t(180, 265, "Trunk representation", 14, INK))
    s.append(_t(180, 287, f"{_m('z', 't', 'base')} = {_m('E', '&#x3B8;')}({_m('x', '1:t')})", 14, INK))

    # Clean academic note above left column
    s.append(_t(180, 162, "Causal masking enables parallel", 13, MUTED))
    s.append(_t(180, 181, "prefill across all positions.", 13, MUTED))
    s.append(_t(180, 200, "Optional frozen or low-LR base.", 13, MUTED))

    # ================= BRIDGE: BOTTLENECK PROJECTION & ANCHOR =================
    s.append(f'  <line x1="278" y1="271" x2="342" y2="271" stroke="{BLUE_STROKE}" stroke-width="1.5" marker-end="url(#arrBlue)"/>\n')

    # Residual skip of z_t^base directly to top fusion node (dashed blue)
    s.append(
        f'  <path d="M 180 240 L 180 218 L 621 218" fill="none" stroke="{BLUE_STROKE}" '
        f'stroke-width="1.4" stroke-dasharray="5,4" marker-end="url(#arrBlue)"/>\n'
    )
    s.append(_t(405, 208, f"identity skip {_m('z', 't', 'base')} (step-0 exact preservation)", 13, BLUE_TEXT))

    # Bottleneck projection box
    s.append(_box(346, 240, 186, 70, BLUE_FILL, BLUE_STROKE))
    s.append(_t(439, 264, "Narrow projection", 14, INK))
    s.append(_t(439, 284, f"{_m('z', 't', '0')} = {_m('W', '&#x2193;')} RMSNorm({_m('z', 't', 'base')})", 13, INK))
    s.append(_t(439, 302, f"{_m('e', 't')} = RMSNorm({_m('z', 't', '0')})", 13, BLUE_TEXT))

    # Wire 1 from Narrow Projection: Input anchor e_t into recursive block
    s.append(f'  <path d="M 456 310 L 456 758 L 538 758" fill="none" stroke="{BLUE_STROKE}" stroke-width="1.45" marker-end="url(#arrBlue)"/>\n')
    s.append(_t(446, 515, f"Input anchor {_m('e', 't')}", 12, BLUE_TEXT, anchor="end"))
    s.append(_t(446, 533, "re-injected at", 12, BLUE_TEXT, anchor="end"))
    s.append(_t(446, 550, f"every step {_m('r')}", 12, BLUE_TEXT, anchor="end"))

    # Wire 2 from Narrow Projection: z_t^0 down to Chunk-Boundary Gated Merge
    s.append(f'  <path d="M 358 310 L 358 946 L 516 946" fill="none" stroke="{BLUE_STROKE}" stroke-width="1.45" marker-end="url(#arrBlue)"/>\n')
    s.append(_t(346, 868, f"{_m('z', 't', '0')}", 14, BLUE_TEXT, anchor="end"))

    # Small clean legend note strictly centered between x=358 and x=456
    s.append(_t(407, 648, "+ denotes residual", 10, MUTED, anchor="middle"))
    s.append(_t(407, 663, "addition. Dashed", 10, MUTED, anchor="middle"))
    s.append(_t(407, 678, "arrows carry stop-grad", 10, MUTED, anchor="middle"))
    s.append(_t(407, 693, "or identity skip.", 10, MUTED, anchor="middle"))

    # ================= CENTER COLUMN: RECURSIVE DEPTH LOOP =================
    # Bottom: Chunk-boundary gated merge (matches Gated merge in reference!)
    s.append(_box(520, 910, 232, 70, AMBER_FILL, AMBER_STROKE))
    s.append(_t(636, 935, "Chunk-boundary gated merge", 14, INK, weight="600"))
    s.append(_t(636, 960, f"{_m('u', 't')} = {_m('z', 't', '0')} + &#x3B1; {_m('g', 't')} &#x2299; {_m('W', 'm')} {_m('r', 'c&#x2212;1')}", 14, INK))
    s.append(
        _t(
            636,
            1004,
            f"{_m('g', 't')} = &#x3C3;({_m('W', 'g')}[{_m('z', 't', '0')}; {_m('r', 'c&#x2212;1')}] + {_m('b', 'g')}),"
            f"   {_m('r', 'c&#x2212;1')} = RMSNorm({_m('M&#x304;', 'c&#x2212;1')})",
            13,
            MUTED,
        )
    )

    # Wire up from Gated Merge into Recursive Container
    s.append(f'  <line x1="636" y1="910" x2="636" y2="796" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(_t(670, 862, f"{_m('h', 't', '(0)')} = {_m('u', 't')}", 14, INK))

    # Outer Amber Container: R weight-tied iterations
    s.append(_box(508, 366, 266, 456, AMBER_CONTAINER, "#F3CFA4", sw=1.15, rx=10))
    s.append(_t(524, 392, f"{_m('R')} weight-tied iterations", 15, INK, anchor="start"))

    # Inside container: Step 1 — Anchor & step injection
    s.append(_box(542, 724, 212, 68, AMBER_FILL, AMBER_STROKE))
    s.append(_t(648, 749, "Anchor &amp; step injection", 14, INK))
    s.append(
        _t(
            648,
            773,
            f"{_m('h&#x303;', 't', '(r)')} = RMSNorm({_m('h', 't', '(r&#x2212;1)')} + &#x3B1;<tspan dy='3' font-size='72%' font-style='italic'>r</tspan><tspan dy='-3'> </tspan>{_m('e', 't')} + {_m('e', 'r')})",
            13,
            INK,
        )
    )

    # Residual bypass inside recursive block: h_t^(r-1) skips to top + node
    s.append(f'  <path d="M 636 806 L 522 806 L 522 440 L 622 440" fill="none" stroke="{WIRE}" stroke-width="1.3" marker-end="url(#arrGray)"/>\n')

    # Wire up to Gated DeltaNet
    s.append(f'  <line x1="636" y1="724" x2="636" y2="682" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')

    # Inside container: Step 2 — Chunk-parallel Gated DeltaNet
    s.append(_box(542, 594, 212, 84, AMBER_FILL, AMBER_STROKE))
    s.append(_t(648, 619, "Chunk-parallel Gated DeltaNet", 14, INK, weight="600"))
    s.append(_t(648, 641, f"Causal conv ({_m('k')}=4) · WY solver", 13, MUTED))
    s.append(_t(648, 663, f"&#x3B2;<tspan dy='3' font-size='72%' font-style='italic'>t</tspan><tspan dy='-3'> = 2&#x3C3;(&#xB7;) &#x2208; (0, 2) · slow decay &#x3B1;</tspan><tspan dy='3' font-size='72%' font-style='italic'>t</tspan>", 13, INK))

    # Wire up from GDN to BDH sparse gate (and direct residual into +)
    s.append(f'  <line x1="636" y1="594" x2="636" y2="554" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(_t(618, 578, f"{_m('y', 't', '(r)')}", 13, INK, anchor="end"))

    # Inside container: Step 3 — BDH sparse interaction gate
    s.append(_box(542, 480, 212, 70, AMBER_FILL, AMBER_STROKE))
    s.append(_t(648, 505, "BDH sparse interaction gate", 14, INK))
    s.append(_t(648, 529, f"{_m('W', 'dec')}(ReLU({_m('W', 'x')} {_m('h&#x303;', 't', '(r)')}) &#x2299; ReLU({_m('W', 'y')} {_m('y', 't', '(r)')}))", 12, INK))

    # Direct y_t^(r) skip around BDH into + node
    s.append(f'  <path d="M 636 574 L 764 574 L 764 440 L 650 440" fill="none" stroke="{WIRE}" stroke-width="1.2" marker-end="url(#arrGray)"/>\n')

    # Wire up from BDH into + node
    s.append(f'  <line x1="636" y1="480" x2="636" y2="454" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(_plus_node(636, 440))

    # Recursive iteration loop-back arrow (cleanly placed along right of amber container)
    s.append(
        f'  <path d="M 636 428 L 636 408 L 792 408 L 792 806 L 642 806" fill="none" '
        f'stroke="{AMBER_STROKE}" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#arrAmber)"/>\n'
    )
    s.append(
        f'  <text x="807" y="518" font-family="{SERIF}" font-size="12" fill="{AMBER_TEXT}" '
        f'text-anchor="middle" transform="rotate(-90 807 518)">repeat for r &lt; R (weight-tied)</text>\n'
    )

    # Wire up out of container (r = R) into Zero-init projection
    s.append(f'  <line x1="636" y1="408" x2="636" y2="334" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(_t(660, 355, f"{_m('h', 't', '(R)')}", 14, INK))

    # Zero-init up-projection box
    s.append(_box(534, 266, 204, 64, AMBER_FILL, AMBER_STROKE))
    s.append(_t(636, 291, "Zero-init residual projection", 14, INK))
    s.append(_t(636, 314, f"&#x394;{_m('z', 't')} = {_m('W', '&#x2191;')} RMSNorm({_m('h', 't', '(R)')}),  {_m('W', '&#x2191;')}|<tspan dy='3' font-size='72%'>0</tspan><tspan dy='-3'> = 0</tspan>", 13, INK))

    # Wire up to top fusion + node (meets blue skip z_t^base!)
    s.append(f'  <line x1="636" y1="266" x2="636" y2="232" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(_plus_node(636, 218))

    # Wire up to Readout + softmax
    s.append(f'  <line x1="636" y1="206" x2="636" y2="162" stroke="{WIRE}" stroke-width="1.35" marker-end="url(#arrGray)"/>\n')
    s.append(_t(648, 176, f"{_m('z&#x302;', 't')} = RMSNorm({_m('z', 't', 'base')} + &#x394;{_m('z', 't')})", 13, INK, anchor="start"))

    # Top Readout + softmax box (exact counterpart to reference diagram!)
    s.append(_box(534, 96, 204, 62, AMBER_FILL, AMBER_STROKE))
    s.append(_t(636, 121, "Readout + softmax", 15, INK, weight="600"))
    s.append(_t(636, 144, f"{_m('p', '&#x398;')}({_m('x', 't+1')} | {_m('x', '1:t')})", 15, INK))

    # Clean MSP uncertainty annotation beside Readout
    s.append(_t(520, 120, "Calibrated uncertainty:", 12, MUTED, anchor="end"))
    s.append(_t(520, 138, f"{_m('u', 't')} = 1 &#x2212; max {_m('p', '&#x398;')}(&#xB7; / {_m('T', '*')})", 12, TEAL_TEXT, anchor="end"))

    # ================= RIGHT COLUMN: O(1) STATES & AUXILIARY HEADS =================
    # 1) Top-Right: Auxiliary heads (MTP-4 + Decoupled K3)
    s.append(_box(856, 96, 234, 62, TEAL_FILL, TEAL_STROKE))
    s.append(_t(973, 121, "Multi-byte heads (MTP-4)", 14, INK, weight="600"))
    s.append(_t(973, 144, f"{_m('p', '&#x398;', '(k)')}({_m('x', 't+k')} | {_m('x', '1:t')}),  {_m('k')} &#x2208; &#x7B;2, 3, 4&#x7D;", 13, TEAL_TEXT))
    s.append(f'  <path d="M 636 188 L 832 188 L 832 127 L 852 127" fill="none" stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n')

    s.append(_box(856, 186, 234, 64, TEAL_FILL, TEAL_STROKE))
    s.append(_t(973, 211, "Decoupled early-exit (K3)", 14, INK, weight="600"))
    s.append(_t(973, 234, f"sg({_m('h', 't', '(1)')}) &#x2192; {_m('p', '&#x398;', '(1)')} (speculative draft)", 13, TEAL_TEXT))
    s.append(
        f'  <path d="M 774 418 L 832 418 L 832 218 L 852 218" fill="none" stroke="{TEAL_STROKE}" '
        f'stroke-width="1.3" stroke-dasharray="4,3" marker-end="url(#arrTeal)"/>\n'
    )
    s.append(_t(824, 286, "stop-grad", 11, TEAL_TEXT, anchor="end", style="italic"))

    # 2) Mid-Right: Constant O(1) DeltaNet Matrix State Cache (exact counterpart to green SWA cache!)
    s.append(_t(973, 466, "Constant O(1) state per step;", 13, TEAL_TEXT))
    s.append(_t(973, 484, "exact chunk / step equivalence", 13, TEAL_TEXT))

    s.append(_box(856, 498, 234, 72, TEAL_FILL, TEAL_STROKE))
    s.append(_t(973, 523, f"Next recurrent state {_m('S', 't', '(r)')}", 14, INK, weight="600"))
    s.append(
        _t(
            973,
            548,
            f"&#x3B1;<tspan dy='3' font-size='72%' font-style='italic'>t</tspan><tspan dy='-3'> </tspan>{_m('S', 't&#x2212;1', '(r)')}"
            f" + &#x3B2;<tspan dy='3' font-size='72%' font-style='italic'>t</tspan><tspan dy='-3'>(</tspan>{_m('v', 't')} &#x2212; &#x3B1;<tspan dy='3' font-size='72%' font-style='italic'>t</tspan><tspan dy='-3'> </tspan>{_m('S', 't&#x2212;1', '(r)')}{_m('k', 't')}){_m('k', 't', '&#x22A4;')}",
            12,
            INK,
        )
    )

    s.append(_box(856, 614, 234, 76, TEAL_FILL, TEAL_STROKE))
    s.append(_t(973, 639, f"Recurrent matrix state {_m('S', 't&#x2212;1', '(r)')}", 14, INK))
    s.append(_t(973, 661, f"{_m('S', '', '(r)')} &#x2208; &#x211D;<tspan dy='-5' font-size='72%' font-style='italic'>H&#xD7;d&#xD7;d</tspan><tspan dy='5'> + causal conv cache</tspan>", 12, MUTED))
    s.append(_t(973, 679, "independent of sequence length T", 12, TEAL_TEXT, style="italic"))

    # Teal wires between GDN and State Cache
    s.append(f'  <path d="M 754 648 L 852 648" fill="none" stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n')
    s.append(f'  <path d="M 856 628 L 758 628" fill="none" stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n')
    s.append(f'  <line x1="973" y1="614" x2="973" y2="574" stroke="{TEAL_STROKE}" stroke-width="1.35" marker-end="url(#arrTeal)"/>\n')

    # 3) Bottom-Right: Chunk-Boundary FP32 SlotCore State Carry (exact counterpart to right-hand orange carry!)
    s.append(_box(856, 326, 234, 74, AMBER_FILL, AMBER_STROKE))
    s.append(_t(973, 350, "FP32 SlotCore commit", 14, INK, weight="600"))
    s.append(_t(973, 372, f"{_m('M', 'c')} = SlotAttn<tspan dy='3' font-size='70%'>FP32</tspan><tspan dy='-3'>(</tspan>{_m('M', 'c&#x2212;1')}, {_m('h', 'end', '(R)')})", 13, INK))
    s.append(_t(973, 391, f"updated at boundary {_m('t')} &#x2261; 0 (mod {_m('B')})", 12, MUTED))

    # Orange arrow from h_t^(R) out to FP32 SlotCore commit
    s.append(f'  <path d="M 636 362 L 852 362" fill="none" stroke="{AMBER_STROKE}" stroke-width="1.5" marker-end="url(#arrAmber)"/>\n')

    # Bottom-right Previous chunk slots box
    s.append(_box(856, 902, 234, 86, AMBER_FILL, AMBER_STROKE))
    s.append(_t(973, 926, "Previous chunk slots", 14, INK, weight="600"))
    s.append(_t(973, 949, f"{_m('M', 'c&#x2212;1')} &#x2208; &#x211D;<tspan dy='-5' font-size='72%' font-style='italic'>S&#xD7;d</tspan><tspan dy='5'>  (</tspan>{_m('S')} = 4 slots)", 13, INK))
    s.append(_t(973, 973, f"{_m('M&#x304;', 'c&#x2212;1')} = (1/{_m('S')}) &#x2211;<tspan dy='3' font-size='70%' font-style='italic'>s</tspan><tspan dy='-3'> </tspan>{_m('M', 'c&#x2212;1,s')}", 13, MUTED))

    # Long orange delay loop on far right: FP32 SlotCore commit -> Previous chunk slots -> Gated merge
    s.append(
        f'  <path d="M 1090 363 L 1114 363 L 1114 945 L 1094 945" fill="none" '
        f'stroke="{AMBER_STROKE}" stroke-width="1.65" marker-end="url(#arrAmber)"/>\n'
    )
    s.append(
        f'  <text x="1128" y="655" font-family="{SERIF}" font-size="13" fill="{AMBER_TEXT}" '
        f'text-anchor="middle" transform="rotate(-90 1128 655)">chunk-boundary delay: c &#x2190; c + 1</text>\n'
    )
    s.append(f'  <line x1="856" y1="945" x2="756" y2="945" stroke="{AMBER_STROKE}" stroke-width="1.65" marker-end="url(#arrAmber)"/>\n')

    s.append("</svg>\n")
    return "".join(s)


def build_benchmarks_1_2m_svg() -> str:
    """Clean white academic paper-style 2-panel benchmark figure (1.20M + O(1) memory)."""
    w, h = 1140, 470
    s = [_svg_open(w, h, "Iso-Parameter 1.20M Benchmarks and O(1) Decode Memory Scaling")]

    # Panel A: Grouped Bar Chart
    s.append(_t(36, 36, "(a) Iso-parameter algorithmic state &amp; memory accuracy (1.20M; 3 seeds)", 15, INK, anchor="start", weight="700"))

    arms = [
        ("RLT-0", "#CBD5E1"),
        ("RLT-1", "#6BA3D6"),
        ("RLT-2", "#E6A259"),
        ("GDN (2.57M)", "#9CA3AF"),
        ("RDFL (1.20M)", BLUE_TEXT),
    ]
    lx = 38
    for name, col in arms:
        s.append(f'  <rect x="{lx}" y="52" width="11" height="11" rx="2" fill="{col}" stroke="#64748B" stroke-width="0.6"/>\n')
        s.append(_t(lx + 16, 62, name, 12, INK, anchor="start"))
        lx += 116 if "2.57" not in name else 132

    x0, y0, pw, ph = 64, 386, 580, 290
    for pct in (0, 25, 50, 75, 100):
        yy = y0 - ph * (pct / 100.0)
        sw = "1.1" if pct == 0 else "0.7"
        col = INK if pct == 0 else "#E5E7EB"
        s.append(f'  <line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="{col}" stroke-width="{sw}"/>\n')
        s.append(_t(x0 - 10, yy + 4, f"{pct}", 12, MUTED, anchor="end"))
    s.append(f'  <line x1="{x0}" y1="{y0 - ph}" x2="{x0}" y2="{y0}" stroke="{INK}" stroke-width="1.1"/>\n')
    s.append(
        f'  <text x="22" y="{y0 - ph / 2:.0f}" font-family="{SERIF}" font-size="13" fill="{INK}" '
        f'text-anchor="middle" transform="rotate(-90 22 {y0 - ph / 2:.0f})">Accuracy (%)</text>\n'
    )

    tasks = [
        ("Parity-64", [50.18, 79.78, 50.15, 100.0, 100.0]),
        ("S5-Perm32", [19.45, 99.93, 20.27, 100.0, 100.0]),
        ("MQAR Overwrite", [22.40, 21.96, 20.92, 23.63, 100.0]),
        ("Multi-Hop ICL", [7.13, 7.32, 7.03, 3.97, 22.72]),
    ]
    grp_w = pw / len(tasks)
    bar_w, gap = 20, 4
    for ti, (tname, vals) in enumerate(tasks):
        gx = x0 + ti * grp_w + 16
        s.append(_t(gx + 2.5 * (bar_w + gap) - 2, y0 + 22, tname, 13, INK))
        for bi, val in enumerate(vals):
            bx = gx + bi * (bar_w + gap)
            bh = max(1.5, ph * (val / 100.0))
            by = y0 - bh
            col = arms[bi][1]
            s.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.6"/>\n')
            if bi == 4:
                lbl = "100" if val >= 99.9 else f"{val:.1f}"
                s.append(_t(bx + bar_w / 2, by - 6, lbl, 11, BLUE_TEXT, weight="700"))

    s.append(_t(354, 442, "Byte-level LM (bpb ↓):  RLT-2 2.4207 · RLT-0 2.4019 · RLT-1 2.3648 · RDFL 2.2896 (-0.1123 vs RLT-0)", 12, MUTED))

    # Panel B: Log-Scale Decode Memory T = 512 .. 32768
    s.append(_t(700, 36, "(b) Decode state memory vs. context length T", 15, INK, anchor="start", weight="700"))

    mx0, my0, mpw, mph = 758, 386, 344, 290
    for lv in (1.0, 10.0, 100.0, 1000.0):
        yy = my0 - mph * (math.log10(lv) / 3.0)
        sw = "1.1" if lv == 1.0 else "0.7"
        col = INK if lv == 1.0 else "#E5E7EB"
        s.append(f'  <line x1="{mx0}" y1="{yy:.1f}" x2="{mx0 + mpw}" y2="{yy:.1f}" stroke="{col}" stroke-width="{sw}"/>\n')
        s.append(_t(mx0 - 10, yy + 4, f"{int(lv)}", 12, MUTED, anchor="end"))
    s.append(f'  <line x1="{mx0}" y1="{my0 - mph}" x2="{mx0}" y2="{my0}" stroke="{INK}" stroke-width="1.1"/>\n')
    s.append(
        f'  <text x="710" y="{my0 - mph / 2:.0f}" font-family="{SERIF}" font-size="13" fill="{INK}" '
        f'text-anchor="middle" transform="rotate(-90 710 {my0 - mph / 2:.0f})">State Memory (MiB, log scale)</text>\n'
    )

    seq_lens = [512, 2048, 8192, 32768]
    kv_mibs = [9.0, 36.0, 144.0, 576.0]
    gdn_mibs = [1.71, 1.71, 1.71, 1.71]
    xs = [mx0 + mpw * (i / 3.0) for i in range(4)]

    def _ly(v: float) -> float:
        return my0 - mph * (math.log10(max(1.0, v)) / 3.0)

    kv_pts = " ".join(f"{x:.1f},{_ly(v):.1f}" for x, v in zip(xs, kv_mibs, strict=True))
    gdn_pts = " ".join(f"{x:.1f},{_ly(v):.1f}" for x, v in zip(xs, gdn_mibs, strict=True))
    s.append(f'  <polyline points="{kv_pts}" fill="none" stroke="{AMBER_STROKE}" stroke-width="2.4"/>\n')
    s.append(f'  <polyline points="{gdn_pts}" fill="none" stroke="{TEAL_STROKE}" stroke-width="2.4"/>\n')
    for x, sl, kv, gd in zip(xs, seq_lens, kv_mibs, gdn_mibs, strict=True):
        s.append(f'  <circle cx="{x:.1f}" cy="{_ly(kv):.1f}" r="4.2" fill="{AMBER_STROKE}"/>\n')
        s.append(f'  <circle cx="{x:.1f}" cy="{_ly(gd):.1f}" r="4.2" fill="{TEAL_STROKE}"/>\n')
        lbl = f"{sl // 1024}K" if sl >= 1024 else str(sl)
        s.append(_t(x, my0 + 22, lbl, 12, INK))

    s.append(_t(mx0 + mpw / 2, my0 + 46, "Sequence Length T (bytes)", 13, MUTED))
    s.append(_t(xs[-1] - 4, _ly(576.0) - 10, "Full-KV Transformer (6L): 576 MiB", 12, AMBER_TEXT, anchor="end", weight="700"))
    s.append(_t(xs[-1] - 4, _ly(1.71) - 12, "RDFL O(1) State: 1.71 MiB (337x smaller)", 12, TEAL_TEXT, anchor="end", weight="700"))

    s.append("</svg>\n")
    return "".join(s)


def build_self_capabilities_svg() -> str:
    """Clean white academic paper-style 3-panel figure for audited self-capabilities V2 & 55M crossover."""
    w, h = 1140, 430
    s = [_svg_open(w, h, "Audited Self-Capabilities V2 and 46x MTP-4 Scale Crossover")]

    # Panel (a): FP32 SlotCore
    s.append(_t(190, 34, "(a) FP32 SlotCore (3-hop graph)", 14, INK, weight="700"))
    ax0, ay0, aw, ah = 54, 340, 276, 250
    for pct in (0, 50, 100):
        yy = ay0 - ah * (pct / 100.0)
        s.append(f'  <line x1="{ax0}" y1="{yy:.1f}" x2="{ax0 + aw}" y2="{yy:.1f}" stroke="{"#1F2937" if pct == 0 else "#E5E7EB"}" stroke-width="{"1.1" if pct == 0 else "0.7"}"/>\n')
        s.append(_t(ax0 - 8, yy + 4, f"{pct}%", 11, MUTED, anchor="end"))
    s.append(f'  <line x1="{ax0}" y1="{ay0 - ah}" x2="{ax0}" y2="{ay0}" stroke="{INK}" stroke-width="1.1"/>\n')
    p1 = [("Base", 68.09, "±54.7", "#CBD5E1"), ("Titans-A", 81.46, "2x state", "#6BA3D6"), ("FP32 Slot", 96.89, "±1.47", TEAL_STROKE)]
    for i, (nm, val, sub, col) in enumerate(p1):
        bx = ax0 + 26 + i * 84
        bh = ah * (val / 100.0)
        s.append(f'  <rect x="{bx}" y="{ay0 - bh:.1f}" width="54" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.7"/>\n')
        s.append(_t(bx + 27, ay0 - bh - 7, f"{val:.1f}%", 12, TEAL_TEXT if i == 2 else INK, weight="700" if i == 2 else "400"))
        s.append(_t(bx + 27, ay0 + 18, nm, 12, INK))
        s.append(_t(bx + 27, ay0 + 34, sub, 11, MUTED))
    s.append(_t(190, 405, "0 FP16 NaN (36/36 runs) · 37.2x lower std", 12, MUTED))

    # Panel (b): Verifier-Gated RSI vs Blind Collapse
    s.append(_t(570, 34, "(b) Bounded RSI: IPT vs. LM drift", 14, INK, weight="700"))
    bx0, by0, bw, bh_max = 426, 340, 290, 250
    for pct in (0, 50, 100):
        yy = by0 - bh_max * (pct / 100.0)
        s.append(f'  <line x1="{bx0}" y1="{yy:.1f}" x2="{bx0 + bw}" y2="{yy:.1f}" stroke="{"#1F2937" if pct == 0 else "#E5E7EB"}" stroke-width="{"1.1" if pct == 0 else "0.7"}"/>\n')
        s.append(_t(bx0 - 8, yy + 4, f"{pct}%", 11, MUTED, anchor="end"))
    s.append(f'  <line x1="{bx0}" y1="{by0 - bh_max}" x2="{bx0}" y2="{by0}" stroke="{INK}" stroke-width="1.1"/>\n')
    p2 = [
        ("Blind RSI1", 63.15, "LM 7.52!", "#DC2626"),
        ("Control", 66.59, "LM 2.94", "#CBD5E1"),
        ("Oracle", 71.26, "LM 2.93", "#6BA3D6"),
        ("AZR RSI3", 80.45, "LM 2.93", BLUE_TEXT),
    ]
    for i, (nm, val, sub, col) in enumerate(p2):
        bx = bx0 + 16 + i * 68
        bh = bh_max * (val / 100.0)
        s.append(f'  <rect x="{bx}" y="{by0 - bh:.1f}" width="46" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.7"/>\n')
        s.append(_t(bx + 23, by0 - bh - 7, f"{val:.1f}%", 11, col if i in (0, 3) else INK, weight="700" if i in (0, 3) else "400"))
        s.append(_t(bx + 23, by0 + 18, nm, 11, INK))
        s.append(_t(bx + 23, by0 + 34, sub, 11, "#DC2626" if i == 0 else MUTED))
    s.append(_t(570, 405, "0 rollbacks · permanent ID anchor + exact 3D KL", 12, MUTED))

    # Panel (c): MTP-4 46x Scale Crossover (1.2M -> 55M)
    s.append(_t(945, 34, "(c) MTP-4 scale crossover (1.2M &#x2192; 55M)", 14, INK, weight="700"))
    cx0, cy0, cw, ch = 804, 340, 294, 250
    # Zero line at y = 278 (allows small negative -0.0005 to dip below zero)
    zy = 290.0
    s.append(f'  <line x1="{cx0}" y1="{zy}" x2="{cx0 + cw}" y2="{zy}" stroke="{INK}" stroke-width="1.1"/>\n')
    s.append(f'  <line x1="{cx0}" y1="{cy0 - ch}" x2="{cx0}" y2="{cy0}" stroke="{INK}" stroke-width="1.1"/>\n')
    for dv in (0.0, 0.01, 0.02, 0.03):
        yy = zy - (dv / 0.035) * 185.0
        if dv > 0:
            s.append(f'  <line x1="{cx0}" y1="{yy:.1f}" x2="{cx0 + cw}" y2="{yy:.1f}" stroke="#E5E7EB" stroke-width="0.7"/>\n')
        s.append(_t(cx0 - 8, yy + 4, f"+{dv:.2f}" if dv > 0 else "0.00", 11, MUTED, anchor="end"))
    s.append(_t(cx0 - 8, 92, "&#x394;bpb", 11, MUTED, anchor="end"))

    p3 = [
        ("1.2M LM", 0.0323, "#DC2626"),
        ("55M LM", 0.0010, AMBER_STROKE),
        ("55M SFT", -0.0005, TEAL_STROKE),
    ]
    for i, (nm, dv, col) in enumerate(p3):
        bx = cx0 + 28 + i * 88
        bh = max(4.0, abs(dv) / 0.035 * 185.0)
        by = zy - bh if dv >= 0 else zy
        s.append(f'  <rect x="{bx}" y="{by:.1f}" width="54" height="{bh:.1f}" fill="{col}" stroke="#374151" stroke-width="0.7"/>\n')
        lbl_y = by - 7 if dv >= 0 else by + bh + 15
        sign = "+" if dv >= 0 else ""
        s.append(_t(bx + 27, lbl_y, f"{sign}{dv:.4f}", 11, col, weight="700"))
        s.append(_t(bx + 27, cy0 + 18, nm, 12, INK))
    s.append(_t(945, 405, "32x lower t+1 tax at 55M · t+2 = 51.6% at 7K steps", 12, MUTED))

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
