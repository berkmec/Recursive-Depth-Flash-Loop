"""Shared academic LaTeX/TikZ-style SVG primitives and palette constants."""

from __future__ import annotations

SERIF = "'CMU Serif', 'STIX Two Text', 'Nimbus Roman', 'Liberation Serif', Georgia, serif"
SANS = "Inter, 'Liberation Sans', -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif"

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


def svg_open(w: int, h: int, label: str) -> str:
    """Open an academic white-canvas SVG with reusable arrowheads."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-label="{label}">\n'
        "  <defs>\n"
        f'    <marker id="arrGray" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="6.5" markerHeight="6.5" orient="auto">'
        f'<path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="{WIRE}"/></marker>\n'
        f'    <marker id="arrBlue" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="6.5" markerHeight="6.5" orient="auto">'
        f'<path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="{BLUE_STROKE}"/></marker>\n'
        f'    <marker id="arrAmber" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="6.5" markerHeight="6.5" orient="auto">'
        f'<path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="{AMBER_STROKE}"/></marker>\n'
        f'    <marker id="arrTeal" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="6.5" markerHeight="6.5" orient="auto">'
        f'<path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="{TEAL_STROKE}"/></marker>\n'
        "  </defs>\n"
        f'  <rect x="0" y="0" width="{w}" height="{h}" fill="#FFFFFF"/>\n'
    )


def box(
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
    """Emit a rounded academic diagram node box."""
    d_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d_attr}/>\n'
    )


def plus_node(cx: float, cy: float, r: float = 11.5) -> str:
    """Emit a residual addition circle-plus node."""
    return (
        f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="#FFFFFF" stroke="{WIRE}" '
        f'stroke-width="1.2"/>\n'
        f'  <line x1="{cx - 5.5}" y1="{cy}" x2="{cx + 5.5}" y2="{cy}" '
        f'stroke="{INK}" stroke-width="1.2"/>\n'
        f'  <line x1="{cx}" y1="{cy - 5.5}" x2="{cx}" y2="{cy + 5.5}" '
        f'stroke="{INK}" stroke-width="1.2"/>\n'
    )


def txt(
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
    """Emit a crisp SVG text element."""
    st = f' font-style="{style}"' if style != "normal" else ""
    return (
        f'  <text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{st}>{markup}</text>\n'
    )


def math_var(var: str, sub: str = "", sup: str = "") -> str:
    """Academic italic math variable with crisp subscript and superscript tspans."""
    out = f'<tspan font-style="italic">{var}</tspan>'
    if sub:
        out += (
            f'<tspan dy="3.5" font-size="72%" font-style="italic">{sub}</tspan>'
            f'<tspan dy="-3.5" font-size="100%">&#8203;</tspan>'
        )
    if sup:
        out += (
            f'<tspan dy="-5.5" font-size="72%" font-style="italic">{sup}</tspan>'
            f'<tspan dy="5.5" font-size="100%">&#8203;</tspan>'
        )
    return out
