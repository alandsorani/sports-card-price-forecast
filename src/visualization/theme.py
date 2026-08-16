"""Chart design tokens and the shared Plotly template.

Light, warm palette. The three categorical slots were checked with the palette
validator against the #fdfcfa chart surface: lightness band, chroma floor,
colorblind separation, and normal-vision separation all pass. Aqua sits at
2.75:1 contrast, below the 3:1 bar, so it is never the only way to read a value
(direct labels and a table view accompany every chart).
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Categorical slots (fixed order, never cycled, never assigned by rank).
SERIES_1 = "#2a78d6"  # blue, observed history
SERIES_2 = "#eb6834"  # orange, forecast / projection
SERIES_3 = "#1baf7a"  # aqua, third series

# Status palette (reserved; always paired with a word, never color alone).
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

# Readable text steps for status labels on the light surface.
TEXT_GOOD = "#006300"
TEXT_WARNING = "#8a5a00"
TEXT_CRITICAL = "#c0392b"

# Surfaces and ink.
SURFACE = "#fdfcfa"       # card and chart surface
PAGE = "#f6f4f0"          # warm page plane
INK_PRIMARY = "#1c1a17"
INK_SECONDARY = "#55524c"
INK_MUTED = "#85817a"
GRIDLINE = "#eceae4"
AXIS = "#d8d5cd"
HAIRLINE = "rgba(28,26,23,0.10)"

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'

TEMPLATE_NAME = "cardviz"


def _register() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_STACK, size=13, color=INK_SECONDARY),
        title=dict(font=dict(size=15, color=INK_PRIMARY), x=0, xanchor="left",
                   pad=dict(b=12)),
        margin=dict(l=56, r=24, t=56, b=44),
        colorway=[SERIES_1, SERIES_2, SERIES_3],
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=HAIRLINE,
                        font=dict(family=FONT_STACK, color=INK_PRIMARY, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left",
                    x=0, font=dict(color=INK_SECONDARY, size=12),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            showgrid=False,
            linecolor=AXIS, linewidth=1, ticks="outside", ticklen=4,
            tickcolor=AXIS, tickfont=dict(color=INK_MUTED, size=11),
            showspikes=True, spikemode="across", spikethickness=1,
            spikecolor=AXIS, spikedash="solid",
        ),
        yaxis=dict(
            # Hairline, solid, recessive. Never dashed.
            showgrid=True, gridcolor=GRIDLINE, gridwidth=1, griddash="solid",
            zeroline=False, linecolor=AXIS, linewidth=1,
            tickfont=dict(color=INK_MUTED, size=11),
        ),
    )
    pio.templates[TEMPLATE_NAME] = template


_register()


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def apply(fig: go.Figure, *, title: str | None = None,
          y_title: str | None = None, height: int = 420) -> go.Figure:
    """Apply the shared template plus per-figure chrome."""
    fig.update_layout(template=TEMPLATE_NAME, height=height)
    if title:
        fig.update_layout(title=dict(text=title))
    if y_title:
        fig.update_yaxes(title=dict(text=y_title,
                                    font=dict(color=INK_MUTED, size=11)))
    return fig
