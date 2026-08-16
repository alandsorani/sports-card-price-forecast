"""Presentation layer: page chrome, CSS, and small HTML components.

Values and labels wear text tokens; color never carries meaning on its own, so
every status badge pairs its color with a word.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

# Pages may import this module first, so the project root goes on the path here
# rather than relying on common.py having been imported already.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.visualization.theme import (GRIDLINE, INK_MUTED, INK_PRIMARY,  # noqa: E402
                                     INK_SECONDARY, PAGE, STATUS_CRITICAL,
                                     STATUS_GOOD, STATUS_WARNING, SURFACE,
                                     TEXT_CRITICAL, TEXT_GOOD, TEXT_WARNING)

CSS = f"""
<style>
:root {{
  --surface: {SURFACE};
  --page: {PAGE};
  --ink-1: {INK_PRIMARY};
  --ink-2: {INK_SECONDARY};
  --ink-3: {INK_MUTED};
  --hairline: rgba(28,26,23,0.09);
  --good: {STATUS_GOOD};
  --warning: {STATUS_WARNING};
  --critical: {STATUS_CRITICAL};
  --text-good: {TEXT_GOOD};
  --text-warning: {TEXT_WARNING};
  --text-critical: {TEXT_CRITICAL};
  --radius: 18px;
  --shadow: 0 1px 2px rgba(28,26,23,0.05), 0 4px 14px rgba(28,26,23,0.05);
}}

.stApp {{ background: var(--page); }}

.block-container {{ padding-top: 3rem; padding-bottom: 5rem; max-width: 1180px; }}

[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

h1, h2, h3 {{ letter-spacing: -0.018em; color: var(--ink-1); }}
h1 {{ font-size: 2.05rem !important; font-weight: 660 !important;
  margin-bottom: .2rem !important; }}
h2 {{ font-size: 1.2rem !important; font-weight: 620 !important;
  margin-top: 2.25rem !important; }}
h3 {{ font-size: 1.02rem !important; font-weight: 620 !important; }}

.page-sub {{ color: var(--ink-2); font-size: .96rem; line-height: 1.6;
  margin: 0 0 1.9rem; max-width: 66ch; }}

section[data-testid="stSidebar"] {{ background: var(--surface);
  border-right: 1px solid var(--hairline); }}
[data-testid="stSidebarNav"] {{ padding-top: .75rem; }}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; }}

/* --- Stat tiles ------------------------------------------------------- */
.tile-row {{ display: grid; gap: 14px; margin: 0 0 1.4rem;
  grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); }}
.tile {{ background: var(--surface); border: 1px solid var(--hairline);
  border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow); }}
.tile-label {{ color: var(--ink-3); font-size: .76rem; font-weight: 500;
  margin-bottom: .4rem; }}
.tile-value {{ color: var(--ink-1); font-size: 1.6rem; font-weight: 640;
  line-height: 1.15; }}   /* proportional figures, not tabular */
.tile-delta {{ font-size: .82rem; margin-top: .3rem; font-weight: 560; }}
.tile-delta.up {{ color: var(--text-good); }}
.tile-delta.down {{ color: var(--text-critical); }}
.tile-delta.flat {{ color: var(--ink-3); }}
.tile-note {{ color: var(--ink-3); font-size: .76rem; margin-top: .35rem; }}

/* --- Forecast cards --------------------------------------------------- */
.fc-row {{ display: grid; gap: 14px; margin: .3rem 0 1.6rem;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
.fc {{ background: var(--surface); border: 1px solid var(--hairline);
  border-radius: var(--radius); padding: 18px; box-shadow: var(--shadow); }}
.fc-h {{ color: var(--ink-3); font-size: .76rem; margin-bottom: .35rem; }}
.fc-v {{ color: var(--ink-1); font-size: 1.75rem; font-weight: 640; line-height: 1.1; }}
.fc-d {{ font-size: .84rem; font-weight: 560; margin-top: .25rem; }}
.fc-d.up {{ color: var(--text-good); }}
.fc-d.down {{ color: var(--text-critical); }}
.fc-d.flat {{ color: var(--ink-3); }}
.fc-i {{ color: var(--ink-2); font-size: .78rem; margin-top: .7rem;
  padding-top: .7rem; border-top: 1px solid var(--hairline);
  font-variant-numeric: tabular-nums; }}

/* --- Status badge (dot + word, never color alone) --------------------- */
.badge {{ display: inline-flex; align-items: center; gap: .4rem;
  font-size: .75rem; font-weight: 620; padding: .28rem .6rem;
  border-radius: 999px; margin-top: .7rem; }}
.badge .dot {{ width: .48rem; height: .48rem; border-radius: 50%; }}
.badge.high {{ color: var(--text-good); background: rgba(12,163,12,.10); }}
.badge.high .dot {{ background: var(--good); }}
.badge.medium {{ color: var(--text-warning); background: rgba(250,178,25,.18); }}
.badge.medium .dot {{ background: var(--warning); }}
.badge.low {{ color: var(--text-critical); background: rgba(208,59,59,.10); }}
.badge.low .dot {{ background: var(--critical); }}

/* --- Charts, tables, controls ----------------------------------------- */
.js-plotly-plot {{ border: 1px solid var(--hairline); border-radius: var(--radius);
  box-shadow: var(--shadow); background: var(--surface); }}

[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums;
  border-radius: 14px; overflow: hidden; }}

[data-testid="stExpander"] {{ border: 1px solid var(--hairline) !important;
  border-radius: 14px !important; background: var(--surface); box-shadow: var(--shadow); }}

[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
  background: var(--surface); border-radius: 10px; }}
div[data-baseweb="select"] > div {{ background: var(--surface);
  border-radius: 10px; border-color: var(--hairline); }}
.stButton button, [data-testid="stFormSubmitButton"] button {{
  border-radius: 999px; padding: .45rem 1.15rem; font-weight: 580; }}

[data-testid="stAlert"] {{ border-radius: 14px; border: 1px solid var(--hairline); }}
/* Secondary ink, not muted: captions carry real explanation and need to be
   comfortably readable against the warm page. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  color: var(--ink-2); font-size: .84rem; line-height: 1.6; }}
hr {{ border-color: {GRIDLINE}; }}
</style>
"""


def page(title: str, subtitle: str | None = None) -> None:
    """Standard page header. Call once at the top of every page."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<p class="page-sub">{html.escape(subtitle)}</p>',
                    unsafe_allow_html=True)


def _delta_class(delta: float | None) -> str:
    if delta is None:
        return "flat"
    if delta > 0.0005:
        return "up"
    if delta < -0.0005:
        return "down"
    return "flat"


def tiles(items: list[dict]) -> None:
    """Render a KPI row. Each item: {label, value, delta?, note?}."""
    cards = []
    for it in items:
        parts = [f'<div class="tile-label">{html.escape(str(it["label"]))}</div>',
                 f'<div class="tile-value">{html.escape(str(it["value"]))}</div>']
        if it.get("delta") is not None:
            cls = _delta_class(it["delta"])
            arrow = {"up": "↑", "down": "↓", "flat": "→"}[cls]
            parts.append(
                f'<div class="tile-delta {cls}">{arrow} {it["delta"]:+.1%}</div>')
        if it.get("note"):
            parts.append(f'<div class="tile-note">{html.escape(str(it["note"]))}</div>')
        cards.append(f'<div class="tile">{"".join(parts)}</div>')
    st.markdown(f'<div class="tile-row">{"".join(cards)}</div>',
                unsafe_allow_html=True)


def badge_html(level: str) -> str:
    cls = level.lower()
    return (f'<span class="badge {cls}"><span class="dot"></span>'
            f'{html.escape(level)} confidence</span>')


def forecast_cards(forecast) -> None:
    """One card per horizon: value, change vs today, interval, reliability."""
    cards = []
    for h in sorted(forecast.horizons):
        v = forecast.horizons[h]
        change = v["point"] / forecast.current_price - 1 if forecast.current_price else None
        cls = _delta_class(change)
        arrow = {"up": "↑", "down": "↓", "flat": "→"}[cls]
        rel = v.get("reliability")
        cards.append(
            f'<div class="fc">'
            f'<div class="fc-h">In {h} days</div>'
            f'<div class="fc-v">${v["point"]:,.2f}</div>'
            f'<div class="fc-d {cls}">{arrow} {change:+.1%} vs today</div>'
            f'<div class="fc-i">Likely between<br>'
            f'${v["lo"]:,.2f} and ${v["hi"]:,.2f}</div>'
            f'{badge_html(rel.level) if rel else ""}'
            f'</div>'
        )
    st.markdown(f'<div class="fc-row">{"".join(cards)}</div>',
                unsafe_allow_html=True)


def synthetic_banner(panel) -> None:
    """Mark the session as demo data.

    Kept deliberately small and out of the way: a sidebar footnote rather than
    a banner on every page. It stays because the app prices real, named cards,
    and these figures are generated, so a viewer has no other way to tell.
    """
    if panel["is_synthetic"].any():
        st.sidebar.caption("Demo data. Figures are generated, not real prices.")
