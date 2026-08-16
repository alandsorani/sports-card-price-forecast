"""Plotly figures for the app.

Mark specs follow the shared design system: 2px lines with round caps, >=8px
markers carrying a 2px surface ring, area fills as a ~10% wash, hairline solid
gridlines, selective direct labels (never one per point), and a legend whenever
two or more series share a plot.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.visualization.theme import (INK_MUTED, INK_PRIMARY, SERIES_1, SERIES_2,
                                     SURFACE, apply, rgba)

RANGE_SELECTOR = dict(
    buttons=[
        dict(count=30, label="30D", step="day", stepmode="backward"),
        dict(count=90, label="90D", step="day", stepmode="backward"),
        dict(count=180, label="180D", step="day", stepmode="backward"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=3, label="3Y", step="year", stepmode="backward"),
        dict(step="all", label="All"),
    ],
    bgcolor=SURFACE,
    activecolor=rgba(SERIES_1, 0.30),
    bordercolor="rgba(255,255,255,0.10)",
    borderwidth=1,
    font=dict(color=INK_MUTED, size=11),
    # Sits above the legend row; the page heading names the card, so these
    # figures carry no in-plot title to collide with.
    x=0, y=1.16, xanchor="left", yanchor="bottom",
)


def history_figure(history: pd.DataFrame) -> go.Figure:
    """Single series, so no legend box; the page heading names the card."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["date"], y=history["price"], mode="lines",
        name="Observed price",
        line=dict(width=2, color=SERIES_1, shape="linear"),
        hovertemplate="$%{y:,.2f}<extra></extra>",
    ))
    _label_last_point(fig, history["date"], history["price"], SERIES_1)
    apply(fig, y_title="Price (USD)")
    fig.update_layout(showlegend=False, margin=dict(t=76),
                      xaxis=dict(rangeselector=RANGE_SELECTOR))
    return fig


def _label_last_point(fig: go.Figure, x, y, color: str) -> None:
    """Direct-label the endpoint only — labels work because they are sparing."""
    if len(x) == 0:
        return
    fig.add_trace(go.Scatter(
        x=[list(x)[-1]], y=[list(y)[-1]], mode="markers+text",
        marker=dict(size=9, color=color,
                    line=dict(width=2, color=SURFACE)),  # 2px surface ring
        text=[f"  ${list(y)[-1]:,.0f}"], textposition="middle right",
        textfont=dict(color=INK_PRIMARY, size=12),
        showlegend=False, hoverinfo="skip", cliponaxis=False,
    ))


def forecast_figure(history: pd.DataFrame, forecast) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["date"], y=history["price"], mode="lines", name="Observed",
        line=dict(width=2, color=SERIES_1),
        hovertemplate="$%{y:,.2f}<extra>Observed</extra>",
    ))
    if not forecast.horizons:
        apply(fig, y_title="Price (USD)")
        fig.update_layout(margin=dict(t=76),
                          xaxis=dict(rangeselector=RANGE_SELECTOR))
        return fig

    asof = forecast.asof
    hs = sorted(forecast.horizons)
    xs = [asof] + [asof + pd.Timedelta(days=h) for h in hs]
    pts = [forecast.current_price] + [forecast.horizons[h]["point"] for h in hs]
    los = [forecast.current_price] + [forecast.horizons[h]["lo"] for h in hs]
    his = [forecast.current_price] + [forecast.horizons[h]["hi"] for h in hs]

    fig.add_trace(go.Scatter(
        x=xs + xs[::-1], y=his + los[::-1], fill="toself",
        fillcolor=rgba(SERIES_2, 0.10),  # ~10% wash, never a saturated block
        # Explicit mode: Plotly draws markers by default on short traces, which
        # would scatter stray dots at every polygon vertex.
        mode="lines", line=dict(width=0), name="90% prediction interval",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=pts, mode="lines+markers", name="Forecast",
        line=dict(width=2, color=SERIES_2, dash="dash"),  # dash = projection
        marker=dict(size=8, color=SERIES_2, line=dict(width=2, color=SURFACE)),
        hovertemplate="$%{y:,.2f}<extra>Forecast</extra>",
    ))
    # Label only the far endpoint, not every horizon.
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[pts[-1]], mode="text",
        text=[f"  ${pts[-1]:,.0f}"], textposition="middle right",
        textfont=dict(color=INK_PRIMARY, size=12),
        showlegend=False, hoverinfo="skip", cliponaxis=False,
    ))
    fig.add_vline(x=asof.to_pydatetime(), line_width=1, line_color=INK_MUTED,
                  line_dash="solid", opacity=0.5)
    apply(fig, y_title="Price (USD)", height=460)
    fig.update_layout(margin=dict(t=76),
                      xaxis=dict(rangeselector=RANGE_SELECTOR))
    return fig


def player_overlay_figure(history: pd.DataFrame, stats: pd.DataFrame,
                          player: str, title: str) -> go.Figure:
    """Card price and player scoring as SMALL MULTIPLES sharing one x-axis.

    Deliberately not a dual-axis chart: aligning two y-scales on one plot
    invents a correlation the data does not contain. Two stacked panels let the
    reader compare timing without the chart asserting a relationship.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=("Card price (USD)", "Player points per game (season)"),
    )
    fig.add_trace(go.Scatter(
        x=history["date"], y=history["price"], mode="lines", name="Card price",
        line=dict(width=2, color=SERIES_1),
        hovertemplate="$%{y:,.2f}<extra>Card price</extra>",
    ), row=1, col=1)

    ps = stats[stats["player"] == player].sort_values("season_end_date")
    if not ps.empty:
        fig.add_trace(go.Scatter(
            x=ps["season_end_date"], y=ps["points_per_game"], mode="lines+markers",
            name="Points per game",
            line=dict(width=2, color=SERIES_2),
            marker=dict(size=8, color=SERIES_2, line=dict(width=2, color=SURFACE)),
            hovertemplate="%{y:.1f} PPG<extra>Season</extra>",
        ), row=2, col=1)

    apply(fig, title=title, height=520)
    fig.update_layout(showlegend=False, hovermode="x")
    for ann in fig.layout.annotations:  # subplot titles: left-align to match
        ann.font.update(size=12, color=INK_MUTED)
        ann.update(x=0, xanchor="left")
    fig.update_yaxes(tickprefix="$", row=1, col=1)
    return fig


def market_figure(market: pd.DataFrame) -> go.Figure:
    """Two series on one axis — both are percentages, so no second scale."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=market["date"], y=market["market_return_30d"], mode="lines",
        name="Median 30-day return", line=dict(width=2, color=SERIES_1),
        hovertemplate="%{y:.1%}<extra>Median 30d return</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=market["date"], y=market["market_momentum"], mode="lines",
        name="Momentum (13-obs median)", line=dict(width=2, color=SERIES_2),
        hovertemplate="%{y:.1%}<extra>Momentum</extra>",
    ))
    apply(fig, y_title="Return")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(margin=dict(t=76),
                      xaxis=dict(rangeselector=RANGE_SELECTOR))
    return fig
