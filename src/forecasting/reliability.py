"""Rule-based forecast reliability rating (High / Medium / Low).

This is deliberately NOT a calibrated probability — the spec forbids quoting
one unless validated. Each rule inspects data sufficiency, freshness, and
uncertainty; the worst bucket wins the headline with reasons attached.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Reliability:
    level: str  # "High" | "Medium" | "Low"
    reasons: list[str]


def assess(card_history: pd.DataFrame, *, interval_rel_width: float | None,
           asof: pd.Timestamp | None = None) -> Reliability:
    asof = asof or card_history["date"].max()
    reasons: list[str] = []
    score = 2  # start at High, demote on findings

    n_obs = len(card_history)
    if n_obs < 10:
        score = 0
        reasons.append(f"Only {n_obs} price observations to learn from.")
    elif n_obs < 30:
        score = min(score, 1)
        reasons.append(f"Not much history yet, just {n_obs} observations.")

    staleness = (asof - card_history["date"].max()).days
    if staleness > 90:
        score = 0
        reasons.append(f"The most recent price is {staleness} days old.")
    elif staleness > 30:
        score = min(score, 1)
        reasons.append(f"The most recent price is {staleness} days old.")

    recent = card_history[card_history["date"] >= asof - pd.Timedelta(days=90)]
    if len(recent) < 4:
        score = min(score, 1)
        reasons.append(f"Only {len(recent)} observations in the last 90 days.")

    vol = card_history.get("volatility_90d")
    if vol is not None and pd.notna(vol.iloc[-1]) and vol.iloc[-1] > 0.6:
        score = min(score, 1)
        reasons.append(f"This card's price moves a lot "
                       f"({vol.iloc[-1]:.0%} a year recently).")

    if interval_rel_width is not None:
        if interval_rel_width > 1.0:
            score = 0
            reasons.append("The likely range is wider than the price itself.")
        elif interval_rel_width > 0.5:
            score = min(score, 1)
            reasons.append("The likely range is wide, over half the forecast.")

    if not reasons:
        reasons.append("Plenty of history, recent data, and a reasonably "
                       "tight range.")
    return Reliability(["Low", "Medium", "High"][score], reasons)
