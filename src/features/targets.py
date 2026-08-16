"""Forecast targets: future price and future return at each horizon.

Methodology (documented per spec section 10): for a row at date T and horizon H,
the target is the card's observation closest to T+H among those within
T+H +/- tol, where tol = max(10 days, 15% of H) (src.config). If no observation
falls in the window the target is NaN and the row is excluded from training for
that horizon. `target_gap_days` records the actual distance used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def add_targets(df: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    horizons = horizons or config.HORIZONS
    df = df.sort_values(["card_id", "date"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("card_id", sort=False):
        g = g.copy()
        dates = g["date"].values
        prices = g["price"].values
        for h in horizons:
            tol = np.timedelta64(config.target_tolerance_days(h), "D")
            target_dates = dates + np.timedelta64(h, "D")
            lo = np.searchsorted(dates, target_dates - tol, side="left")
            hi = np.searchsorted(dates, target_dates + tol, side="right")
            future_price = np.full(len(g), np.nan)
            gap = np.full(len(g), np.nan)
            for i in range(len(g)):
                if hi[i] <= lo[i]:
                    continue
                window = dates[lo[i]:hi[i]]
                offsets = np.abs((window - target_dates[i]).astype("timedelta64[D]").astype(int))
                best = int(np.argmin(offsets))
                future_price[i] = prices[lo[i] + best]
                gap[i] = offsets[best]
            g[f"future_price_{h}d"] = future_price
            g[f"future_return_{h}d"] = future_price / prices - 1
            g[f"target_gap_days_{h}d"] = gap
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
