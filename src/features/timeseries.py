"""Time-aligned features. Every feature at row (card, T) uses only rows with
date <= T for that card, or dates <= T across cards for market features.
See docs/leakage_audit.md for the per-feature audit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACKS = [7, 30, 60, 90, 180, 365]


def _asof_lag(group: pd.DataFrame, days: int) -> pd.Series:
    """Price at the closest observation at least `days` before each row."""
    dates = group["date"]
    target = dates - pd.Timedelta(days=days)
    # searchsorted over the group's own sorted dates: index of last date <= target
    idx = np.searchsorted(dates.values, target.values, side="right") - 1
    out = np.full(len(group), np.nan)
    valid = idx >= 0
    out[valid] = group["price"].values[idx[valid]]
    return pd.Series(out, index=group.index)


def add_card_features(series: pd.DataFrame) -> pd.DataFrame:
    """Add per-card rolling/lag features. `series` is the to_series() output."""
    df = series.sort_values(["card_id", "date"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("card_id", sort=False):
        g = g.copy()
        for days in LOOKBACKS:
            lag = _asof_lag(g, days)
            g[f"price_{days}d_ago"] = lag
            g[f"return_{days}d"] = g["price"] / lag - 1
        logret = np.log(g["price"] / g["price"].shift(1))
        dt_days = g["date"].diff().dt.days
        daily_logret = logret / dt_days.replace(0, np.nan)
        g_index = g.set_index("date")
        for days in [30, 90, 180]:
            vol = (
                daily_logret.set_axis(g_index.index)
                .rolling(f"{days}D")
                .std()
            )
            g[f"volatility_{days}d"] = (vol * np.sqrt(365)).values
        expanding_max = g["price"].cummax()
        expanding_min = g["price"].cummin()
        g["historical_high"] = expanding_max
        g["historical_low"] = expanding_min
        g["distance_from_high"] = g["price"] / expanding_max - 1
        g["distance_from_low"] = g["price"] / expanding_min - 1
        g["price_momentum"] = g["return_30d"]
        g["price_acceleration"] = g["return_30d"] - (
            g["price_30d_ago"] / g["price_90d_ago"] - 1
        )
        counts = pd.Series(1.0, index=g_index.index).rolling("90D").sum()
        g["obs_count_90d"] = counts.values
        g["days_since_last_obs"] = dt_days
        g["obs_number"] = np.arange(1, len(g) + 1)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-card market index built from data available at each date.

    The market index at date D is the median 30-day return across all cards
    whose latest observation is on or before D — never future rows.
    """
    daily = (
        df.dropna(subset=["return_30d"])
        .groupby("date")["return_30d"]
        .agg(market_return_30d="median", market_vol_cross="std", market_volume="size")
        .sort_index()
    )
    # cumulative history only: expanding stats over past dates
    daily["market_momentum"] = daily["market_return_30d"].rolling(13, min_periods=4).median()
    daily = daily.reset_index()
    df = df.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(df, daily, on="date", direction="backward")
    return merged.sort_values(["card_id", "date"]).reset_index(drop=True)


def add_card_age(df: pd.DataFrame) -> pd.DataFrame:
    """Card age from the set year (Jan 1 of `year`; exact release dates are
    rarely public, and this approximation is documented in the README)."""
    year = pd.to_numeric(df["year"], errors="coerce")
    release = pd.to_datetime(year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    df = df.copy()
    df["card_age_days"] = (df["date"] - release).dt.days
    df["card_age_years"] = df["card_age_days"] / 365.25
    df["is_rookie_card"] = (
        df["rookie"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    ).astype(int)
    return df


FEATURE_COLUMNS = (
    [f"price_{d}d_ago" for d in LOOKBACKS]
    + [f"return_{d}d" for d in LOOKBACKS]
    + [f"volatility_{d}d" for d in [30, 90, 180]]
    + [
        "historical_high",
        "historical_low",
        "distance_from_high",
        "distance_from_low",
        "price_momentum",
        "price_acceleration",
        "obs_count_90d",
        "days_since_last_obs",
        "obs_number",
        "market_return_30d",
        "market_vol_cross",
        "market_volume",
        "market_momentum",
        "card_age_days",
        "card_age_years",
        "is_rookie_card",
    ]
)

PLAYER_FEATURE_COLUMNS = [
    "prev_season_points_per_game",
    "prev_season_rebounds_per_game",
    "prev_season_assists_per_game",
    "prev_season_field_goal_percentage",
    "prev_season_three_point_percentage",
    "prev_season_free_throw_percentage",
    "prev_season_games_played",
    "prev_season_all_star",
    "card_year_relative_to_debut",
]
