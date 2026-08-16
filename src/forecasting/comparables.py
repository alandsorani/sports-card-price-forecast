"""Comparable-card similarity (spec section 33). Label output 'Comparable
Cards' — similarity is heuristic, these are never identical cards."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _latest_rows(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values("date").groupby("card_id").tail(1).reset_index(drop=True)


def find_comparables(panel: pd.DataFrame, card_id: str, top_n: int = 5) -> pd.DataFrame:
    """Score cards by shared metadata and similar price/volatility level."""
    latest = _latest_rows(panel)
    target = latest[latest["card_id"] == card_id]
    if target.empty:
        return pd.DataFrame()
    t = target.iloc[0]
    pool = latest[latest["card_id"] != card_id].copy()
    if pool.empty:
        return pool

    score = np.zeros(len(pool))
    score += 3.0 * (pool["player"] == t["player"]).values
    score += 1.5 * (pool["set"] == t["set"]).values
    score += 1.0 * (pool["grade"].astype(str) == str(t["grade"])).values
    score += 1.0 * (pool["grading_company"].astype(str) == str(t["grading_company"])).values
    year_diff = (pd.to_numeric(pool["year"], errors="coerce")
                 - pd.to_numeric(pd.Series([t["year"]]), errors="coerce").iloc[0]).abs()
    score += np.clip(1.5 - 0.25 * year_diff.fillna(6).values, 0, 1.5)
    score += 1.0 * (pool["is_rookie_card"] == t["is_rookie_card"]).values
    log_ratio = np.abs(np.log10(pool["price"].values / max(t["price"], 1e-9)))
    score += np.clip(1.5 - log_ratio, 0, 1.5)  # same order of magnitude
    if "volatility_90d" in pool.columns and pd.notna(t.get("volatility_90d")):
        vol_diff = (pool["volatility_90d"] - t["volatility_90d"]).abs().fillna(1.0)
        score += np.clip(1.0 - vol_diff.values, 0, 1.0)

    pool["similarity_score"] = np.round(score, 2)
    cols = ["card_id", "card_name", "player", "year", "set", "grade",
            "grading_company", "price", "date", "similarity_score"]
    return (pool.sort_values("similarity_score", ascending=False)
            .head(top_n)[[c for c in cols if c in pool.columns]]
            .reset_index(drop=True))
