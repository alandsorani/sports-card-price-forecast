"""Leakage audit (spec section 24).

`AUDIT` documents every feature's information cutoff. `mutation_test` is the
programmatic check: mutate all rows strictly after a cutoff date, rebuild
features, and assert rows at or before the cutoff are unchanged. It runs in the
test suite (tests/test_leakage.py) and via `python -m src.evaluation.leakage`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import to_series
from src.data.player_stats import merge_player_features
from src.features.timeseries import (
    FEATURE_COLUMNS,
    add_card_age,
    add_card_features,
    add_market_features,
)

AUDIT = [
    # feature (pattern), definition, cutoff, risk, test
    ("price", "current observed price at T", "<= T", "none", "mutation_test"),
    ("price_{d}d_ago", "closest observation >= d days before T", "<= T", "none", "mutation_test"),
    ("return_{d}d", "price / price_{d}d_ago - 1", "<= T", "none", "mutation_test"),
    ("volatility_{d}d", "annualized std of daily log returns over trailing d days", "<= T", "none", "mutation_test"),
    ("historical_high/low", "expanding max/min of past prices", "<= T", "none", "mutation_test"),
    ("distance_from_high/low", "price relative to expanding extremes", "<= T", "none", "mutation_test"),
    ("price_momentum/acceleration", "trailing 30d return and its change", "<= T", "none", "mutation_test"),
    ("obs_count_90d, days_since_last_obs, obs_number", "trailing observation bookkeeping", "<= T", "none", "mutation_test"),
    ("market_*", "cross-card medians merged as-of T (backward)", "<= T", "low: verify merge direction", "mutation_test"),
    ("card_age_*, is_rookie_card", "static metadata + T - release year", "static", "none", "n/a"),
    ("prev_season_*", "latest season with season_end_date <= T (merge_asof backward)", "season end <= T", "low: verify season_end_date is correct", "mutation_test"),
    ("future_price_{h}d / future_return_{h}d", "TARGETS: only compared against, never features", "> T by design", "must never enter feature list", "feature-list check in tests"),
]


def audit_table() -> pd.DataFrame:
    return pd.DataFrame(AUDIT, columns=["feature", "definition", "cutoff", "risk", "test"])


def _build(series_input: pd.DataFrame, stats: pd.DataFrame | None) -> pd.DataFrame:
    panel = add_card_features(to_series(series_input))
    panel = add_market_features(panel)
    panel = add_card_age(panel)
    if stats is not None:
        panel = merge_player_features(panel, stats)
    return panel


def mutation_test(raw: pd.DataFrame, cutoff: pd.Timestamp,
                  stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return rows (<= cutoff) whose features changed after future data mutated.

    An empty result means no future information leaks into past features.
    """
    base = _build(raw, stats)
    mutated_raw = raw.copy()
    future = mutated_raw["date"] > cutoff
    mutated_raw.loc[future, "price"] = mutated_raw.loc[future, "price"] * 3.7 + 123.0
    mutated = _build(mutated_raw, stats)

    cols = [c for c in FEATURE_COLUMNS if c in base.columns]
    key = ["card_id", "date"]
    a = base[base["date"] <= cutoff].set_index(key)[cols].sort_index()
    b = mutated[mutated["date"] <= cutoff].set_index(key)[cols].sort_index()
    a, b = a.align(b, join="inner")
    diff = ~np.isclose(a.values, b.values, equal_nan=True)
    bad = a[diff.any(axis=1)]
    return bad.reset_index()


if __name__ == "__main__":
    from src.data.load import load_prices

    raw = load_prices()
    cutoff = raw["date"].quantile(0.5)
    bad = mutation_test(raw, cutoff)
    print(audit_table().to_string(index=False))
    print(f"\nMutation test at cutoff {cutoff.date()}: "
          f"{'PASS — no leakage detected' if bad.empty else f'FAIL — {len(bad)} rows leaked'}")
