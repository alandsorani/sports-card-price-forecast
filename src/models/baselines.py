"""Naive baselines every ML model must beat (spec section 19)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def last_price(panel: pd.DataFrame) -> pd.Series:
    """Baseline 1: the future price equals today's price."""
    return panel["price"].copy()


def moving_median(panel: pd.DataFrame, days: int) -> pd.Series:
    """Baseline 2/3: trailing `days`-day median price per card."""
    out = pd.Series(np.nan, index=panel.index)
    for _, g in panel.groupby("card_id", sort=False):
        med = (
            g.set_index("date")["price"].rolling(f"{days}D", min_periods=1).median()
        )
        out.loc[g.index] = med.values
    return out


def momentum_adjusted(panel: pd.DataFrame, horizon: int) -> pd.Series:
    """Baseline 4: extrapolate the trailing 90d return over the horizon,
    clipped to +/-50% to avoid absurd extrapolations."""
    r90 = panel["return_90d"].clip(-0.5, 0.5)
    scaled = (1 + r90.fillna(0)) ** (horizon / 90.0)
    return panel["price"] * scaled


def baseline_predictions(panel: pd.DataFrame, horizon: int) -> dict[str, pd.Series]:
    return {
        "last_price": last_price(panel),
        "moving_median_30d": moving_median(panel, 30),
        "moving_median_90d": moving_median(panel, 90),
        "momentum_adjusted": momentum_adjusted(panel, horizon),
    }


BASELINE_NAMES = ("last_price", "moving_median_30d", "moving_median_90d",
                  "momentum_adjusted")


class BaselineModel:
    """sklearn-shaped wrapper so a baseline can be deployed like any model.

    Deployed when no ML model beats it out-of-sample — the spec requires
    selecting on genuine performance, not on model sophistication.

    `moving_median_*` needs a card's trailing window, which a single row cannot
    provide; at predict time it falls back to that row's current price, which
    equals the trailing median whenever the latest observation is the median.
    The deployed choice and this caveat are recorded in the model bundle.
    """

    def __init__(self, name: str, horizon: int):
        if name not in BASELINE_NAMES:
            raise ValueError(f"unknown baseline: {name}")
        self.name = name
        self.horizon = horizon

    def fit(self, X: pd.DataFrame, y=None) -> "BaselineModel":
        return self  # baselines are rules, nothing to estimate

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.name == "momentum_adjusted":
            r90 = X["return_90d"].clip(-0.5, 0.5).fillna(0)
            return (X["price"] * (1 + r90) ** (self.horizon / 90.0)).to_numpy()
        return X["price"].to_numpy()
