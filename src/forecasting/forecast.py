"""Generate forecasts for a card as of its latest observation, with intervals,
reliability, and a plain-language explanation of feature influence."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src import config
from src.forecasting.reliability import Reliability, assess
from src.models.intervals import interval
from src.models.train import load_bundle, predict_with_bundle

MIN_OBS_FOR_FORECAST = 10


@dataclass
class CardForecast:
    card_id: str
    card_name: str
    asof: pd.Timestamp
    current_price: float
    horizons: dict = field(default_factory=dict)  # h -> {point, lo, hi}
    reliability: Reliability | None = None
    limited_history: bool = False
    message: str = ""


def bundle_path(horizon: int):
    return config.MODELS_DIR / f"model_{horizon}d.pkl"


def forecast_card(panel: pd.DataFrame, card_id: str) -> CardForecast:
    history = panel[panel["card_id"] == card_id].sort_values("date")
    if history.empty:
        raise KeyError(f"card_id not found: {card_id}")
    latest = history.iloc[-1]
    out = CardForecast(
        card_id=card_id,
        card_name=latest["card_name"],
        asof=latest["date"],
        current_price=float(latest["price"]),
    )
    if len(history) < MIN_OBS_FOR_FORECAST:
        out.limited_history = True
        out.message = (
            f"There isn't enough history for this card yet, only "
            f"{len(history)} observations. Rather than show a falsely precise "
            "number, we've skipped the forecast. Take a look at Similar Cards "
            "for context instead."
        )
        out.reliability = assess(history, interval_rel_width=None)
        return out

    for h in config.HORIZONS:
        path = bundle_path(h)
        if not path.exists():
            out.message = "Models not trained yet. Run `python train.py` first."
            continue
        bundle = load_bundle(path)
        row = latest.to_frame().T
        point = float(predict_with_bundle(bundle, row)[0])
        lo, hi = interval(point, bundle["conformal_quantiles"])
        rel_width = (float(hi) - float(lo)) / point if point > 0 else None
        out.horizons[h] = {
            "point": point,
            "lo": float(lo),
            "hi": float(hi),
            "model": bundle["model_name"],
            "trained_through": bundle["trained_through"],
            # Reliability is per-horizon: the data-quality signals are shared,
            # but interval width grows with the horizon, so a single rating
            # would let the 365d uncertainty mask a solid 30d forecast.
            "reliability": assess(history, interval_rel_width=rel_width),
        }
    # Headline rating covers data quality only (no horizon-specific width).
    out.reliability = assess(history, interval_rel_width=None)
    return out


def explain_forecast(horizon: int, latest_row: pd.Series, top_n: int = 6) -> pd.DataFrame:
    """Permutation-importance-based explanation for the deployed model.

    Importances are computed once during training (on validation data, stored in
    the bundle); here they are paired with the card's current feature values.
    """
    path = bundle_path(horizon)
    if not path.exists():
        return pd.DataFrame()
    bundle = load_bundle(path)
    imp = bundle.get("feature_importance")
    if imp is None:
        return pd.DataFrame()
    rows = []
    for feat, weight in imp[:top_n]:
        rows.append({
            "feature": feat,
            "importance": round(weight, 4),
            "card_value": latest_row.get(feat),
        })
    return pd.DataFrame(rows)
