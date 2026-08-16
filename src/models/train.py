"""Model zoo, walk-forward validation, and final model training.

Validation NEVER uses random splits. Folds are expanding windows over calendar
years present in the data (spec section 22): train on all rows whose date falls
before the fold year, validate on the fold year. The most recent year is held
out as the final test period.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluation.metrics import compute_metrics
from src.models.baselines import BASELINE_NAMES, BaselineModel, baseline_predictions


def model_zoo(random_state: int = 0) -> dict[str, object]:
    zoo = {
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                    random_state=random_state,
                )),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=random_state)),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=random_state
        ),
    }
    try:  # optional extras; used only if installed
        from xgboost import XGBRegressor

        zoo["xgboost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            random_state=random_state, n_jobs=-1,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor

        zoo["lightgbm"] = LGBMRegressor(random_state=random_state, verbose=-1)
    except ImportError:
        pass
    return zoo


def usable_features(train: pd.DataFrame, features: list[str]) -> list[str]:
    """Drop columns that are entirely missing in this training set.

    Early walk-forward folds and early backtest windows have no history for the
    long lookbacks (e.g. return_365d), leaving all-NaN columns that carry no
    information and that HistGradientBoosting's binner cannot fit.
    """
    return [c for c in features if train[c].notna().any()]


def build_model(model_name: str, horizon: int):
    """Instantiate an ML model from the zoo, or a deployable baseline."""
    if model_name in BASELINE_NAMES:
        return BaselineModel(model_name, horizon)
    return model_zoo()[model_name]


@dataclass
class FoldResult:
    model: str
    horizon: int
    fold_year: int
    target_type: str  # "price" or "return"
    log_price: bool
    metrics: dict = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0
    residuals: np.ndarray | None = None  # dollar-space residuals (actual - pred)


def year_folds(dates: pd.Series, min_train_years: int = 2) -> list[int]:
    years = sorted(dates.dt.year.unique())
    return [y for y in years[min_train_years:]]


def _encode_target(y_price: pd.Series, price_now: pd.Series, target_type: str,
                   log_price: bool) -> pd.Series:
    if target_type == "return":
        return y_price / price_now - 1
    if log_price:
        return np.log1p(y_price)
    return y_price


def _decode_prediction(pred: np.ndarray, price_now: pd.Series, target_type: str,
                       log_price: bool) -> np.ndarray:
    if target_type == "return":
        return price_now.values * (1 + pred)
    if log_price:
        return np.expm1(pred)
    return pred


def walk_forward(
    panel: pd.DataFrame,
    features: list[str],
    horizon: int,
    *,
    target_type: str = "price",
    log_price: bool = True,
    models: dict[str, object] | None = None,
    include_baselines: bool = True,
) -> list[FoldResult]:
    """Expanding-window walk-forward validation for one horizon."""
    target_col = f"future_price_{horizon}d"
    data = panel.dropna(subset=[target_col, "price"]).reset_index(drop=True)
    results: list[FoldResult] = []
    if data.empty:
        return results
    models = models if models is not None else model_zoo()

    for fold_year in year_folds(data["date"]):
        train = data[data["date"].dt.year < fold_year]
        val = data[data["date"].dt.year == fold_year]
        if len(train) < 50 or len(val) == 0:
            continue

        if include_baselines:
            for name, preds in baseline_predictions(val, horizon).items():
                res = FoldResult(name, horizon, fold_year, "n/a", False,
                                 n_train=len(train), n_val=len(val))
                res.metrics = compute_metrics(
                    val[target_col].values, preds.values, val["price"].values
                )
                res.residuals = val[target_col].values - preds.values
                results.append(res)

        y_train = _encode_target(train[target_col], train["price"], target_type, log_price)
        fold_features = usable_features(train, features)
        for name, template in models.items():
            model = pickle.loads(pickle.dumps(template))  # fresh clone per fold
            model.fit(train[fold_features], y_train)
            raw_pred = model.predict(val[fold_features])
            pred = _decode_prediction(raw_pred, val["price"], target_type, log_price)
            pred = np.clip(pred, 0.0, None)
            res = FoldResult(name, horizon, fold_year, target_type, log_price,
                             n_train=len(train), n_val=len(val))
            res.metrics = compute_metrics(val[target_col].values, pred, val["price"].values)
            res.residuals = val[target_col].values - pred
            results.append(res)
    return results


def results_frame(results: list[FoldResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.model,
                "horizon": r.horizon,
                "fold_year": r.fold_year,
                "target_type": r.target_type,
                "log_price": r.log_price,
                "n_train": r.n_train,
                "n_val": r.n_val,
                **r.metrics,
            }
        )
    return pd.DataFrame(rows)


def fit_final_model(
    panel: pd.DataFrame,
    features: list[str],
    horizon: int,
    *,
    model_name: str,
    target_type: str = "price",
    log_price: bool = True,
) -> dict:
    """Fit the chosen model on all labeled data for deployment."""
    target_col = f"future_price_{horizon}d"
    data = panel.dropna(subset=[target_col, "price"])
    model = build_model(model_name, horizon)
    features = usable_features(data, features)
    y = _encode_target(data[target_col], data["price"], target_type, log_price)
    model.fit(data[features], y)
    return {
        "model": model,
        "model_name": model_name,
        "horizon": horizon,
        "features": features,
        "target_type": target_type,
        "log_price": log_price,
        "trained_through": str(data["date"].max().date()),
        "n_train": len(data),
    }


def predict_with_bundle(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    raw = bundle["model"].predict(rows[bundle["features"]])
    pred = _decode_prediction(raw, rows["price"], bundle["target_type"], bundle["log_price"])
    return np.clip(pred, 0.0, None)


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
