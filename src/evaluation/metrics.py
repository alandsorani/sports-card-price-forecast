"""Forecast accuracy metrics (spec section 25)."""
from __future__ import annotations

import numpy as np


def compute_metrics(actual: np.ndarray, pred: np.ndarray,
                    price_now: np.ndarray | None = None) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[mask], pred[mask]
    if len(actual) == 0:
        return {}
    err = actual - pred
    abs_err = np.abs(err)
    out = {
        "n": int(len(actual)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "medae": float(np.median(abs_err)),
        "mape": float((abs_err / np.abs(actual)).mean() * 100),
        "smape": float((2 * abs_err / (np.abs(actual) + np.abs(pred))).mean() * 100),
    }
    ss_res = float((err ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    out["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if price_now is not None:
        price_now = np.asarray(price_now, dtype=float)[mask]
        actual_dir = np.sign(actual - price_now)
        pred_dir = np.sign(pred - price_now)
        moved = actual_dir != 0
        out["directional_accuracy"] = (
            float((actual_dir[moved] == pred_dir[moved]).mean()) if moved.any() else float("nan")
        )
    return out
