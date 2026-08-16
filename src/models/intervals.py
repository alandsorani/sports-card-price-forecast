"""Split-conformal prediction intervals in relative-error space.

Approach: collect walk-forward (out-of-sample) residuals expressed as relative
errors (actual/pred - 1). The interval for a new prediction p is
[p * (1 + q_lo), p * (1 + q_hi)] where q_lo, q_hi are the alpha/2 and
1 - alpha/2 empirical quantiles with the standard conformal finite-sample
correction. Relative errors are used because card price scales span orders of
magnitude. Coverage is evaluated on held-out folds (`evaluate_coverage`).
"""
from __future__ import annotations

import numpy as np


def relative_residuals(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred) & (pred > 0)
    return actual[mask] / pred[mask] - 1


def conformal_quantiles(rel_resid: np.ndarray, alpha: float = 0.10) -> tuple[float, float]:
    r = np.asarray(rel_resid, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20:
        raise ValueError(f"Need >=20 out-of-sample residuals for intervals, got {n}")
    # Finite-sample-corrected empirical quantiles.
    lo_q = np.quantile(r, max(0.0, np.floor(alpha / 2 * (n + 1)) / n))
    hi_q = np.quantile(r, min(1.0, np.ceil((1 - alpha / 2) * (n + 1)) / n))
    return float(lo_q), float(hi_q)


def interval(pred: float | np.ndarray, quantiles: tuple[float, float]) -> tuple:
    lo_q, hi_q = quantiles
    lo = np.maximum(np.asarray(pred) * (1 + lo_q), 0.0)
    hi = np.asarray(pred) * (1 + hi_q)
    return lo, hi


def evaluate_coverage(actual: np.ndarray, pred: np.ndarray,
                      quantiles: tuple[float, float]) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[mask], pred[mask]
    lo, hi = interval(pred, quantiles)
    inside = (actual >= lo) & (actual <= hi)
    width = (hi - lo) / np.where(pred > 0, pred, np.nan)
    return {
        "n": int(len(actual)),
        "coverage": float(inside.mean()) if len(actual) else float("nan"),
        "mean_relative_width": float(np.nanmean(width)) if len(actual) else float("nan"),
    }
