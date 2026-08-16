import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics
from src.models.intervals import (conformal_quantiles, evaluate_coverage,
                                  interval, relative_residuals)


def test_metrics_known_values():
    actual = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 330.0])
    m = compute_metrics(actual, pred, price_now=np.array([100.0, 210.0, 250.0]))
    assert m["mae"] == pytest.approx(50 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert m["medae"] == pytest.approx(10.0)
    # directions: actual vs now = (0, -10, +50); pred vs now = (+10, -20, +80)
    # row 1 didn't move -> excluded; rows 2,3 both correct
    assert m["directional_accuracy"] == pytest.approx(1.0)


def test_metrics_ignore_nans():
    m = compute_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    assert m["n"] == 1


def test_conformal_coverage_on_known_noise():
    rng = np.random.default_rng(0)
    pred = rng.uniform(50, 500, 4000)
    actual = pred * (1 + rng.normal(0, 0.1, 4000))
    calib_r = relative_residuals(actual[:2000], pred[:2000])
    q = conformal_quantiles(calib_r, alpha=0.10)
    cov = evaluate_coverage(actual[2000:], pred[2000:], q)
    assert 0.87 <= cov["coverage"] <= 0.93


def test_interval_never_negative():
    lo, hi = interval(10.0, (-1.5, 0.5))
    assert lo == 0.0 and hi == 15.0


def test_conformal_needs_enough_residuals():
    with pytest.raises(ValueError):
        conformal_quantiles(np.zeros(5))
