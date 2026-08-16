"""Train forecasting models for every horizon.

Steps per horizon:
1. Build the feature panel from data/raw/sports_card_prices.csv.
2. Walk-forward validate baselines + model zoo, for price and return targets,
   raw and log1p price encodings.
3. Pick the configuration with the best mean out-of-sample MAE, but only if it
   beats the best baseline; otherwise the baseline is deployed.
4. Compute split-conformal interval quantiles from out-of-sample residuals.
5. Fit the winner on all labeled data, attach permutation importance, save to
   models/model_{h}d.pkl, and write the comparison table to reports/.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.features.pipeline import build_panel, feature_columns, save_panel
from src.models.intervals import conformal_quantiles, evaluate_coverage
from src.models.baselines import BASELINE_NAMES
from src.models.train import (
    fit_final_model,
    results_frame,
    save_bundle,
    walk_forward,
    year_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="*", default=config.HORIZONS)
    args = parser.parse_args()

    panel = build_panel()
    if panel["is_synthetic"].any():
        print("*** WARNING: panel contains SYNTHETIC demo rows. Results describe "
              "the synthetic random walk, NOT any real card market. ***")
    save_panel(panel)
    features = feature_columns(panel)
    print(f"Panel: {len(panel)} rows, {panel['card_id'].nunique()} cards, "
          f"{panel['date'].min().date()} .. {panel['date'].max().date()}, "
          f"{len(features)} features")

    all_results = []
    for horizon in args.horizons:
        target_col = f"future_price_{horizon}d"
        labeled = panel.dropna(subset=[target_col])
        if len(labeled) < 100:
            print(f"[{horizon}d] Only {len(labeled)} labeled rows — skipping "
                  "(need at least 100). Add more historical data.")
            continue

        configs = [("price", False), ("price", True), ("return", False)]
        best = None  # (mae, results, target_type, log_price)
        baseline_results = None
        for target_type, log_price in configs:
            res = walk_forward(panel, features, horizon, target_type=target_type,
                               log_price=log_price,
                               include_baselines=baseline_results is None)
            if baseline_results is None:
                baseline_results = [r for r in res if r.model in BASELINE_NAMES]
            model_res = [r for r in res if r.model not in BASELINE_NAMES]
            frame = results_frame(model_res)
            if frame.empty:
                continue
            frame_tagged = frame.assign(target_type=target_type, log_price=log_price)
            all_results.append(frame_tagged)
            by_model = frame.groupby("model")["mae"].mean()
            top_model, top_mae = by_model.idxmin(), by_model.min()
            if best is None or top_mae < best[0]:
                best = (top_mae, model_res, target_type, log_price, top_model)

        all_results.append(results_frame(baseline_results))
        base_frame = results_frame(baseline_results)
        base_by_model = base_frame.groupby("model")["mae"].mean()
        best_baseline, best_baseline_mae = base_by_model.idxmin(), base_by_model.min()

        if best is None:
            print(f"[{horizon}d] no model results; skipping")
            continue
        top_mae, model_res, target_type, log_price, top_model = best
        print(f"[{horizon}d] best model: {top_model} (target={target_type}, "
              f"log={log_price}) MAE={top_mae:.2f} | best baseline: "
              f"{best_baseline} MAE={best_baseline_mae:.2f}")

        beat_baseline = top_mae < best_baseline_mae
        if not beat_baseline:
            # Selection is on genuine out-of-sample performance, so the baseline
            # is the better model and gets deployed.
            print(f"[{horizon}d] no ML model beat the {best_baseline} baseline "
                  f"({top_mae:.2f} vs {best_baseline_mae:.2f}) — deploying the "
                  "baseline.")
            top_model, target_type, log_price = best_baseline, "price", False

        # Conformal quantiles from the winner's out-of-sample relative
        # residuals; the last fold is excluded and used for honest coverage.
        rel_resid = _oos_relative_residuals(panel, features, horizon,
                                            top_model, target_type, log_price)
        quantiles = conformal_quantiles(rel_resid, alpha=config.INTERVAL_ALPHA)
        coverage = _holdout_coverage(panel, features, horizon, top_model,
                                     target_type, log_price, quantiles)
        print(f"[{horizon}d] conformal q=({quantiles[0]:+.2%}, {quantiles[1]:+.2%}), "
              f"holdout coverage={coverage.get('coverage', float('nan')):.1%} "
              f"(target {1 - config.INTERVAL_ALPHA:.0%}, n={coverage.get('n')})")

        bundle = fit_final_model(panel, features, horizon, model_name=top_model,
                                 target_type=target_type, log_price=log_price)
        bundle["conformal_quantiles"] = quantiles
        bundle["holdout_coverage"] = coverage
        bundle["beat_baseline"] = bool(beat_baseline)
        bundle["deployed_kind"] = "ml" if beat_baseline else "baseline"
        bundle["best_baseline"] = {"name": best_baseline, "mae": float(best_baseline_mae)}
        bundle["cv_mae"] = float(top_mae)
        bundle["is_synthetic_data"] = bool(panel["is_synthetic"].any())
        bundle["feature_importance"] = _importance(bundle, panel, features, horizon)
        save_bundle(bundle, config.MODELS_DIR / f"model_{horizon}d.pkl")
        print(f"[{horizon}d] saved models/model_{horizon}d.pkl")

    if all_results:
        table = pd.concat(all_results, ignore_index=True)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = config.REPORTS_DIR / "model_comparison.csv"
        table.to_csv(out, index=False)
        print(f"Wrote {out}")


def _last_fold_split(panel: pd.DataFrame, horizon: int):
    target_col = f"future_price_{horizon}d"
    data = panel.dropna(subset=[target_col, "price"]).reset_index(drop=True)
    folds = year_folds(data["date"])
    if not folds:
        raise SystemExit("Not enough distinct years for walk-forward validation.")
    last = folds[-1]
    return (data[data["date"].dt.year < last], data[data["date"].dt.year == last],
            target_col)


def _fit_predict(train, val, features, horizon, model_name, target_type, log_price):
    from src.models.train import (build_model, usable_features, _encode_target,
                                  _decode_prediction)

    model = build_model(model_name, horizon)
    features = usable_features(train, features)
    y = _encode_target(train[f"future_price_{horizon}d"], train["price"],
                       target_type, log_price)
    model.fit(train[features], y)
    raw = model.predict(val[features])
    pred = _decode_prediction(raw, val["price"], target_type, log_price)
    return np.clip(pred, 1e-9, None), model


def _oos_relative_residuals(panel, features, horizon, model_name, target_type,
                            log_price) -> np.ndarray:
    """Relative residuals pooled over every walk-forward fold except the last
    (the last is reserved to evaluate coverage honestly)."""
    target_col = f"future_price_{horizon}d"
    data = panel.dropna(subset=[target_col, "price"]).reset_index(drop=True)
    folds = year_folds(data["date"])
    rel = []
    for fold_year in folds[:-1] or folds:
        train = data[data["date"].dt.year < fold_year]
        val = data[data["date"].dt.year == fold_year]
        if len(train) < 50 or val.empty:
            continue
        pred, _ = _fit_predict(train, val, features, horizon, model_name,
                               target_type, log_price)
        rel.append(val[target_col].values / pred - 1)
    if not rel:
        raise SystemExit("No folds available for conformal calibration.")
    return np.concatenate(rel)


def _holdout_coverage(panel, features, horizon, model_name, target_type,
                      log_price, quantiles) -> dict:
    train, val, target_col = _last_fold_split(panel, horizon)
    if len(train) < 50 or val.empty:
        return {}
    pred, _ = _fit_predict(train, val, features, horizon, model_name,
                           target_type, log_price)
    return evaluate_coverage(val[target_col].values, pred, quantiles)


def _importance(bundle, panel, features, horizon) -> list[tuple[str, float]]:
    """Permutation importance on the last walk-forward fold's validation year."""
    if bundle["model_name"] in BASELINE_NAMES:
        return []  # a baseline is a fixed rule; permuting features is meaningless
    train, val, target_col = _last_fold_split(panel, horizon)
    if val.empty or len(train) < 50:
        return []
    pred, model = _fit_predict(train, val, features, horizon,
                               bundle["model_name"], bundle["target_type"],
                               bundle["log_price"])
    from src.models.train import _encode_target, usable_features

    # _fit_predict drops all-NaN columns, so score the same column set.
    features = usable_features(train, features)
    y_val = _encode_target(val[target_col], val["price"], bundle["target_type"],
                           bundle["log_price"])
    try:
        imp = permutation_importance(model, val[features], y_val, n_repeats=5,
                                     random_state=0, n_jobs=-1)
    except Exception:
        return []
    order = np.argsort(-imp.importances_mean)
    return [(features[i], float(imp.importances_mean[i])) for i in order]


if __name__ == "__main__":
    main()
