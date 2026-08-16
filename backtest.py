"""True historical simulation (spec section 23).

For each backtest date D (every `--step` days across the panel's date range):
1. Truncate the RAW data to observations dated <= D.
2. Rebuild features from that truncated data only.
3. Train on rows whose target date also falls on or before D (so the training
   labels themselves were observable at D).
4. Forecast every card's latest row as of D.
5. Store forecasts; after the loop, compare against the actual future prices.

Writes reports/backtest_forecasts.csv and reports/backtest_metrics.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config
from src.data.load import load_prices, to_series
from src.data.player_stats import load_player_stats, merge_player_features
from src.evaluation.metrics import compute_metrics
from src.features.pipeline import feature_columns
from src.features.targets import add_targets
from src.features.timeseries import add_card_age, add_card_features, add_market_features
from src.models.train import (_decode_prediction, _encode_target, build_model,
                              usable_features)


def build_panel_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    panel = add_card_features(to_series(raw))
    panel = add_market_features(panel)
    panel = add_card_age(panel)
    panel = merge_player_features(panel, load_player_stats())
    return add_targets(panel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hist_gradient_boosting")
    parser.add_argument("--horizons", type=int, nargs="*", default=config.HORIZONS)
    parser.add_argument("--step", type=int, default=90, help="days between backtest dates")
    parser.add_argument("--min-train-rows", type=int, default=150)
    parser.add_argument("--log-price", action="store_true", default=True)
    args = parser.parse_args()

    raw_full = load_prices()
    if raw_full["is_synthetic"].any():
        print("*** WARNING: SYNTHETIC demo data — backtest results describe the "
              "synthetic random walk, not any real market. ***")
    full_panel = build_panel_from_raw(raw_full)
    dates = pd.date_range(
        raw_full["date"].min() + pd.Timedelta(days=540),
        raw_full["date"].max() - pd.Timedelta(days=30),
        freq=f"{args.step}D",
    )
    if len(dates) == 0:
        raise SystemExit("Date range too short for a backtest.")

    forecasts = []
    for asof in dates:
        raw = raw_full[raw_full["date"] <= asof]
        if raw.empty:
            continue
        panel = build_panel_from_raw(raw)
        features = feature_columns(panel)
        latest = panel.sort_values("date").groupby("card_id").tail(1)
        for horizon in args.horizons:
            target_col = f"future_price_{horizon}d"
            # training labels must have matured by `asof`
            train = panel.dropna(subset=[target_col, "price"])
            train = train[train["date"] + pd.Timedelta(days=horizon) <= asof]
            if len(train) < args.min_train_rows:
                continue
            model = build_model(args.model, horizon)
            fit_features = usable_features(train, features)
            y = _encode_target(train[target_col], train["price"], "price", args.log_price)
            model.fit(train[fit_features], y)
            pred = _decode_prediction(model.predict(latest[fit_features]),
                                      latest["price"], "price", args.log_price)
            pred = np.clip(pred, 0.0, None)
            for card_id, row_date, row_price, p in zip(
                latest["card_id"], latest["date"], latest["price"], pred
            ):
                forecasts.append({
                    "asof": asof, "card_id": card_id, "obs_date": row_date,
                    "horizon": horizon, "current_price": row_price, "forecast": p,
                })

    fc = pd.DataFrame(forecasts)
    if fc.empty:
        raise SystemExit("No forecasts generated — not enough matured training data.")

    # Match each forecast to the actual future observation in the full panel.
    actuals = full_panel[["card_id", "date"] + [f"future_price_{h}d" for h in args.horizons]]
    fc = fc.merge(actuals, left_on=["card_id", "obs_date"],
                  right_on=["card_id", "date"], how="left")
    fc["actual"] = np.nan
    for h in args.horizons:
        m = fc["horizon"] == h
        fc.loc[m, "actual"] = fc.loc[m, f"future_price_{h}d"]
    fc = fc.drop(columns=["date"] + [f"future_price_{h}d" for h in args.horizons])

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fc.to_csv(config.REPORTS_DIR / "backtest_forecasts.csv", index=False)

    rows = []
    for h in args.horizons:
        sub = fc[(fc["horizon"] == h) & fc["actual"].notna()]
        if sub.empty:
            continue
        m = compute_metrics(sub["actual"].values, sub["forecast"].values,
                            sub["current_price"].values)
        rows.append({"model": args.model, "horizon": h, **m})
        print(f"[{h}d] n={m['n']} MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} "
              f"R2={m['r2']:.3f} dir_acc={m.get('directional_accuracy', float('nan')):.1%}")
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "backtest_metrics.csv", index=False)
    print(f"Wrote {config.REPORTS_DIR / 'backtest_forecasts.csv'} and backtest_metrics.csv")


if __name__ == "__main__":
    main()
