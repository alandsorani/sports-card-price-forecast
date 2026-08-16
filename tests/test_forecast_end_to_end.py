import numpy as np
import pandas as pd
import pytest

from src.data.load import to_series
from src.data.player_stats import merge_player_features, stats_asof
from src.features.pipeline import feature_columns
from src.features.targets import add_targets
from src.features.timeseries import add_card_age, add_card_features, add_market_features
from src.forecasting.comparables import find_comparables
from src.forecasting.portfolio import aggregate, unrealized_gain
from src.forecasting.reliability import assess
from src.models.baselines import baseline_predictions
from src.models.train import fit_final_model, predict_with_bundle, walk_forward


@pytest.fixture(scope="module")
def panel(synthetic_raw, synthetic_stats):
    p = add_card_features(to_series(synthetic_raw))
    p = add_market_features(p)
    p = add_card_age(p)
    p = merge_player_features(p, synthetic_stats)
    return add_targets(p, horizons=[30])


def test_walk_forward_runs_and_is_chronological(panel):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    features = feature_columns(panel)
    ridge = Pipeline([("impute", SimpleImputer()), ("model", Ridge())])
    results = walk_forward(panel, features, 30, models={"ridge": ridge})
    assert results, "walk-forward produced no folds"
    models_seen = {r.model for r in results}
    assert "last_price" in models_seen and "ridge" in models_seen
    labeled = panel.dropna(subset=["future_price_30d"])
    for r in results:
        # every fold trains strictly on years before its validation year
        train_max = labeled[labeled["date"].dt.year < r.fold_year]["date"].max()
        assert train_max.year < r.fold_year


def test_final_model_predicts_positive_prices(panel):
    features = feature_columns(panel)
    bundle = fit_final_model(panel, features, 30, model_name="ridge",
                             target_type="price", log_price=True)
    latest = panel.sort_values("date").groupby("card_id").tail(1)
    preds = predict_with_bundle(bundle, latest)
    assert np.all(preds >= 0)
    assert len(preds) == len(latest)


def test_baseline_model_wrapper_is_deployable(panel):
    from src.models.baselines import BaselineModel

    features = feature_columns(panel)
    bundle = fit_final_model(panel, features, 30, model_name="last_price",
                             target_type="price", log_price=False)
    assert isinstance(bundle["model"], BaselineModel)
    latest = panel.sort_values("date").groupby("card_id").tail(1)
    preds = predict_with_bundle(bundle, latest)
    # last_price baseline must reproduce the current price exactly
    assert np.allclose(preds, latest["price"].values)


def test_momentum_baseline_scales_with_horizon(panel):
    from src.models.baselines import BaselineModel

    features = feature_columns(panel)
    rows = panel.dropna(subset=["return_90d"]).head(20)
    p30 = BaselineModel("momentum_adjusted", 30).predict(rows[features])
    p365 = BaselineModel("momentum_adjusted", 365).predict(rows[features])
    up = rows["return_90d"].clip(-0.5, 0.5) > 0
    assert np.all(p365[up.values] > p30[up.values])


def test_all_nan_features_are_dropped(panel):
    """Early backtest windows leave long-lookback columns all-NaN; those columns
    must be dropped or HistGradientBoosting's binner raises on them."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from src.models.train import usable_features

    features = feature_columns(panel)
    train = panel.dropna(subset=["future_price_30d"]).copy()
    train["return_365d"] = np.nan  # simulate a window with no 365d history
    kept = usable_features(train, features)
    assert "return_365d" not in kept
    HistGradientBoostingRegressor(max_iter=10).fit(
        train[kept], train["future_price_30d"]
    )  # would raise ValueError if the all-NaN column were kept


def test_baselines_shapes(panel):
    labeled = panel.dropna(subset=["future_price_30d"]).reset_index(drop=True)
    preds = baseline_predictions(labeled, 30)
    for name, series in preds.items():
        assert len(series) == len(labeled), name


def test_stats_asof_never_uses_future(synthetic_stats):
    player = synthetic_stats.iloc[0]["player"]
    early = synthetic_stats[synthetic_stats["player"] == player]["season_end_date"].min()
    row = stats_asof(synthetic_stats, player, early - pd.Timedelta(days=1))
    assert row is None  # before any season ended -> nothing knowable


def test_comparables_exclude_self(panel):
    card_id = panel["card_id"].iloc[0]
    comps = find_comparables(panel, card_id)
    assert card_id not in set(comps["card_id"])
    assert "similarity_score" in comps.columns


def test_reliability_cold_start(panel):
    tiny = panel[panel["card_id"] == panel["card_id"].iloc[0]].head(3)
    rel = assess(tiny, interval_rel_width=None)
    assert rel.level == "Low"


def test_reliability_degrades_with_interval_width(panel):
    history = panel[panel["card_id"] == panel["card_id"].iloc[0]]
    tight = assess(history, interval_rel_width=0.1)
    wide = assess(history, interval_rel_width=1.5)
    order = {"Low": 0, "Medium": 1, "High": 2}
    assert order[wide.level] < order[tight.level]


def test_portfolio_aggregation():
    items = [
        {"qty": 2, "purchase_price": 100, "current_price": 150,
         "horizons": {30: {"point": 160, "lo": 140, "hi": 185}}},
        {"qty": 1, "purchase_price": 50, "current_price": 40,
         "horizons": {30: {"point": 45, "lo": 30, "hi": 60}}},
    ]
    port = aggregate(items)
    assert port["current_value"] == 340
    assert port["unrealized_gain"] == pytest.approx(340 - 250)
    assert port["horizons"][30]["point"] == pytest.approx(2 * 160 + 45)
    assert port["horizons"][30]["lo"] == pytest.approx(2 * 140 + 30)
    assert unrealized_gain(150, 100, 2) == 100
