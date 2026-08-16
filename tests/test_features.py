import numpy as np
import pandas as pd

from src.data.load import to_series
from src.features.targets import add_targets
from src.features.timeseries import add_card_features, add_market_features


def _tiny_series():
    """Hand-built weekly series with known values."""
    dates = pd.date_range("2023-01-01", periods=60, freq="7D")
    prices = np.linspace(100, 218, 60)  # +2 per week
    df = pd.DataFrame({
        "card_id": "test-card", "date": dates, "price": prices,
        "card_name": "Test", "player": "P", "year": 2020, "manufacturer": "M",
        "set": "S", "card_number": "1", "parallel": "", "rookie": "true",
        "autograph": "", "memorabilia": "", "grading_company": "PSA",
        "grade": "10", "sport": "basketball", "is_synthetic": True,
        "mean_price": prices, "min_price": prices, "max_price": prices,
        "sales_count": 1, "price_std": 0.0, "price_iqr": 0.0,
    })
    return df


def test_lag_uses_only_past():
    feats = add_card_features(_tiny_series())
    # 30d lag at row i should be the price ~5 weeks earlier (35 days >= 30)
    row = feats.iloc[10]
    expected = feats.iloc[10 - 5]["price"]
    assert row["price_30d_ago"] == expected
    # first row has no history
    assert np.isnan(feats.iloc[0]["price_30d_ago"])


def test_historical_high_is_expanding():
    df = _tiny_series()
    df.loc[30:, "price"] = 50  # crash after row 30
    feats = add_card_features(df)
    assert feats.iloc[59]["historical_high"] == df.iloc[30 - 1]["price"]
    assert feats.iloc[59]["price"] == 50


def test_targets_within_tolerance():
    feats = add_targets(_tiny_series(), horizons=[30])
    # weekly data: closest obs to +30d is +28d (4 rows ahead), gap 2 <= tol 10
    row = feats.iloc[10]
    assert row["future_price_30d"] == feats.iloc[14]["price"]
    assert row["target_gap_days_30d"] == 2
    # tail rows have no future observation
    assert np.isnan(feats.iloc[-1]["future_price_30d"])


def test_future_return_formula():
    feats = add_targets(_tiny_series(), horizons=[30])
    row = feats.iloc[10]
    assert np.isclose(row["future_return_30d"],
                      row["future_price_30d"] / row["price"] - 1)


def test_market_features_backward_merge(synthetic_raw):
    panel = add_market_features(add_card_features(to_series(synthetic_raw)))
    assert "market_return_30d" in panel.columns
    # market value at each row must not depend on rows after that date:
    # spot-check that the merge is backward — market col for the earliest date
    # with no prior return data is NaN rather than filled from the future.
    first_date = panel["date"].min()
    assert panel.loc[panel["date"] == first_date, "market_return_30d"].isna().all()
