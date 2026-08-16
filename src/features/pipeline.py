"""End-to-end feature build: raw CSV -> model-ready panel."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config
from src.data.load import load_prices, to_series
from src.data.player_stats import load_player_stats, merge_player_features
from src.features.targets import add_targets
from src.features.timeseries import (
    FEATURE_COLUMNS,
    PLAYER_FEATURE_COLUMNS,
    add_card_age,
    add_card_features,
    add_market_features,
)


def build_panel(prices_path: Path | None = None, *, with_targets: bool = True) -> pd.DataFrame:
    raw = load_prices(prices_path)
    series = to_series(raw)
    panel = add_card_features(series)
    panel = add_market_features(panel)
    panel = add_card_age(panel)
    stats = load_player_stats()
    panel = merge_player_features(panel, stats)
    if with_targets:
        panel = add_targets(panel)
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    cols = ["price"] + FEATURE_COLUMNS + PLAYER_FEATURE_COLUMNS
    return [c for c in cols if c in panel.columns]


def save_panel(panel: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (config.DATA_PROCESSED / "panel.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return path
