"""Player season statistics, time-aligned so no future seasons leak.

Data comes from data/raw/player_season_stats.csv. Fill it from the Kaggle
"NBA Database" (wyattowalsh/basketball, CC BY-SA 4.0) or another legitimate
source; `python -m src.data.synthetic` writes a SYNTHETIC demo version.

Alignment rule: a season's statistics become usable only after
`season_end_date`. When predicting at date T, a card gets the most recent
season that ENDED on or before T (previous_season_* features). This is the
strict interpretation of the spec's section 15 for season-level data; in-season
to-date stats would require game-level data and can be added later.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

REQUIRED = [
    "player",
    "season_start_year",
    "season_end_date",
    "games_played",
    "points_per_game",
    "rebounds_per_game",
    "assists_per_game",
]

STAT_FEATURES = [
    "points_per_game",
    "rebounds_per_game",
    "assists_per_game",
    "field_goal_percentage",
    "three_point_percentage",
    "free_throw_percentage",
    "games_played",
    "all_star",
]


def load_player_stats(path: Path | None = None) -> pd.DataFrame | None:
    """Return season stats sorted by player/season, or None if file is absent."""
    path = path or config.PLAYER_STATS_CSV
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"player_season_stats.csv missing columns: {missing}")
    df["season_end_date"] = pd.to_datetime(df["season_end_date"])
    return df.sort_values(["player", "season_end_date"]).reset_index(drop=True)


def stats_asof(stats: pd.DataFrame, player: str, asof: pd.Timestamp) -> pd.Series | None:
    """Most recent season for `player` that ended on or before `asof`."""
    subset = stats[(stats["player"] == player) & (stats["season_end_date"] <= asof)]
    if subset.empty:
        return None
    return subset.iloc[-1]


def merge_player_features(panel: pd.DataFrame, stats: pd.DataFrame | None) -> pd.DataFrame:
    """As-of merge of previous-season stats onto a (player, date) panel."""
    if stats is None:
        return panel
    stats_sorted = stats.sort_values("season_end_date").copy()
    keep = ["player", "season_end_date", "debut_year"] + [
        c for c in STAT_FEATURES if c in stats_sorted.columns
    ]
    stats_sorted = stats_sorted[[c for c in keep if c in stats_sorted.columns]]
    rename = {c: f"prev_season_{c}" for c in STAT_FEATURES if c in stats_sorted.columns}
    stats_sorted = stats_sorted.rename(columns=rename)

    panel = panel.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(
        panel,
        stats_sorted.rename(columns={"season_end_date": "date"}).sort_values("date"),
        on="date",
        by="player",
        direction="backward",  # only seasons that already ended are visible
    )
    if "debut_year" in merged.columns:
        merged["card_year_relative_to_debut"] = (
            pd.to_numeric(merged["year"], errors="coerce")
            - pd.to_numeric(merged["debut_year"], errors="coerce")
        )
    return merged
