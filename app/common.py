"""Shared helpers for the Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.features.pipeline import build_panel  # noqa: E402
from ui import synthetic_banner  # noqa: E402,F401  (re-exported for pages)


@st.cache_data(show_spinner="Building feature panel...")
def get_panel() -> pd.DataFrame:
    return build_panel()


def latest_cards(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values("date").groupby("card_id").tail(1).reset_index(drop=True)


def card_picker(panel: pd.DataFrame, key: str = "card") -> str | None:
    cards = latest_cards(panel)
    options = cards.sort_values("card_name")[["card_id", "card_name"]]
    return st.selectbox(
        "Card", options["card_id"],
        format_func=lambda cid: options.set_index("card_id").loc[cid, "card_name"],
        key=key,
    )


def load_collection() -> pd.DataFrame:
    if config.COLLECTION_CSV.exists():
        df = pd.read_csv(config.COLLECTION_CSV)
        df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
        return df
    return pd.DataFrame(columns=[
        "card_id", "player", "year", "set", "card_number", "grade",
        "purchase_price", "purchase_date", "quantity",
    ])


def save_collection(df: pd.DataFrame) -> None:
    config.COLLECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.COLLECTION_CSV, index=False)
