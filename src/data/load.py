"""Load and clean the manual price template into a tidy time series.

The pipeline runs on data/raw/sports_card_prices.csv, which the project owner
fills with legitimately collected observations (see README "Data sources").
Rows whose `source` equals SYNTHETIC are demo data and are surfaced as such in
the app; they exist only to exercise the pipeline and tests.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data.schema import PRICE_COLUMNS, build_card_id, display_name

__all__ = ["load_prices", "to_series", "validate_price_frame"]

MIN_VALID_PRICE = 0.01
MAX_VALID_PRICE = 50_000_000  # above any card ever sold; flags typos
MIN_VALID_DATE = pd.Timestamp("1980-01-01")


def load_prices(path: Path | None = None, *, keep_flagged: bool = False) -> pd.DataFrame:
    """Return the cleaned long-format price table.

    Adds: card_id (if blank), card_name, sport, quality flags. Invalid rows are
    dropped unless keep_flagged=True (then returned with `flag` set).
    """
    path = path or config.PRICES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fill in the template data/raw/sports_card_prices.csv "
            "with real observations, or run `python -m src.data.synthetic` to generate "
            "clearly-labeled synthetic demo data."
        )
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Price CSV is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    today = pd.Timestamp.today().normalize()
    flags = np.select(
        [
            df["date"].isna(),
            df["price"].isna(),
            df["date"] < MIN_VALID_DATE,
            df["date"] > today,
            df["price"] < MIN_VALID_PRICE,
            df["price"] > MAX_VALID_PRICE,
        ],
        [
            "unparseable_date",
            "unparseable_price",
            "date_too_old",
            "date_in_future",
            "price_too_low",
            "price_too_high",
        ],
        default="",
    )
    df["flag"] = flags

    needs_id = df["card_id"].isna() | (df["card_id"].str.strip() == "")
    if needs_id.any():
        df.loc[needs_id, "card_id"] = df.loc[needs_id].apply(build_card_id, axis=1)
    df["card_name"] = df.apply(display_name, axis=1)
    df["sport"] = config.SPORT_DEFAULT
    df["is_synthetic"] = df["source"].str.strip().str.upper() == config.SYNTHETIC_SOURCE_LABEL

    # Exact duplicates (same card, date, price, source) are collapsed; same-day
    # repeat observations for a card are aggregated to their median later.
    df = df.drop_duplicates(subset=["card_id", "date", "price", "source"])

    if not keep_flagged:
        df = df[df["flag"] == ""].copy()
    return df.sort_values(["card_id", "date"]).reset_index(drop=True)


def validate_price_frame(df: pd.DataFrame) -> dict:
    """Inspect a candidate price table without writing anything.

    Runs the same parsing and quality rules as `load_prices` so what the import
    screen reports matches what the pipeline will actually accept.
    """
    report: dict = {"ok": False, "missing_columns": [], "extra_columns": [],
                    "n_rows": len(df), "n_valid": 0, "n_flagged": 0,
                    "flags": {}, "n_cards": 0, "date_range": None,
                    "n_synthetic": 0}
    report["missing_columns"] = [c for c in PRICE_COLUMNS if c not in df.columns]
    report["extra_columns"] = [c for c in df.columns if c not in PRICE_COLUMNS]
    if report["missing_columns"]:
        return report

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    flags = np.select(
        [
            work["date"].isna(), work["price"].isna(),
            work["date"] < MIN_VALID_DATE, work["date"] > today,
            work["price"] < MIN_VALID_PRICE, work["price"] > MAX_VALID_PRICE,
        ],
        ["unparseable_date", "unparseable_price", "date_too_old",
         "date_in_future", "price_too_low", "price_too_high"],
        default="",
    )
    valid = work[flags == ""]
    report["n_valid"] = int(len(valid))
    report["n_flagged"] = int((flags != "").sum())
    report["flags"] = {k: int(v) for k, v in
                       pd.Series(flags[flags != ""]).value_counts().items()}
    if not valid.empty:
        # A blank card_id arrives as NaN, and str(nan) is the truthy "nan",
        # so test for missingness explicitly rather than truthiness.
        given = valid["card_id"]
        has_id = given.notna() & (given.astype(str).str.strip() != "")
        ids = given.where(has_id, valid.apply(build_card_id, axis=1))
        report["n_cards"] = int(ids.nunique())
        report["date_range"] = (valid["date"].min(), valid["date"].max())
    report["n_synthetic"] = int(
        (df["source"].astype(str).str.strip().str.upper()
         == config.SYNTHETIC_SOURCE_LABEL).sum())
    report["ok"] = report["n_valid"] > 0
    return report


def to_series(df: pd.DataFrame) -> pd.DataFrame:
    """One row per card_id per date (median of same-day observations).

    Values from a price-guide chart are point-in-time estimates, not individual
    transactions, and are used directly; transaction rows get median-aggregated.
    """
    grouped = (
        df.groupby(["card_id", "date"])
        .agg(
            price=("price", "median"),
            mean_price=("price", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            sales_count=("price", "size"),
            price_std=("price", "std"),
            card_name=("card_name", "first"),
            player=("player", "first"),
            year=("year", "first"),
            manufacturer=("manufacturer", "first"),
            set=("set", "first"),
            card_number=("card_number", "first"),
            parallel=("parallel", "first"),
            rookie=("rookie", "first"),
            autograph=("autograph", "first"),
            memorabilia=("memorabilia", "first"),
            grading_company=("grading_company", "first"),
            grade=("grade", "first"),
            sport=("sport", "first"),
            is_synthetic=("is_synthetic", "any"),
        )
        .reset_index()
        .sort_values(["card_id", "date"])
        .reset_index(drop=True)
    )
    q75 = df.groupby(["card_id", "date"])["price"].quantile(0.75)
    q25 = df.groupby(["card_id", "date"])["price"].quantile(0.25)
    grouped["price_iqr"] = (q75 - q25).values
    return grouped
