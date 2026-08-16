"""Canonical schema for the manual price template and card identification.

A card is NEVER identified by player name alone. `card_id` is built from every
identity field the row provides, so `2003 Topps Chrome LeBron James #111 PSA 10`,
`... PSA 9`, and `... Refractor PSA 10` are three distinct series.
"""
from __future__ import annotations

import re

import pandas as pd

# Columns of data/raw/sports_card_prices.csv (section 46 of the spec).
PRICE_COLUMNS = [
    "date",
    "card_id",        # optional; derived from identity fields when blank
    "player",
    "year",
    "manufacturer",
    "set",
    "card_number",
    "parallel",
    "rookie",
    "autograph",
    "memorabilia",
    "serial_number",
    "grading_company",
    "grade",
    "price",
    "source",
    "source_url",
]

IDENTITY_FIELDS = [
    "year",
    "manufacturer",
    "set",
    "player",
    "card_number",
    "parallel",
    "serial_number",
    "autograph",
    "memorabilia",
    "grading_company",
    "grade",
]

BOOL_FIELDS = ["rookie", "autograph", "memorabilia"]


def _slug(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9.]+", "-", text)
    return text.strip("-")


def build_card_id(row: pd.Series) -> str:
    """Deterministic card id from all populated identity fields."""
    parts = []
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if pd.isna(value) or str(value).strip() == "":
            continue
        if field in BOOL_FIELDS:
            if _truthy(value):
                parts.append(field)
            continue
        parts.append(_slug(value))
    return "-".join(parts)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def display_name(row: pd.Series) -> str:
    """Human-readable name, e.g. '2003 Topps Chrome LeBron James #111 PSA 10'."""
    bits = []
    year = row.get("year")
    if pd.notna(year) and str(year).strip():
        bits.append(str(year).strip())
    manufacturer = str(row.get("manufacturer") or "").strip()
    set_name = str(row.get("set") or "").strip()
    # "Topps" + "Topps Chrome" reads as "Topps Topps Chrome"; keep the set only.
    if manufacturer and not set_name.lower().startswith(manufacturer.lower()):
        bits.append(manufacturer)
    if set_name:
        bits.append(set_name)
    player = row.get("player")
    if pd.notna(player) and str(player).strip():
        bits.append(str(player).strip())
    number = row.get("card_number")
    if pd.notna(number) and str(number).strip():
        bits.append(f"#{str(number).strip()}")
    parallel = row.get("parallel")
    if pd.notna(parallel) and str(parallel).strip():
        bits.append(str(parallel).strip())
    company = str(row.get("grading_company") or "").strip()
    grade = str(row.get("grade") or "").strip()
    if company or grade:
        bits.append(f"{company} {grade}".strip())
    return " ".join(bits)
