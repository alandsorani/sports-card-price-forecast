"""Portfolio aggregation for the Collection page.

Portfolio ranges sum each card's interval bounds. This is documented as a rough
band, NOT an exact joint 90% interval: it ignores cross-card correlation.
"""
from __future__ import annotations

import pandas as pd


def aggregate(items: list[dict]) -> dict:
    """items: [{qty, purchase_price, current_price, horizons: {h: {point, lo, hi}}}]"""
    out = {
        "current_value": 0.0,
        "cost_basis": 0.0,
        "horizons": {},
    }
    for item in items:
        qty = int(item["qty"])
        out["current_value"] += float(item["current_price"]) * qty
        out["cost_basis"] += float(item["purchase_price"]) * qty
        for h, v in item.get("horizons", {}).items():
            slot = out["horizons"].setdefault(h, {"point": 0.0, "lo": 0.0, "hi": 0.0})
            slot["point"] += v["point"] * qty
            slot["lo"] += v["lo"] * qty
            slot["hi"] += v["hi"] * qty
    out["unrealized_gain"] = out["current_value"] - out["cost_basis"]
    return out


def unrealized_gain(current_price: float, purchase_price: float, qty: int) -> float:
    return (float(current_price) - float(purchase_price)) * int(qty)
