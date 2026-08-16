"""Forecast a single card from the command line.

Example:
    python predict.py --player "LeBron James" --year 2003 \
        --set "Topps Chrome" --card-number 111 --grade "10" --grading-company PSA
"""
from __future__ import annotations

import argparse

from src.features.pipeline import build_panel
from src.forecasting.comparables import find_comparables
from src.forecasting.forecast import forecast_card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card-id", help="exact card_id (overrides other filters)")
    parser.add_argument("--player")
    parser.add_argument("--year", type=int)
    parser.add_argument("--set", dest="set_name")
    parser.add_argument("--card-number")
    parser.add_argument("--grade")
    parser.add_argument("--grading-company")
    args = parser.parse_args()

    panel = build_panel()
    if panel["is_synthetic"].any():
        print("*** SYNTHETIC demo data loaded — outputs do not describe any real market. ***\n")

    cards = panel.sort_values("date").groupby("card_id").tail(1)
    if args.card_id:
        matches = cards[cards["card_id"] == args.card_id]
    else:
        matches = cards
        if args.player:
            matches = matches[matches["player"].str.contains(args.player, case=False, na=False)]
        if args.year:
            matches = matches[matches["year"] == args.year]
        if args.set_name:
            matches = matches[matches["set"].str.contains(args.set_name, case=False, na=False)]
        if args.card_number:
            matches = matches[matches["card_number"].astype(str) == str(args.card_number)]
        if args.grade:
            matches = matches[matches["grade"].astype(str).str.replace("PSA ", "") ==
                              str(args.grade).replace("PSA ", "")]
        if args.grading_company:
            matches = matches[matches["grading_company"].astype(str).str.lower() ==
                              args.grading_company.lower()]

    if matches.empty:
        raise SystemExit("No card matches those filters.")
    if len(matches) > 1:
        print("Multiple matches — pass --card-id to disambiguate:")
        for _, r in matches.iterrows():
            print(f"  {r['card_id']}  ({r['card_name']})")
        raise SystemExit(1)

    card_id = matches.iloc[0]["card_id"]
    fc = forecast_card(panel, card_id)
    print(f"## {fc.card_name}\n")
    print(f"Data through: {fc.asof.date()}")
    print(f"Current observed price: ${fc.current_price:,.2f}\n")
    if fc.limited_history:
        print(fc.message)
    for h in sorted(fc.horizons):
        v = fc.horizons[h]
        print(f"{h:>3}-day forecast: ${v['point']:,.2f}   "
              f"90% interval: ${v['lo']:,.2f} - ${v['hi']:,.2f}   "
              f"reliability: {v['reliability'].level}   "
              f"(model: {v['model']}, trained through {v['trained_through']})")
    if fc.message and not fc.limited_history:
        print(fc.message)
    if fc.reliability:
        print(f"\nData quality: {fc.reliability.level}")
        for reason in fc.reliability.reasons:
            print(f"  - {reason}")

    comps = find_comparables(panel, card_id)
    if not comps.empty:
        print("\nComparable Cards (similar, NOT identical):")
        for _, r in comps.iterrows():
            print(f"  {r['card_name']}  latest ${r['price']:,.2f}  "
                  f"(similarity {r['similarity_score']})")


if __name__ == "__main__":
    main()
