"""Generate CLEARLY-LABELED synthetic demo data.

Every row carries source=SYNTHETIC and source_url=about:synthetic. This data
exists ONLY to demonstrate the pipeline and to power unit tests. It is not, and
must never be presented as, real market data. The Streamlit app shows a warning
banner whenever synthetic rows are loaded.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config
from src.data.schema import PRICE_COLUMNS, build_card_id

# Card identities here are plausible-looking labels for demo purposes only;
# their prices below are random walks, not observations of the real cards.
_DEMO_CARDS = [
    ("LeBron James", 2003, "Topps", "Topps Chrome", "111", "", True, "PSA", "10"),
    ("LeBron James", 2003, "Topps", "Topps Chrome", "111", "", True, "PSA", "9"),
    ("LeBron James", 2003, "Topps", "Topps Chrome", "111", "Refractor", True, "PSA", "10"),
    ("Michael Jordan", 1986, "Fleer", "Fleer", "57", "", True, "PSA", "8"),
    ("Michael Jordan", 1986, "Fleer", "Fleer", "57", "", True, "PSA", "9"),
    ("Kobe Bryant", 1996, "Topps", "Topps Chrome", "138", "", True, "PSA", "9"),
    ("Kobe Bryant", 1996, "Topps", "Topps Chrome", "138", "Refractor", True, "PSA", "9"),
    ("Stephen Curry", 2009, "Topps", "Topps Chrome", "101", "", True, "PSA", "10"),
    ("Stephen Curry", 2009, "Panini", "Prizm", "307", "", True, "PSA", "9"),
    ("Luka Doncic", 2018, "Panini", "Prizm", "280", "", True, "PSA", "10"),
    ("Luka Doncic", 2018, "Panini", "Prizm", "280", "Silver", True, "PSA", "10"),
    ("Victor Wembanyama", 2023, "Panini", "Prizm", "136", "", True, "PSA", "10"),
    ("Giannis Antetokounmpo", 2013, "Panini", "Prizm", "290", "", True, "PSA", "10"),
    ("Kevin Durant", 2007, "Topps", "Topps Chrome", "131", "", True, "PSA", "9"),
    ("Nikola Jokic", 2015, "Panini", "Prizm", "335", "", True, "PSA", "10"),
    ("Jayson Tatum", 2017, "Panini", "Prizm", "25", "", True, "PSA", "10"),
    ("Anthony Edwards", 2020, "Panini", "Prizm", "258", "", True, "PSA", "10"),
    ("Ja Morant", 2019, "Panini", "Prizm", "249", "", True, "PSA", "10"),
    ("Tim Duncan", 1997, "Topps", "Topps Chrome", "115", "", True, "PSA", "9"),
    ("Dirk Nowitzki", 1998, "Topps", "Topps Chrome", "154", "", True, "PSA", "9"),
]


def generate_prices(
    *,
    start: str = "2019-01-05",
    end: str = "2026-08-01",
    freq: str = "7D",
    seed: int = 7,
) -> pd.DataFrame:
    """Weekly random-walk price series per demo card, labeled SYNTHETIC."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq=freq)
    n = len(dates)
    t = np.arange(n)
    # One shared "market" factor so market features have signal.
    market = np.cumsum(rng.normal(0.001, 0.02, n)) + 0.15 * np.sin(t / 26 * np.pi)

    rows = []
    for player, year, manufacturer, set_name, number, parallel, rookie, company, grade in _DEMO_CARDS:
        base = float(rng.uniform(2.2, 4.8))  # log10 of starting price
        beta = float(rng.uniform(0.5, 1.6))
        drift = float(rng.normal(0.0005, 0.001))
        noise = rng.normal(0, rng.uniform(0.01, 0.04), n)
        logp = base + beta * market + drift * t + np.cumsum(noise)
        prices = np.round(10 ** logp, 2)
        # Randomly drop ~15% of observations to simulate irregular sampling.
        keep = rng.random(n) > 0.15
        for date, price, k in zip(dates, prices, keep):
            if not k:
                continue
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "card_id": "",
                    "player": player,
                    "year": year,
                    "manufacturer": manufacturer,
                    "set": set_name,
                    "card_number": number,
                    "parallel": parallel,
                    "rookie": "true" if rookie else "",
                    "autograph": "",
                    "memorabilia": "",
                    "serial_number": "",
                    "grading_company": company,
                    "grade": grade,
                    "price": price,
                    "source": config.SYNTHETIC_SOURCE_LABEL,
                    "source_url": "about:synthetic",
                }
            )
    df = pd.DataFrame(rows, columns=PRICE_COLUMNS)
    df["card_id"] = df.apply(build_card_id, axis=1)
    return df


def generate_player_stats(seed: int = 11) -> pd.DataFrame:
    """Synthetic per-season stats for the demo players, labeled via `source`."""
    rng = np.random.default_rng(seed)
    debut = {
        "LeBron James": 2003, "Michael Jordan": 1984, "Kobe Bryant": 1996,
        "Stephen Curry": 2009, "Luka Doncic": 2018, "Victor Wembanyama": 2023,
        "Giannis Antetokounmpo": 2013, "Kevin Durant": 2007, "Nikola Jokic": 2015,
        "Jayson Tatum": 2017, "Anthony Edwards": 2020, "Ja Morant": 2019,
        "Tim Duncan": 1997, "Dirk Nowitzki": 1998,
    }
    rows = []
    for player, first_year in debut.items():
        for season_start in range(max(first_year, 2016), 2026):
            rows.append(
                {
                    "player": player,
                    "season_start_year": season_start,
                    "season_end_date": f"{season_start + 1}-06-30",
                    "games_played": int(rng.integers(45, 82)),
                    "points_per_game": round(float(rng.uniform(12, 33)), 1),
                    "rebounds_per_game": round(float(rng.uniform(3, 12)), 1),
                    "assists_per_game": round(float(rng.uniform(2, 10)), 1),
                    "field_goal_percentage": round(float(rng.uniform(0.42, 0.62)), 3),
                    "three_point_percentage": round(float(rng.uniform(0.30, 0.44)), 3),
                    "free_throw_percentage": round(float(rng.uniform(0.70, 0.92)), 3),
                    "all_star": int(rng.random() < 0.6),
                    "debut_year": first_year,
                    "source": config.SYNTHETIC_SOURCE_LABEL,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write synthetic demo CSVs (labeled SYNTHETIC).")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    prices = generate_prices(seed=args.seed)
    prices.to_csv(config.PRICES_CSV, index=False)
    stats = generate_player_stats()
    stats.to_csv(config.PLAYER_STATS_CSV, index=False)
    print(
        f"Wrote {len(prices)} SYNTHETIC price rows for {prices['card_id'].nunique()} cards "
        f"to {config.PRICES_CSV}\nWrote {len(stats)} SYNTHETIC player-season rows to "
        f"{config.PLAYER_STATS_CSV}\nAll rows are labeled source=SYNTHETIC. Replace with real, "
        "legitimately collected observations before drawing any market conclusions."
    )


if __name__ == "__main__":
    main()
