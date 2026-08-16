import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import generate_prices, generate_player_stats  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_raw(tmp_path_factory) -> pd.DataFrame:
    """SYNTHETIC price rows (labeled), loaded through the real cleaning path."""
    from src.data.load import load_prices

    path = tmp_path_factory.mktemp("data") / "sports_card_prices.csv"
    generate_prices(seed=3).to_csv(path, index=False)
    return load_prices(path)


@pytest.fixture(scope="session")
def synthetic_stats() -> pd.DataFrame:
    df = generate_player_stats()
    df["season_end_date"] = pd.to_datetime(df["season_end_date"])
    return df
