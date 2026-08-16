"""Central configuration: paths, horizons, tolerances."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

PRICES_CSV = DATA_RAW / "sports_card_prices.csv"
PLAYER_STATS_CSV = DATA_RAW / "player_season_stats.csv"
COLLECTION_CSV = DATA_PROCESSED / "collection.csv"

# Forecast horizons in days.
HORIZONS = [30, 90, 180, 365]

# A future observation matches a horizon if it falls within
# horizon +/- max(TARGET_TOL_MIN_DAYS, TARGET_TOL_FRAC * horizon) days,
# taking the closest observation. Documented in README ("Target construction").
TARGET_TOL_MIN_DAYS = 10
TARGET_TOL_FRAC = 0.15

# Nominal coverage for prediction intervals.
INTERVAL_ALPHA = 0.10  # -> 90% intervals

SPORT_DEFAULT = "basketball"

SYNTHETIC_SOURCE_LABEL = "SYNTHETIC"


def target_tolerance_days(horizon: int) -> int:
    return int(max(TARGET_TOL_MIN_DAYS, TARGET_TOL_FRAC * horizon))
