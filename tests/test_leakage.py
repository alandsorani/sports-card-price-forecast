import pandas as pd

from src.evaluation.leakage import audit_table, mutation_test
from src.features.pipeline import feature_columns
from src.features.timeseries import FEATURE_COLUMNS, PLAYER_FEATURE_COLUMNS


def test_mutation_test_finds_no_leakage(synthetic_raw, synthetic_stats):
    cutoff = synthetic_raw["date"].quantile(0.5)
    bad = mutation_test(synthetic_raw, cutoff, synthetic_stats)
    assert bad.empty, f"Leaking rows:\n{bad.head()}"


def test_no_target_in_feature_list():
    fake = pd.DataFrame(columns=["price"] + FEATURE_COLUMNS + PLAYER_FEATURE_COLUMNS
                        + [f"future_price_{h}d" for h in (30, 90, 180, 365)]
                        + [f"future_return_{h}d" for h in (30, 90, 180, 365)])
    cols = feature_columns(fake)
    assert not any("future" in c for c in cols)


def test_audit_table_covers_every_feature_family():
    table = audit_table()
    assert {"feature", "definition", "cutoff", "risk", "test"} <= set(table.columns)
    assert len(table) >= 10
