import pandas as pd
import pytest

from src.data.load import load_prices, to_series
from src.data.schema import PRICE_COLUMNS


def _write(tmp_path, rows):
    df = pd.DataFrame(rows, columns=PRICE_COLUMNS)
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    return path


def _base_row(**overrides):
    row = {c: "" for c in PRICE_COLUMNS}
    row.update({
        "date": "2024-01-05", "player": "Test Player", "year": "2020",
        "manufacturer": "Panini", "set": "Prizm", "card_number": "1",
        "grading_company": "PSA", "grade": "10", "price": "100",
        "source": "manual", "source_url": "https://example.com",
    })
    row.update(overrides)
    return row


def test_flags_impossible_rows(tmp_path):
    path = _write(tmp_path, [
        _base_row(),
        _base_row(date="2054-01-01"),          # future date
        _base_row(date="1875-01-01"),          # too old
        _base_row(price="-5"),                 # negative price
        _base_row(price="not-a-number"),
        _base_row(date="garbage"),
    ])
    clean = load_prices(path)
    assert len(clean) == 1
    flagged = load_prices(path, keep_flagged=True)
    assert (flagged["flag"] != "").sum() == 5


def test_duplicate_rows_collapse(tmp_path):
    path = _write(tmp_path, [_base_row(), _base_row()])
    assert len(load_prices(path)) == 1


def test_missing_column_raises(tmp_path):
    df = pd.DataFrame([_base_row()]).drop(columns=["grade"])
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="grade"):
        load_prices(path)


def test_to_series_same_day_median(tmp_path):
    path = _write(tmp_path, [
        _base_row(price="100"), _base_row(price="300", source="other"),
        _base_row(price="200", source="third"),
    ])
    series = to_series(load_prices(path))
    assert len(series) == 1
    assert series.iloc[0]["price"] == 200  # median, robust to the outlier
    assert series.iloc[0]["sales_count"] == 3


def test_synthetic_rows_are_labeled(synthetic_raw):
    assert synthetic_raw["is_synthetic"].all()


def test_validate_reports_missing_columns():
    from src.data.load import validate_price_frame

    df = pd.DataFrame([_base_row()]).drop(columns=["price", "grade"])
    report = validate_price_frame(df)
    assert not report["ok"]
    assert set(report["missing_columns"]) == {"price", "grade"}


def test_validate_counts_valid_and_flagged():
    from src.data.load import validate_price_frame

    df = pd.DataFrame([
        _base_row(),
        _base_row(date="2024-02-01", price="150"),
        _base_row(date="2054-01-01"),   # future
        _base_row(price="-5"),          # non-positive
    ], columns=PRICE_COLUMNS)
    report = validate_price_frame(df)
    assert report["ok"]
    assert report["n_rows"] == 4
    assert report["n_valid"] == 2
    assert report["n_flagged"] == 2
    assert report["flags"] == {"date_in_future": 1, "price_too_low": 1}
    assert report["n_cards"] == 1  # same card observed twice


def test_validate_counts_cards_when_card_id_blank(tmp_path):
    """A blank card_id column reads back as NaN, which must not be mistaken
    for a real identifier."""
    from src.data.load import validate_price_frame

    path = _write(tmp_path, [
        _base_row(card_id=""),
        _base_row(card_id="", date="2024-02-01", price="150"),
        _base_row(card_id="", player="Someone Else", date="2024-02-01"),
    ])
    report = validate_price_frame(pd.read_csv(path, dtype=str))
    assert report["n_cards"] == 2


def test_validate_flags_all_bad_file():
    from src.data.load import validate_price_frame

    df = pd.DataFrame([_base_row(price="nope")], columns=PRICE_COLUMNS)
    report = validate_price_frame(df)
    assert not report["ok"]
    assert report["n_valid"] == 0


def test_validate_counts_synthetic_rows():
    from src.data.load import validate_price_frame

    df = pd.DataFrame([
        _base_row(source="SYNTHETIC"),
        _base_row(date="2024-02-01", source="my records"),
    ], columns=PRICE_COLUMNS)
    report = validate_price_frame(df)
    assert report["n_synthetic"] == 1


def test_validate_reports_extra_columns():
    from src.data.load import validate_price_frame

    df = pd.DataFrame([_base_row()], columns=PRICE_COLUMNS)
    df["notes"] = "hello"
    assert validate_price_frame(df)["extra_columns"] == ["notes"]
