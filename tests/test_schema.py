import pandas as pd

from src.data.schema import build_card_id, display_name


def _row(**overrides):
    base = {
        "player": "LeBron James", "year": 2003, "manufacturer": "Topps",
        "set": "Topps Chrome", "card_number": "111", "parallel": "",
        "rookie": "true", "autograph": "", "memorabilia": "",
        "serial_number": "", "grading_company": "PSA", "grade": "10",
    }
    base.update(overrides)
    return pd.Series(base)


def test_grade_changes_card_id():
    assert build_card_id(_row(grade="10")) != build_card_id(_row(grade="9"))


def test_parallel_changes_card_id():
    assert build_card_id(_row()) != build_card_id(_row(parallel="Refractor"))


def test_same_identity_same_id():
    assert build_card_id(_row()) == build_card_id(_row())


def test_player_alone_is_not_the_id():
    a = build_card_id(_row(set="Topps Chrome"))
    b = build_card_id(_row(set="Upper Deck"))
    assert a != b


def test_display_name():
    name = display_name(_row())
    assert "2003" in name and "LeBron James" in name and "#111" in name and "PSA 10" in name


def test_display_name_does_not_repeat_manufacturer_in_set():
    # manufacturer "Topps" + set "Topps Chrome" must not read "Topps Topps Chrome"
    name = display_name(_row(manufacturer="Topps", set="Topps Chrome"))
    assert "Topps Topps" not in name
    assert "Topps Chrome" in name


def test_display_name_keeps_distinct_manufacturer():
    name = display_name(_row(manufacturer="Panini", set="Prizm"))
    assert "Panini Prizm" in name
