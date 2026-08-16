import streamlit as st

import ui

st.set_page_config(page_title="About", layout="wide")
ui.page("About",
        "What this app forecasts, the data it runs on, and the limits worth "
        "knowing about.")

st.subheader("The question")
st.markdown(
    "Given everything knowable about a sports card today, what will it be "
    "worth in 30, 90, 180, and 365 days?"
)

st.subheader("The data")
st.markdown(
    """
Forecasts run on the price observations in `data/raw/sports_card_prices.csv`.
Load your own through the **Import Data** page, or edit the file directly. Each
row records one observation of one card on one date, along with a `source` and
`source_url` so the provenance travels with the price.

Any row marked `source=SYNTHETIC` is placeholder data for trying out the app,
and every page flags it when present.

Player statistics are optional and live in
`data/raw/player_season_stats.csv`. One free source is the Kaggle
[NBA Database](https://www.kaggle.com/datasets/wyattowalsh/basketball),
licensed CC BY-SA 4.0 and built from public NBA API responses.
"""
)

st.subheader("How the forecasting works")
st.markdown(
    """
Models train on the past and are tested on the future, never on a random
shuffle of the two. Four simple rules (last price, a 30-day and a 90-day moving
median, and a momentum estimate) compete against Ridge, Random Forest, Gradient
Boosting, and HistGradientBoosting. Whichever performs best on unseen data is
the one that runs, and at some horizons that turns out to be one of the simple
rules.

Ranges come from conformal calibration, which measures how far off predictions
have been historically and converts that into a likely interval. How often the
real price actually landed inside that interval is then checked against a test
period the calibration never saw.

Every input is documented with the cutoff date it respects, and an automated
test scrambles future data to confirm that nothing about the past changes as a
result.
"""
)

st.subheader("Limits worth knowing")
st.markdown(
    """
- Forecasts are only as good as the data behind them. Cards with sparse
  history get a plain "not enough data" message instead of a falsely precise
  number.
- Confidence ratings come from rules of thumb, not calibrated probabilities.
- Portfolio ranges add up individual card ranges, which is an approximation.
- Card age is estimated from January 1 of the printed year.
- Basketball only for now, though the structure supports other sports.
- **None of this is investment advice.**
"""
)
