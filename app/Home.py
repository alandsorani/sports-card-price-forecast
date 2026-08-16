import streamlit as st

import ui
from common import get_panel

st.set_page_config(page_title="Card Price Forecast", layout="wide", initial_sidebar_state="expanded")

ui.page(
    "Sports Card Price Forecasting",
    "Pick a card and see where its price might go over the next 30, 90, 180, "
    "and 365 days. Every forecast uses only what was knowable on the day it "
    "was made, and comes with an honest range rather than a single number.",
)

try:
    panel = get_panel()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

ui.synthetic_banner(panel)

ui.tiles([
    {"label": "Cards tracked", "value": f"{panel['card_id'].nunique():,}"},
    {"label": "Price observations", "value": f"{len(panel):,}"},
    {"label": "Players", "value": f"{panel['player'].nunique():,}"},
    {"label": "Data through", "value": str(panel["date"].max().date())},
    {"label": "History begins", "value": str(panel["date"].min().date())},
])

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Where to start")
    st.markdown(
        """
**Card Search** lets you filter by player, set, year, number, or grade.

**Forecast** is the main event: price projections with a likely range,
recent momentum, comparable cards, and a plain explanation of what drove
the numbers.

**Collection** tracks what you own and totals it up.

**Model Performance** shows how well the forecasts have actually done.
"""
    )

with right:
    st.subheader("How it works")
    st.markdown(
        """
Models train on the past and are scored on the future, never on a random
shuffle of the two. Simple rules compete against gradient boosting and
random forests, and whichever wins on unseen data is the one that runs.

Every forecast comes with a range, not just a number.
"""
    )

st.caption(
    "A quick reminder: these are statistical estimates with wide, honest "
    "uncertainty ranges, not investment advice."
)
