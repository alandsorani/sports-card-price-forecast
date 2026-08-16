import streamlit as st

import ui
from common import card_picker, get_panel
from src.forecasting.comparables import find_comparables

st.set_page_config(page_title="Similar Cards", layout="wide")
ui.page("Similar Cards",
        "Cards that resemble the one you picked. Especially useful when a card "
        "has too little history of its own to forecast confidently.")

panel = get_panel()
ui.synthetic_banner(panel)

card_id = card_picker(panel)
if not card_id:
    st.stop()
top_n = st.slider("How many to show", 3, 15, 5)
comps = find_comparables(panel, card_id, top_n=top_n)
if comps.empty:
    st.info("No similar cards found in the data.")
else:
    st.dataframe(comps, hide_index=True, use_container_width=True)
    st.caption("`similarity_score` adds up matches on player, set, grade, "
               "grading company, how close the years are, rookie status, price "
               "level, and price movement. Higher means more alike. These are "
               "comparable cards, never identical ones.")
