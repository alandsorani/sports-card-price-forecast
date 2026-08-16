import streamlit as st

import ui
from common import get_panel, latest_cards

st.set_page_config(page_title="Card Search", layout="wide")
ui.page("Card Search",
        "Find a card by any detail you remember. Leave a box empty to ignore it.")

panel = get_panel()
ui.synthetic_banner(panel)
cards = latest_cards(panel)

c1, c2, c3 = st.columns(3)
player = c1.text_input("Player")
set_name = c2.text_input("Set")
manufacturer = c3.text_input("Manufacturer")
c4, c5, c6 = st.columns(3)
year = c4.text_input("Year")
number = c5.text_input("Card number")
grade = c6.text_input("Grade")

f = cards
if player:
    f = f[f["player"].str.contains(player, case=False, na=False)]
if set_name:
    f = f[f["set"].str.contains(set_name, case=False, na=False)]
if manufacturer:
    f = f[f["manufacturer"].str.contains(manufacturer, case=False, na=False)]
if year:
    f = f[f["year"].astype(str).str.contains(year, na=False)]
if number:
    f = f[f["card_number"].astype(str).str.contains(number, na=False)]
if grade:
    f = f[f["grade"].astype(str).str.contains(grade, case=False, na=False)]

obs_counts = panel.groupby("card_id").size().rename("observations")
f = f.merge(obs_counts, on="card_id")

ui.tiles([
    {"label": "Cards matched", "value": f"{len(f):,}"},
    {"label": "Median latest price",
     "value": f"${f['price'].median():,.2f}" if len(f) else "n/a"},
    {"label": "Observations",
     "value": f"{int(f['observations'].sum()):,}" if len(f) else "0"},
])

if f.empty:
    st.info("Nothing matched those filters. Try clearing one of the boxes.")
else:
    st.dataframe(
        f[["card_name", "year", "set", "grade", "grading_company", "price",
           "date", "observations", "card_id"]]
        .rename(columns={"price": "latest_price", "date": "latest_date"})
        .sort_values("latest_price", ascending=False),
        use_container_width=True, hide_index=True,
    )
    st.caption("Copy a `card_id` to use on the Forecast page, or with "
               "`python predict.py --card-id ...`")
