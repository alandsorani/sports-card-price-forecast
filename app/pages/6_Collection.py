import pandas as pd
import streamlit as st

import ui
from common import get_panel, latest_cards, load_collection, save_collection
from src.forecasting.forecast import forecast_card
from src.forecasting.portfolio import aggregate, unrealized_gain

st.set_page_config(page_title="Collection", layout="wide")
ui.page("My Collection",
        "Keep track of what you own and see what the forecasts add up to "
        "across the whole shelf.")

panel = get_panel()
ui.synthetic_banner(panel)
cards = latest_cards(panel)

with st.expander("Add a card", expanded=load_collection().empty):
    with st.form("add_card"):
        options = cards.sort_values("card_name")
        card_id = st.selectbox(
            "Card", options["card_id"],
            format_func=lambda cid: options.set_index("card_id").loc[cid, "card_name"],
        )
        c1, c2, c3 = st.columns(3)
        purchase_price = c1.number_input("What you paid ($)", min_value=0.0, step=1.0)
        purchase_date = c2.date_input("When you bought it")
        quantity = c3.number_input("How many", min_value=1, step=1, value=1)
        if st.form_submit_button("Add to collection", type="primary"):
            row = cards[cards["card_id"] == card_id].iloc[0]
            coll = load_collection()
            coll = pd.concat([coll, pd.DataFrame([{
                "card_id": card_id, "player": row["player"], "year": row["year"],
                "set": row["set"], "card_number": row["card_number"],
                "grade": f"{row['grading_company']} {row['grade']}".strip(),
                "purchase_price": purchase_price,
                "purchase_date": pd.Timestamp(purchase_date), "quantity": quantity,
            }])], ignore_index=True)
            save_collection(coll)
            st.success("Added it.")

coll = load_collection()
if coll.empty:
    st.info("Your collection is empty for now. Add a card above and the "
            "portfolio totals will appear here.")
    st.stop()

rows, items = [], []
for _, item in coll.iterrows():
    if item["card_id"] not in set(panel["card_id"]):
        continue
    fc = forecast_card(panel, item["card_id"])
    qty = int(item["quantity"])
    row = {
        "card": fc.card_name, "qty": qty,
        "paid": item["purchase_price"],
        "current_price": fc.current_price,
        "current_value": fc.current_price * qty,
        "gain": unrealized_gain(fc.current_price, item["purchase_price"], qty),
    }
    for h, v in fc.horizons.items():
        row[f"in_{h}d"] = v["point"] * qty
    items.append({"qty": qty, "purchase_price": item["purchase_price"],
                  "current_price": fc.current_price, "horizons": fc.horizons})
    rows.append(row)

port = aggregate(items)
cost = port["cost_basis"]
tiles = [
    {"label": "What it's worth now", "value": f"${port['current_value']:,.2f}"},
    {"label": "What you paid", "value": f"${cost:,.2f}"},
    {"label": "Gain so far", "value": f"${port['unrealized_gain']:+,.2f}",
     "delta": (port["unrealized_gain"] / cost) if cost else None},
]
for h in sorted(port["horizons"]):
    v = port["horizons"][h]
    change = (v["point"] / port["current_value"] - 1) if port["current_value"] else None
    tiles.append({"label": f"In {h} days", "value": f"${v['point']:,.2f}",
                  "delta": change,
                  "note": f"${v['lo']:,.0f} to ${v['hi']:,.0f}"})
ui.tiles(tiles)

if port["horizons"]:
    st.caption(
        "About those portfolio ranges: they add up each card's individual range. "
        "Card prices tend to rise and fall together, so read this as a rough "
        "band rather than an exact one. Adding the bounds ignores the benefit of "
        "spreading risk, and pinning down a tighter number would mean modeling "
        "how cards move relative to each other, which this version does not do."
    )
else:
    st.info("Run `python train.py` to see what the forecasts do to these totals.")

st.subheader("What you own")
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

if st.button("Clear collection"):
    save_collection(load_collection().iloc[0:0])
    st.rerun()
