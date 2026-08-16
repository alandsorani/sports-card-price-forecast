import pandas as pd
import streamlit as st

import ui
from common import card_picker, get_panel
from src.visualization.charts import history_figure

st.set_page_config(page_title="Historical Prices", layout="wide")
ui.page("Price History",
        "Every observation on record for a card. This is the raw material "
        "behind each forecast.")

panel = get_panel()
ui.synthetic_banner(panel)

card_id = card_picker(panel)
if not card_id:
    st.stop()
history = panel[panel["card_id"] == card_id].sort_values("date")
latest = history.iloc[-1]

ui.tiles([
    {"label": "Latest price", "value": f"${latest['price']:,.2f}",
     "note": str(latest["date"].date())},
    {"label": "All-time high", "value": f"${history['price'].max():,.2f}"},
    {"label": "All-time low", "value": f"${history['price'].min():,.2f}"},
    {"label": "Off its peak",
     "value": f"{latest['distance_from_high']:+.1%}"
     if pd.notna(latest.get("distance_from_high")) else "n/a"},
    {"label": "Observations", "value": f"{len(history):,}"},
])

st.subheader(latest["card_name"])
st.plotly_chart(history_figure(history), use_container_width=True)

st.subheader("Every observation")
st.dataframe(
    history[["date", "price", "sales_count", "min_price", "max_price",
             "return_30d", "volatility_90d"]].sort_values("date", ascending=False),
    hide_index=True, use_container_width=True,
)
