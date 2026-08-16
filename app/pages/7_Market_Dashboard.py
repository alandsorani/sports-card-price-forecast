import streamlit as st

import ui
from common import get_panel, latest_cards
from src.visualization.charts import market_figure

st.set_page_config(page_title="Market Dashboard", layout="wide")
ui.page("Market Overview",
        "How the tracked cards are doing as a group. This covers only the cards "
        "in your data, so treat it as your slice rather than the whole hobby.")

panel = get_panel()
ui.synthetic_banner(panel)

market = (
    panel.dropna(subset=["market_return_30d"])
    .groupby("date")[["market_return_30d", "market_momentum", "market_volume"]]
    .first()
    .reset_index()
)

if not market.empty:
    latest = market.iloc[-1]
    mom = latest["market_momentum"]
    ui.tiles([
        {"label": "Typical 30-day move",
         "value": f"{latest['market_return_30d']:+.1%}"},
        {"label": "Momentum", "value": f"{mom:+.1%}" if mom == mom else "n/a",
         "note": "median of last 13 observations"},
        {"label": "Cards reporting", "value": f"{int(latest['market_volume']):,}"},
        {"label": "As of", "value": str(latest["date"].date())},
    ])
    st.plotly_chart(market_figure(market), use_container_width=True)
    with st.expander("See the numbers"):
        st.dataframe(market.sort_values("date", ascending=False),
                     hide_index=True, use_container_width=True)

st.subheader("Biggest movers this past month")
cards = latest_cards(panel).dropna(subset=["return_30d"])
cols = ["card_name", "price", "return_30d", "volatility_90d", "date"]
left, right = st.columns(2, gap="large")
left.markdown("**Climbing**")
left.dataframe(cards.nlargest(10, "return_30d")[cols], hide_index=True,
               use_container_width=True)
right.markdown("**Sliding**")
right.dataframe(cards.nsmallest(10, "return_30d")[cols], hide_index=True,
                use_container_width=True)
