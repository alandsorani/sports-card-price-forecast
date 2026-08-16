import streamlit as st

import ui
from common import get_panel
from src.data.player_stats import load_player_stats
from src.visualization.charts import player_overlay_figure

st.set_page_config(page_title="Player Analysis", layout="wide")
ui.page("Player Analysis",
        "Card price and on-court production side by side over the same years. "
        "Handy for spotting timing, though it cannot tell you that one caused "
        "the other.")

panel = get_panel()
ui.synthetic_banner(panel)

stats = load_player_stats()
if stats is None:
    st.error("There's no player stats file at `data/raw/player_season_stats.csv` "
             "yet. You can fill it from the Kaggle NBA Database (CC BY-SA 4.0), "
             "or run `python -m src.data.synthetic` to get demo values.")
    st.stop()
if "source" in stats.columns and (stats["source"].astype(str).str.upper() == "SYNTHETIC").any():
    st.info("These player statistics are demo values, not real career numbers.")

c1, c2 = st.columns(2)
player = c1.selectbox("Player", sorted(panel["player"].dropna().unique()))
cards = panel[panel["player"] == player]
card_name = c2.selectbox("Card", sorted(cards["card_name"].unique()))
history = cards[cards["card_name"] == card_name].sort_values("date")

st.plotly_chart(
    player_overlay_figure(history, stats, player, card_name),
    use_container_width=True,
)
st.caption("These are two separate panels on a shared timeline rather than two "
           "scales stacked on one plot. Overlaying dollars and points per game "
           "would suggest a relationship the data does not actually show.")

st.subheader("Season by season")
st.dataframe(stats[stats["player"] == player].sort_values("season_start_year",
                                                          ascending=False),
             hide_index=True, use_container_width=True)
