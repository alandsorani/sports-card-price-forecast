import pandas as pd
import streamlit as st

import ui
from common import card_picker, get_panel
from src.forecasting.comparables import find_comparables
from src.forecasting.forecast import bundle_path, explain_forecast, forecast_card
from src.models.train import load_bundle
from src.visualization.charts import forecast_figure

st.set_page_config(page_title="Forecast", layout="wide")
ui.page("Card Forecast",
        "Where this card's price might go, with the range it will most likely "
        "land in and the evidence behind it.")

panel = get_panel()
ui.synthetic_banner(panel)

card_id = card_picker(panel)
if not card_id:
    st.stop()
history = panel[panel["card_id"] == card_id].sort_values("date")
latest = history.iloc[-1]
fc = forecast_card(panel, card_id)

st.subheader(fc.card_name)

mom = latest.get("return_30d")
vol = latest.get("volatility_90d")
ui.tiles([
    {"label": "Current price", "value": f"${fc.current_price:,.2f}",
     "note": f"as of {fc.asof.date()}"},
    {"label": "Last 30 days", "value": f"{mom:+.1%}" if pd.notna(mom) else "n/a"},
    {"label": "Price swing", "value": f"{vol:.0%}" if pd.notna(vol) else "n/a",
     "note": "yearly, based on last 90 days"},
    {"label": "Observations", "value": f"{len(history):,}"},
    {"label": "Data quality", "value": fc.reliability.level if fc.reliability else "n/a"},
])

if fc.limited_history:
    st.warning(fc.message)
elif not fc.horizons:
    st.warning(fc.message or "No models trained yet. Run `python train.py` first.")
else:
    ui.forecast_cards(fc)

st.plotly_chart(forecast_figure(history, fc), use_container_width=True)

if fc.horizons:
    with st.expander("See the numbers"):
        st.dataframe(pd.DataFrame([
            {"horizon": f"{h} days",
             "forecast": round(v["point"], 2),
             "range_low": round(v["lo"], 2),
             "range_high": round(v["hi"], 2),
             "confidence": v["reliability"].level,
             "model": v["model"]}
            for h, v in sorted(fc.horizons.items())
        ]), hide_index=True, use_container_width=True)

if fc.reliability:
    with st.expander("Why these confidence ratings?"):
        st.markdown("**About the data itself, which affects every horizon**")
        for reason in fc.reliability.reasons:
            st.write(f"- {reason}")
        for h in sorted(fc.horizons):
            rel = fc.horizons[h]["reliability"]
            st.markdown(f"**{h} days: {rel.level}**")
            for reason in rel.reasons:
                st.write(f"- {reason}")
        st.caption("These ratings come from a set of rules, so treat them as a "
                   "guide rather than a calibrated probability. Ranges widen "
                   "the further out you look, which is why longer horizons "
                   "tend to rate lower.")

if fc.horizons:
    st.subheader("What drove this forecast")
    h_choice = st.selectbox("Horizon", sorted(fc.horizons),
                            format_func=lambda h: f"{h} days")
    bundle = load_bundle(bundle_path(h_choice))
    if bundle.get("deployed_kind") == "baseline":
        st.info(
            f"At this horizon we simply use the **{bundle['model_name']}** rule. "
            "No machine-learning model managed to beat it in testing (its "
            f"average error was {bundle['best_baseline']['mae']:,.0f}), and a "
            "fancier model that performs worse would only add false confidence.",
        )
    else:
        st.caption("How much each input mattered, measured on data the model "
                   "had never seen, shown next to this card's own values.")
        table = explain_forecast(h_choice, latest)
        if table.empty:
            st.info("No importance data was stored for this model.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)

st.subheader("Similar cards")
st.caption("These are comparable, not identical. They are scored on player, "
           "set, grade, year, rookie status, price level, and how much the "
           "price tends to move.")
comps = find_comparables(panel, card_id)
if comps.empty:
    st.info("No similar cards found in the data.")
else:
    st.dataframe(comps, hide_index=True, use_container_width=True)
