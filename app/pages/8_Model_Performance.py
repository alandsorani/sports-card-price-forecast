import pandas as pd
import streamlit as st

import ui
from common import get_panel
from src import config
from src.forecasting.forecast import bundle_path
from src.models.train import load_bundle

st.set_page_config(page_title="Model Performance", layout="wide")
ui.page("How Good Are These Forecasts?",
        "Everything here is measured on data the models had never seen. Where a "
        "simple rule beats machine learning, the simple rule is what we use.")

panel = get_panel()
ui.synthetic_banner(panel)

st.subheader("What's running right now")
rows = []
for h in config.HORIZONS:
    p = bundle_path(h)
    if not p.exists():
        continue
    b = load_bundle(p)
    cov = b.get("holdout_coverage", {})
    rows.append({
        "horizon": f"{h} days", "using": b["model_name"],
        "kind": b.get("deployed_kind", "n/a"),
        "predicts": b["target_type"], "log_price": b["log_price"],
        "avg_error": round(b.get("cv_mae", float("nan")), 1),
        "beat_simple_rule": b.get("beat_baseline"),
        "simple_rule": b.get("best_baseline", {}).get("name"),
        "its_error": round(b.get("best_baseline", {}).get("mae", float("nan")), 1),
        "range_accuracy": cov.get("coverage"),
        "trained_through": b["trained_through"], "training_rows": b["n_train"],
    })
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("`range_accuracy` is how often the real price landed inside the "
               "predicted range during the final test year. We aim for 0.90. "
               "When `kind` says baseline, it means no machine-learning model "
               "beat the simple rule, so we kept the simple rule.")
else:
    st.info("No trained models yet. Run `python train.py` to create them.")

comparison = config.REPORTS_DIR / "model_comparison.csv"
if comparison.exists():
    st.subheader("Everything we tried")
    st.caption("Average results across expanding time windows, always training "
               "on the past and testing on the future. Never a random split.")
    df = pd.read_csv(comparison)
    summary = (
        df.groupby(["horizon", "model", "target_type", "log_price"])
        [["mae", "rmse", "r2", "directional_accuracy"]]
        .mean().round(3).reset_index().sort_values(["horizon", "mae"])
    )
    tabs = st.tabs([f"{h} days" for h in sorted(summary["horizon"].unique())])
    for tab, h in zip(tabs, sorted(summary["horizon"].unique())):
        with tab:
            st.dataframe(summary[summary["horizon"] == h].drop(columns="horizon"),
                         hide_index=True, use_container_width=True)
    st.caption(
        "`target_type` says whether the model predicted the future price or the "
        "future percentage change. `log_price` says whether prices were "
        "compressed before training. One quirk worth knowing: the `last_price` "
        "rule always scores zero on direction, because predicting no change "
        "never picks a direction at all."
    )
else:
    st.info("No comparison table yet. Run `python train.py` to create one.")

backtest = config.REPORTS_DIR / "backtest_metrics.csv"
if backtest.exists():
    st.subheader("The full rehearsal")
    st.caption("A replay of history: on each simulated day, the models were "
               "rebuilt and retrained using only what was known back then, then "
               "graded against what actually happened.")
    st.dataframe(pd.read_csv(backtest), hide_index=True, use_container_width=True)
else:
    st.info("No backtest results yet. Run `python backtest.py` to create them.")
