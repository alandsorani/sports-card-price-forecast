"""Upload your own price observations and player statistics."""
import pandas as pd
import streamlit as st

import ui
from common import get_panel
from src import config
from src.data.load import validate_price_frame
from src.data.schema import PRICE_COLUMNS

st.set_page_config(page_title="Import Data", layout="wide")
ui.page("Import Your Data",
        "Load your own price observations. Nothing is saved until you confirm, "
        "and you'll see exactly what passed validation first.")

tab_prices, tab_players, tab_format = st.tabs(
    ["Price observations", "Player statistics", "File format"])

with tab_prices:
    current_rows = 0
    if config.PRICES_CSV.exists():
        current = pd.read_csv(config.PRICES_CSV, dtype=str)
        current_rows = len(current)
        is_demo = (current["source"].astype(str).str.strip().str.upper()
                   == config.SYNTHETIC_SOURCE_LABEL).all()
        st.caption(f"Currently loaded: {current_rows:,} rows"
                   f"{' of demo data' if is_demo else ''}.")

    upload = st.file_uploader("Choose a CSV of price observations", type="csv",
                              key="prices_upload")
    if upload is not None:
        try:
            incoming = pd.read_csv(upload, dtype=str)
        except Exception as exc:
            st.error(f"That file could not be read as CSV: {exc}")
            st.stop()

        report = validate_price_frame(incoming)

        if report["missing_columns"]:
            st.error("This file is missing required columns: "
                     + ", ".join(report["missing_columns"])
                     + ". Check the File format tab for a template.")
            st.stop()

        ui.tiles([
            {"label": "Rows in file", "value": f"{report['n_rows']:,}"},
            {"label": "Usable rows", "value": f"{report['n_valid']:,}"},
            {"label": "Rejected rows", "value": f"{report['n_flagged']:,}"},
            {"label": "Distinct cards", "value": f"{report['n_cards']:,}"},
        ])

        if report["date_range"]:
            lo, hi = report["date_range"]
            st.caption(f"Dates run from {lo.date()} to {hi.date()}.")
        if report["extra_columns"]:
            st.caption("Extra columns that will be ignored: "
                       + ", ".join(report["extra_columns"]))
        if report["n_synthetic"]:
            st.warning(f"{report['n_synthetic']:,} rows are labeled "
                       "source=SYNTHETIC and will be treated as demo data.")

        if report["flags"]:
            with st.expander(f"Why {report['n_flagged']:,} rows were rejected"):
                st.dataframe(
                    pd.DataFrame(sorted(report["flags"].items()),
                                 columns=["reason", "rows"]),
                    hide_index=True, use_container_width=True)
                st.caption("Rejected rows are skipped, not silently corrected. "
                           "Fix them in your file and upload again if they matter.")

        if not report["ok"]:
            st.error("No usable rows in this file, so there is nothing to import.")
            st.stop()

        st.subheader("Preview")
        st.dataframe(incoming.head(10), hide_index=True, use_container_width=True)

        mode = st.radio(
            "How should this be saved?",
            ["Replace everything currently loaded", "Add to what is already there"],
            help="Adding keeps existing rows. Exact duplicates are removed when "
                 "the data is loaded.",
        )
        replacing = mode.startswith("Replace")
        if replacing and current_rows:
            st.warning(f"This will overwrite the {current_rows:,} rows currently "
                       "in data/raw/sports_card_prices.csv. A copy of the old "
                       "file is kept as sports_card_prices.backup.csv.")

        if st.button("Save to the project", type="primary"):
            config.DATA_RAW.mkdir(parents=True, exist_ok=True)
            if config.PRICES_CSV.exists():
                backup = config.PRICES_CSV.with_suffix(".backup.csv")
                backup.write_text(config.PRICES_CSV.read_text())
            out = incoming[PRICE_COLUMNS]
            if not replacing and config.PRICES_CSV.exists():
                existing = pd.read_csv(config.PRICES_CSV, dtype=str)[PRICE_COLUMNS]
                out = pd.concat([existing, out], ignore_index=True)
            out.to_csv(config.PRICES_CSV, index=False)
            get_panel.clear()
            st.success(f"Saved {len(out):,} rows. Retrain with `python train.py` "
                       "so the forecasts reflect this data.")

with tab_players:
    st.markdown(
        "Player statistics are optional. Without them the forecasts still run, "
        "just without the player performance inputs."
    )
    p_upload = st.file_uploader("Choose a CSV of player season statistics",
                                type="csv", key="players_upload")
    if p_upload is not None:
        try:
            stats = pd.read_csv(p_upload)
        except Exception as exc:
            st.error(f"That file could not be read as CSV: {exc}")
            st.stop()
        required = ["player", "season_start_year", "season_end_date",
                    "games_played", "points_per_game", "rebounds_per_game",
                    "assists_per_game"]
        missing = [c for c in required if c not in stats.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            ui.tiles([
                {"label": "Rows", "value": f"{len(stats):,}"},
                {"label": "Players", "value": f"{stats['player'].nunique():,}"},
            ])
            st.dataframe(stats.head(10), hide_index=True, use_container_width=True)
            st.caption("season_end_date matters most. A season only becomes "
                       "visible to the model after that date, which is what "
                       "keeps future performance out of past forecasts.")
            if st.button("Save player statistics", type="primary"):
                config.DATA_RAW.mkdir(parents=True, exist_ok=True)
                stats.to_csv(config.PLAYER_STATS_CSV, index=False)
                get_panel.clear()
                st.success("Saved. Retrain with `python train.py` to use it.")

with tab_format:
    st.markdown(
        "Your price file needs these columns. Only `date`, `price`, and enough "
        "identifying fields to tell one card from another are strictly "
        "required, but the more you fill in, the better the card matching and "
        "the features get."
    )
    st.dataframe(pd.DataFrame({
        "column": PRICE_COLUMNS,
        "example": ["2024-03-15", "(leave blank to auto-generate)",
                    "LeBron James", "2003", "Topps", "Topps Chrome", "111",
                    "Refractor", "true", "", "", "", "PSA", "10", "1450.00",
                    "my records", "https://example.com/listing"],
    }), hide_index=True, use_container_width=True)

    st.download_button(
        "Download a blank template",
        data=",".join(PRICE_COLUMNS) + "\n",
        file_name="sports_card_prices_template.csv",
        mime="text/csv",
    )
    st.caption(
        "One row per observation. Use the `source` and `source_url` columns to "
        "record where each price came from, so the provenance stays with the "
        "data. Only add observations you can legitimately collect, such as your "
        "own purchase and sale records or prices you noted manually."
    )
