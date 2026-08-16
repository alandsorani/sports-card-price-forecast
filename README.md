# Sports Card Price Forecast

Forecasts the future price of physical basketball cards at **30 / 90 / 180 /
365 days** with calibrated prediction intervals, using only information
knowable at the prediction date.

**Research question:** *Given everything that was knowable about a sports card
at time T, can we predict its price at T+30, T+90, T+180, and T+365 days?*

## What's in here

- **Leakage-audited feature engineering.** Lagged prices, returns,
  volatilities, expanding highs and lows, momentum, a cross-card market index,
  card metadata, and player statistics, each documented with the cutoff date it
  respects. An automated mutation test perturbs all future observations and
  asserts that no past feature value changes.
- **Walk-forward validation over expanding calendar-year windows**, never a
  random split, with four naive baselines that machine-learning models have to
  beat. When they don't, the baseline is what ships.
- **Split-conformal prediction intervals** calibrated on out-of-sample relative
  residuals, with coverage measured on a fold the calibration never touched.
- **A true historical backtest** that rebuilds features and retrains from each
  simulation date's information set, using only labels that had matured by then.
- **A Streamlit app** with card search, forecasts, price history, player
  analysis, comparable cards, a collection tracker, a market dashboard, and a
  model performance page. It ships loaded with generated demo data.
- **Three CLIs**: `train.py`, `backtest.py`, `predict.py`.

## Running it locally

Everything runs on your own machine. There are no API keys, no accounts, and
no network calls at runtime.

**Prerequisites:** Python 3.12 (what this is developed and tested against;
3.10+ should work) and git. Check with `python3 --version`.

### 1. Clone and install

```bash
git clone https://github.com/alandsorani/sports-card-price-forecast.git
cd sports-card-price-forecast
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Installation pulls
down pandas, scikit-learn, plotly, and streamlit, and takes a minute or two.

### 2. Get some data in

The repo ships no price data, so start with the generated demo set:

```bash
python -m src.data.synthetic
```

This writes ~6,700 rows across 20 cards to `data/raw/sports_card_prices.csv`,
every row labeled `source=SYNTHETIC`. It is random-walk data for exercising the
pipeline, not real prices. Replace that file with real observations whenever
you have them.

### 3. Train the models

```bash
python train.py
```

Walk-forward validation across four horizons and several model families, so
expect around five minutes. It prints which model won at each horizon and
saves them to `models/`. Until this finishes, the app runs but has no forecasts
to show.

### 4. Start the app

```bash
streamlit run app/Home.py
```

Your browser opens at `http://localhost:8501`. Stop the server with `Ctrl+C`.

### Optional

```bash
# Replay history and grade the forecasts against what actually happened
python backtest.py

# Forecast a single card from the terminal
python predict.py --player "LeBron James" --year 2003 --set "Topps Chrome" \
    --card-number 111 --grade 10 --grading-company PSA

# Run the test suite
pytest
```

### If something goes wrong

- **`FileNotFoundError` about `sports_card_prices.csv`** — step 2 has not run.
- **"Models not trained yet" in the app** — step 3 has not run, or did not finish.
- **Port already in use** — `streamlit run app/Home.py --server.port 8502`.
- **Edited something under `src/` and the app did not pick it up** — restart the
  server. Streamlit re-runs the page script but keeps already-imported modules
  in memory.

## Pipeline

1. **Load & clean** (`src/data/load.py`) — parses the template CSV, flags
   impossible dates/prices (future dates, pre-1980, non-positive), collapses
   duplicates, aggregates same-day observations to their **median** (robust to
   extreme sales). Chart-style point-in-time values are used directly and never
   presented as individual transactions.
2. **Card identity** (`src/data/schema.py`) — `card_id` concatenates every
   populated identity field (year, manufacturer, set, player, number, parallel,
   serial, autograph/memorabilia, grading company, grade). `PSA 10` ≠ `PSA 9` ≠
   `Refractor PSA 10`. No mathematical relationship between grades is assumed —
   each card/grade combination is its own series.
3. **Features** (`src/features/timeseries.py`) — lagged prices and returns (7,
   30, 60, 90, 180, 365d), rolling volatilities (30/90/180d, annualized from
   irregular gaps), expanding high/low and distances, momentum, acceleration,
   observation bookkeeping, cross-card market index (median 30d return,
   momentum, volume) merged **backward** as-of each date, card age (from Jan 1
   of the printed year — exact release dates are rarely public), rookie flag
   from metadata (never inferred from debut season alone).
4. **Player features** (`src/data/player_stats.py`) — season stats join via
   `merge_asof(direction="backward")` on `season_end_date`, so a season is
   visible only after it ended. Predicting on Jan 1, 2022 sees the 2020-21
   season, never the in-progress or final 2021-22 season.
5. **Targets** (`src/features/targets.py`) — for horizon H at date T: the
   card's observation closest to T+H within ±max(10, 0.15·H) days; otherwise
   NaN (row unlabeled for that horizon). `target_gap_days_*` records the actual
   distance. Both `future_price_*` and `future_return_*` are created; training
   compares forecasting price (raw and log1p) vs return and keeps the best
   out-of-sample configuration per horizon.

## Models & validation

- **Baselines** — last price, 30d moving median, 90d moving median,
  momentum-adjusted extrapolation.
- **ML zoo** — Ridge, Random Forest, Gradient Boosting,
  HistGradientBoosting (+ XGBoost/LightGBM if installed). No deep learning:
  the expected data volumes (10²–10⁴ labeled rows) do not justify it.
- **Validation** — expanding-window walk-forward over calendar years; never a
  random split. The newest year is the final holdout.
- **Selection** — lowest mean out-of-sample MAE; if no ML model beats the best
  baseline, that fact is printed and stored in the model bundle
  (`beat_baseline: false`) and shown in the app.
- **Intervals** — split-conformal on out-of-sample *relative* residuals
  (finite-sample corrected quantiles), calibrated on all folds except the last;
  coverage is then measured on the untouched final fold and reported.
- **Backtest** (`backtest.py`) — at each simulation date, the raw data is
  truncated, features rebuilt, and models retrained using only labels that had
  matured by that date; forecasts are stored and only afterwards compared to
  actual future prices.
- **Leakage audit** — `src/evaluation/leakage.py` documents every feature's
  cutoff and runs a mutation test (perturb all future rows → past features must
  be bit-identical). Enforced in `tests/test_leakage.py`.

## Results on the synthetic demo data

**These numbers describe the synthetic random walk shipped by
`python -m src.data.synthetic`, not any real card market.** They are included
to show the pipeline runs end to end and that model selection behaves
correctly. Panel: 6,733 rows, 20 cards, 2019-01-05 → 2026-08-01, 41 features.

| Horizon | Deployed | Why | Interval coverage (final holdout) |
|---|---|---|---|
| 30d | `random_forest` (target=return) | Beat best baseline: MAE 2538 vs 2723 | 91.0% (n=476) |
| 90d | `last_price` **baseline** | No ML model beat it: 6280 vs 6073 | 92.9% (n=336) |
| 180d | `moving_median_90d` **baseline** | No ML model beat it: 7550 vs 7103 | 96.7% (n=151) |
| 365d | `gradient_boosting` (target=return) | Beat best baseline: MAE 6807 vs 9143 | 93.7% (n=654) |

Historical backtest (`python backtest.py`, `hist_gradient_boosting`, retrained
from scratch at each simulation date on labels that had matured by that date):

| Horizon | n | MAE | RMSE | R² | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| 30d | 499 | 2495.40 | 16404.84 | 0.885 | 62.1% |
| 90d | 482 | 5253.86 | 30301.78 | 0.639 | 63.5% |
| 180d | 480 | 7072.43 | 37288.95 | 0.490 | 62.9% |
| 365d | 440 | 8604.06 | 41456.71 | 0.406 | 46.8% |

Accuracy decays monotonically with horizon, and directional accuracy at 365d
(46.8%) is below a coin flip — the expected result on a random walk, and the
kind of honest negative finding this pipeline is built to surface.

Two things worth reading from the deployment table. First, **baselines win at 90d and
180d, and the baseline is what gets deployed** — random-walk data is exactly
the case where last price is hard to beat, and the pipeline reports that
instead of shipping a fancier model that scores worse. Second, 90% conformal
intervals land at 91–97% observed coverage: correctly calibrated at 30d/365d
and conservative (too wide) at 180d, where only 151 holdout points are
available. `reports/model_comparison.csv` has the full per-fold grid.

## Forecast presentation

- Point forecast + 90% conformal interval per horizon.
- **Reliability** (High/Medium/Low) — rule-based on observation count,
  freshness, recent density, volatility, and interval width. Deliberately *not*
  quoted as a probability; nothing here is calibrated to be one.
- **Cold start** — under 10 observations: no model forecast, an explicit
  "Limited historical data" message, and comparable cards instead.
- **Comparable cards** — metadata + price-level + volatility similarity,
  always labeled *Comparable*, never *Identical*.
- **Collection** — per-card current value, forecasts, unrealized gain;
  portfolio totals sum per-card interval bounds and are labeled as a rough
  band, not an exact joint interval (cross-card correlation is not modeled).

## Interface & chart design

The Streamlit app runs on a small shared design system rather than default
widgets: `src/visualization/theme.py` holds the tokens and the Plotly template,
`app/ui.py` the CSS and components (KPI tiles, forecast cards, status badges).

- **Palette is validated, not eyeballed.** The three categorical slots (blue
  `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`) were checked against the
  `#fdfcfa` chart surface for lightness band, chroma floor, colorblind
  separation, and normal-vision separation, all of which pass. Aqua sits at
  2.75:1 contrast, under the 3:1 bar, so it is never the sole carrier of a
  value: direct labels and a table view accompany every chart.
- **Warm light surfaces** (`#f6f4f0` page, `#fdfcfa` cards), 18px radii, and
  soft shadows rather than hard borders.
- **Plain, friendly copy with no em dashes.** Ranges read "likely between $X
  and $Y" rather than using dash notation.
- **No dual-axis charts.** Card price versus player scoring is drawn as small
  multiples on a shared time axis. Putting two different units on one plot with
  two y-scales invents a correlation the data does not contain, and the
  alignment of the two scales would be arbitrary.
- **Reliability never rides on color alone.** Each badge pairs its status color
  with the word High / Medium / Low.
- **Marks:** 2px lines, ≥8px markers with a 2px surface ring, interval bands as
  a ~10% wash, hairline solid gridlines, and endpoint-only direct labels rather
  than a number on every point. Every chart has a table view beside it.
- **Forecasts read as projections:** the forecast line is dashed and separated
  from observed history by a vertical rule at the as-of date.

## Project structure

```
├── README.md
├── requirements.txt / .env.example / .gitignore
├── data/{raw,interim,processed,external}/
├── notebooks/01..07 (audit → EDA → features → baselines → modeling → backtest → analysis)
├── src/{data,features,models,forecasting,evaluation,visualization}/
├── models/          # trained bundles (model_{30,90,180,365}d.pkl)
├── reports/         # model_comparison.csv, backtest_*.csv
├── app/             # Streamlit (Home + 9 pages)
├── tests/
├── train.py / backtest.py / predict.py
```

## Limitations

- **Results shipped in this repo describe synthetic demo data only.** Real
  accuracy numbers exist only after you supply real observations and rerun
  `train.py`/`backtest.py`.
- Reported prices in the template are whatever your sources are — chart values
  and transactions must not be mixed carelessly (keep `source` accurate).
- Card age uses Jan-1-of-year as release proxy; in-season player stats are not
  yet implemented (season-level only); population reports are not included.
- Basketball only for now; the schema carries a `sport` column so baseball,
  football, hockey, and soccer can be added without structural change.
- Not investment advice.

## Future improvements

- Game-level player stats for in-season to-date features.
- Cross-card correlation model for honest portfolio intervals.
- Quantile-regression intervals as an alternative to split conformal.

## License

Released under the [MIT License](LICENSE).

The license covers this source code. It does not extend to any price data you
load into it: whatever you place in `data/raw/` stays governed by the terms of
wherever you obtained it.
