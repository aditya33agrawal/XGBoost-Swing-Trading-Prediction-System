# Nifty 50 Swing-Trading Prediction Pipeline

An end-to-end machine-learning pipeline that predicts short-term (swing) moves for
the Nifty 50 universe using **XGBoost**, walk-forward validation, Indian-market
cost modelling, paper trading, and a **Streamlit** dashboard.

The system is designed to be **retrained weekly** (typically on Google Colab),
auto-deploy only when the new model beats the current one, log every prediction,
resolve outcomes after the swing horizon, and surface everything in a dashboard.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Project layout](#project-layout)
3. [Quick start](#quick-start)
4. [Configuration & environment](#configuration--environment)
5. [Training the model](#training-the-model)
6. [Generating signals (inference)](#generating-signals-inference)
7. [Weekly retrain & auto-deploy](#weekly-retrain--auto-deploy)
8. [Running on Google Colab](#running-on-google-colab)
9. [The dashboard](#the-dashboard)
10. [Outputs & data files](#outputs--data-files)
11. [How the model works](#how-the-model-works)
12. [Interpreting the results](#interpreting-the-results)
13. [Troubleshooting](#troubleshooting)
14. [Disclaimer](#disclaimer)

---

## What it does

For a fixed universe of 50 large-cap NSE stocks, on every run the pipeline:

1. **Ingests** daily OHLCV (via `yfinance`, with a synthetic GBM fallback so the
   pipeline never hard-fails offline).
2. **Validates** the data (OHLCV sanity gates, spike detection, date-gap checks).
3. **Engineers ~52 features** — momentum, oscillators, volatility, volume, regime,
   and cross-sectional rank features.
4. **Labels** each row with either a **triple-barrier** (ATR-based, classification)
   or **forward log-return** (regression) target over the swing `horizon`.
5. **Tunes** XGBoost hyperparameters with **Optuna** (IC objective, purged
   walk-forward CV).
6. **Walk-forward backtests** with **purge + embargo** to avoid look-ahead leakage,
   applying a realistic **Indian round-trip cost model** (STT + stamp duty + GST + DP).
7. **Generates today's signals** — top-N LONG picks with entry / stop-loss / target.
8. **Paper-trades** the signals in a tracked portfolio.
9. **Logs** predictions + run metadata to **Supabase** (or local JSON fallback),
   and **resolves** past predictions once the horizon has elapsed.

---

## Project layout

```
predictive_modelling/
├── run_pipeline.py            # ← main CLI entry point
├── requirements.txt           # full dependency set
├── .env.example               # copy to .env and fill in Supabase creds
│
├── src/
│   ├── config.py              # Config dataclass — ALL tunable knobs
│   ├── exchange_calendar.py   # NSE holidays / trading days
│   ├── logging_setup.py
│   ├── data/
│   │   ├── ingestion.py       # yfinance + synthetic fallback, UNIVERSE list
│   │   ├── storage.py         # DuckDB + Parquet (optional/best-effort)
│   │   └── validation.py      # data-quality gates
│   ├── features/engineer.py   # ~52-feature catalog
│   ├── labels/
│   │   ├── targets.py         # triple_barrier + forward_log_return
│   │   └── weights.py         # uniqueness × time-decay sample weights
│   ├── models/
│   │   ├── trainer.py         # XGBoost + Optuna, device handling
│   │   ├── calibration.py     # isotonic probability calibration
│   │   └── improvement.py     # model versioning + deploy decision
│   ├── validation/
│   │   ├── walk_forward.py    # PurgedWalkForward splitter
│   │   └── metrics.py         # IC, Sharpe, Sortino, Calmar, ECE, …
│   ├── backtest/
│   │   ├── costs.py           # Indian transaction-cost model
│   │   └── engine.py          # vectorized cross-sectional backtest
│   ├── trading/
│   │   ├── signals.py         # enrich signals w/ entry/stop/target, save
│   │   └── paper_trader.py    # PaperPortfolio
│   ├── tracking/
│   │   ├── prediction_journal.py  # save predictions + run metadata
│   │   └── outcome_tracker.py     # resolve outcomes, rolling IC
│   ├── db/supabase_client.py
│   └── pipeline/runner.py     # the orchestrator (all phases)
│
├── scripts/weekly_retrain.py  # Sunday retrain + auto-deploy
├── notebooks/
│   ├── colab_train.ipynb      # one-shot training on Colab (GPU)
│   └── colab_weekly.ipynb     # scheduled weekly retrain on Colab
│
├── app/                       # Streamlit dashboard
│   ├── streamlit_app.py       # entry point
│   ├── pages/                 # Signals / Portfolio / Performance / Health / Settings
│   └── requirements.txt       # minimal deps for Streamlit Cloud
│
├── data/                      # DuckDB + raw Parquet (gitignored data)
├── models/                    # best_params.json, model artifacts
└── outputs/                   # signals JSON/CSV, portfolio.json (created on run)
```

---

## Quick start

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) configure Supabase for live DB sync
cp .env.example .env        # then edit .env — see "Configuration" below

# 4. fastest end-to-end smoke test (no hyperparameter tuning)
python run_pipeline.py --trials 0

# 5. real run with tuning + walk-forward backtest
python run_pipeline.py --trials 50
```

> **No Supabase?** That's fine — everything falls back to local JSON files in
> `outputs/`. Supabase is only needed if you want the dashboard to read live,
> shared data.

---

## Configuration & environment

All defaults live in [`src/config.py`](src/config.py). Every CLI flag in
`run_pipeline.py` maps onto a field there. The most important knobs:

| Field | Default | Meaning |
|-------|---------|---------|
| `start` / `end` | `2015-01-01` / `2025-12-31` | history window |
| `horizon` | `5` | swing horizon in **trading days** |
| `label_type` | `triple_barrier` | `triple_barrier` (classification) or `fwd_ret` (regression) |
| `mode` | `long_only` | `long_only` or `long_short` (needs F&O) |
| `xgb_n_trials` | `50` | Optuna trials (`0` = skip, load saved params) |
| `device` | `auto` | `auto` / `cuda` / `cpu` for XGBoost |
| `cost_bps_per_side` | `20.0` | ~40 bps round-trip assumption |
| `initial_capital` | `1_000_000` | paper-trading capital (INR) |
| `position_size_pct` | `0.05` | 5% of portfolio per position |
| `max_positions` | `10` | max concurrent open positions |

### Environment variables (`.env`)

Copy `.env.example` → `.env`:

```ini
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SECRET_KEY=sb_secret_xxx          # needed for pipeline writes
SUPABASE_PUBLISHABLE_KEY=sb_publishable_xxx # enough for read-only dashboard
# SUPABASE_KEY=...                          # legacy key still works
# MODEL_VERSION=v20260617                   # auto = v{YYYYMMDD} if unset
# LOG_LEVEL=INFO
```

- The **pipeline** needs the *secret* key (write access).
- The **dashboard** only needs the *publishable* key (read-only).
- If no key is set, the pipeline runs in **JSON-only** mode.

---

## Training the model

Training, tuning, validation, and signal generation all happen in a single run of
the orchestrator. Use the flags to control how much work it does.

```bash
# Full production train: 50 Optuna trials + walk-forward backtest + paper trading
python run_pipeline.py --trials 50

# More tuning for a (marginally) better model
python run_pipeline.py --trials 100

# Train on a specific window
python run_pipeline.py --start 2018-01-01 --end 2024-12-31

# Different swing horizon (e.g. 10 trading days)
python run_pipeline.py --horizon 10

# Regression mode — predict forward returns instead of barrier labels
python run_pipeline.py --label-type fwd_ret

# Long/short mode (requires F&O capability to act on shorts)
python run_pipeline.py --mode long_short

# Force CPU or GPU
python run_pipeline.py --device cpu
python run_pipeline.py --device cuda
```

### What happens during a full run (phases)

1. **Data ingestion** → fetch prices + index, persist to DuckDB/Parquet (best-effort).
2. **Validation gates** → abort on bad data.
3. **Resolve past predictions** → score predictions whose horizon has elapsed.
4. **Feature engineering** → ~52 features.
5. **Labels + sample weights**.
6. **Optuna hyperparameter search** → best params saved to `models/best_params.json`.
7. **Probability calibration** (classification) → isotonic, ECE reported.
8. **Walk-forward OOF predictions + cost-adjusted backtest** → IC, Sharpe, etc.
9. **Latest signals** → top-N picks, enriched with entry/stop/target.
10. **Paper trading** → portfolio updated and saved.

The tuned hyperparameters persist to `models/best_params.json`, so later
`--trials 0` / `--fast-signals` runs reuse them without re-tuning.

---

## Generating signals (inference)

Once you have a tuned `models/best_params.json`, you can produce fresh signals
cheaply without re-running Optuna or the backtest:

```bash
# Fast signals only — loads saved params, skips Optuna + backtest (~2 min on CPU)
python run_pipeline.py --fast-signals

# Tune-skipped but still backtest
python run_pipeline.py --trials 0

# Generate signals without touching the paper portfolio
python run_pipeline.py --fast-signals --no-paper-trade

# Don't write any output files
python run_pipeline.py --fast-signals --no-save
```

Signals are saved to `outputs/` as JSON + CSV (and to Supabase if configured),
and printed as a formatted table. Each signal includes the model score
(`prob_up` for classification, `pred_return` for regression) plus ATR-based
**entry**, **stop-loss**, and **target** levels.

### Useful flags reference

| Flag | Effect |
|------|--------|
| `--fast-signals` | skip Optuna + walk-forward; only score today (forces `--trials 0`) |
| `--skip-backtest` | still tune, but skip the walk-forward backtest |
| `--trials N` | number of Optuna trials (`0` loads saved params) |
| `--no-paper-trade` | don't update the paper portfolio |
| `--no-save` | don't write `outputs/` files |
| `--no-supabase` | force JSON-only mode |
| `--capital / --position-size / --max-positions` | paper-trading sizing |
| `--model-version` | override the auto `v{YYYYMMDD}` tag |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` |

Run `python run_pipeline.py --help` for the complete list.

---

## Weekly retrain & auto-deploy

[`scripts/weekly_retrain.py`](scripts/weekly_retrain.py) is the intended
production cadence — run it every **Sunday** (manually, on Colab, or via a
scheduler). It:

1. Connects to Supabase.
2. Fetches fresh prices and **resolves** last week's predictions.
3. Computes the **4-week rolling IC** of the currently deployed model.
4. Runs a **full retrain** (Optuna + walk-forward).
5. **Compares** new model IC vs deployed model IC.
6. **Deploys only if the new model is better** (updates Supabase + saves artifacts;
   syncs to Google Drive when on Colab).
7. Generates next week's signals (fast mode).
8. Prints a summary report with the 8-week IC trend.

```bash
export SUPABASE_URL=https://xxx.supabase.co
export SUPABASE_KEY=your-key

python scripts/weekly_retrain.py                 # standard run
python scripts/weekly_retrain.py --trials 100    # more tuning
python scripts/weekly_retrain.py --force-deploy  # deploy regardless of IC
python scripts/weekly_retrain.py --skip-deploy   # compare but don't flip the flag
```

---

## Running on Google Colab

Two ready-made notebooks live in `notebooks/`:

- **`colab_train.ipynb`** — one-shot training (use a GPU runtime; the pipeline
  auto-detects CUDA via `device=auto`).
- **`colab_weekly.ipynb`** — the scheduled weekly retrain flow.

Typical Colab steps:

1. Open the notebook in Colab and select a **GPU** runtime.
2. Mount Google Drive (for persisting `models/` and `outputs/`).
3. Set `SUPABASE_URL` / `SUPABASE_KEY` (Colab secrets or env).
4. Run all cells — the notebook clones/loads the repo, installs deps, and runs
   the pipeline / weekly retrain.

The last production Colab run (50 trials, 2015–2025, CUDA, 50 tickers / ~129k rows)
produced **OOF IC ≈ 0.014, directional accuracy ≈ 54%, Sharpe ≈ 0.64**.

---

## The dashboard

A Streamlit app in `app/` visualizes signals, portfolio, and model health.

```bash
streamlit run app/streamlit_app.py
```

Pages:

- **📡 Signals** — this week's LONG picks with entry / stop / target and confidence.
- **💼 Portfolio** — paper-trading equity curve, open/closed trades, win rate, P&L.
- **📈 Performance** — IC trend, directional accuracy.
- **🩺 Model Health** — IC trend, feature importance, version history
  (GREEN = IC > 0.01, AMBER = 0–0.01, RED < 0).
- **⚙️ Settings** — controls and config.

It reads from Supabase when configured (publishable key is enough), otherwise from
the local JSON files in `outputs/`. For **Streamlit Cloud**, put credentials in
the app's *Secrets* and use `app/requirements.txt` (the trimmed dependency set).

---

## Outputs & data files

| Path | Created by | Contents |
|------|-----------|----------|
| `models/best_params.json` | Optuna phase | tuned XGBoost hyperparameters |
| `outputs/signals_*.json` / `.csv` | signals phase | latest top-N signals |
| `outputs/portfolio.json` | paper trading | open/closed positions, equity |
| `data/market.duckdb` | storage | upserted OHLCV |
| `data/raw/prices.parquet` | storage | raw price snapshot |
| Supabase tables | tracking | predictions, model_runs, paper_trades |

Storage to DuckDB/Parquet is **best-effort** — a storage hiccup never kills a run.

---

## How the model works

- **Universe**: 50 large-cap NSE tickers (see `UNIVERSE` in `src/data/ingestion.py`).
- **Features (~52)**: momentum, oscillators (RSI etc.), realized volatility, volume,
  market-regime, and **cross-sectional rank** features computed per date.
- **Labels**:
  - *triple_barrier* (default, classification): up/down/timeout barriers placed at
    `±barrier_mult × ATR` over the horizon → predicts probability of an up move.
  - *fwd_ret* (regression): forward log-return over the horizon.
- **Sample weights**: label-uniqueness × time-decay (à la López de Prado) so
  overlapping labels don't double-count.
- **Validation**: `PurgedWalkForward` with **purge + embargo** equal to the horizon
  to remove look-ahead leakage between train and test.
- **Tuning**: Optuna maximizing **Information Coefficient** across CV folds.
- **Calibration**: time-ordered isotonic regression; ECE measured on held-out data.
  > Note: the calibrator is **not** applied to the final latest-signal scoring —
  > raw probabilities preserve the cross-sectional ranking that signal selection
  > depends on (see `predict_latest` in `runner.py`).
- **Backtest**: vectorized cross-sectional quintile strategy with the full Indian
  round-trip cost model, plus a cost-sensitivity sweep (0.5× / 1× / 2×).

---

## Interpreting the results

This is a **small-edge** problem. On real Nifty data the achievable edge is thin
(roughly **52–56%** directional accuracy, **IC ~0.01–0.02**). Treat the metrics
accordingly:

- **IC (Information Coefficient)** — rank correlation of predictions vs realized
  returns. Above ~0.01 is meaningful here; below 0 is worse than random.
- **Sharpe / Sortino / Calmar** — risk-adjusted returns after costs.
- **Cost sensitivity** — if Sharpe collapses at 2× costs, the edge is fragile.

On synthetic / low-signal data, a **near-zero Sharpe after costs is the correct
result** — the pipeline prints a reminder to this effect. Don't over-fit to a
single lucky window; the walk-forward + cost model exist precisely to keep you
honest.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `yfinance` returns nothing | network/rate-limit — the pipeline falls back to synthetic GBM data; retry later for real data |
| `duckdb` import / storage warning | storage is optional; the run continues. Install `duckdb` to enable persistence |
| All signals collapse to the same score | usually early-stopping stopping at round 0 on sparse data — use `--trials >0` to get proper `n_estimators`; the latest-signal path already avoids this |
| "Supabase not configured" | expected without `.env` creds — runs in JSON mode |
| GPU not used on Colab | ensure a GPU runtime and `--device cuda` (or `auto`) |
| `--fast-signals` says no saved params | run a full `--trials 50` once to create `models/best_params.json` |

---

## Disclaimer

This project is for **research and education**. It is **not** investment advice.
Paper-trade for a meaningful period (the dashboard suggests ~2 months) before
even considering live capital, and understand that the historical edge here is
small and may not persist. Trade at your own risk.
