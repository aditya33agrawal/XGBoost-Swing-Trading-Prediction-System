Auto-Improving Swing Trading System

Context

The existing pipeline (Colab + XGBoost + Nifty 50) already trains, backtests, and generates signals — but predictions disappear after each run. The goal is to close the feedback loop: save every
prediction, resolve actuals after horizon days, compare model performance week-over-week, and surface everything in a Streamlit UI. The user will paper-trade for 2 months before going live, so the trust
layer (performance history, IC trend, model health) is as important as the trading logic itself.

Decisions: Supabase (PostgreSQL), Streamlit UI, weekly retraining every Sunday.

---
Architecture

Colab (Sunday)
  → resolve last week's outcomes
  → retrain model (50 Optuna trials)
  → compare new OOF IC vs deployed model's 4-wk IC
  → deploy if better → save to Drive + Supabase
  → generate next week's signals → push to Supabase

Streamlit (always-on, Streamlit Community Cloud)
  ← reads Supabase (predictions, outcomes, paper_trades, model_runs)
  ← falls back to local JSON if no credentials
  → 5 pages: Signals, Portfolio, Performance, Model Health, Settings

---
Supabase SQL Schema

Run these in the Supabase dashboard SQL editor before wiring up the client.

-- One row per training run
CREATE TABLE model_runs (
    run_id          TEXT PRIMARY KEY,          -- e.g. "v20260617"
    model_version   TEXT NOT NULL,
    run_date        DATE NOT NULL,
    train_start     DATE, train_end DATE,
    horizon_days    INTEGER NOT NULL DEFAULT 5,
    label_type      TEXT NOT NULL DEFAULT 'triple_barrier',
    n_trials        INTEGER,
    oof_ic          FLOAT, oof_dir_acc FLOAT,
    bt_cagr FLOAT, bt_sharpe FLOAT, bt_sortino FLOAT, bt_calmar FLOAT,
    bt_max_drawdown FLOAT, bt_hit_rate FLOAT, bt_profit_factor FLOAT,
    best_params     JSONB,
    is_deployed     BOOLEAN NOT NULL DEFAULT FALSE,
    deployed_at     TIMESTAMPTZ,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per ticker per prediction date
CREATE TABLE predictions (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES model_runs(run_id),
    model_version   TEXT NOT NULL,
    signal_date     DATE NOT NULL,
    ticker          TEXT NOT NULL,
    signal          TEXT NOT NULL CHECK (signal IN ('LONG','NEUTRAL')),
    prob_up         FLOAT, entry_price FLOAT,
    stop_loss FLOAT, target_price FLOAT,
    risk_reward FLOAT, atr14 FLOAT,
    horizon_days    INTEGER NOT NULL DEFAULT 5,
    outcome_resolved   BOOLEAN NOT NULL DEFAULT FALSE,
    outcome_date       DATE,
    actual_close       FLOAT,
    actual_fwd_ret     FLOAT,
    hit_target BOOLEAN, hit_stop BOOLEAN,
    label_direction    INTEGER,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(signal_date, ticker, run_id)
);
CREATE INDEX ON predictions(signal_date DESC);
CREATE INDEX ON predictions(outcome_resolved, signal_date) WHERE outcome_resolved = FALSE;

-- Denormalised for fast IC queries (populated by outcome_tracker)
CREATE TABLE outcomes (
    id              BIGSERIAL PRIMARY KEY,
    prediction_id   BIGINT NOT NULL REFERENCES predictions(id),
    run_id          TEXT NOT NULL,
    signal_date     DATE NOT NULL,
    outcome_date    DATE NOT NULL,
    ticker          TEXT NOT NULL,
    prob_up         FLOAT,
    actual_fwd_ret  FLOAT NOT NULL,
    label_direction INTEGER NOT NULL,
    is_correct      BOOLEAN NOT NULL,
    exit_reason     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(prediction_id)
);
CREATE INDEX ON outcomes(signal_date DESC);

-- Synced from portfolio.json (upsert on trade_id)
CREATE TABLE paper_trades (
    trade_id        TEXT PRIMARY KEY,
    run_id          TEXT REFERENCES model_runs(run_id),
    ticker TEXT, entry_date DATE, entry_price FLOAT, shares INTEGER,
    signal TEXT DEFAULT 'LONG',
    stop_loss FLOAT, target_price FLOAT, horizon_days INTEGER, prob_up FLOAT,
    status          TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','closed')),
    exit_date DATE, exit_price FLOAT, exit_reason TEXT,
    pnl FLOAT, pnl_pct FLOAT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Top N features per run (for drift detection)
CREATE TABLE feature_importance (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES model_runs(run_id),
    feature_name    TEXT NOT NULL,
    importance_score FLOAT NOT NULL,
    importance_rank  INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id, feature_name)
);

---
New Files

src/
  db/
    __init__.py
    supabase_client.py          # get_client(), upsert_rows(), fetch_rows()
  tracking/
    __init__.py
    prediction_journal.py       # save_predictions(), save_run_metadata(), sync_paper_trades()
    outcome_tracker.py          # resolve_outcomes(), compute_recent_ic(), compute_weekly_ic_series()
  models/
    improvement.py              # get_model_version(), compare_models(), load_current_model_ic()

scripts/
  weekly_retrain.py             # standalone Sunday orchestrator

app/
  streamlit_app.py              # entry point, sidebar nav, session_state init
  pages/
    1_Signals.py
    2_Portfolio.py
    3_Performance.py
    4_Model_Health.py
    5_Settings.py
  utils/
    data_loader.py              # load_signals/trades/runs/ic/features — Supabase + JSON fallback
    charts.py                   # shared Plotly helpers (equity_curve, ic_trend, feature_bar)

notebooks/
  colab_weekly.ipynb            # 9-cell Sunday training notebook

.env.example

---
Modified Files

src/config.py

Add at the bottom of the dataclass:
# Supabase
supabase_url:              str  = ""   # from os.getenv("SUPABASE_URL", "")
supabase_key:              str  = ""   # from os.getenv("SUPABASE_KEY", "")
model_version:             str  = ""   # auto-set to v{YYYYMMDD} if empty
drive_output_dir:          str  = "MyDrive/swing_outputs"
resolve_outcomes_on_start: bool = True
save_to_supabase:          bool = True

src/pipeline/runner.py

After Phase 1b (price_df is clean, before Phase 2 training):
_sb_client = get_supabase_client(cfg.supabase_url, cfg.supabase_key)
if cfg.resolve_outcomes_on_start:
    n_resolved = resolve_outcomes(price_df, _sb_client, cfg.output_dir, cfg.horizon)
    print(f"  Resolved {n_resolved} past predictions")

After Phase 5c (signals saved to JSON, before Phase 6 paper trading):
if cfg.save_to_supabase and not signals.empty:
    _run_id = cfg.model_version or get_model_version()
    save_predictions(signals, _run_id, cfg.model_version, _sb_client, cfg.output_dir)
    save_run_metadata(stats, _run_id, cfg.model_version, best_params, None, _sb_client, cfg.output_dir)

After Phase 6 (portfolio saved):
if cfg.save_to_supabase and cfg.paper_trade:
    sync_paper_trades(portfolio, _run_id, _sb_client)

requirements.txt

Add: supabase>=2.0, streamlit>=1.35, plotly>=5.0, python-dotenv>=1.0

---
Key Function Signatures

src/db/supabase_client.py

def get_supabase_client(url: str = "", key: str = "") -> "Client | None":
    # reads SUPABASE_URL/KEY from env if args empty
    # returns None (never raises) — callers gracefully skip on None

def upsert_rows(client, table: str, rows: list[dict], on_conflict: str = "id") -> bool:
    # silently logs errors, returns False on failure

def fetch_rows(client, table: str, filters: dict | None = None,
               order_by: str | None = None, limit: int | None = None) -> list[dict]:
    # returns [] on any failure including client=None

src/tracking/prediction_journal.py

def save_predictions(signals_df, run_id, model_version, supabase_client, fallback_dir="outputs") -> bool
def save_run_metadata(stats_dict, run_id, model_version, best_params, feat_imp, supabase_client, fallback_dir="outputs") -> bool
def sync_paper_trades(portfolio, run_id, supabase_client) -> int  # returns rows synced

src/tracking/outcome_tracker.py

def resolve_outcomes(price_df, supabase_client, fallback_dir="outputs", horizon_days=5) -> int
    # finds unresolved predictions with signal_date <= today - horizon_days
    # computes actual_fwd_ret = ln(close_{t+h} / entry_price), hit_target, hit_stop
    # updates predictions table + inserts into outcomes table

def compute_recent_ic(supabase_client, n_weeks=4, fallback_dir="outputs") -> float
    # Spearman IC between prob_up and actual_fwd_ret on last n_weeks of outcomes

def compute_weekly_ic_series(supabase_client, n_weeks=12, fallback_dir="outputs") -> pd.DataFrame
    # returns DataFrame[week_start, ic, n_obs, dir_accuracy]

src/models/improvement.py

def get_model_version(prefix="v") -> str  # "v20260617"
def compare_models(new_ic, current_ic, min_improvement=0.005) -> bool
    # deploy only if new_ic >= current_ic + min_improvement OR current_ic is NaN
def load_current_model_ic(supabase_client, fallback_dir="outputs") -> tuple[float, str]
    # returns (4-week rolling IC of deployed model, run_id)

---
Streamlit Pages (what each reads + shows)

┌────────────────┬────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────┐
│      Page      │                Data Source                 │                                        Key Content                                         │
├────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1_Signals      │ predictions table / signals_latest.csv     │ Sortable signal table (LONG=green), prob_up gradient, entry/stop/target                    │
├────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2_Portfolio    │ paper_trades table / portfolio.json        │ Portfolio value gauge, open positions + unrealised P&L, equity curve, closed trade history │
├────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3_Performance  │ outcomes table grouped by week             │ IC over time (line), directional accuracy (bar), exit reason breakdown (pie)               │
├────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4_Model_Health │ outcomes + model_runs + feature_importance │ IC health banner (GREEN/AMBER/RED), top-20 feature importance, model comparison table      │
├────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5_Settings     │ open paper_trades                          │ Manual open/close position, "Trigger resolve outcomes" button, config display              │
└────────────────┴────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────┘

Supabase credentials in Streamlit Cloud: Set under App Settings > Secrets as SUPABASE_URL and SUPABASE_KEY. data_loader.py reads st.secrets first, then os.getenv.

---
Colab Weekly Notebook (notebooks/colab_weekly.ipynb) — 9 Cells

1. GPU check + load Colab Secrets (SUPABASE_URL, SUPABASE_KEY) into os.environ
2. Mount Drive, git pull repo, copy models/best_params.json from Drive → repo
3. pip install -q optuna yfinance supabase
4. Resolve outcomes: fetch recent prices → resolve_outcomes() → compute_recent_ic() → print 4-wk IC
5. Retrain: run(cfg) with --trials 50, resolve_outcomes_on_start=False (already done)
6. Compare: load_current_model_ic() vs new OOF IC → compare_models()
7. Deploy: if better → upsert is_deployed=True in model_runs, copy models/ to Drive
8. Generate signals: run(cfg) with skip_backtest=True, xgb_n_trials=0 → push to Supabase
9. Summary: print IC trend table, deployed model version, signal count

---
Implementation Order (Phased)

Phase A — Foundation (2 days): Predictions persisted

1. src/config.py — add 6 fields
2. src/db/supabase_client.py — client + upsert/fetch
3. Run Supabase SQL schema
4. src/tracking/prediction_journal.py — save_predictions, save_run_metadata
5. Wire Phase 5e into runner.py
6. ✅ Test: python run_pipeline.py --fast-signals → rows appear in Supabase predictions table

Phase B — Feedback loop (2 days): Outcomes resolved

7. src/tracking/outcome_tracker.py — resolve_outcomes, compute_recent_ic
8. Wire Phase 1b call into runner.py (after validation gates)
9. sync_paper_trades() + wire after Phase 6
10. ✅ Test: second run resolves first week's predictions; outcomes table populated

Phase C — Self-improvement (1 day): Deploy decision works

11. src/models/improvement.py — all 3 functions
12. scripts/weekly_retrain.py
13. ✅ Test: python scripts/weekly_retrain.py completes, marks run as deployed in Supabase

Phase D — Colab notebook (1 day)

14. notebooks/colab_weekly.ipynb — 9 cells
15. ✅ Test: run cells 1–4 in Colab (skip training cell with --trials 5 for speed)

Phase E — Streamlit UI (4 days): Full observation layer

16. app/utils/data_loader.py + app/utils/charts.py
17. Pages 1–5 in order of value: Signals → Portfolio → Performance → Model Health → Settings
18. app/streamlit_app.py entry point
19. ✅ Test locally: streamlit run app/streamlit_app.py
20. ✅ Deploy: push to GitHub → connect to Streamlit Community Cloud → add secrets

Phase F — Polish (1-2 days)

21. requirements.txt additions, .env.example
22. run_pipeline.py new flags: --model-version, --no-supabase

---
Environment Variables

┌───────────────┬────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│   Variable    │               Where Set                │                            Notes                            │
├───────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ SUPABASE_URL  │ Colab Secrets, .env, Streamlit Secrets │ Required for DB                                             │
├───────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ SUPABASE_KEY  │ Colab Secrets, .env, Streamlit Secrets │ Use anon key (Row Level Security disabled for this project) │
├───────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ MODEL_VERSION │ Optional                               │ Auto-generates v{YYYYMMDD} if empty                         │
├───────────────┼────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ LOG_LEVEL     │ Optional                               │ Default INFO                                                │
└───────────────┴────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘

Graceful fallback contract: If SUPABASE_URL/KEY are empty, get_supabase_client() returns None. Every caller that receives None skips the write/read and falls back to JSON files in outputs/. Pipeline
runs fully without any credentials — essential for development and CI.

---
Verification

1. Phase A: Run python run_pipeline.py --fast-signals --no-paper-trade → check Supabase predictions table has rows
2. Phase B: Run pipeline twice 5+ days apart → check outcomes table populated after second run
3. Phase C: python scripts/weekly_retrain.py → model_runs table shows is_deployed=True for new run
4. End-to-end IC: After 2 weeks of resolved predictions, compute_recent_ic() returns a non-NaN float
5. Streamlit: streamlit run app/streamlit_app.py with SUPABASE_URL unset → app loads from local JSON fallback; with creds set → shows live Supabase data
6. Deploy gate: Manually set current_ic = 0.05 and new_ic = 0.04 in compare_models test → confirms new model NOT deployed
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
