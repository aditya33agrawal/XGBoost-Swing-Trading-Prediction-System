"""Central configuration — all tuneable knobs live here."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # --- universe & dates ------------------------------------------------
    start: str = "2015-01-01"
    end: str = "2025-12-31"

    # --- target / label --------------------------------------------------
    horizon: int = 5                      # swing horizon in trading days
    label_type: Literal["fwd_ret", "triple_barrier"] = "triple_barrier"
    barrier_up_mult: float = 2.0          # ×ATR upper barrier
    barrier_dn_mult: float = 2.0          # ×ATR lower barrier

    # --- walk-forward ----------------------------------------------------
    train_min_days: int = 504             # ~2 years before first prediction
    embargo: int = 5                      # ≥ horizon bars purge gap
    n_wf_splits: int = 8                  # walk-forward folds for tuning

    # --- rebalance -------------------------------------------------------
    rebalance_every: int = 5              # days between rebalance points
    n_quantile: int = 5                   # quintile for long/short selection
    mode: Literal["long_only", "long_short"] = "long_only"

    # --- costs -----------------------------------------------------------
    cost_bps_per_side: float = 20.0       # ~40 bps round-trip (STT+slip)

    # --- model -----------------------------------------------------------
    xgb_n_trials: int = 50               # Optuna trials
    xgb_early_stopping: int = 50
    xgb_seed: int = 42
    # "auto" → use GPU if one is visible (Colab), else CPU. Force with "cuda"/"cpu".
    device: Literal["auto", "cuda", "cpu"] = "auto"

    # --- storage ---------------------------------------------------------
    data_dir: str = "data"
    db_path: str = "data/market.duckdb"
    raw_dir: str = "data/raw"
    model_dir: str = "models"
    mlflow_uri: str = "mlruns"

    # --- outputs ---------------------------------------------------------
    output_dir:    str  = "outputs"
    save_outputs:  bool = True           # write signals JSON + CSV on every run

    # --- paper trading ---------------------------------------------------
    paper_trade:       bool  = True
    portfolio_path:    str   = "outputs/portfolio.json"
    initial_capital:   float = 1_000_000    # INR
    position_size_pct: float = 0.05         # 5 % of portfolio per trade
    max_positions:     int   = 10

    # --- fast-signals mode -----------------------------------------------
    skip_backtest: bool = False   # skip walk-forward + backtest, only gen signals
    params_path:   str  = "models/best_params.json"

    # --- Supabase (read from env if not passed explicitly) ---------------
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    # prefer the secret (server-side, full-access) key; fall back to a
    # publishable key or legacy SUPABASE_KEY if that's what's set.
    supabase_key: str = field(default_factory=lambda: (
        os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or ""
    ))

    # --- model versioning ------------------------------------------------
    model_version:   str  = ""    # auto-set to v{YYYYMMDD} at runtime if empty
    drive_output_dir: str = "MyDrive/swing_outputs"

    # --- outcome + Supabase feature flags --------------------------------
    resolve_outcomes_on_start: bool = True
    save_to_supabase:          bool = True

    # --- runtime ---------------------------------------------------------
    feature_cols: list = field(default_factory=list)
