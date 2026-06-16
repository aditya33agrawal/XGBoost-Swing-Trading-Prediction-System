"""Central configuration — all tuneable knobs live here."""
from __future__ import annotations
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

    # --- storage ---------------------------------------------------------
    data_dir: str = "data"
    db_path: str = "data/market.duckdb"
    raw_dir: str = "data/raw"
    model_dir: str = "models"
    mlflow_uri: str = "mlruns"

    # --- runtime ---------------------------------------------------------
    feature_cols: list = field(default_factory=list)
