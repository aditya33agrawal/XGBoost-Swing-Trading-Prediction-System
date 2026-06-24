"""Central configuration — all tuneable knobs live here."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


def _today() -> str:
    return date.today().isoformat()


@dataclass
class Config:
    # --- universe & dates ------------------------------------------------
    start: str = "2015-01-01"
    end: str = field(default_factory=_today)

    # --- target / label --------------------------------------------------
    horizon: int = 5                      # swing horizon in trading days
    label_type: Literal["fwd_ret", "triple_barrier"] = "triple_barrier"
    barrier_up_mult: float = 2.0          # ×ATR upper barrier (label only)
    barrier_dn_mult: float = 2.0          # ×ATR lower barrier (label only)

    # --- signal exit levels (paper/live trades — decoupled from label) ---
    # The label barriers above define the classification target; they are NOT
    # the stop/target the paper trader executes.  A symmetric ±2 ATR pair gives
    # a 1:1 payoff which bleeds at a sub-0.5 hit rate.  These let winners run.
    signal_stop_atr_mult:   float = 1.5   # ×ATR below entry → stop loss
    signal_target_atr_mult: float = 3.0   # ×ATR above entry → take profit (RR≈2.0)

    # --- walk-forward ----------------------------------------------------
    train_min_days: int = 504             # ~2 years before first prediction
    embargo: int = 5                      # ≥ horizon bars purge gap
    n_wf_splits: int = 8                  # walk-forward folds for tuning
    wf_es_val_days: int = 63              # early-stopping val window (~3 months)

    # --- rebalance -------------------------------------------------------
    rebalance_every: int = 5              # days between rebalance points
    n_quantile: int = 5                   # quintile for long/short selection
    mode: Literal["long_only", "long_short"] = "long_only"
    # Weight each basket member by score-rank conviction instead of equal
    # weight (plan §Phase 4.19) — targets weak Calmar/drawdown without
    # touching the underlying alpha signal. False = old equal-weight behaviour.
    conviction_weighted_sizing: bool = True

    # --- risk overlay ----------------------------------------------------
    # Go flat (no new longs) when the Nifty index is below its long SMA.
    # A hard risk overlay on top of the model — distinct from the
    # nifty_dist_sma200 *feature* — that caps drawdown in sustained bear
    # markets.  Applied in both the backtest and live signal generation.
    regime_filter:  bool = True
    regime_sma_col: str  = "nifty_dist_sma200"   # <0 ⇒ index below 200-SMA ⇒ risk-off

    # --- costs -----------------------------------------------------------
    cost_bps_per_side: float = 20.0       # ~40 bps round-trip (STT+slip)

    # --- dynamic horizon & RR (docs/dynamic-horizon-rr-plan.md) -----------
    # Master switch: everything below is inert when False. The legacy
    # fixed-horizon (cfg.horizon) / fixed-RR (signal_*_atr_mult) path is
    # untouched so the frozen baseline stays reproducible (plan §Phase 6
    # ablations need it). Flip on only for an explicit dynamic-horizon run.
    dynamic_horizon_enabled: bool = False
    horizon_grid: list = field(default_factory=lambda: [5, 21, 63])  # coarse grid (plan §2 item 6)
    quantile_taus: list = field(default_factory=lambda: [0.1, 0.5, 0.9])
    horizon_lambda_t: float = 0.0005   # per-day time-decay penalty in select_horizon
    horizon_h_max: int = 63            # hard cap on h* (plan §2 item 7)
    rr_k: float = 1.0                  # stop=entry-k*|q10|*entry, target=entry+k*q90*entry
    stop_atr_clamp: tuple = (0.8, 3.0)    # sanity floor/ceiling on derived stop, in ATR mult
    target_atr_clamp: tuple = (1.0, 6.0)  # sanity floor/ceiling on derived target, in ATR mult
    # Separate (smaller) bagging count for quantile heads — the grid x tau
    # training matrix already multiplies cost by len(horizon_grid)*len(quantile_taus);
    # don't also multiply by cfg.ensemble_size unless explicitly raised.
    quantile_ensemble_size: int = 1

    # --- model -----------------------------------------------------------
    xgb_n_trials: int = 50               # Optuna trials
    xgb_early_stopping: int = 50
    xgb_seed: int = 42
    # Multi-seed bagging per walk-forward step (plan §Phase 3.16) — averaging
    # independent noisy XGBoost fits reduces the run-to-run OOF IC instability
    # observed across retrains (same data/hyperparams, different random_state).
    # 1 = old single-model behaviour. Keep modest (3-5); cost scales linearly
    # with the walk-forward loop, which already runs hundreds of steps.
    ensemble_size: int = 3
    # "auto" → use GPU if one is visible (Colab), else CPU. Force with "cuda"/"cpu".
    device: Literal["auto", "cuda", "cpu"] = "auto"
    # Concurrent XGBoost fits (Optuna trials, walk-forward rebalance steps).
    # Each fit is tiny (one stock-day slice, <=1500 shallow trees), so on a
    # single GPU the bottleneck is per-call launch/transfer overhead, not
    # compute — running several fits concurrently overlaps that overhead
    # instead of paying it serially. 1 = old fully-sequential behaviour.
    max_parallel_fits: int = 8

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

    # --- registry & drift (bot-implementaion-plan §7, §9) ----------------
    registry_root:   str  = "."        # bundles → {root}/registry/bundles/model_<v>
    reports_dir:     str  = "reports"  # drift + weekly reports
    save_bundle:     bool = True       # persist a reproducible model bundle each run
    keep_bundles:    int  = 8          # retain last N for instant rollback

    # --- outcome + Supabase feature flags --------------------------------
    resolve_outcomes_on_start: bool = True
    save_to_supabase:          bool = True

    # --- runtime ---------------------------------------------------------
    feature_cols: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str = "config/strategy.yaml", **overrides) -> "Config":
        """Build a Config from config/strategy.yaml (plan §2 — versioned config).

        Maps the nested YAML schema onto the flat dataclass; unknown keys are
        ignored so the file can carry extra documentation. Any keyword override
        wins over the file. Falls back to dataclass defaults if the file/PyYAML
        is unavailable.
        """
        data: dict = {}
        try:
            import yaml
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            data = {}

        flat: dict = {}
        if "horizon" in data:    flat["horizon"] = data["horizon"]
        if "label_type" in data: flat["label_type"] = data["label_type"]
        b = data.get("barriers", {})
        if "up_mult" in b: flat["barrier_up_mult"] = b["up_mult"]
        if "dn_mult" in b: flat["barrier_dn_mult"] = b["dn_mult"]
        if "n_quantile" in data: flat["n_quantile"] = data["n_quantile"]
        if "mode" in data:       flat["mode"] = data["mode"]
        if "conviction_weighted_sizing" in data: flat["conviction_weighted_sizing"] = data["conviction_weighted_sizing"]
        for k in ("train_min_days", "embargo", "n_wf_splits", "ensemble_size"):
            if k in data: flat[k] = data[k]
        c = data.get("costs", {})
        if "cost_bps_per_side" in c: flat["cost_bps_per_side"] = c["cost_bps_per_side"]
        r = data.get("risk", {})
        if "max_positions" in r:     flat["max_positions"] = r["max_positions"]
        if "position_size_pct" in r: flat["position_size_pct"] = r["position_size_pct"]
        if "initial_capital" in r:   flat["initial_capital"] = r["initial_capital"]
        if "regime_filter" in r:     flat["regime_filter"] = r["regime_filter"]
        if "signal_stop_atr_mult" in r:   flat["signal_stop_atr_mult"] = r["signal_stop_atr_mult"]
        if "signal_target_atr_mult" in r: flat["signal_target_atr_mult"] = r["signal_target_atr_mult"]
        reg = data.get("registry", {})
        if "keep_bundles" in reg:    flat["keep_bundles"] = reg["keep_bundles"]

        dh = data.get("dynamic_horizon", {})
        if "enabled" in dh:           flat["dynamic_horizon_enabled"] = dh["enabled"]
        if "horizon_grid" in dh:      flat["horizon_grid"] = dh["horizon_grid"]
        if "quantile_taus" in dh:     flat["quantile_taus"] = dh["quantile_taus"]
        if "lambda_t" in dh:          flat["horizon_lambda_t"] = dh["lambda_t"]
        if "h_max" in dh:             flat["horizon_h_max"] = dh["h_max"]
        if "rr_k" in dh:              flat["rr_k"] = dh["rr_k"]
        if "stop_atr_clamp" in dh:    flat["stop_atr_clamp"] = tuple(dh["stop_atr_clamp"])
        if "target_atr_clamp" in dh:  flat["target_atr_clamp"] = tuple(dh["target_atr_clamp"])
        if "quantile_ensemble_size" in dh: flat["quantile_ensemble_size"] = dh["quantile_ensemble_size"]

        flat.update(overrides)
        return cls(**flat)
