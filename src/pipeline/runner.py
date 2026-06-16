"""Main pipeline orchestrator (plan §11).

Phases executed in order:
  1. Ingest — fetch prices for universe
  2. Store  — upsert into DuckDB + Parquet
  3. Validate — run quality gates (abort on failure)
  4. Features — compute full feature catalog
  5. Labels — triple-barrier or forward return
  6. Walk-forward training + OOF predictions
  7. Backtest — cost-adjusted metrics
  8. Latest signals — score today's bar

Usage:
    from src.pipeline.runner import run
    stats, signals = run(cfg)
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data.ingestion import fetch_prices, fetch_index_prices, UNIVERSE
from src.data.storage import init_db, upsert_prices, upsert_index, load_prices, save_parquet
from src.data.validation import run_all_gates, DataQualityError
from src.features.engineer import build_features
from src.labels.targets import add_labels
from src.labels.weights import sample_weights
from src.models.trainer import train_xgb, tune_hyperparameters, fit_final_model, _BASE_PARAMS_CLF, _BASE_PARAMS_REG
from src.models.calibration import TimeOrderedCalibrator
from src.validation.walk_forward import PurgedWalkForward
from src.validation.metrics import information_coefficient, directional_accuracy, summarise
from src.backtest.engine import run_backtest, sensitivity_analysis

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Walk-forward loop: generate OOF predictions across all dates
# ---------------------------------------------------------------------------
def _walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: Config,
    task: str,
    sw: np.ndarray,
    best_params: dict,
) -> pd.DataFrame:
    """Expanding-window walk-forward: retrain at each rebalance date, predict next."""
    splitter = PurgedWalkForward(
        n_splits=cfg.n_wf_splits,
        embargo=cfg.embargo,
        label_h=cfg.horizon,
        min_train_size=cfg.train_min_days,
    )

    dates = np.sort(df["date"].unique())
    all_preds = []
    last_train_idx = None

    # Use all unique dates as rebalance points spaced rebalance_every apart
    for i in range(cfg.train_min_days, len(dates) - cfg.horizon, cfg.rebalance_every):
        rebal_date = dates[i]
        cutoff_date = dates[max(0, i - cfg.embargo - cfg.horizon)]

        train_mask = df["date"] <= cutoff_date
        test_mask = df["date"] == rebal_date
        train_df = df[train_mask].dropna(subset=feature_cols + [target_col])
        test_df = df[test_mask].dropna(subset=feature_cols)

        if len(train_df) < cfg.train_min_days or test_df.empty:
            continue

        # Use a small recent slice as early-stopping validation
        val_cutoff = dates[max(0, i - cfg.embargo - cfg.horizon - 21)]
        val_mask = (df["date"] > val_cutoff) & (df["date"] <= cutoff_date)
        val_df = df[val_mask].dropna(subset=feature_cols + [target_col])
        if val_df.empty:
            continue

        X_tr = train_df[feature_cols]
        y_tr = train_df[target_col]
        X_vl = val_df[feature_cols]
        y_vl = val_df[target_col]
        X_te = test_df[feature_cols]

        sw_tr = sw[train_df.index] if sw is not None else None

        try:
            model = train_xgb(
                X_tr, y_tr, X_vl, y_vl,
                params=best_params,
                sample_weight=sw_tr,
                early_stopping=cfg.xgb_early_stopping,
                task=task,
            )
        except Exception as e:
            print(f"[runner] training failed at {rebal_date}: {e}")
            continue

        if task == "classification":
            scores = model.predict_proba(X_te)[:, 1]
        else:
            scores = model.predict(X_te)

        pred_df = test_df[["date", "ticker", "fwd_ret"]].copy()
        pred_df["pred"] = scores
        all_preds.append(pred_df)

    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


# ---------------------------------------------------------------------------
# Latest-bar signal generation
# ---------------------------------------------------------------------------
def predict_latest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: Config,
    task: str,
    sw: np.ndarray,
    best_params: dict,
    calibrator: TimeOrderedCalibrator | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Retrain on all available data and score the most-recent bar."""
    df_train = df.dropna(subset=feature_cols + [target_col])
    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].dropna(subset=feature_cols)

    if df_train.empty or latest_df.empty:
        return pd.DataFrame(columns=["ticker", "score", "signal"])

    # Small validation window for early stopping
    dates_sorted = np.sort(df_train["date"].unique())
    val_date = dates_sorted[max(0, len(dates_sorted) - 21)]
    val_df = df_train[df_train["date"] >= val_date]
    train_df = df_train[df_train["date"] < val_date]

    if train_df.empty:
        return pd.DataFrame()

    sw_tr = sw[train_df.index] if sw is not None else None

    model = train_xgb(
        train_df[feature_cols], train_df[target_col],
        val_df[feature_cols], val_df[target_col],
        params=best_params,
        sample_weight=sw_tr,
        early_stopping=cfg.xgb_early_stopping,
        task=task,
    )

    if task == "classification":
        raw_probs = model.predict_proba(latest_df[feature_cols])[:, 1]
        scores = calibrator.predict_proba(raw_probs) if calibrator else raw_probs
        signal_col = "prob_up"
    else:
        scores = model.predict(latest_df[feature_cols])
        signal_col = "pred_return"

    out = latest_df[["ticker"]].copy()
    out[signal_col] = scores
    out["signal"] = np.where(scores > 0.5 if task == "classification" else scores > 0,
                              "LONG", "NEUTRAL")
    out = out.sort_values(signal_col, ascending=False).reset_index(drop=True)
    return out.head(top_n)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(cfg: Config | None = None) -> tuple[dict, pd.DataFrame]:
    """Execute the full pipeline end-to-end.

    Returns (backtest_stats, latest_signals).
    """
    if cfg is None:
        cfg = Config()

    # Ensure directories exist
    for d in [cfg.data_dir, cfg.raw_dir, cfg.model_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("NIFTY 50 SWING PREDICTION PIPELINE")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Phase 1: Data ingestion
    # ------------------------------------------------------------------
    print("\n[Phase 1] Data ingestion …")
    t0 = time.time()
    price_df = fetch_prices(UNIVERSE, cfg.start, cfg.end)
    index_df = fetch_index_prices(cfg.start, cfg.end)
    print(f"  fetched {len(price_df):,} rows in {time.time()-t0:.1f}s")

    # Persist raw (best-effort — continue if DuckDB not installed)
    try:
        save_parquet(price_df, cfg.raw_dir, "prices")
        init_db(cfg.db_path)
        n_new = upsert_prices(price_df, cfg.db_path)
        if not index_df.empty:
            upsert_index(index_df, cfg.db_path)
        print(f"  {n_new} new rows upserted into DuckDB")
    except ImportError as e:
        print(f"  [storage] skipped — {e}")

    # ------------------------------------------------------------------
    # Phase 1b: Validation gates
    # ------------------------------------------------------------------
    print("\n[Phase 1b] Validation gates …")
    try:
        price_df = run_all_gates(price_df)
    except DataQualityError as e:
        print(f"  ABORTED: {e}")
        raise

    # ------------------------------------------------------------------
    # Phase 2: Features & labels
    # ------------------------------------------------------------------
    print("\n[Phase 2] Feature engineering …")
    df, feature_cols = build_features(
        price_df,
        index_df=index_df if not index_df.empty else None,
    )
    cfg.feature_cols = feature_cols
    print(f"  {len(feature_cols)} features computed")

    print("\n[Phase 2b] Label generation …")
    task = "classification" if cfg.label_type == "triple_barrier" else "regression"
    df = add_labels(df, cfg.horizon, cfg.label_type)
    target_col = "target"

    # Drop last h rows (no valid label) and rows with NaN label
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    print(f"  {len(df):,} rows with valid labels")

    # Sample weights
    print("\n[Phase 2c] Computing sample weights …")
    sw = sample_weights(df, cfg.horizon, label_col=target_col)

    # ------------------------------------------------------------------
    # Phase 3: Hyperparameter optimisation (CV on training period)
    # ------------------------------------------------------------------
    splitter = PurgedWalkForward(
        n_splits=cfg.n_wf_splits,
        embargo=cfg.embargo,
        label_h=cfg.horizon,
        min_train_size=cfg.train_min_days,
    )
    train_idx_final, test_idx_final = splitter.final_train_test_split(df, test_fraction=0.2)
    df_hpt = df.iloc[train_idx_final]
    sw_hpt = sw[train_idx_final]

    cv_splits = splitter.split(df_hpt)

    print(f"\n[Phase 3] Hyperparameter search ({cfg.xgb_n_trials} Optuna trials) …")
    if cfg.xgb_n_trials > 0 and len(cv_splits) >= 2:
        best_params = tune_hyperparameters(
            df_hpt, feature_cols, target_col,
            splits=cv_splits,
            task=task,
            n_trials=cfg.xgb_n_trials,
            sample_weights=sw_hpt,
        )
    else:
        best_params = _BASE_PARAMS_CLF if task == "classification" else _BASE_PARAMS_REG
        print("  skipping tuning — using default params")

    # ------------------------------------------------------------------
    # Phase 3b: Probability calibration on the final validation slice
    # ------------------------------------------------------------------
    calibrator = None
    if task == "classification":
        cal_idx = test_idx_final[:len(test_idx_final) // 2]
        oos_idx = test_idx_final[len(test_idx_final) // 2:]
        df_cal = df.iloc[cal_idx]
        if len(df_cal) > 50:
            cal_model = fit_final_model(
                df, feature_cols, target_col,
                train_idx_final, cal_idx,
                best_params, task, sw,
            )
            raw_probs = cal_model.predict_proba(df_cal[feature_cols])[:, 1]
            cal_labels = (df_cal[target_col] == 1).astype(int).values
            calibrator = TimeOrderedCalibrator()
            calibrator.fit(raw_probs, cal_labels)
            ece = calibrator.calibration_error(raw_probs, cal_labels)
            print(f"\n[calibration] ECE = {ece:.4f}")
        else:
            oos_idx = test_idx_final
    else:
        oos_idx = test_idx_final

    # ------------------------------------------------------------------
    # Phase 4: Walk-forward OOF predictions + backtest
    # ------------------------------------------------------------------
    print("\n[Phase 4] Walk-forward prediction loop …")
    oof_preds = _walk_forward_predict(df, feature_cols, target_col, cfg, task, sw, best_params)
    print(f"  generated predictions for {oof_preds['date'].nunique() if not oof_preds.empty else 0} dates")

    if not oof_preds.empty:
        ic = information_coefficient(
            oof_preds["pred"].values, oof_preds["fwd_ret"].values
        )
        da = directional_accuracy(
            oof_preds["pred"].values, oof_preds["fwd_ret"].values
        )
        print(f"  OOF Information Coefficient = {ic:.4f}")
        print(f"  OOF Directional Accuracy    = {da:.4f}")

    # ------------------------------------------------------------------
    # Phase 4b: Cost-adjusted backtest
    # ------------------------------------------------------------------
    print("\n[Phase 4b] Backtest with Indian transaction costs …")
    stats = run_backtest(oof_preds, cfg)

    print("\n--- BACKTEST RESULTS ---")
    for k, v in stats.items():
        if k in ("equity_curve", "period_returns"):
            continue
        print(f"  {k:>16}: {v}")

    # Sensitivity to costs
    if not oof_preds.empty:
        sens = sensitivity_analysis(oof_preds, cfg)
        print("\n--- COST SENSITIVITY ---")
        print(sens.to_string(index=False))

    # ------------------------------------------------------------------
    # Phase 5: Latest signals
    # ------------------------------------------------------------------
    print("\n[Phase 5] Generating latest signals …")
    signals = predict_latest(
        df, feature_cols, target_col, cfg, task, sw, best_params,
        calibrator=calibrator,
    )
    print("\n--- TOP SIGNALS (today) ---")
    print(signals.to_string(index=False) if not signals.empty else "  (no signals)")

    print("\n" + "=" * 72)
    print("Pipeline complete.")
    print(
        "NOTE: On synthetic/low-signal data, near-zero Sharpe after costs is "
        "the correct result.  Real edge, if it exists, is small (52–56% accuracy)."
    )
    print("=" * 72)
    return stats, signals
