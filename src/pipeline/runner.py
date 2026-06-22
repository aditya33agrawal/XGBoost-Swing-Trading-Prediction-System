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

import json
import logging
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
from src.models.trainer import (
    train_xgb, tune_hyperparameters, fit_final_model,
    _BASE_PARAMS_CLF, _BASE_PARAMS_REG, _get_xgb,
    set_device, apply_device, get_device,
)
from src.models.calibration import TimeOrderedCalibrator
from src.validation.walk_forward import PurgedWalkForward
from src.validation.metrics import information_coefficient, directional_accuracy, summarise
from src.backtest.engine import run_backtest, sensitivity_analysis
from src.trading.signals import enrich_signals, save_signals, print_signal_table
from src.trading.paper_trader import PaperPortfolio
from src.db.supabase_client import get_supabase_client
from src.tracking.prediction_journal import save_predictions, save_run_metadata, sync_paper_trades, sync_ledger
from src.models.improvement import get_model_version
from src.registry.bundle import save_bundle, prune_old_bundles
from src.monitoring.drift import (
    feature_drift_report, concept_drift_from_outcomes,
    calibration_drift, build_drift_report, write_drift_report,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _print_backtest_results(stats: dict) -> None:
    skip = {"equity_curve", "period_returns", "error"}
    w = 60
    print(f"\n{'─' * w}")
    print("  BACKTEST RESULTS")
    print(f"{'─' * w}")
    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
    else:
        labels = {
            "label": "Strategy", "n_periods": "Periods",
            "CAGR": "CAGR", "Sharpe": "Sharpe ratio",
            "Sortino": "Sortino ratio", "Calmar": "Calmar ratio",
            "max_drawdown": "Max drawdown", "hit_rate": "Hit rate",
            "profit_factor": "Profit factor",
            "avg_period_ret": "Avg period return", "final_equity": "Final equity (×)",
        }
        for k, label in labels.items():
            if k in stats and k not in skip:
                v = stats[k]
                if isinstance(v, float):
                    fmt = f"{v:>+.2%}" if k in ("CAGR", "max_drawdown", "avg_period_ret") else f"{v:>.3f}"
                else:
                    fmt = str(v)
                print(f"  {label:<28}  {fmt}")
    print(f"{'─' * w}")


def _print_cost_sensitivity(sens: pd.DataFrame) -> None:
    print(f"\n  COST SENSITIVITY")
    print(f"  {'Mult':>6}  {'Sharpe':>8}  {'CAGR':>8}  {'MaxDD':>9}")
    print(f"  {'─' * 36}")
    for _, row in sens.iterrows():
        print(
            f"  {row['cost_mult']:>5.1f}×  {row['Sharpe']:>8.3f}  "
            f"{row['CAGR']:>+7.2%}  {row['max_drawdown']:>+8.2%}"
        )


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

    rebal_points = list(range(cfg.train_min_days, len(dates) - cfg.horizon, cfg.rebalance_every))
    n_steps = len(rebal_points)
    logger.info(
        "Walk-forward: %d rebalance steps from %s to %s (device=%s)",
        n_steps,
        pd.Timestamp(dates[rebal_points[0]]).date() if n_steps else "n/a",
        pd.Timestamp(dates[rebal_points[-1]]).date() if n_steps else "n/a",
        get_device(),
    )
    t_wf = time.time()

    # Use all unique dates as rebalance points spaced rebalance_every apart
    for step, i in enumerate(rebal_points):
        rebal_date = dates[i]
        cutoff_date = dates[max(0, i - cfg.embargo - cfg.horizon)]

        train_mask = df["date"] <= cutoff_date
        test_mask = df["date"] == rebal_date
        train_df = df[train_mask].dropna(subset=feature_cols + [target_col])
        test_df = df[test_mask].dropna(subset=feature_cols)

        if len(train_df) < cfg.train_min_days or test_df.empty:
            continue

        # Early-stopping validation window.  The old 21-day slice gave early
        # stopping almost no signal: with the low learning rates Optuna favours
        # (~0.007) AUCPR never improved within `patience` rounds on such a noisy
        # window, so best_iteration collapsed to 0 and every 2017–2020 fold was
        # dropped below (~160 / 441 walk-forward steps wasted).  A ~3-month
        # window (cfg.wf_es_val_days) gives a stable early-stop signal and
        # recovers those early folds.
        val_cutoff = dates[max(0, i - cfg.embargo - cfg.horizon - cfg.wf_es_val_days)]
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
            logger.warning("training failed at %s: %s", rebal_date, e)
            continue

        # Early stopping can land on best_iteration == 0 when the training
        # window is too sparse (notably the early 2017–2020 folds).  A 0-tree
        # model predicts a constant base rate: zero cross-sectional dispersion,
        # which adds pure noise to OOF IC and feeds random quintiles into the
        # backtest.  Drop those degenerate folds rather than poison the metrics.
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None and best_iter == 0:
            logger.warning("skipping fold %s — best_iteration=0 (sparse train, no signal)", rebal_date)
            continue

        if task == "classification":
            scores = model.predict_proba(X_te)[:, 1]
        else:
            scores = model.predict(X_te)

        pred_cols = ["date", "ticker", "fwd_ret"]
        if cfg.regime_sma_col in test_df.columns:
            pred_cols.append(cfg.regime_sma_col)
        pred_df = test_df[pred_cols].copy()
        pred_df["pred"] = scores
        all_preds.append(pred_df)

        # Periodic progress with a rough ETA so long runs aren't a black box.
        if step % 10 == 0 or step == n_steps - 1:
            elapsed = time.time() - t_wf
            rate = elapsed / (step + 1)
            eta = rate * (n_steps - step - 1)
            logger.info(
                "  step %3d/%d | %s | train=%d rows | best_iter=%s | ETA %.0fs",
                step + 1, n_steps, pd.Timestamp(rebal_date).date(),
                len(train_df),
                getattr(model, "best_iteration", "n/a"),
                eta,
            )

    logger.info("Walk-forward done in %.1fs", time.time() - t_wf)
    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


# ---------------------------------------------------------------------------
# Latest-bar signal generation
# ---------------------------------------------------------------------------
def predict_latest(
    df_labeled: pd.DataFrame,          # training data — has valid target column
    df_full: pd.DataFrame,             # full feature frame — includes post-label dates
    feature_cols: list[str],
    target_col: str,
    cfg: Config,
    task: str,
    sw: np.ndarray,
    best_params: dict,
    calibrator: TimeOrderedCalibrator | None = None,
    top_n: int = 10,
    return_model: bool = False,
):
    """Retrain on all labeled data and score the actual latest price date.

    The distinction between df_labeled and df_full matters:
      df_labeled has targets for dates up to (end - horizon); used for training.
      df_full has features for all dates including the most-recent `horizon` bars
      that have no label yet; used for scoring.
    """
    df_train = df_labeled.dropna(subset=feature_cols + [target_col])

    # Actual latest date in the price feed (may be horizon bars beyond last label)
    latest_date = df_full["date"].max()
    latest_df = df_full[df_full["date"] == latest_date].dropna(subset=feature_cols)

    if df_train.empty or latest_df.empty:
        empty = pd.DataFrame(columns=["ticker", "score", "signal"])
        return (empty, None) if return_model else empty

    # For the final production model we use a fixed n_estimators (no early stopping).
    # Early stopping requires a held-out val set; the val window here is only ~21 days
    # which is too small and causes the model to stop at round 0, collapsing all
    # predictions to the base rate.  We use the n_estimators chosen by Optuna instead.
    params_no_es = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}
    params_no_es = apply_device(params_no_es)

    xgb = _get_xgb()
    sw_tr = sw[df_train.index] if sw is not None else None

    if task == "classification":
        y_tr = (df_train[target_col] == 1).astype(int)
        model = xgb.XGBClassifier(**params_no_es)
        model.fit(df_train[feature_cols], y_tr, sample_weight=sw_tr, verbose=False)
        raw_probs = model.predict_proba(latest_df[feature_cols])[:, 1]
        # Do NOT apply calibrator here.  The calibrator was fitted on a different
        # model's output (trained on the HPT subset); applying it to a model trained
        # on the full dataset maps all scores to near-constant values, collapsing
        # the ranking.  Raw probabilities preserve cross-sectional ordering.
        scores = raw_probs
        signal_col = "prob_up"
    else:
        model = xgb.XGBRegressor(**params_no_es)
        model.fit(df_train[feature_cols], df_train[target_col],
                  sample_weight=sw_tr, verbose=False)
        scores = model.predict(latest_df[feature_cols])
        signal_col = "pred_return"

    out = latest_df[["ticker"]].copy()
    out[signal_col] = scores
    out = out.sort_values(signal_col, ascending=False).reset_index(drop=True)

    # Selection MUST mirror the backtest engine, which longs the top
    # cross-sectional quantile by score (engine.py: q == n_quantile-1) — NOT an
    # absolute threshold.  With the triple-barrier 'up' base rate ~0.23,
    # calibrated probabilities almost never exceed 0.5, so an absolute
    # prob>0.5 cutoff emits ZERO long signals every day and the live layer
    # never trades the strategy that was actually validated.
    n = len(out)
    n_long = max(1, n // cfg.n_quantile) if n else 0
    out["signal"] = "NEUTRAL"
    if n_long:
        out.iloc[:n_long, out.columns.get_loc("signal")] = "LONG"

    # Risk overlay (mirror the backtest): if the index is below its long SMA
    # today, suppress all LONGs and stay flat regardless of model scores.
    if getattr(cfg, "regime_filter", False) and cfg.regime_sma_col in latest_df.columns:
        regime_val = float(latest_df[cfg.regime_sma_col].iloc[0])
        if regime_val < 0:
            out["signal"] = "NEUTRAL"

    result = out.head(top_n)
    return (result, model) if return_model else result


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

    logger.info("=" * 60)
    logger.info("NIFTY 50 SWING PREDICTION PIPELINE")
    logger.info("=" * 60)

    # Supabase client — shared across all phases; None = JSON-only mode
    _sb = get_supabase_client(cfg.supabase_url, cfg.supabase_key)
    if _sb:
        logger.info("Supabase connected")
    else:
        logger.info("Supabase not configured — outputs written to %s/ only", cfg.output_dir)

    # Auto-generate model version if not set
    _run_id = cfg.model_version or get_model_version()
    cfg.model_version = _run_id

    # Resolve CPU vs GPU once and report it. Every XGBoost model built downstream
    # picks this up via the trainer's module-level device.
    dev = set_device(cfg.device)
    logger.info(
        "Config: %s..%s | horizon=%d | label=%s | trials=%d | device=%s",
        cfg.start, cfg.end, cfg.horizon, cfg.label_type, cfg.xgb_n_trials, dev,
    )

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
        logger.info("  %d new rows upserted into DuckDB", n_new)
    except Exception as e:
        # Persistence is best-effort — never let a storage hiccup kill training.
        logger.warning("  [storage] skipped — %s: %s", type(e).__name__, e)

    # ------------------------------------------------------------------
    # Phase 1b: Validation gates
    # ------------------------------------------------------------------
    print("\n[Phase 1b] Validation gates …")
    try:
        price_df = run_all_gates(price_df)
    except DataQualityError as e:
        print(f"  ABORTED: {e}")
        raise

    # Phase 1c: Resolve past predictions now that clean price data is available
    if cfg.resolve_outcomes_on_start:
        from src.tracking.outcome_tracker import resolve_outcomes
        print("\n[Phase 1c] Resolving past predictions …")
        n_resolved = resolve_outcomes(price_df, _sb, cfg.output_dir, cfg.horizon)
        print(f"  {n_resolved} prediction(s) resolved")

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

    # Keep df_full (features for ALL dates, including last horizon bars which have
    # no valid label) so predict_latest can score the actual latest price date.
    df_full = df.copy()

    # Drop last h rows (no valid label) from the training frame
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    print(f"  {len(df):,} rows with valid labels  |  latest inference date: {df_full['date'].max().date()}")

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
    # Try to load saved params when skipping Optuna (fast-signals mode)
    _params_path = Path(cfg.params_path)
    if cfg.xgb_n_trials > 0 and len(cv_splits) >= 2:
        best_params = tune_hyperparameters(
            df_hpt, feature_cols, target_col,
            splits=cv_splits,
            task=task,
            n_trials=cfg.xgb_n_trials,
            sample_weights=sw_hpt,
        )
        # Persist so fast-signals runs can reuse them
        _params_path.parent.mkdir(parents=True, exist_ok=True)
        _params_path.write_text(json.dumps(best_params, indent=2))
        logger.info("Best params saved → %s", _params_path)
    elif _params_path.exists():
        best_params = json.loads(_params_path.read_text())
        best_params = apply_device(best_params)
        logger.info("Loaded saved params from %s (device=%s)", _params_path, get_device())
    else:
        base = _BASE_PARAMS_CLF if task == "classification" else _BASE_PARAMS_REG
        best_params = apply_device(base)
        logger.info("Skipping Optuna — using default params on device=%s", get_device())

    # ------------------------------------------------------------------
    # Phase 3b: Probability calibration on the final validation slice
    # ------------------------------------------------------------------
    calibrator = None
    _oos_ece = float("nan")          # held-out calibration error (for promotion gate / bundle)
    if task == "classification":
        # Split the held-out test period into two halves:
        #   first half  → fit the calibrator (time-ordered)
        #   second half → measure ECE on truly unseen data
        mid = len(test_idx_final) // 2
        cal_idx = test_idx_final[:mid]
        oos_idx = test_idx_final[mid:]
        df_cal = df.iloc[cal_idx]
        df_oos = df.iloc[oos_idx]
        if len(df_cal) > 50:
            cal_model = fit_final_model(
                df, feature_cols, target_col,
                train_idx_final, cal_idx,
                best_params, task, sw,
            )
            # Fit calibrator on first half
            raw_probs_cal = cal_model.predict_proba(df_cal[feature_cols])[:, 1]
            cal_labels = (df_cal[target_col] == 1).astype(int).values
            calibrator = TimeOrderedCalibrator()
            calibrator.fit(raw_probs_cal, cal_labels)
            # Measure ECE on second half (truly held-out)
            if len(df_oos) > 10:
                raw_probs_oos = cal_model.predict_proba(df_oos[feature_cols])[:, 1]
                oos_labels = (df_oos[target_col] == 1).astype(int).values
                ece = calibrator.calibration_error(raw_probs_oos, oos_labels)
                _oos_ece = float(ece)
                print(f"\n[calibration] ECE on held-out OOS = {ece:.4f}")
            else:
                print(f"\n[calibration] fitted (OOS too small to measure ECE)")
        else:
            oos_idx = test_idx_final
    else:
        oos_idx = test_idx_final

    # ------------------------------------------------------------------
    # Phase 4: Walk-forward OOF predictions + backtest
    # ------------------------------------------------------------------
    stats: dict = {}
    oof_preds: pd.DataFrame = pd.DataFrame()
    sens: pd.DataFrame = pd.DataFrame()

    if cfg.skip_backtest:
        print("\n[Phase 4] Walk-forward skipped (--fast-signals mode)")
    else:
        print("\n[Phase 4] Walk-forward prediction loop …")
        oof_preds = _walk_forward_predict(df, feature_cols, target_col, cfg, task, sw, best_params)
        n_dates = oof_preds["date"].nunique() if not oof_preds.empty else 0
        print(f"  generated predictions for {n_dates} dates")

        oof_ic = oof_dir_acc = None
        if not oof_preds.empty:
            oof_ic = float(information_coefficient(oof_preds["pred"].values, oof_preds["fwd_ret"].values))
            oof_dir_acc = float(directional_accuracy(oof_preds["pred"].values, oof_preds["fwd_ret"].values))
            print(f"  OOF Information Coefficient = {oof_ic:.4f}")
            print(f"  OOF Directional Accuracy    = {oof_dir_acc:.4f}")

        # Phase 4b: Cost-adjusted backtest
        print("\n[Phase 4b] Backtest with Indian transaction costs …")
        stats = run_backtest(oof_preds, cfg)   # returns a fresh dict
        if oof_ic is not None:
            stats["oof_ic"] = oof_ic
            stats["oof_dir_acc"] = oof_dir_acc
        _print_backtest_results(stats)

        if not oof_preds.empty:
            sens = sensitivity_analysis(oof_preds, cfg)
            _print_cost_sensitivity(sens)

    # ------------------------------------------------------------------
    # Phase 5: Latest signals
    # ------------------------------------------------------------------
    print("\n[Phase 5] Generating latest signals …")
    signals, final_model = predict_latest(
        df_labeled=df,
        df_full=df_full,
        feature_cols=feature_cols,
        target_col=target_col,
        cfg=cfg,
        task=task,
        sw=sw,
        best_params=best_params,
        calibrator=calibrator,
        return_model=True,
    )

    # Phase 5b: Enrich with ATR-based entry / stop / target levels
    if not signals.empty:
        signals = enrich_signals(signals, price_df, cfg)

    # Phase 5c: Persist signals to disk
    if cfg.save_outputs and not signals.empty:
        json_path = save_signals(signals, cfg.output_dir)
        print(f"  Signals saved → {json_path}")

    # Phase 5d: Formatted signal table
    print_signal_table(signals, title="TOP SIGNALS (today)")

    # Phase 5e: Persist run metadata + predictions to Supabase / JSON.
    # ORDER MATTERS: model_runs MUST be written before predictions, because
    # predictions.run_id is a FOREIGN KEY → model_runs.run_id.  Writing the
    # predictions first raises a 23503 FK violation (the parent run row does
    # not exist yet) and the day's signals are silently dropped from the DB.
    if cfg.save_to_supabase and not signals.empty:
        # Enrich stats_dict with OOF metrics so they appear in model_runs
        _stats_enriched = dict(stats)
        _stats_enriched["horizon_days"] = cfg.horizon
        _stats_enriched["label_type"]   = cfg.label_type

        # Feature importances from the production model → model_runs / feature_importance
        _feat_imp = None
        try:
            _imp = getattr(final_model, "feature_importances_", None)
            if _imp is not None:
                _feat_imp = {f: float(v) for f, v in zip(feature_cols, _imp)}
        except Exception:
            _feat_imp = None

        save_run_metadata(
            _stats_enriched, _run_id, cfg.model_version,
            best_params, _feat_imp, _sb, cfg.output_dir,
        )
        save_predictions(signals, _run_id, cfg.model_version, _sb, cfg.output_dir)

    # ------------------------------------------------------------------
    # Phase 5f: Persist a reproducible model bundle to the registry (§7)
    # ------------------------------------------------------------------
    # Persist the bundle whenever we have a trained production model — even on a
    # day where the regime overlay suppresses every LONG (all-NEUTRAL signals).
    # The reproducible artifact (booster + manifest + metrics) is what the daily
    # VM loop loads; it must not depend on whether today happened to trade.
    if cfg.save_bundle and final_model is not None:
        try:
            bundle_metrics = {
                "oof_ic":       stats.get("oof_ic"),
                "oof_dir_acc":  stats.get("oof_dir_acc"),
                "sharpe_net":   stats.get("Sharpe"),
                "sortino":      stats.get("Sortino"),
                "calmar":       stats.get("Calmar"),
                "max_drawdown": stats.get("max_drawdown"),
                "hit_rate":     stats.get("hit_rate"),
                "calib_err":    None if (_oos_ece != _oos_ece) else _oos_ece,
            }
            bundle_dir = save_bundle(
                cfg.registry_root,
                model=final_model,
                calibrator=calibrator,
                features=feature_cols,
                hyperparams=best_params,
                metrics=bundle_metrics,
                train_window={"start": cfg.start, "end": cfg.end},
                horizon_days=cfg.horizon,
                embargo_days=cfg.embargo,
                model_version=_run_id,
                label_type=cfg.label_type,
                task=task,
            )
            print(f"  Model bundle saved → {bundle_dir}")
            prune_old_bundles(cfg.registry_root, keep=cfg.keep_bundles)
        except Exception as exc:
            logger.warning("Bundle save failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Phase 5g: Drift report — feature (PSI/KS), concept (ledger), calibration (§9)
    # ------------------------------------------------------------------
    drift_report = None
    if not cfg.skip_backtest:
        try:
            df_feat = df.dropna(subset=feature_cols)
            n = len(df_feat)
            drift_fd = pd.DataFrame()
            if n > 200:
                ref = df_feat.iloc[: n // 2]           # older half = training reference
                cur = df_feat.iloc[n // 2 :]           # recent half = current regime
                drift_fd = feature_drift_report(ref, cur, feature_cols)

            outcomes_df = pd.DataFrame()
            try:
                from src.tracking.outcome_tracker import _load_outcomes_df
                outcomes_df = _load_outcomes_df(_sb, n_weeks=12, fallback_dir=cfg.output_dir)
            except Exception:
                pass

            report = build_drift_report(
                feature_drift=drift_fd,
                concept=concept_drift_from_outcomes(outcomes_df, backtest_ic=stats.get("oof_ic")),
                calibration=calibration_drift(outcomes_df),
                extra={"model_version": _run_id},
            )
            _, html_path = write_drift_report(report, cfg.reports_dir, tag=_run_id)
            print(f"  Drift report → {html_path}"
                  f"{'  [RETRAIN RECOMMENDED]' if report['retrain_recommended'] else ''}")
            drift_report = report
        except Exception as exc:
            logger.warning("Drift report failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Phase 5h: Detailed backtest report — what this run found, persisted
    # to reports/ so it doesn't just scroll away in the console.
    # ------------------------------------------------------------------
    if not cfg.skip_backtest:
        try:
            from src.backtest.report import build_backtest_report, write_backtest_report
            bt_report = build_backtest_report(
                stats=stats,
                sensitivity_df=sens,
                cfg=cfg,
                oof_preds=oof_preds,
                price_df=price_df,
                drift_report=drift_report,
            )
            _, md_path = write_backtest_report(bt_report, cfg.reports_dir, tag=_run_id)
            print(f"  Backtest report → {md_path}")
        except Exception as exc:
            logger.warning("Backtest report failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Phase 6: Paper trading — exits only. New positions are NEVER opened
    # automatically by the pipeline; only a manual "Take Trade" click in the
    # UI (app/utils/writer.open_trade) opens a position, and only at CMP
    # during market hours. This phase just settles stop / target / horizon
    # exits on positions that were opened that way.
    # ------------------------------------------------------------------
    if cfg.paper_trade:
        print("\n[Phase 6] Paper trading — settling exits (no auto-entries) …")
        portfolio = PaperPortfolio.load(cfg.portfolio_path)
        portfolio.max_positions     = cfg.max_positions
        portfolio.position_size_pct = cfg.position_size_pct
        if portfolio.initial_capital == 1_000_000 and cfg.initial_capital != 1_000_000:
            portfolio.initial_capital = cfg.initial_capital

        closed = portfolio.update(price_df)
        if closed:
            print(f"  Closed {len(closed)} position(s): "
                  f"{[(t.ticker, t.exit_reason) for t in closed]}")

        portfolio.print_summary(price_df)
        portfolio.save(cfg.portfolio_path)

        # Sync all trades to Supabase paper_trades table
        if cfg.save_to_supabase:
            n_synced = sync_paper_trades(portfolio, _run_id, _sb)
            if n_synced:
                logger.info("Synced %d paper trades to Supabase", n_synced)
            n_ledger = sync_ledger(portfolio, _run_id, _sb, fallback_dir=cfg.output_dir)
            if n_ledger:
                logger.info("Synced %d ledger rows to Supabase", n_ledger)

    print("\n" + "=" * 72)
    print("Pipeline complete.")
    if not cfg.skip_backtest:
        print(
            "NOTE: On synthetic/low-signal data, near-zero Sharpe after costs is "
            "the correct result.  Real edge, if it exists, is small (52–56% accuracy)."
        )
    print("=" * 72)
    return stats, signals
