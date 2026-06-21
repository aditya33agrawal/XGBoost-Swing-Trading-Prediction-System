#!/usr/bin/env python3
"""Sunday weekly retrain orchestrator.

Run every Sunday (manually on Colab or scheduled):
    python scripts/weekly_retrain.py

Flow:
  1. Connect to Supabase
  2. Fetch fresh price data
  3. Resolve last week's predictions → compute 4-week rolling IC
  4. Full retrain (50 Optuna trials + walk-forward)
  5. Compare new model IC vs deployed model IC
  6. Deploy if new model is better (update Supabase + save to Drive/local)
  7. Generate signals for next week (fast mode)
  8. Print summary report

Set env vars before running:
    export SUPABASE_URL=https://xxx.supabase.co
    export SUPABASE_KEY=your-anon-key
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.ingestion import fetch_prices, fetch_index_prices, UNIVERSE
from src.db.supabase_client import get_supabase_client, upsert_rows
from src.logging_setup import setup_logging
from src.models.improvement import (
    compare_models,
    get_model_version,
    load_current_model_ic,
    mark_deployed,
    should_retrain,
)
from src.pipeline.runner import run
from src.registry.bundle import set_prod_pointer
from src.registry.promotion import evaluate_promotion
from src.tracking.outcome_tracker import compute_recent_ic, compute_weekly_ic_series, resolve_outcomes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weekly Nifty 50 model retrain + deploy")
    p.add_argument("--trials",       type=int,   default=50)
    p.add_argument("--horizon",      type=int,   default=5)
    p.add_argument("--capital",      type=float, default=1_000_000)
    p.add_argument("--max-positions",type=int,   default=10)
    p.add_argument("--drive-dir",    default="",
                   help="Google Drive output dir (Colab only)")
    p.add_argument("--skip-deploy",  action="store_true",
                   help="Run comparison but do not update is_deployed flag")
    p.add_argument("--force-deploy", action="store_true",
                   help="Deploy regardless of IC comparison")
    p.add_argument("--log-level",    default="INFO")
    p.add_argument("--no-paper-trade", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Drive sync helper (no-op when not on Colab)
# ---------------------------------------------------------------------------
def _sync_to_drive(src_dir: str, drive_dir: str, label: str = "") -> None:
    if not drive_dir:
        return
    try:
        dest = Path(drive_dir)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, str(dest / Path(src_dir).name), dirs_exist_ok=True)
        logger.info("Synced %s → %s/%s", label or src_dir, drive_dir, Path(src_dir).name)
    except Exception as exc:
        logger.warning("Drive sync failed (%s): %s", label, exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    today      = date.today()
    new_run_id = get_model_version()
    logger.info("=" * 64)
    logger.info("WEEKLY RETRAIN  —  %s  —  run_id=%s", today, new_run_id)
    logger.info("=" * 64)

    # ---- Supabase client ------------------------------------------------
    sb = get_supabase_client()
    if sb:
        logger.info("Supabase connected")
    else:
        logger.info("No Supabase credentials — running in JSON-only mode")

    # ---- Step 1: Resolve last week's outcomes ---------------------------
    logger.info("\n[Step 1] Resolving past predictions …")
    base_cfg = Config(
        horizon         = args.horizon,
        supabase_url    = os.getenv("SUPABASE_URL", ""),
        supabase_key    = os.getenv("SUPABASE_KEY", ""),
        model_version   = new_run_id,
        save_to_supabase= True,
        resolve_outcomes_on_start = False,  # we'll do it manually here
    )
    price_df = fetch_prices(UNIVERSE, base_cfg.start, base_cfg.end)
    n_resolved = resolve_outcomes(price_df, sb, base_cfg.output_dir, args.horizon)
    logger.info("  Resolved %d prediction(s)", n_resolved)

    current_ic, current_run_id = load_current_model_ic(sb, base_cfg.output_dir)
    logger.info("  Current deployed model: %s  |  4-wk IC = %s",
                current_run_id or "none",
                f"{current_ic:.4f}" if not math.isnan(current_ic) else "n/a")

    # Emergency retrain check
    _needs_emergency, _msg = should_retrain(current_ic)
    if _needs_emergency:
        logger.warning("EMERGENCY: %s", _msg)

    # ---- Step 2: Full retrain -------------------------------------------
    logger.info("\n[Step 2] Full retrain (trials=%d) …", args.trials)
    train_cfg = Config(
        horizon          = args.horizon,
        xgb_n_trials     = args.trials,
        skip_backtest    = False,
        save_outputs     = True,
        save_to_supabase = True,
        paper_trade      = not args.no_paper_trade,
        initial_capital  = args.capital,
        max_positions    = args.max_positions,
        supabase_url     = os.getenv("SUPABASE_URL", ""),
        supabase_key     = os.getenv("SUPABASE_KEY", ""),
        model_version    = new_run_id,
        resolve_outcomes_on_start = False,  # already done above
        rebalance_every  = args.horizon,
        embargo          = args.horizon,
    )
    new_stats, new_signals = run(train_cfg)

    new_oof_ic  = new_stats.get("oof_ic",  float("nan"))
    new_sharpe  = new_stats.get("Sharpe",  float("nan"))
    new_cagr    = new_stats.get("CAGR",    float("nan"))
    logger.info("  New model — OOF IC=%.4f  Sharpe=%.3f  CAGR=%.2f%%",
                new_oof_ic if not math.isnan(new_oof_ic) else 0,
                new_sharpe if not math.isnan(new_sharpe) else 0,
                (new_cagr or 0) * 100)

    # ---- Step 3: Champion/challenger gate (§8) --------------------------
    logger.info("\n[Step 3] Champion/challenger gate …")

    # Was a drift alarm raised by this run's drift report?
    drift_alarm = False
    try:
        drift_path = Path(train_cfg.reports_dir) / f"drift_{new_run_id}.json"
        if drift_path.exists():
            import json as _json
            drift_alarm = bool(_json.loads(drift_path.read_text()).get("retrain_recommended"))
    except Exception:
        pass

    if args.force_deploy:
        should_deploy = True
        decision = {"promote": True, "reasons": ["--force-deploy"]}
        logger.info("  --force-deploy flag set — deploying unconditionally")
    else:
        decision = evaluate_promotion(
            challenger={"sharpe_net": new_sharpe, "ic": new_oof_ic,
                        "calib_err": new_stats.get("calib_err")},
            champion=None if math.isnan(current_ic) else {"ic": current_ic},
            drift_alarm=drift_alarm,
        )
        should_deploy = decision["promote"]
        logger.info("  Decision: %s — %s",
                    "PROMOTE" if should_deploy else "KEEP CHAMPION",
                    "; ".join(decision["reasons"]))

    if should_deploy and not args.skip_deploy:
        mark_deployed(new_run_id, sb, base_cfg.output_dir, previous_run_id=current_run_id)
        # Flip the registry prod pointer to the freshly-saved bundle (atomic, rollback-safe)
        new_bundle = Path(train_cfg.registry_root) / "registry" / "bundles" / f"model_{new_run_id}"
        if new_bundle.exists():
            set_prod_pointer(train_cfg.registry_root, str(new_bundle))
        logger.info("  Deployed: %s (replaces %s)", new_run_id, current_run_id or "none")

        # Sync to Google Drive if running on Colab
        _sync_to_drive("models",   args.drive_dir, "models")
        _sync_to_drive("outputs",  args.drive_dir, "outputs")
        _sync_to_drive("registry", args.drive_dir, "registry")
        _sync_to_drive("reports",  args.drive_dir, "reports")
    else:
        reason = "--skip-deploy flag" if args.skip_deploy else "IC did not improve sufficiently"
        logger.info("  Not deploying: %s", reason)

    # ---- Step 4: Generate next-week signals (fast mode) ----------------
    logger.info("\n[Step 4] Generating next-week signals (fast mode) …")
    sig_cfg = Config(
        horizon          = args.horizon,
        xgb_n_trials     = 0,   # use saved best_params.json
        skip_backtest    = True,
        save_outputs     = True,
        save_to_supabase = True,
        paper_trade      = not args.no_paper_trade,
        initial_capital  = args.capital,
        max_positions    = args.max_positions,
        supabase_url     = os.getenv("SUPABASE_URL", ""),
        supabase_key     = os.getenv("SUPABASE_KEY", ""),
        model_version    = new_run_id if should_deploy else current_run_id,
        resolve_outcomes_on_start = False,
        rebalance_every  = args.horizon,
        embargo          = args.horizon,
    )
    _, week_signals = run(sig_cfg)

    # ---- Summary --------------------------------------------------------
    logger.info("\n" + "=" * 64)
    logger.info("WEEKLY RETRAIN COMPLETE")
    logger.info("=" * 64)
    logger.info("  Date:                 %s", today)
    logger.info("  New run_id:           %s", new_run_id)
    logger.info("  Outcomes resolved:    %d", n_resolved)
    logger.info("  Previous IC (4-wk):   %s",
                f"{current_ic:.4f}" if not math.isnan(current_ic) else "n/a")
    logger.info("  New model OOF IC:     %s",
                f"{new_oof_ic:.4f}" if not math.isnan(new_oof_ic) else "n/a")
    logger.info("  Deployment decision:  %s",
                "DEPLOYED" if (should_deploy and not args.skip_deploy) else "KEPT PREVIOUS")
    logger.info("  Signals generated:    %d (%d LONG)",
                len(week_signals),
                int((week_signals.get("signal", "") == "LONG").sum()) if not week_signals.empty else 0)

    # Print IC trend
    ic_series = compute_weekly_ic_series(sb, n_weeks=8, fallback_dir=base_cfg.output_dir)
    if not ic_series.empty:
        logger.info("\n  IC Trend (last 8 weeks):")
        logger.info("  %-14s  %8s  %8s  %10s", "Week", "IC", "N obs", "Dir acc")
        for _, row in ic_series.iterrows():
            logger.info("  %-14s  %8s  %8d  %10s",
                        row["week_start"],
                        f"{row['ic']:.4f}" if row["ic"] is not None else "n/a",
                        int(row["n_obs"]),
                        f"{row['dir_accuracy']:.1%}" if row["dir_accuracy"] is not None else "n/a")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
