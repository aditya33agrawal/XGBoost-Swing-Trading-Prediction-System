"""Prediction journal — persist signals and model metadata to Supabase + JSON fallback.

Three public functions:
  save_predictions()   — write this run's signals to predictions table
  save_run_metadata()  — write model metrics + params to model_runs table
  sync_paper_trades()  — upsert all trades from PaperPortfolio to paper_trades table
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any

import pandas as pd

from src.db.supabase_client import upsert_rows

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to JSON-serialisable types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)[:10]
    if isinstance(obj, float) and (obj != obj):   # NaN check
        return None
    return obj


def _fallback_path(fallback_dir: str, filename: str) -> Path:
    p = Path(fallback_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / filename


# ---------------------------------------------------------------------------
# 1. Save predictions
# ---------------------------------------------------------------------------
def save_predictions(
    signals_df: pd.DataFrame,
    run_id: str,
    model_version: str,
    supabase_client,
    fallback_dir: str = "outputs",
) -> bool:
    """Persist each row of signals_df to the Supabase predictions table.

    Falls back to appending outputs/predictions_journal.json if client is None.
    Returns True if at least one persistence path succeeded.
    """
    if signals_df.empty:
        logger.info("No signals to save")
        return False

    rows: list[dict] = []
    for _, r in signals_df.iterrows():
        row = {
            "run_id":        run_id,
            "model_version": model_version or run_id,
            "signal_date":   str(r.get("signal_date", date.today())),
            "ticker":        str(r["ticker"]),
            "signal":        str(r.get("signal", "NEUTRAL")),
            "prob_up":       _json_safe(r.get("prob_up")),
            "entry_price":   _json_safe(r.get("entry_price")),
            "stop_loss":     _json_safe(r.get("stop_loss")),
            "target_price":  _json_safe(r.get("target_price")),
            "risk_reward":   _json_safe(r.get("risk_reward")),
            "atr14":         _json_safe(r.get("atr14")),
            "horizon_days":  int(r.get("horizon_days", 5)),
        }
        rows.append(row)

    # Supabase write
    sb_ok = upsert_rows(supabase_client, "predictions", rows, on_conflict="signal_date,ticker,run_id")
    if sb_ok:
        logger.info("Saved %d predictions to Supabase (run_id=%s)", len(rows), run_id)

    # JSON fallback — always write so local mode works
    fpath = _fallback_path(fallback_dir, "predictions_journal.json")
    try:
        existing: list = []
        if fpath.exists():
            existing = json.loads(fpath.read_text())
        # Remove stale entries for the same (signal_date, run_id) combo
        signal_date = rows[0]["signal_date"] if rows else ""
        existing = [
            e for e in existing
            if not (e.get("signal_date") == signal_date and e.get("run_id") == run_id)
        ]
        existing.extend(rows)
        fpath.write_text(json.dumps(existing, indent=2, default=str))
        logger.info("Predictions appended → %s", fpath)
        return True
    except Exception as exc:
        logger.warning("JSON fallback write failed: %s", exc)
        return sb_ok


# ---------------------------------------------------------------------------
# 2. Save run metadata
# ---------------------------------------------------------------------------
def save_run_metadata(
    stats_dict: dict,
    run_id: str,
    model_version: str,
    best_params: dict,
    feat_imp: dict | None,
    supabase_client,
    fallback_dir: str = "outputs",
) -> bool:
    """Insert/update a row in model_runs with this run's performance stats.

    Also saves feature_importance rows if feat_imp is provided.
    Falls back to outputs/model_runs.json if client is None.
    """
    today = date.today().isoformat()

    # Build model_runs row
    run_row = {
        "run_id":         run_id,
        "model_version":  model_version or run_id,
        "run_date":       today,
        "horizon_days":   int(stats_dict.get("horizon_days", 5)),
        "label_type":     str(stats_dict.get("label_type", "triple_barrier")),
        "n_trials":       _json_safe(best_params.get("n_estimators")),
        # OOF metrics (set by runner.py in stats_dict if not skip_backtest)
        "oof_ic":         _json_safe(stats_dict.get("oof_ic")),
        "oof_dir_acc":    _json_safe(stats_dict.get("oof_dir_acc")),
        # Backtest metrics
        "bt_cagr":        _json_safe(stats_dict.get("CAGR")),
        "bt_sharpe":      _json_safe(stats_dict.get("Sharpe")),
        "bt_sortino":     _json_safe(stats_dict.get("Sortino")),
        "bt_calmar":      _json_safe(stats_dict.get("Calmar")),
        "bt_max_drawdown":_json_safe(stats_dict.get("max_drawdown")),
        "bt_hit_rate":    _json_safe(stats_dict.get("hit_rate")),
        "bt_profit_factor":_json_safe(stats_dict.get("profit_factor")),
        "best_params":    _json_safe(best_params),
        "is_deployed":    False,
    }

    sb_ok = upsert_rows(supabase_client, "model_runs", [run_row], on_conflict="run_id")
    if sb_ok:
        logger.info("Run metadata saved to Supabase (run_id=%s)", run_id)

    # Feature importance
    if feat_imp and supabase_client:
        fi_rows = [
            {
                "run_id":          run_id,
                "feature_name":    name,
                "importance_score": _json_safe(score),
                "importance_rank": rank + 1,
            }
            for rank, (name, score) in enumerate(
                sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:50]
            )
        ]
        upsert_rows(supabase_client, "feature_importance", fi_rows, on_conflict="run_id,feature_name")

    # JSON fallback
    fpath = _fallback_path(fallback_dir, "model_runs.json")
    try:
        existing: list = []
        if fpath.exists():
            existing = json.loads(fpath.read_text())
        existing = [e for e in existing if e.get("run_id") != run_id]
        existing.append(run_row)
        fpath.write_text(json.dumps(existing, indent=2, default=str))

        # Also save feature importance to a local JSON
        if feat_imp:
            fi_path = _fallback_path(fallback_dir, f"feature_importance_{run_id}.json")
            fi_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:50]
            fi_path.write_text(json.dumps(
                [{"feature": n, "score": float(s)} for n, s in fi_sorted],
                indent=2,
            ))
        return True
    except Exception as exc:
        logger.warning("model_runs JSON fallback failed: %s", exc)
        return sb_ok


# ---------------------------------------------------------------------------
# 3. Sync paper trades
# ---------------------------------------------------------------------------
def sync_paper_trades(
    portfolio,
    run_id: str,
    supabase_client,
) -> int:
    """Upsert all trades from PaperPortfolio into the paper_trades table.

    Returns count of rows synced. Silently skips if client is None.
    The upsert is idempotent — safe to call multiple times per day.
    """
    if not portfolio.trades:
        return 0

    from dataclasses import asdict
    rows = []
    for t in portfolio.trades:
        d = asdict(t)
        d["run_id"] = run_id
        d = _json_safe(d)
        rows.append(d)

    if upsert_rows(supabase_client, "paper_trades", rows, on_conflict="trade_id"):
        logger.info("Synced %d paper trades to Supabase", len(rows))
        return len(rows)
    return 0


# ---------------------------------------------------------------------------
# 4. Sync account ledger (funds statement)
# ---------------------------------------------------------------------------
def sync_ledger(
    portfolio,
    run_id: str,
    supabase_client,
    fallback_dir: str = "outputs",
) -> int:
    """Upsert all funds-ledger rows from PaperPortfolio into account_ledger.

    Idempotent (each ledger row has a stable uuid `id`). Always mirrors to
    outputs/ledger.json so local/offline mode has the same statement.
    Returns count of rows synced (Supabase path) or written (JSON-only path).
    """
    if not getattr(portfolio, "ledger", None):
        return 0

    rows = [_json_safe({**row, "run_id": run_id}) for row in portfolio.ledger]

    sb_ok = upsert_rows(supabase_client, "account_ledger", rows, on_conflict="id")
    if sb_ok:
        logger.info("Synced %d ledger rows to Supabase", len(rows))

    fpath = _fallback_path(fallback_dir, "ledger.json")
    try:
        fpath.write_text(json.dumps(rows, indent=2, default=str))
        return len(rows)
    except Exception as exc:
        logger.warning("ledger.json write failed: %s", exc)
        return len(rows) if sb_ok else 0
