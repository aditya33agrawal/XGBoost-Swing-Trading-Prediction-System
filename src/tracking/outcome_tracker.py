"""Outcome tracker — resolve predictions after the horizon has elapsed.

After `horizon_days` trading days, we know whether a signal was correct.
This module:
  1. Finds unresolved predictions (outcome_resolved=False, signal_date <= today - horizon)
  2. Looks up the actual closing price at signal_date + horizon_days from price_df
  3. Computes actual_fwd_ret, hit_target, hit_stop, is_correct
  4. Updates the predictions row and inserts an outcomes row in Supabase
  5. Falls back to local JSON when Supabase is not configured

Public API:
  resolve_outcomes(price_df, client, fallback_dir, horizon_days) -> int
  compute_recent_ic(client, n_weeks, fallback_dir) -> float
  compute_weekly_ic_series(client, n_weeks, fallback_dir) -> pd.DataFrame
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.db.supabase_client import fetch_rows, upsert_rows

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_date(val) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return datetime.strptime(str(val)[:10], _DATE_FMT).date()
    except Exception:
        return None


def _build_price_lookup(price_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Build {(ticker, date_str): close_price} for fast look-up."""
    if price_df.empty or "close" not in price_df.columns:
        return {}
    price_df = price_df.copy()
    price_df["_date_str"] = pd.to_datetime(price_df["date"]).dt.strftime(_DATE_FMT)
    return {
        (row["ticker"], row["_date_str"]): float(row["close"])
        for _, row in price_df[["ticker", "_date_str", "close"]].iterrows()
    }


def _nearest_close_after(
    ticker: str,
    target_date: date,
    lookup: dict[tuple[str, str], float],
    max_search: int = 5,
) -> float | None:
    """Return close price on target_date or the next available trading day."""
    for delta in range(max_search + 1):
        d = (target_date + timedelta(days=delta)).strftime(_DATE_FMT)
        price = lookup.get((ticker, d))
        if price is not None:
            return price
    return None


def _load_local_predictions(fallback_dir: str) -> list[dict]:
    fpath = Path(fallback_dir) / "predictions_journal.json"
    if not fpath.exists():
        return []
    try:
        return json.loads(fpath.read_text())
    except Exception as exc:
        logger.warning("Could not load predictions_journal.json: %s", exc)
        return []


def _save_local_predictions(preds: list[dict], fallback_dir: str) -> None:
    fpath = Path(fallback_dir) / "predictions_journal.json"
    Path(fallback_dir).mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(preds, indent=2, default=str))


def _load_local_outcomes(fallback_dir: str) -> list[dict]:
    fpath = Path(fallback_dir) / "outcomes.json"
    if not fpath.exists():
        return []
    try:
        return json.loads(fpath.read_text())
    except Exception as exc:
        logger.warning("Could not load outcomes.json: %s", exc)
        return []


def _save_local_outcomes(outcomes: list[dict], fallback_dir: str) -> None:
    fpath = Path(fallback_dir) / "outcomes.json"
    Path(fallback_dir).mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(outcomes, indent=2, default=str))


# ---------------------------------------------------------------------------
# 1. Resolve outcomes
# ---------------------------------------------------------------------------
def resolve_outcomes(
    price_df: pd.DataFrame,
    supabase_client,
    fallback_dir: str = "outputs",
    horizon_days: int = 5,
) -> int:
    """Resolve predictions whose horizon has elapsed.

    Finds unresolved predictions with signal_date <= today - horizon_days.
    For each: fetches actual close, computes return, updates predictions +
    inserts an outcomes row.

    Returns count of newly resolved predictions.
    """
    today = date.today()
    cutoff = today - timedelta(days=horizon_days + 2)   # +2 for weekends buffer

    # ---- Load unresolved predictions ------------------------------------
    if supabase_client:
        raw = fetch_rows(
            supabase_client, "predictions",
            filters={"outcome_resolved": False},
        )
        unresolved = [
            r for r in raw
            if _to_date(r.get("signal_date")) is not None
            and _to_date(r["signal_date"]) <= cutoff
        ]
    else:
        all_preds = _load_local_predictions(fallback_dir)
        unresolved = [
            p for p in all_preds
            if not p.get("outcome_resolved", False)
            and _to_date(p.get("signal_date")) is not None
            and _to_date(p["signal_date"]) <= cutoff
        ]

    if not unresolved:
        logger.info("No unresolved predictions to settle")
        return 0

    logger.info("Resolving %d predictions (cutoff %s) …", len(unresolved), cutoff)

    # ---- Build price look-up -------------------------------------------
    lookup = _build_price_lookup(price_df)
    existing_outcomes = _load_local_outcomes(fallback_dir) if not supabase_client else []
    existing_pred_ids = {o.get("prediction_id") for o in existing_outcomes}

    prediction_updates: list[dict] = []
    outcome_rows:        list[dict] = []
    n_resolved = 0

    for pred in unresolved:
        signal_date  = _to_date(pred["signal_date"])
        target_date  = signal_date + timedelta(days=horizon_days + 2)  # approx h trading days
        ticker       = pred["ticker"]
        entry_price  = pred.get("entry_price") or 0.0
        stop_loss    = pred.get("stop_loss") or 0.0
        target_price = pred.get("target_price") or 0.0
        prob_up      = pred.get("prob_up") or 0.5
        pred_id      = pred.get("id")

        actual_close = _nearest_close_after(ticker, target_date, lookup, max_search=5)
        if actual_close is None:
            logger.debug("No price found for %s @ %s+%d", ticker, signal_date, horizon_days)
            continue

        # Check intraday barriers using min/max over the horizon window
        # (simple: use close price only — high/low would be more accurate)
        fwd_ret      = math.log(actual_close / entry_price) if entry_price > 0 else 0.0
        hit_target   = (actual_close >= target_price) if target_price > 0 else False
        hit_stop     = (actual_close <= stop_loss)     if stop_loss > 0    else False
        exit_reason  = "target" if hit_target else ("stop" if hit_stop else "expired")
        label_dir    = 1 if fwd_ret > 0 else -1
        is_correct   = (pred.get("signal") == "LONG" and fwd_ret > 0) or \
                       (pred.get("signal") == "NEUTRAL" and fwd_ret <= 0)
        outcome_date = target_date.isoformat()

        # Build Supabase update for predictions row
        pred_update = {
            "id":                pred_id,
            "outcome_resolved":  True,
            "outcome_date":      outcome_date,
            "actual_close":      round(actual_close, 2),
            "actual_fwd_ret":    round(fwd_ret, 6),
            "hit_target":        hit_target,
            "hit_stop":          hit_stop,
            "label_direction":   label_dir,
        }
        prediction_updates.append(pred_update)

        # Build outcomes row
        outcome_row = {
            "prediction_id":  pred_id,
            "run_id":         pred.get("run_id", ""),
            "signal_date":    pred["signal_date"],
            "outcome_date":   outcome_date,
            "ticker":         ticker,
            "prob_up":        prob_up,
            "actual_fwd_ret": round(fwd_ret, 6),
            "label_direction": label_dir,
            "is_correct":     is_correct,
            "exit_reason":    exit_reason,
        }
        outcome_rows.append(outcome_row)
        n_resolved += 1

    if not n_resolved:
        return 0

    # ---- Persist --------------------------------------------------------
    if supabase_client:
        upsert_rows(supabase_client, "predictions", prediction_updates, on_conflict="id")
        upsert_rows(supabase_client, "outcomes", outcome_rows, on_conflict="prediction_id")
    else:
        # Update local predictions
        all_preds = _load_local_predictions(fallback_dir)
        updated_ids = {r["id"] for r in prediction_updates if r.get("id")}
        all_preds = [p for p in all_preds if p.get("id") not in updated_ids]
        for orig, upd in zip(unresolved, prediction_updates):
            merged = {**orig, **upd}
            all_preds.append(merged)
        _save_local_predictions(all_preds, fallback_dir)

        # Append outcomes
        existing_outcomes = [o for o in existing_outcomes
                             if o.get("prediction_id") not in {r.get("prediction_id") for r in outcome_rows}]
        existing_outcomes.extend(outcome_rows)
        _save_local_outcomes(existing_outcomes, fallback_dir)

    logger.info("Resolved %d predictions", n_resolved)
    return n_resolved


# ---------------------------------------------------------------------------
# 2. Rolling IC
# ---------------------------------------------------------------------------
def compute_recent_ic(
    supabase_client,
    n_weeks: int = 4,
    fallback_dir: str = "outputs",
) -> float:
    """Spearman IC between prob_up and actual_fwd_ret over the last n_weeks."""
    df = _load_outcomes_df(supabase_client, n_weeks=n_weeks, fallback_dir=fallback_dir)
    if df.empty or len(df) < 5:
        logger.info("Insufficient resolved outcomes for IC computation (n=%d)", len(df))
        return float("nan")

    mask = df["prob_up"].notna() & df["actual_fwd_ret"].notna()
    df = df[mask]
    if len(df) < 5:
        return float("nan")

    ic, _ = scipy_stats.spearmanr(df["prob_up"].values, df["actual_fwd_ret"].values)
    logger.info("Recent %d-week IC = %.4f  (n=%d observations)", n_weeks, ic, len(df))
    return float(ic)


# ---------------------------------------------------------------------------
# 3. Weekly IC series for charting
# ---------------------------------------------------------------------------
def compute_weekly_ic_series(
    supabase_client,
    n_weeks: int = 12,
    fallback_dir: str = "outputs",
) -> pd.DataFrame:
    """Return DataFrame[week_start, ic, n_obs, dir_accuracy] for the last n_weeks."""
    df = _load_outcomes_df(supabase_client, n_weeks=n_weeks, fallback_dir=fallback_dir)
    if df.empty:
        return pd.DataFrame(columns=["week_start", "ic", "n_obs", "dir_accuracy"])

    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["week_start"]  = df["signal_date"].dt.to_period("W").dt.start_time

    rows = []
    for week, grp in df.groupby("week_start"):
        grp = grp.dropna(subset=["prob_up", "actual_fwd_ret"])
        if len(grp) < 3:
            continue
        ic, _ = scipy_stats.spearmanr(grp["prob_up"].values, grp["actual_fwd_ret"].values)
        dir_acc = float(grp["is_correct"].mean()) if "is_correct" in grp.columns else float("nan")
        rows.append({
            "week_start":    str(week.date()) if hasattr(week, "date") else str(week)[:10],
            "ic":            round(float(ic), 4) if not math.isnan(ic) else None,
            "n_obs":         len(grp),
            "dir_accuracy":  round(dir_acc, 3),
        })

    result = pd.DataFrame(rows).sort_values("week_start")
    return result


# ---------------------------------------------------------------------------
# Internal: load outcomes as DataFrame
# ---------------------------------------------------------------------------
def _load_outcomes_df(
    supabase_client,
    n_weeks: int = 12,
    fallback_dir: str = "outputs",
) -> pd.DataFrame:
    """Load resolved outcomes from Supabase or local JSON, up to n_weeks back."""
    cutoff = (date.today() - timedelta(weeks=n_weeks)).isoformat()

    if supabase_client:
        rows = fetch_rows(supabase_client, "outcomes", order_by="-signal_date")
        rows = [r for r in rows if (r.get("signal_date") or "") >= cutoff]
    else:
        rows = _load_local_outcomes(fallback_dir)
        rows = [r for r in rows if (r.get("signal_date") or "") >= cutoff]

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
