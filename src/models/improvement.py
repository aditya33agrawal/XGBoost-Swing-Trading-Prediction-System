"""Model versioning, comparison, and deployment decision logic.

Used by:
  - runner.py: get_model_version() at startup
  - scripts/weekly_retrain.py: compare_models(), load_current_model_ic()
  - notebooks/colab_weekly.ipynb: all three
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version string
# ---------------------------------------------------------------------------
def get_model_version(prefix: str = "v") -> str:
    """Return a date-stamped version string: 'v20260617'."""
    return f"{prefix}{datetime.today().strftime('%Y%m%d')}"


# ---------------------------------------------------------------------------
# Deployment decision
# ---------------------------------------------------------------------------
def compare_models(
    new_ic: float,
    current_ic: float,
    min_improvement: float = 0.005,
) -> bool:
    """Return True if the new model should replace the current deployed model.

    Rules:
      - If current_ic is NaN (no deployed model): always deploy.
      - Deploy only if new_ic >= current_ic + min_improvement.
      - Prevents marginal regressions from reaching production.
    """
    if math.isnan(current_ic):
        logger.info("No deployed model — new model will be deployed")
        return True
    if math.isnan(new_ic):
        logger.warning("New model IC is NaN — not deploying")
        return False
    deploy = new_ic >= current_ic + min_improvement
    logger.info(
        "Model comparison: new IC=%.4f, current IC=%.4f, min_improvement=%.4f → %s",
        new_ic, current_ic, min_improvement,
        "DEPLOY" if deploy else "KEEP CURRENT",
    )
    return deploy


def should_retrain(
    recent_ic: float,
    threshold: float = -0.01,
    warn_threshold: float = 0.01,
) -> tuple[bool, str]:
    """Return (needs_emergency_retrain, message).

    Threshold checks (independent of weekly schedule):
      - recent_ic < threshold   → True, model is anti-predictive
      - recent_ic < warn_threshold → False but warning message
    """
    if math.isnan(recent_ic):
        return False, "IC is NaN — insufficient resolved outcomes yet"
    if recent_ic < threshold:
        msg = f"IC={recent_ic:.4f} is below emergency threshold {threshold} — retrain immediately"
        logger.warning(msg)
        return True, msg
    if recent_ic < warn_threshold:
        msg = f"IC={recent_ic:.4f} is degrading (warn threshold={warn_threshold}) — watch closely"
        logger.warning(msg)
        return False, msg
    return False, f"IC={recent_ic:.4f} is healthy"


# ---------------------------------------------------------------------------
# Load deployed model's rolling IC from Supabase or local fallback
# ---------------------------------------------------------------------------
def load_current_model_ic(
    supabase_client,
    fallback_dir: str = "outputs",
    n_recent_weeks: int = 4,
) -> tuple[float, str]:
    """Return (4-week rolling IC of deployed model, run_id).

    Falls back to local model_runs.json → returns the most recent deployed row.
    Returns (nan, "") if no deployed model exists or insufficient data.
    """
    from src.db.supabase_client import fetch_rows

    # Try Supabase first
    rows = fetch_rows(supabase_client, "model_runs", filters={"is_deployed": True}, limit=1)
    if rows:
        run_id = rows[0].get("run_id", "")
        oof_ic = rows[0].get("oof_ic")
        if oof_ic is not None:
            logger.info("Current deployed model: %s  OOF IC=%.4f", run_id, oof_ic)
            return float(oof_ic), run_id
        return float("nan"), run_id

    # JSON fallback
    fpath = Path(fallback_dir) / "model_runs.json"
    if fpath.exists():
        try:
            runs = json.loads(fpath.read_text())
            deployed = [r for r in runs if r.get("is_deployed")]
            if deployed:
                r = sorted(deployed, key=lambda x: x.get("run_date", ""), reverse=True)[0]
                oof_ic = r.get("oof_ic")
                run_id = r.get("run_id", "")
                if oof_ic is not None:
                    return float(oof_ic), run_id
                return float("nan"), run_id
        except Exception as exc:
            logger.warning("Could not load model_runs.json: %s", exc)

    return float("nan"), ""


# ---------------------------------------------------------------------------
# Mark a model as deployed in Supabase and local JSON
# ---------------------------------------------------------------------------
def mark_deployed(
    run_id: str,
    supabase_client,
    fallback_dir: str = "outputs",
    previous_run_id: str = "",
) -> None:
    """Set is_deployed=True for run_id and False for previous_run_id."""
    from src.db.supabase_client import upsert_rows

    now = datetime.utcnow().isoformat()

    if supabase_client:
        upsert_rows(supabase_client, "model_runs",
                    [{"run_id": run_id, "is_deployed": True, "deployed_at": now}],
                    on_conflict="run_id")
        if previous_run_id:
            upsert_rows(supabase_client, "model_runs",
                        [{"run_id": previous_run_id, "is_deployed": False}],
                        on_conflict="run_id")

    # JSON fallback
    fpath = Path(fallback_dir) / "model_runs.json"
    if fpath.exists():
        try:
            runs = json.loads(fpath.read_text())
            for r in runs:
                if r.get("run_id") == run_id:
                    r["is_deployed"] = True
                    r["deployed_at"] = now
                elif r.get("run_id") == previous_run_id:
                    r["is_deployed"] = False
            fpath.write_text(json.dumps(runs, indent=2, default=str))
        except Exception as exc:
            logger.warning("Could not update model_runs.json: %s", exc)

    logger.info("Model %s marked as deployed (previous: %s)", run_id, previous_run_id or "none")
