"""Drift monitoring — §9 of the implementation plan / §11.3 of the swing-bot plan.

Three families of drift, all computed in the weekly job and written to /reports:

  1. Feature drift   — PSI / KS of each feature vs the training reference.
                       PSI > 0.25 ⇒ retrain trigger.
  2. Concept drift   — rolling live IC / hit-rate from the ledger vs the backtest
                       expectation; a CUSUM on the prediction-error stream.
  3. Calibration drift — reliability of recent resolved `prob_up` vs realised.

`write_drift_report` serialises everything to JSON + a tiny standalone HTML so the
weekly run leaves an auditable artefact even with no dashboard server.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_RETRAIN = 0.25      # plan §9: PSI above this triggers a retrain
PSI_WATCH = 0.20        # investigate


# ---------------------------------------------------------------------------
# 1. Feature drift — PSI + KS
# ---------------------------------------------------------------------------
def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """PSI between a reference and current sample of one feature.

    Bins on reference quantiles (so bins are populated), then sums
    (cur% - ref%) * ln(cur% / ref%). Returns 0.0 if not enough data.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 20 or len(cur) < 20:
        return 0.0

    # quantile edges from the reference, dedup to avoid zero-width bins
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (0 = identical, 1 = disjoint)."""
    try:
        from scipy import stats
        ref = np.asarray(reference, float); cur = np.asarray(current, float)
        ref = ref[~np.isnan(ref)]; cur = cur[~np.isnan(cur)]
        if len(ref) < 20 or len(cur) < 20:
            return 0.0
        return float(stats.ks_2samp(ref, cur).statistic)
    except Exception:
        return 0.0


def feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Per-feature PSI + KS vs the training reference, flagged & sorted by PSI."""
    rows = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = population_stability_index(reference_df[col].values, current_df[col].values)
        ks = ks_statistic(reference_df[col].values, current_df[col].values)
        flag = "RETRAIN" if psi > PSI_RETRAIN else ("WATCH" if psi > PSI_WATCH else "ok")
        rows.append({"feature": col, "psi": round(psi, 4), "ks": round(ks, 4), "flag": flag})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 2. Concept drift — rolling IC/hit-rate from the ledger + CUSUM
# ---------------------------------------------------------------------------
def cusum_drift(errors: np.ndarray, threshold: float = 5.0, drift: float = 0.0) -> dict:
    """Two-sided CUSUM on a prediction-error stream.

    `errors` is e.g. |prob_up - realised_label| in time order. Returns the max
    cumulative deviation and whether it breached `threshold` (alarm).
    """
    e = np.asarray(errors, float)
    e = e[~np.isnan(e)]
    if len(e) < 5:
        return {"alarm": False, "g_max": 0.0, "n": int(len(e))}
    mean = float(np.mean(e))
    s_hi = s_lo = 0.0
    g_max = 0.0
    for x in e:
        s_hi = max(0.0, s_hi + (x - mean) - drift)
        s_lo = min(0.0, s_lo + (x - mean) + drift)
        g_max = max(g_max, s_hi, -s_lo)
    return {"alarm": bool(g_max > threshold), "g_max": round(g_max, 4), "n": int(len(e)),
            "mean_error": round(mean, 4)}


def concept_drift_from_outcomes(
    outcomes_df: pd.DataFrame,
    backtest_ic: float | None = None,
) -> dict:
    """Rolling live IC / hit-rate from the resolved ledger vs backtest expectation.

    Expects columns: prob_up, actual_fwd_ret, is_correct (as written by
    outcome_tracker). Returns live IC, hit-rate, CUSUM alarm, and the gap vs
    the backtest IC.
    """
    if outcomes_df is None or outcomes_df.empty:
        return {"live_ic": float("nan"), "hit_rate": float("nan"),
                "n": 0, "cusum": {"alarm": False}, "ic_gap": None}

    df = outcomes_df.dropna(subset=["prob_up", "actual_fwd_ret"])
    if len(df) < 5:
        return {"live_ic": float("nan"), "hit_rate": float("nan"),
                "n": int(len(df)), "cusum": {"alarm": False}, "ic_gap": None}

    from scipy import stats
    live_ic = float(stats.spearmanr(df["prob_up"], df["actual_fwd_ret"]).correlation)
    hit = float(df["is_correct"].mean()) if "is_correct" in df.columns else float("nan")

    # prediction-error stream for CUSUM: |prob_up - 1{ret>0}|
    realised = (df["actual_fwd_ret"] > 0).astype(float).values
    errors = np.abs(df["prob_up"].values - realised)
    cusum = cusum_drift(errors)

    ic_gap = None if backtest_ic is None else round(live_ic - float(backtest_ic), 4)
    return {"live_ic": round(live_ic, 4), "hit_rate": round(hit, 4),
            "n": int(len(df)), "cusum": cusum, "ic_gap": ic_gap}


# ---------------------------------------------------------------------------
# 3. Calibration drift — reliability of recent prob_up
# ---------------------------------------------------------------------------
def calibration_drift(outcomes_df: pd.DataFrame, bins: int = 10) -> dict:
    """Expected Calibration Error of recent resolved predictions.

    Compares predicted prob_up to the realised up-rate in each probability bin.
    """
    if outcomes_df is None or outcomes_df.empty:
        return {"ece": float("nan"), "n": 0}
    df = outcomes_df.dropna(subset=["prob_up", "actual_fwd_ret"])
    if len(df) < 10:
        return {"ece": float("nan"), "n": int(len(df))}

    p = df["prob_up"].values
    y = (df["actual_fwd_ret"].values > 0).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return {"ece": round(float(ece), 4), "n": int(len(df))}


# ---------------------------------------------------------------------------
# Report assembly + persistence
# ---------------------------------------------------------------------------
def build_drift_report(
    *,
    feature_drift: pd.DataFrame | None = None,
    concept: dict | None = None,
    calibration: dict | None = None,
    extra: dict | None = None,
) -> dict:
    fd = feature_drift if feature_drift is not None else pd.DataFrame()
    n_retrain = int((fd["flag"] == "RETRAIN").sum()) if not fd.empty else 0
    alarm = (
        n_retrain > 0
        or bool((concept or {}).get("cusum", {}).get("alarm", False))
    )
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retrain_recommended": alarm,
        "n_features_drifting": n_retrain,
        "feature_drift": fd.to_dict(orient="records") if not fd.empty else [],
        "concept_drift": concept or {},
        "calibration_drift": calibration or {},
        **(extra or {}),
    }


def write_drift_report(report: dict, reports_dir: str = "reports", tag: str | None = None) -> tuple[str, str]:
    """Write the report to JSON + a small standalone HTML. Returns (json_path, html_path)."""
    tag = tag or datetime.today().strftime("%Y-%m-%d")
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    json_path = d / f"drift_{tag}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    fd_rows = report.get("feature_drift", [])
    rows_html = "".join(
        f"<tr><td>{r['feature']}</td><td>{r['psi']}</td><td>{r['ks']}</td>"
        f"<td class='{r['flag']}'>{r['flag']}</td></tr>"
        for r in fd_rows[:30]
    )
    banner = "RETRAIN RECOMMENDED" if report.get("retrain_recommended") else "stable"
    html = f"""<!doctype html><meta charset=utf-8>
<title>Drift report {tag}</title>
<style>body{{font-family:system-ui;margin:2rem;color:#222}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px}}
.RETRAIN{{color:#b00;font-weight:700}}.WATCH{{color:#b80}}.ok{{color:#070}}
.banner{{padding:.5rem 1rem;border-radius:6px;display:inline-block;
background:{'#fee' if report.get('retrain_recommended') else '#efe'}}}</style>
<h1>Drift report — {tag}</h1>
<p class=banner><b>{banner}</b> · {report.get('n_features_drifting',0)} feature(s) drifting</p>
<h2>Concept drift (live ledger)</h2>
<pre>{json.dumps(report.get('concept_drift', {}), indent=2)}</pre>
<h2>Calibration drift</h2>
<pre>{json.dumps(report.get('calibration_drift', {}), indent=2)}</pre>
<h2>Feature drift (top 30 by PSI)</h2>
<table><tr><th>feature</th><th>PSI</th><th>KS</th><th>flag</th></tr>{rows_html}</table>
"""
    html_path = d / f"drift_{tag}.html"
    html_path.write_text(html)
    logger.info("Drift report → %s", html_path)
    return str(json_path), str(html_path)
