"""Detailed backtest report — persists what each run actually found.

Mirrors the pattern in `src.monitoring.drift` (build_*_report / write_*_report):
assemble a plain dict, then serialise it to JSON + a human-readable Markdown
file under `reports/` so every backtest leaves an auditable artefact, not just
console output that scrolls away.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Known historical baselines (from prior real-data runs) to compare against.
_BASELINE_OOF_IC = {"2026-06-17": 0.0139, "2026-06-21": 0.0281, "2026-06-22": 0.0308}


def _regime_overlay_stats(oof_preds: pd.DataFrame, cfg) -> dict:
    regime_col = getattr(cfg, "regime_sma_col", "nifty_dist_sma200")
    use_regime = getattr(cfg, "regime_filter", False)
    if not use_regime or oof_preds is None or oof_preds.empty or regime_col not in oof_preds.columns:
        return {"enabled": use_regime, "flat_fraction": None}
    per_date = oof_preds.groupby("date")[regime_col].first()
    flat_fraction = float((per_date < 0).mean()) if len(per_date) else None
    return {
        "enabled": True,
        "regime_col": regime_col,
        "n_rebalance_dates": int(len(per_date)),
        "n_flat_dates": int((per_date < 0).sum()),
        "flat_fraction": round(flat_fraction, 3) if flat_fraction is not None else None,
    }


def _findings(stats: dict, sensitivity_df: pd.DataFrame | None, regime_stats: dict) -> list[str]:
    bullets: list[str] = []
    oof_ic = stats.get("oof_ic")
    if oof_ic is not None:
        prior = sorted(_BASELINE_OOF_IC.items())
        prior_str = ", ".join(f"{d}={v:.4f}" for d, v in prior)
        bullets.append(f"OOF IC = {oof_ic:.4f} (prior runs for comparison: {prior_str})")

    sharpe = stats.get("Sharpe")
    if sharpe is not None:
        bullets.append(f"Net Sharpe = {sharpe:.2f}, CAGR = {stats.get('CAGR', float('nan')):.2%}, "
                        f"max drawdown = {stats.get('max_drawdown', float('nan')):.2%}")

    if sensitivity_df is not None and not sensitivity_df.empty:
        row_2x = sensitivity_df[sensitivity_df["cost_mult"] == 2.0]
        if not row_2x.empty:
            sh2x = row_2x["Sharpe"].iloc[0]
            verdict = "survives" if pd.notna(sh2x) and sh2x > 0 else "dies"
            bullets.append(f"Edge {verdict} at 2x assumed transaction costs (Sharpe={sh2x:.2f})")

    if regime_stats.get("enabled") and regime_stats.get("flat_fraction") is not None:
        bullets.append(
            f"Regime overlay forced flat on {regime_stats['n_flat_dates']}/"
            f"{regime_stats['n_rebalance_dates']} rebalance dates "
            f"({regime_stats['flat_fraction']:.0%})"
        )

    n_periods = stats.get("n_periods")
    if n_periods is not None:
        bullets.append(f"{n_periods} tradable rebalance periods in the walk-forward OOF window")

    return bullets


def build_backtest_report(
    *,
    stats: dict,
    sensitivity_df: pd.DataFrame | None,
    cfg,
    oof_preds: pd.DataFrame | None,
    price_df: pd.DataFrame | None = None,
    drift_report: dict | None = None,
) -> dict:
    """Assemble a full backtest report dict — config, metrics, findings."""
    regime_stats = _regime_overlay_stats(oof_preds, cfg)

    data_coverage = {}
    if price_df is not None and not price_df.empty:
        data_coverage = {
            "n_tickers": int(price_df["ticker"].nunique()),
            "n_rows": int(len(price_df)),
            "date_min": str(price_df["date"].min().date()),
            "date_max": str(price_df["date"].max().date()),
        }

    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": {
            "model_version": getattr(cfg, "model_version", None),
            "start": getattr(cfg, "start", None),
            "end": getattr(cfg, "end", None),
            "horizon": getattr(cfg, "horizon", None),
            "label_type": getattr(cfg, "label_type", None),
            "mode": getattr(cfg, "mode", None),
            "device": getattr(cfg, "device", None),
            "xgb_n_trials": getattr(cfg, "xgb_n_trials", None),
        },
        "data_coverage": data_coverage,
        "oof_metrics": {
            "oof_ic": stats.get("oof_ic"),
            "oof_dir_acc": stats.get("oof_dir_acc"),
        },
        "stats": {k: v for k, v in stats.items()
                  if k not in {"equity_curve", "period_returns", "error"}},
        "error": stats.get("error"),
        "cost_sensitivity": sensitivity_df.to_dict(orient="records") if sensitivity_df is not None and not sensitivity_df.empty else [],
        "regime_overlay": regime_stats,
        "retrain_recommended": (drift_report or {}).get("retrain_recommended"),
    }
    report["findings"] = _findings(stats, sensitivity_df, regime_stats)
    return report


def write_backtest_report(report: dict, reports_dir: str = "reports", tag: str | None = None) -> tuple[str, str]:
    """Write the report to JSON + Markdown. Returns (json_path, md_path)."""
    tag = tag or datetime.today().strftime("%Y-%m-%d")
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)

    json_path = d / f"backtest_{tag}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    run = report.get("run", {})
    stats = report.get("stats", {})
    cov = report.get("data_coverage", {})
    findings = report.get("findings", [])
    sens = report.get("cost_sensitivity", [])

    sens_table = "\n".join(
        f"| {r['cost_mult']}x | {r.get('Sharpe')} | {r.get('CAGR')} | {r.get('max_drawdown')} |"
        for r in sens
    )

    retrain = report.get("retrain_recommended")
    retrain_line = (
        "**RETRAIN RECOMMENDED** (drift alarm)" if retrain
        else ("Stable — no retrain signal" if retrain is not None else "No drift report available")
    )

    md = f"""# Backtest report — {tag}

Generated: {report.get('generated_utc')}

## Run config
- model_version: {run.get('model_version')}
- period: {run.get('start')} → {run.get('end')}
- horizon: {run.get('horizon')} days, label_type: {run.get('label_type')}, mode: {run.get('mode')}
- device: {run.get('device')}, optuna trials: {run.get('xgb_n_trials')}

## Data coverage
- tickers: {cov.get('n_tickers')}, rows: {cov.get('n_rows')}
- date range fetched: {cov.get('date_min')} → {cov.get('date_max')}

## OOF metrics
- OOF IC: {report.get('oof_metrics', {}).get('oof_ic')}
- OOF directional accuracy: {report.get('oof_metrics', {}).get('oof_dir_acc')}

## Backtest stats (cost-adjusted)
{json.dumps(stats, indent=2)}

## Cost sensitivity
| cost multiplier | Sharpe | CAGR | max_drawdown |
|---|---|---|---|
{sens_table}

## Regime overlay
{json.dumps(report.get('regime_overlay', {}), indent=2)}

## Findings
{chr(10).join(f"- {b}" for b in findings) if findings else "- (none)"}

## Retrain recommendation
{retrain_line}
"""
    md_path = d / f"backtest_{tag}.md"
    md_path.write_text(md)
    logger.info("Backtest report -> %s", md_path)
    return str(json_path), str(md_path)
