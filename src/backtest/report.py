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

from src.models.horizon_selection import diagnose_horizon_distribution
from src.validation.quantile_calibration import quantile_calibration_from_surface

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

    ic_ir = stats.get("oof_ic_ir")
    t_stat = stats.get("oof_ic_t_stat")
    if ic_ir is not None and t_stat is not None and ic_ir == ic_ir:  # not NaN
        sig = "statistically significant" if abs(t_stat) >= 2.0 else "NOT statistically distinguishable from zero"
        bullets.append(
            f"Daily cross-sectional IC-IR = {ic_ir:.3f} (t-stat={t_stat:.2f}, "
            f"n_days={stats.get('oof_ic_n_days')}) — {sig} at the usual |t|>=2 bar"
        )

    dsr = stats.get("deflated_sharpe")
    if dsr is not None and dsr == dsr:  # not NaN
        verdict = "plausibly real" if dsr >= 0.95 else "NOT yet distinguishable from a lucky draw among the Optuna trials"
        bullets.append(f"Deflated Sharpe Ratio = {dsr:.3f} (P[true Sharpe>0] after multiple-testing correction) — {verdict}")

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
            "oof_ic_daily_mean": stats.get("oof_ic_daily_mean"),
            "oof_ic_ir": stats.get("oof_ic_ir"),
            "oof_ic_t_stat": stats.get("oof_ic_t_stat"),
            "oof_ic_n_days": stats.get("oof_ic_n_days"),
        },
        "stats": {k: v for k, v in stats.items()
                  if k not in {"equity_curve", "period_returns", "error"}},
        "error": stats.get("error"),
        "cost_sensitivity": sensitivity_df.to_dict(orient="records") if sensitivity_df is not None and not sensitivity_df.empty else [],
        "regime_overlay": regime_stats,
        "retrain_recommended": (drift_report or {}).get("retrain_recommended"),
    }
    report["findings"] = _findings(stats, sensitivity_df, regime_stats)

    # Dynamic-horizon diagnostics (docs/dynamic-horizon-rr-plan.md Phase 6) —
    # only meaningful (and only present) when the run used
    # _walk_forward_predict_surface, i.e. oof_preds carries horizon_star.
    if getattr(cfg, "dynamic_horizon_enabled", False) and oof_preds is not None and not oof_preds.empty \
            and "horizon_star" in oof_preds.columns:
        diag = diagnose_horizon_distribution(oof_preds["horizon_star"].to_numpy(), list(cfg.horizon_grid))
        calib = quantile_calibration_from_surface(oof_preds, tuple(cfg.quantile_taus))
        report["dynamic_horizon"] = {
            "horizon_distribution": diag,
            "quantile_calibration": calib.to_dict(orient="records") if not calib.empty else [],
        }

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

    oof = report.get("oof_metrics", {})
    dsr = stats.get("deflated_sharpe")
    boot = stats.get("bootstrap_ci", {})
    boot_block = (
        f"- Sharpe  p05/p50/p95: {boot.get('sharpe_p05')} / {boot.get('sharpe_p50')} / {boot.get('sharpe_p95')}\n"
        f"- CAGR    p05/p50/p95: {boot.get('cagr_p05')} / {boot.get('cagr_p50')} / {boot.get('cagr_p95')}\n"
        f"- MaxDD   p05/p50/p95: {boot.get('max_drawdown_p05')} / {boot.get('max_drawdown_p50')} / {boot.get('max_drawdown_p95')}\n"
        f"({boot.get('n_boot')} block-bootstrap draws, block_size={boot.get('block_size')})"
        if boot and "error" not in boot else "- (not available)"
    )

    retrain = report.get("retrain_recommended")
    retrain_line = (
        "**RETRAIN RECOMMENDED** (drift alarm)" if retrain
        else ("Stable — no retrain signal" if retrain is not None else "No drift report available")
    )

    dh = report.get("dynamic_horizon")
    dh_section = ""
    if dh:
        diag = dh.get("horizon_distribution", {})
        calib_rows = dh.get("quantile_calibration", [])
        calib_table = "\n".join(
            f"| {r['tau']} | {r['realised_rate']} | {r['abs_gap']} | {r['n']} |" for r in calib_rows
        )
        dh_section = f"""
## Dynamic horizon (docs/dynamic-horizon-rr-plan.md Phase 6)
- h* distribution: {diag.get('fractions')}
- collapsed (>=95% one horizon): {diag.get('collapsed')}  |  looks uniform/noise: {diag.get('looks_uniform')}

### Quantile calibration (realised exceedance rate vs nominal tau — large abs_gap means don't trust the dynamic RR yet)
| tau | realised_rate | abs_gap | n |
|---|---|---|---|
{calib_table if calib_table else "| (no data) | | | |"}
"""

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
- OOF IC (pooled, legacy): {oof.get('oof_ic')}
- OOF directional accuracy: {oof.get('oof_dir_acc')}
- OOF daily IC (mean cross-sectional Spearman per date): {oof.get('oof_ic_daily_mean')}
- OOF IC-IR (mean/std of daily IC) / t-stat / n_days: {oof.get('oof_ic_ir')} / {oof.get('oof_ic_t_stat')} / {oof.get('oof_ic_n_days')}

## Backtest stats (cost-adjusted)
{json.dumps({k: v for k, v in stats.items() if k != 'bootstrap_ci'}, indent=2)}

## Robustness (Phase 0 — plan §A1/A9)
- Deflated Sharpe Ratio (P[true Sharpe>0], corrected for {run.get('xgb_n_trials')} Optuna trials): {dsr}
- Block-bootstrap CI (distribution across the one walk-forward path, not a point estimate):
{boot_block}

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
{dh_section}"""
    md_path = d / f"backtest_{tag}.md"
    md_path.write_text(md)
    logger.info("Backtest report -> %s", md_path)
    return str(json_path), str(md_path)
