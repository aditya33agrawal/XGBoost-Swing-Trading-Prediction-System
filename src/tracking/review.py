"""Weekly review — the predict → compare → adapt scorecard
(docs/small-capital-strategy-plan.md §3).

Every weekly retrain, after resolve_outcomes() settles last week's
predictions against realized prices, this module writes one markdown report:

  * last week's predicted-vs-actual table (per ticker: prob_up, signal,
    realized return, correct?, exit reason)
  * rolling live IC / directional-accuracy trend vs the backtest OOF IC
  * paper-portfolio equity, realized P&L and total cost drag
  * a verdict line (ON TRACK / WATCH / DEGRADING) driven by the live IC

This is the artifact to download from Colab each week during the 2-month
paper-trading window — it is the evidence for the go/no-go decision.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.tracking.outcome_tracker import (
    _load_outcomes_df, compute_recent_ic, compute_weekly_ic_series,
)

logger = logging.getLogger(__name__)


def _portfolio_snapshot(portfolio_path: str) -> dict:
    p = Path(portfolio_path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    trades = data.get("trades", [])
    closed = [t for t in trades if t.get("status") == "closed"]
    open_t = [t for t in trades if t.get("status") == "open"]
    realized = sum(t.get("pnl") or 0.0 for t in closed)
    charges = sum((t.get("entry_charges") or 0.0) + (t.get("exit_charges") or 0.0)
                  for t in trades)
    wins = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
    return {
        "initial_capital": data.get("initial_capital"),
        "cash": data.get("cash"),
        "n_open": len(open_t),
        "n_closed": len(closed),
        "realized_pnl": round(realized, 2),
        "total_charges": round(charges, 2),
        "win_rate": round(wins / len(closed), 3) if closed else None,
    }


def build_weekly_review(
    supabase_client,
    fallback_dir: str = "outputs",
    portfolio_path: str = "outputs/portfolio.json",
    backtest_oof_ic: float | None = None,
    n_weeks: int = 8,
) -> dict:
    """Assemble the review payload. Pure read — no writes."""
    outcomes = _load_outcomes_df(supabase_client, n_weeks=n_weeks, fallback_dir=fallback_dir)
    ic_series = compute_weekly_ic_series(supabase_client, n_weeks=n_weeks, fallback_dir=fallback_dir)
    live_ic_4w = compute_recent_ic(supabase_client, n_weeks=4, fallback_dir=fallback_dir)

    latest_week: pd.DataFrame = pd.DataFrame()
    if not outcomes.empty and "signal_date" in outcomes.columns:
        outcomes = outcomes.copy()
        outcomes["signal_date"] = outcomes["signal_date"].astype(str)
        last_date = outcomes["signal_date"].max()
        latest_week = outcomes[outcomes["signal_date"] == last_date]

    # Verdict: live 4-week IC drives the traffic light. Thresholds follow
    # models/improvement.py's emergency-retrain floor (IC < 0 sustained = bad).
    if isinstance(live_ic_4w, float) and not math.isnan(live_ic_4w):
        if live_ic_4w > 0.02:
            verdict = "ON TRACK — live IC positive and material"
        elif live_ic_4w > 0.0:
            verdict = "WATCH — live IC positive but thin"
        else:
            verdict = "DEGRADING — live IC ≤ 0; champion/challenger + drift gates are the backstop"
    else:
        verdict = "INSUFFICIENT DATA — need ≥1 resolved week of predictions"

    return {
        "generated_at": datetime.now().isoformat(),
        "live_ic_4w": None if (isinstance(live_ic_4w, float) and math.isnan(live_ic_4w)) else round(live_ic_4w, 4),
        "backtest_oof_ic": backtest_oof_ic,
        "verdict": verdict,
        "latest_week": latest_week,
        "ic_series": ic_series,
        "portfolio": _portfolio_snapshot(portfolio_path),
    }


def write_weekly_review(review: dict, reports_dir: str = "reports") -> Path:
    """Render the review dict to reports/weekly_review_<date>.md."""
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"weekly_review_{date.today().strftime('%Y%m%d')}.md"

    lines = [
        f"# Weekly Review — {date.today().isoformat()}",
        "",
        f"**Verdict:** {review['verdict']}",
        "",
        f"- Live rolling 4-week IC: **{review['live_ic_4w'] if review['live_ic_4w'] is not None else 'n/a'}**",
    ]
    if review.get("backtest_oof_ic") is not None:
        lines.append(f"- Backtest OOF IC (this model): {review['backtest_oof_ic']:.4f} "
                     f"— live should sit inside this ballpark; a persistent gap is concept drift")
    lines.append("")

    # --- last week predicted vs actual -----------------------------------
    lw: pd.DataFrame = review.get("latest_week", pd.DataFrame())
    lines.append("## Last resolved week — predicted vs actual")
    if lw is None or lw.empty:
        lines.append("")
        lines.append("_No newly resolved predictions this run._")
    else:
        lines.append("")
        lines.append(f"Signal date: **{lw['signal_date'].iloc[0]}** — {len(lw)} predictions resolved")
        lines.append("")
        lines.append("| Ticker | prob_up | Actual fwd ret | Correct | Exit |")
        lines.append("|---|---|---|---|---|")
        lw_sorted = lw.sort_values("prob_up", ascending=False) if "prob_up" in lw.columns else lw
        for _, r in lw_sorted.head(40).iterrows():
            ret = r.get("actual_fwd_ret")
            lines.append(
                f"| {r.get('ticker','?')} | {r.get('prob_up', float('nan')):.3f} "
                f"| {ret:+.2%} | {'✅' if r.get('is_correct') else '❌'} "
                f"| {r.get('exit_reason','-')} |"
            )
        n_correct = int(lw["is_correct"].sum()) if "is_correct" in lw.columns else 0
        lines.append("")
        lines.append(f"Week hit rate: **{n_correct}/{len(lw)}**")
    lines.append("")

    # --- IC trend ----------------------------------------------------------
    ics: pd.DataFrame = review.get("ic_series", pd.DataFrame())
    lines.append("## Live IC trend")
    if ics is None or ics.empty:
        lines.append("")
        lines.append("_Not enough resolved history yet._")
    else:
        lines.append("")
        lines.append("| Week | IC | N | Dir acc |")
        lines.append("|---|---|---|---|")
        for _, r in ics.iterrows():
            ic_s = f"{r['ic']:.4f}" if r["ic"] is not None else "n/a"
            da_s = f"{r['dir_accuracy']:.1%}" if r["dir_accuracy"] is not None else "n/a"
            lines.append(f"| {r['week_start']} | {ic_s} | {int(r['n_obs'])} | {da_s} |")
    lines.append("")

    # --- paper portfolio ----------------------------------------------------
    pf = review.get("portfolio") or {}
    lines.append("## Paper portfolio (₹-real cost accounting)")
    if not pf:
        lines.append("")
        lines.append("_No portfolio file yet._")
    else:
        lines.append("")
        lines.append(f"- Initial capital: ₹{pf.get('initial_capital', 0):,.0f}")
        lines.append(f"- Cash: ₹{pf.get('cash', 0):,.0f}")
        lines.append(f"- Open / closed positions: {pf.get('n_open', 0)} / {pf.get('n_closed', 0)}")
        lines.append(f"- Realized P&L (net): ₹{pf.get('realized_pnl', 0):,.0f}")
        lines.append(f"- **Total charges paid: ₹{pf.get('total_charges', 0):,.0f}** "
                     f"— watch this against P&L; at ₹20k sizing costs are the main enemy")
        if pf.get("win_rate") is not None:
            lines.append(f"- Win rate: {pf['win_rate']:.1%}")
    lines.append("")

    path.write_text("\n".join(lines))
    logger.info("Weekly review → %s", path)
    return path
