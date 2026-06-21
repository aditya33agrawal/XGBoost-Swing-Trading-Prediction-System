"""Vectorised backtest engine (plan §10).

Takes the walk-forward prediction frame and simulates a cross-sectional
long-only (or long/short) quantile portfolio with Indian transaction costs.

Entry: next open after signal (T+1 — T+0 optional for Nifty 50).
Exit: after `horizon` trading days OR triple-barrier hit.
Cost: turnover-based, using indian_round_trip_cost fraction.

Returns period returns, equity curve, and summary stats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import src.backtest.costs as _costs_mod
from src.validation.metrics import summarise


def run_backtest(
    preds: pd.DataFrame,
    cfg,
    rt_cost_override: float | None = None,
) -> dict:
    """Simulate portfolio from walk-forward prediction frame.

    Parameters
    ----------
    preds : DataFrame with columns [date, ticker, pred, fwd_ret]
            `pred` is the model score (higher = more bullish)
            `fwd_ret` is the realised h-day return (for evaluation)
    cfg   : Config object
    rt_cost_override : override round-trip cost fraction (for sensitivity analysis)

    Returns
    -------
    dict with equity curve, period returns, and summary statistics
    """
    if preds.empty or "pred" not in preds.columns or "fwd_ret" not in preds.columns:
        return {"error": "prediction frame is empty or missing required columns"}

    rt_cost = rt_cost_override if rt_cost_override is not None else _costs_mod.APPROX_RT_COST_FRACTION
    period_rets: list[float] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    dates_traded: list[pd.Timestamp] = []

    regime_col = getattr(cfg, "regime_sma_col", "nifty_dist_sma200")
    use_regime = getattr(cfg, "regime_filter", False) and regime_col in preds.columns

    for date, day in preds.groupby("date"):
        day = day.dropna(subset=["pred", "fwd_ret"])
        if len(day) < cfg.n_quantile * 2:
            continue

        # Risk overlay: if the index is below its long SMA on this rebalance
        # date, go flat — hold no longs.  Prior book is closed (turnover cost
        # applies once), then we sit in cash until the regime turns back on.
        if use_regime:
            regime_val = float(day[regime_col].iloc[0])
            if regime_val < 0:
                turn_long = 1.0 if prev_long else 0.0
                period_rets.append(-rt_cost * turn_long)
                dates_traded.append(date)
                prev_long, prev_short = set(), set()
                continue

        # Rank-based quintile assignment
        day = day.copy()
        day["q"] = pd.qcut(
            day["pred"].rank(method="first"),
            cfg.n_quantile,
            labels=False,
        )

        longs = day[day["q"] == cfg.n_quantile - 1]
        long_set = set(longs["ticker"])
        long_ret = float(longs["fwd_ret"].mean()) if not longs.empty else 0.0

        if cfg.mode == "long_short":
            shorts = day[day["q"] == 0]
            short_set = set(shorts["ticker"])
            short_ret = float(shorts["fwd_ret"].mean()) if not shorts.empty else 0.0
            gross = long_ret - short_ret
        else:
            short_set = set()
            gross = long_ret

        # Turnover-based cost: fraction of book that changed vs prior period
        turn_long = 1 - len(long_set & prev_long) / max(len(long_set), 1)
        if cfg.mode == "long_short" and short_set:
            turn_short = 1 - len(short_set & prev_short) / max(len(short_set), 1)
            cost = rt_cost * (turn_long + turn_short) / 2
        else:
            cost = rt_cost * turn_long

        net = gross - cost
        period_rets.append(net)
        dates_traded.append(date)
        prev_long, prev_short = long_set, short_set

    if not period_rets:
        return {"error": "no tradable periods — check data coverage and n_quantile"}

    r = pd.Series(period_rets, index=pd.to_datetime(dates_traded))
    equity = (1 + r).cumprod()
    periods_per_year = 252.0 / cfg.rebalance_every

    stats = summarise(r.values, periods_per_year=periods_per_year, label=cfg.mode)
    stats["equity_curve"] = equity
    stats["period_returns"] = r
    return stats


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------
def sensitivity_analysis(
    preds: pd.DataFrame,
    cfg,
    cost_multipliers: list[float] = (0.5, 1.0, 1.5, 2.0),
) -> pd.DataFrame:
    """Vary cost assumptions and report Sharpe across scenarios."""
    base_cost = _costs_mod.APPROX_RT_COST_FRACTION
    rows = []
    for mult in cost_multipliers:
        result = run_backtest(preds, cfg, rt_cost_override=base_cost * mult)
        rows.append({
            "cost_mult": mult,
            "Sharpe": result.get("Sharpe", np.nan),
            "CAGR": result.get("CAGR", np.nan),
            "max_drawdown": result.get("max_drawdown", np.nan),
        })
    return pd.DataFrame(rows)
