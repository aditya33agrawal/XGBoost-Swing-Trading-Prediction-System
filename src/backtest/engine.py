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


def conviction_weights(basket: pd.DataFrame, reverse: bool = False) -> pd.Series:
    """Per-name conviction weight within `basket`, by score-rank (plan
    §Phase 4.19) — the most confident name gets more capital than one barely
    past the cutoff. Shared by the vectorised backtest (`run_backtest`) and
    the event-driven engine (`event_engine.py`) so sizing logic only lives
    in one place.

    `reverse=True` for the short basket — most-bearish (lowest score) gets
    the highest weight there. Returned weights sum to 1 (or are empty).
    """
    if basket.empty:
        return pd.Series(dtype=float)
    # ascending=True ranks lowest pred as 1 → highest pred gets the largest
    # rank/weight (longs); reverse=True flips it so the lowest pred (most
    # bearish) gets the largest weight (shorts).
    ranks = basket["pred"].rank(method="first", ascending=not reverse)
    return ranks / ranks.sum()


def _conviction_weighted_return(basket: pd.DataFrame, reverse: bool = False) -> float:
    """Weight each name in `basket` by its score-rank conviction *within the
    basket* instead of equal-weighting (plan §Phase 4.19) — the most
    confident name gets more capital than one barely past the cutoff. A
    cheap stand-in for full meta-labeling-based sizing (plan §3.15): same
    primary model score, just used for sizing as well as selection.
    """
    if basket.empty:
        return 0.0
    weights = conviction_weights(basket, reverse=reverse)
    return float((basket["fwd_ret"] * weights).sum())


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

        conviction_weighted = getattr(cfg, "conviction_weighted_sizing", True)

        longs = day[day["q"] == cfg.n_quantile - 1]
        long_set = set(longs["ticker"])
        if longs.empty:
            long_ret = 0.0
        elif conviction_weighted:
            long_ret = _conviction_weighted_return(longs, reverse=False)
        else:
            long_ret = float(longs["fwd_ret"].mean())

        if cfg.mode == "long_short":
            shorts = day[day["q"] == 0]
            short_set = set(shorts["ticker"])
            if shorts.empty:
                short_ret = 0.0
            elif conviction_weighted:
                short_ret = _conviction_weighted_return(shorts, reverse=True)
            else:
                short_ret = float(shorts["fwd_ret"].mean())
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
# Horizon-bucketed sleeves (docs/dynamic-horizon-rr-plan.md Phase 4 item 16)
# — the cheap "does variable horizon help" check, built entirely on the
# existing vectorised `run_backtest`: discretise horizon_star into the grid
# buckets, run the unmodified backtest once per bucket, then combine equity
# curves by capital allocation. Does NOT see stop/target (that needs the
# full event_engine) — only validates the horizon-selection signal.
# ---------------------------------------------------------------------------
def run_backtest_bucketed_sleeves(
    oof_preds: pd.DataFrame,
    cfg,
    rt_cost_override: float | None = None,
) -> dict:
    """Run `run_backtest` once per horizon bucket in `oof_preds["horizon_star"]`,
    then combine the per-bucket equity curves into one capital-weighted curve.

    Requires `oof_preds` to carry `horizon_star` (from
    `_walk_forward_predict_surface`). Capital is split equally across the
    buckets that actually produced tradable periods.
    """
    if "horizon_star" not in oof_preds.columns:
        return {"error": "oof_preds missing 'horizon_star' — run with cfg.dynamic_horizon_enabled"}

    buckets = sorted(oof_preds["horizon_star"].dropna().unique())
    sleeve_stats: dict[int, dict] = {}
    for h in buckets:
        bucket_df = oof_preds[oof_preds["horizon_star"] == h]
        result = run_backtest(bucket_df, cfg, rt_cost_override=rt_cost_override)
        if "error" not in result and "period_returns" in result:
            sleeve_stats[int(h)] = result

    if not sleeve_stats:
        return {"error": "no bucket produced a tradable backtest"}

    # Equal capital allocation across sleeves that traded; combine daily by
    # reindexing each sleeve's return series onto the union of dates (0 return
    # on days a given sleeve didn't trade) and averaging.
    n_sleeves = len(sleeve_stats)
    all_dates = sorted(set().union(*(s["period_returns"].index for s in sleeve_stats.values())))
    combined = pd.Series(0.0, index=pd.Index(all_dates))
    for h, s in sleeve_stats.items():
        r = s["period_returns"].reindex(all_dates, fill_value=0.0)
        combined = combined.add(r / n_sleeves, fill_value=0.0)

    equity = (1 + combined).cumprod()
    periods_per_year = 252.0 / cfg.rebalance_every
    stats = summarise(combined.values, periods_per_year=periods_per_year, label=f"{cfg.mode}_sleeves")
    stats["equity_curve"] = equity
    stats["period_returns"] = combined
    stats["sleeve_stats"] = {h: {k: v for k, v in s.items() if k not in ("equity_curve", "period_returns")}
                              for h, s in sleeve_stats.items()}
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
