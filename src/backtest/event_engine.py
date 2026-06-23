"""Event-driven, overlapping-position backtest (docs/dynamic-horizon-rr-plan.md
Phase 4). Mirrors `src/trading/paper_trader.py::PaperPortfolio.update`'s exit
priority (target > stop > expiry) over historical data, so the backtest and
the live/paper trader finally agree on what "exit" means — today's vectorised
`run_backtest` (src/backtest/engine.py) never simulates a stop or target at
all; it just books the realised `fwd_ret` of a quintile basket. This module is
what makes a dynamic risk-reward evaluable at all.

Highest-overfitting-risk module in the whole plan (plan §4 "what not to
expect"): a path-dependent stop/target/expiry simulator plus a per-name
barrier choice has more degrees of freedom than a quintile-mean backtest.
Treat any large jump in these stats as a bug to hunt, not a result to trust,
until it clears the Phase 0 deflated-Sharpe / CPCV bar (src/validation/metrics.py).

Ablation note: which of the plan's (a)/(b)/(c)/(d) combination a given run
represents is fully determined by whether `oof_preds` carries
`horizon_star`/`q10_star`/`q90_star` (dynamic horizon — from
`_walk_forward_predict_surface`) — both flow from the same
`cfg.dynamic_horizon_enabled` flag, so Colab runs across the 4 settings are
directly comparable via `reports/backtest_v*.md`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import src.backtest.costs as _costs_mod
from src.backtest.engine import conviction_weights
from src.validation.metrics import summarise

logger = logging.getLogger(__name__)


def _atr14_asof(price_by_ticker: dict[str, pd.DataFrame], ticker: str, as_of) -> float:
    """Cheap trailing-20-bar true-range average as of `as_of` (inclusive)."""
    g = price_by_ticker.get(ticker)
    if g is None:
        return float("nan")
    hist = g[g.index <= as_of].tail(20)
    if len(hist) < 2:
        return float("nan")
    hi, lo, cl = hist["high"].to_numpy(float), hist["low"].to_numpy(float), hist["close"].to_numpy(float)
    tr = np.maximum(hi[1:] - lo[1:], np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
    return float(np.mean(tr)) if len(tr) else float("nan")


def run_event_backtest(
    oof_preds: pd.DataFrame,
    price_df: pd.DataFrame,
    cfg,
) -> dict:
    """Simulate an explicit, asynchronous position book with per-trade
    stop/target/horizon exits, marked to market daily.

    Parameters
    ----------
    oof_preds : walk-forward frame, one row per (date, ticker), with column
                `pred` and, when available, `horizon_star`/`q10_star`/`q90_star`
                (dynamic horizon/RR, from `_walk_forward_predict_surface`).
                Falls back to `cfg.horizon` / `cfg.signal_stop_atr_mult` /
                `cfg.signal_target_atr_mult` when those columns are absent —
                so this engine also works as a pure "what if we'd simulated
                exits on the legacy fixed-horizon/fixed-RR strategy" check.
    price_df  : raw OHLCV frame (date, ticker, high, low, close) used to walk
                the daily path and check target/stop, exactly like
                `PaperPortfolio.update`.
    cfg       : Config.

    Returns
    -------
    dict shaped like `run_backtest`'s output (via `summarise`) plus
    `equity_curve`/`period_returns`, so it's a drop-in alternative wherever
    those are consumed (`_print_backtest_results`, `sensitivity_analysis`).
    """
    if oof_preds.empty or "pred" not in oof_preds.columns:
        return {"error": "oof_preds is empty or missing 'pred'"}
    if not {"high", "low", "close"}.issubset(price_df.columns):
        return {"error": "price_df missing high/low/close — cannot walk daily path"}

    has_surface = {"horizon_star", "q10_star", "q90_star"}.issubset(oof_preds.columns)
    fallback_h = cfg.horizon
    fallback_stop_mult = getattr(cfg, "signal_stop_atr_mult", 1.5)
    fallback_target_mult = getattr(cfg, "signal_target_atr_mult", 3.0)
    rr_k = getattr(cfg, "rr_k", 1.0)
    stop_lo, stop_hi = getattr(cfg, "stop_atr_clamp", (0.8, 3.0))
    tgt_lo, tgt_hi = getattr(cfg, "target_atr_clamp", (1.0, 6.0))

    price_by_ticker: dict[str, pd.DataFrame] = {
        t: g.sort_values("date").set_index("date")
        for t, g in price_df.groupby("ticker")
    }

    all_dates = sorted(price_df["date"].unique())
    rebal_dates = set(oof_preds["date"].unique())
    regime_col = getattr(cfg, "regime_sma_col", "nifty_dist_sma200")
    use_regime = getattr(cfg, "regime_filter", False) and regime_col in oof_preds.columns
    rt_cost = _costs_mod.APPROX_RT_COST_FRACTION  # per-leg friction approximation

    open_positions: dict[str, dict] = {}
    cash = 1.0
    nav_dates: list = []
    nav_values: list[float] = []
    n_target = n_stop = n_expired = 0

    def _close(ticker: str, price: float, reason: str) -> None:
        nonlocal cash, n_target, n_stop, n_expired
        pos = open_positions.pop(ticker)
        proceeds = pos["shares"] * price * (1 - rt_cost / 2)
        cash += proceeds
        if reason == "target":
            n_target += 1
        elif reason == "stop":
            n_stop += 1
        else:
            n_expired += 1

    for date in all_dates:
        # 1. Mark every open position's exit condition before opening new ones.
        for ticker in list(open_positions.keys()):
            g = price_by_ticker.get(ticker)
            if g is None or date not in g.index:
                continue  # missing bar — leave position open, don't crash the loop
            row = g.loc[date]
            hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
            pos = open_positions[ticker]
            pos["days_held"] += 1
            if hi >= pos["target"]:
                _close(ticker, pos["target"], "target")
            elif lo <= pos["stop"]:
                _close(ticker, pos["stop"], "stop")
            elif pos["days_held"] >= pos["horizon"]:
                _close(ticker, cl, "expired")

        # 2. Open new positions on rebalance dates (mirrors run_backtest's
        #    quintile selection + conviction-weighted sizing, plus a stop/
        #    target this vectorised engine never sees).
        if date in rebal_dates:
            day = oof_preds[oof_preds["date"] == date].dropna(subset=["pred"])
            if use_regime and not day.empty:
                regime_val = float(day[regime_col].iloc[0])
                if regime_val < 0:
                    day = day.iloc[0:0]

            if len(day) >= cfg.n_quantile * 2:
                day = day.copy()
                day["q"] = pd.qcut(day["pred"].rank(method="first"), cfg.n_quantile, labels=False)
                longs = day[day["q"] == cfg.n_quantile - 1]
                longs = longs[~longs["ticker"].isin(open_positions.keys())]
                slots = max(0, cfg.max_positions - len(open_positions))
                longs = longs.head(slots)

                if not longs.empty:
                    weights = conviction_weights(longs, reverse=False)
                    invested_now = sum(
                        p["shares"] * float(price_by_ticker[t].loc[date, "close"])
                        for t, p in open_positions.items()
                        if t in price_by_ticker and date in price_by_ticker[t].index
                    )
                    portfolio_value = cash + invested_now

                    for (_, row), w in zip(longs.iterrows(), weights):
                        ticker = row["ticker"]
                        g = price_by_ticker.get(ticker)
                        if g is None or date not in g.index:
                            continue
                        entry_price = float(g.loc[date, "close"])
                        if entry_price <= 0:
                            continue
                        atr = _atr14_asof(price_by_ticker, ticker, date)
                        if not np.isfinite(atr) or atr <= 0:
                            atr = entry_price * 0.015  # same fallback as signals.enrich_signals

                        if has_surface and pd.notna(row.get("q10_star")) and pd.notna(row.get("q90_star")):
                            stop_mult = float(np.clip(
                                rr_k * abs(row["q10_star"]) * entry_price / atr, stop_lo, stop_hi))
                            target_mult = float(np.clip(
                                rr_k * max(row["q90_star"], 0.0) * entry_price / atr, tgt_lo, tgt_hi))
                            horizon = int(row["horizon_star"]) if pd.notna(row.get("horizon_star")) else fallback_h
                        else:
                            stop_mult, target_mult, horizon = fallback_stop_mult, fallback_target_mult, fallback_h

                        # Solve notional so that notional + cost == portfolio_value*w
                        # exactly (weights sum to 1 across the basket) — otherwise
                        # the cost leg makes the last name(s) in the loop overdraw
                        # cash that the earlier names already (rightfully) spent.
                        notional = portfolio_value * float(w) / (1 + rt_cost / 2)
                        cost = notional * rt_cost / 2
                        if notional <= 0 or cash < notional + cost - 1e-9:
                            continue
                        cash -= (notional + cost)
                        open_positions[ticker] = {
                            "shares": notional / entry_price,
                            "entry_price": entry_price,
                            "stop": entry_price - stop_mult * atr,
                            "target": entry_price + target_mult * atr,
                            "horizon": horizon,
                            "days_held": 0,
                        }

        # 3. Daily mark-to-market NAV.
        invested = 0.0
        for t, p in open_positions.items():
            g = price_by_ticker.get(t)
            if g is not None and date in g.index:
                invested += p["shares"] * float(g.loc[date, "close"])
            else:
                invested += p["shares"] * p["entry_price"]  # stale fallback, never crashes
        nav_dates.append(date)
        nav_values.append(cash + invested)

    if len(nav_values) < 2:
        return {"error": "event backtest produced no NAV history — check data coverage"}

    nav = pd.Series(nav_values, index=pd.to_datetime(nav_dates))
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"error": "event backtest NAV series too short to compute returns"}

    stats = summarise(rets.values, periods_per_year=252.0, label=f"{cfg.mode}_event")
    stats["equity_curve"] = nav / nav.iloc[0]
    stats["period_returns"] = rets
    stats["n_periods"] = len(rets)
    stats["n_exits_target"] = n_target
    stats["n_exits_stop"] = n_stop
    stats["n_exits_expired"] = n_expired
    stats["n_open_at_end"] = len(open_positions)
    return stats
