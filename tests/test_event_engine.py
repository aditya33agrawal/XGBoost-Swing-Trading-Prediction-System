"""Tests for src.backtest.event_engine and engine.run_backtest_bucketed_sleeves
(docs/dynamic-horizon-rr-plan.md Phase 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.backtest.event_engine import run_event_backtest
from src.backtest.engine import run_backtest, run_backtest_bucketed_sleeves


def _flat_price_panel(tickers, n_days=10, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in tickers:
        rows.append(pd.DataFrame({
            "ticker": t, "date": dates,
            "high": 101.0, "low": 99.0, "close": 100.0,
        }))
    return pd.concat(rows, ignore_index=True)


def test_event_engine_closes_target_and_stop_for_the_right_reason():
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    price_df = _flat_price_panel(tickers, n_days=10)
    dates = sorted(price_df["date"].unique())

    cfg = Config(n_quantile=2, max_positions=10, regime_filter=False,
                 signal_stop_atr_mult=1.5, signal_target_atr_mult=3.0, horizon=5)

    # Fallback ATR (no q10_star/q90_star in oof_preds) = 1.5% of entry price = 1.5.
    # stop = 100 - 1.5*1.5 = 97.75 ; target = 100 + 3*1.5 = 104.5
    entry = 100.0
    atr_fallback = entry * 0.015
    target_price = entry + 3.0 * atr_fallback
    stop_price = entry - 1.5 * atr_fallback

    # AAA hits target on the day after entry; BBB hits stop two days after entry.
    price_df.loc[(price_df["ticker"] == "AAA") & (price_df["date"] == dates[1]), "high"] = target_price + 1.0
    price_df.loc[(price_df["ticker"] == "BBB") & (price_df["date"] == dates[2]), "low"] = stop_price - 1.0

    oof_preds = pd.DataFrame({
        "date": [dates[0]] * 4,
        "ticker": tickers,
        "pred": [0.9, 0.8, 0.2, 0.1],   # AAA, BBB are the top-quintile longs
        "fwd_ret": [0.0, 0.0, 0.0, 0.0],
    })

    stats = run_event_backtest(oof_preds, price_df, cfg)

    assert "error" not in stats
    assert stats["n_exits_target"] == 1
    assert stats["n_exits_stop"] == 1
    assert stats["n_periods"] >= 1
    assert not stats["period_returns"].isna().any()
    assert np.isclose(stats["equity_curve"].iloc[0], 1.0)


def test_event_engine_handles_missing_price_history_without_crashing():
    """A ticker present in oof_preds but with sparse/missing price rows must
    not crash the loop — position just stays open / uses the stale fallback."""
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    price_df = _flat_price_panel(tickers, n_days=8)
    dates = sorted(price_df["date"].unique())

    # Drop AAA's price rows after entry to simulate a data gap.
    price_df = price_df[~((price_df["ticker"] == "AAA") & (price_df["date"] > dates[0]))]

    cfg = Config(n_quantile=2, max_positions=10, regime_filter=False, horizon=3)
    oof_preds = pd.DataFrame({
        "date": [dates[0]] * 4,
        "ticker": tickers,
        "pred": [0.9, 0.8, 0.2, 0.1],
        "fwd_ret": [0.0, 0.0, 0.0, 0.0],
    })
    stats = run_event_backtest(oof_preds, price_df, cfg)
    assert "error" not in stats


def test_event_engine_empty_inputs_return_error_not_crash():
    cfg = Config()
    assert "error" in run_event_backtest(pd.DataFrame(), pd.DataFrame(), cfg)


def test_bucketed_sleeves_combines_two_horizon_buckets():
    cfg = Config(n_quantile=2, regime_filter=False, rebalance_every=5)
    dates = pd.bdate_range("2024-01-01", periods=4)

    # Bucket h=5: 2 dates, bucket h=21: 2 dates — each with >= n_quantile*2 names.
    rows = []
    for i, d in enumerate(dates):
        h = 5 if i < 2 else 21
        for j, t in enumerate(["A", "B", "C", "D"]):
            rows.append({
                "date": d, "ticker": t, "pred": float(j),
                "fwd_ret": 0.01 * (j - 1.5), "horizon_star": h,
            })
    oof_preds = pd.DataFrame(rows)

    combined = run_backtest_bucketed_sleeves(oof_preds, cfg)
    assert "error" not in combined
    assert "sleeve_stats" in combined
    assert set(combined["sleeve_stats"].keys()) == {5, 21}

    # Hand-computed: combined return on each date = (bucket_return)/n_sleeves
    # for whichever sleeve traded that date (the other sleeve contributes 0).
    bucket5 = run_backtest(oof_preds[oof_preds["horizon_star"] == 5], cfg)
    bucket21 = run_backtest(oof_preds[oof_preds["horizon_star"] == 21], cfg)
    n_sleeves = 2
    expected = pd.concat([
        bucket5["period_returns"] / n_sleeves,
        bucket21["period_returns"] / n_sleeves,
    ]).sort_index()
    pd.testing.assert_series_equal(
        combined["period_returns"].sort_index(), expected, check_names=False, atol=1e-9,
    )


def test_bucketed_sleeves_missing_horizon_star_returns_error():
    cfg = Config()
    oof_preds = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "ticker": ["A"], "pred": [0.1], "fwd_ret": [0.0]})
    result = run_backtest_bucketed_sleeves(oof_preds, cfg)
    assert "error" in result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
