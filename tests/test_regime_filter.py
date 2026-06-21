"""Backtest regime overlay: go flat when the index is below its long SMA.

Verifies that with regime_filter on, rebalance dates where nifty_dist_sma200 < 0
hold no longs (return is only the closing cost), and that turning the filter off
restores normal quantile trading.
"""
import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest


class _Cfg:
    n_quantile = 5
    mode = "long_only"
    rebalance_every = 5
    regime_filter = True
    regime_sma_col = "nifty_dist_sma200"


def _make_preds(regime_val: float, n_names: int = 20) -> pd.DataFrame:
    """One rebalance date; pred correlates with fwd_ret so the top quintile wins."""
    rng = np.random.default_rng(0)
    pred = rng.normal(size=n_names)
    fwd = pred * 0.02 + rng.normal(scale=0.001, size=n_names)  # strong positive edge
    return pd.DataFrame({
        "date": pd.Timestamp("2024-01-05"),
        "ticker": [f"T{i}" for i in range(n_names)],
        "fwd_ret": fwd,
        "pred": pred,
        "nifty_dist_sma200": regime_val,
    })


def test_risk_off_goes_flat():
    cfg = _Cfg()
    preds = _make_preds(regime_val=-0.05)  # index below 200-SMA
    stats = run_backtest(preds, cfg)
    r = stats["period_returns"]
    # No prior book on the first date ⇒ flat ⇒ exactly zero return, not the
    # positive top-quintile return it would earn if it traded.
    assert float(r.iloc[0]) == 0.0


def test_risk_on_trades():
    cfg = _Cfg()
    preds = _make_preds(regime_val=+0.05)  # index above 200-SMA
    stats = run_backtest(preds, cfg)
    assert float(stats["period_returns"].iloc[0]) > 0.0


def test_filter_off_ignores_regime():
    cfg = _Cfg()
    cfg.regime_filter = False
    preds = _make_preds(regime_val=-0.05)  # risk-off, but filter disabled
    stats = run_backtest(preds, cfg)
    assert float(stats["period_returns"].iloc[0]) > 0.0
