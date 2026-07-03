"""Tests for the 2026-06-29 retrain hardening: data-quality gates, market-neutral
target, and live-path multi-seed bagging (docs/weekly-retrain-fixes-2026-06-29.md
§2b + Tier 2)."""
import numpy as np
import pandas as pd
import pytest

from src.data.validation import (
    check_latest_bar_coverage, run_index_gates, summarize_data_quality,
    DataQualityError,
)
from src.labels.targets import residualize_fwd_ret, forward_log_return


def _prices(n_days=10, n_tickers=4, last_drop=None):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for t in range(n_tickers):
        # Optionally truncate one ticker's history to make its latest bar stale.
        d = dates[:-last_drop] if (last_drop and t == 0) else dates
        for x in d:
            rows.append({"date": x, "ticker": f"T{t}", "open": 100.0, "high": 101.0,
                         "low": 99.0, "close": 100.0, "volume": 1000.0})
    return pd.DataFrame(rows)


def _index(dates):
    out = []
    for tk in ("^NSEI", "^INDIAVIX"):
        for x in dates:
            out.append({"date": x, "ticker": tk, "open": 20000.0, "high": 20000.0,
                        "low": 20000.0, "close": 20000.0, "volume": 0.0})
    return pd.DataFrame(out)


def test_latest_bar_coverage_flags_stale_ticker():
    df = _prices(n_days=10, last_drop=5)   # T0 lags by 5 business days
    stale = check_latest_bar_coverage(df, max_stale_days=4)
    assert stale == ["T0"]
    # When everything is aligned, nothing is flagged.
    assert check_latest_bar_coverage(_prices(n_days=10), max_stale_days=4) == []


def test_run_index_gates_passes_and_catches_corruption():
    dates = pd.bdate_range("2024-01-01", periods=10)
    run_index_gates(_index(dates))   # clean → no raise
    bad = _index(dates).copy()
    bad.loc[0, "close"] = -1.0
    with pytest.raises(DataQualityError):
        run_index_gates(bad)


def test_summarize_data_quality_shape():
    df = _prices(n_days=10)
    df["spike_flag"] = 0
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    s = summarize_data_quality(df, _index(dates), stale_tickers=["T0"])
    assert s["n_tickers"] == 4 and s["n_stale_latest_tickers"] == 1
    assert s["index_lag_days"]["^NSEI"] == 0 and "^INDIAVIX" in s["index_lag_days"]


def test_residualize_fwd_ret_is_market_excess():
    df = _prices(n_days=12)
    # Make ticker prices drift so fwd_ret is non-zero.
    df["close"] = df["close"] + df.groupby("ticker").cumcount()
    df = forward_log_return(df, h=3)
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    idx = _index(dates)
    idx.loc[idx["ticker"] == "^NSEI", "close"] = np.linspace(20000, 21000, len(dates))
    out = residualize_fwd_ret(df, idx, h=3)
    ok = out.dropna(subset=["fwd_ret", "nifty_fwd_ret"])
    assert np.allclose(ok["fwd_ret_resid"], ok["fwd_ret"] - ok["nifty_fwd_ret"])


def test_bag_no_es_averages_seeds():
    pytest.importorskip("xgboost")
    from src.models.trainer import train_xgb_bag_no_es, predict_bag
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] + rng.normal(0, 0.3, 120))
    params = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1, "verbosity": 0}
    bag = train_xgb_bag_no_es(X, y, params, task="regression", n_seeds=3)
    assert len(bag) == 3
    preds = predict_bag(bag, X, task="regression")
    assert preds.shape == (120,)
    # n_seeds=1 reduces to a single fit (default-behaviour preservation).
    assert len(train_xgb_bag_no_es(X, y, params, task="regression", n_seeds=1)) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
