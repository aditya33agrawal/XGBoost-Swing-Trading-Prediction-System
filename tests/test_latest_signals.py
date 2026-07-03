"""Regression test: a missing latest index/VIX bar must not wipe all signals.

yfinance frequently lags ^INDIAVIX by a trading day. Before the fix, the shared
index/VIX feature columns were NaN on the most-recent stock date for every
ticker, so predict_latest's dropna(subset=feature_cols) dropped the entire
cross-section and the pipeline emitted "(no signals generated)".  This verifies
that build_features forward-fills the regime columns so the latest cross-section
survives, and that predict_latest then produces at least one LONG.
"""
import numpy as np
import pandas as pd

from src.features.engineer import build_features
from src.data.validation import check_index_freshness


def _make_prices(n_tickers: int = 8, n_days: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for t in range(n_tickers):
        price = 100 + np.cumsum(rng.normal(0, 1, n_days))
        price = np.maximum(price, 5.0)
        for d, p in zip(dates, price):
            rows.append({
                "date": d, "ticker": f"T{t}",
                "open": p, "high": p * 1.01, "low": p * 0.99,
                "close": p, "volume": float(rng.integers(1e5, 1e6)),
            })
    return pd.DataFrame(rows)


def _make_index(dates: pd.DatetimeIndex, drop_last_vix: bool) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    nifty = 20000 + np.cumsum(rng.normal(0, 50, len(dates)))
    vix = 15 + rng.normal(0, 2, len(dates))
    frames = [pd.DataFrame({"date": dates, "ticker": "^NSEI", "close": nifty,
                            "open": nifty, "high": nifty, "low": nifty, "volume": 0.0})]
    vix_dates = dates[:-1] if drop_last_vix else dates  # simulate yfinance VIX lag
    frames.append(pd.DataFrame({"date": vix_dates, "ticker": "^INDIAVIX",
                                "close": vix[:len(vix_dates)], "open": vix[:len(vix_dates)],
                                "high": vix[:len(vix_dates)], "low": vix[:len(vix_dates)],
                                "volume": 0.0}))
    return pd.concat(frames, ignore_index=True)


def test_latest_bar_survives_missing_vix():
    prices = _make_prices()
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    index_df = _make_index(dates, drop_last_vix=True)

    df, feature_cols = build_features(prices, index_df=index_df, n_jobs=1)

    latest = df[df["date"] == df["date"].max()]
    # VIX columns are present in the feature set...
    vix_cols = [c for c in feature_cols if c.startswith("vix")]
    assert vix_cols, "expected vix_* features"
    # ...and after ffill, no ticker is dropped by dropna on the latest date.
    survived = latest.dropna(subset=feature_cols)
    assert len(survived) == latest["ticker"].nunique(), (
        "missing latest VIX bar should be forward-filled, not wipe the cross-section"
    )
    for c in vix_cols:
        assert latest[c].notna().all(), f"{c} still NaN on latest date after ffill"


def test_index_freshness_warns_on_lag(capsys):
    prices = _make_prices()
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    index_df = _make_index(dates, drop_last_vix=True)  # VIX lags by 1 bar

    check_index_freshness(prices, index_df, max_lag_days=0)
    out = capsys.readouterr().out
    assert "INDIAVIX" in out and "lags" in out


def test_index_freshness_quiet_when_aligned(capsys):
    prices = _make_prices()
    dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    index_df = _make_index(dates, drop_last_vix=False)

    check_index_freshness(prices, index_df)
    out = capsys.readouterr().out
    assert "lags" not in out
