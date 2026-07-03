"""Tests for the cross-sectional factor features added to make the model
better (docs/weekly-retrain-fixes-2026-06-29.md Tier-2 feature work):
12-1 / 6-1 momentum, 52-week-high proximity, idiosyncratic vol, risk-adjusted
momentum, return skew, MAX effect — plus the z-score winsorization robustifier.

The features are validated for (a) presence in the built feature set, (b)
finiteness after the warmup window, (c) no look-ahead through the full
build_features pipeline, and (d) the winsorization cap.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.engineer import build_features, _cs_zscore, _CS_ZSCORE_CLIP

NEW_FEATURES = [
    "mom_12_1", "mom_6_1", "pos_52w", "sharpe_mom_63d",
    "ret_skew_63d", "max_ret_21d", "idio_vol_63d",
]


def _panel(n_days=400, n_tickers=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    frames = []
    for t in range(n_tickers):
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n_days))
        high = close * (1 + np.abs(rng.normal(0, 0.006, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.006, n_days)))
        frames.append(pd.DataFrame({
            "ticker": f"T{t}", "date": dates,
            "open": close, "high": high, "low": low, "close": close,
            "volume": rng.integers(1e5, 1e6, n_days).astype(float),
        }))
    return pd.concat(frames, ignore_index=True), dates


def _index(dates, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    nse = 20000 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(dates)))
    vix = 15 + rng.normal(0, 1, len(dates)).cumsum() * 0.1
    for tk, series in (("^NSEI", nse), ("^INDIAVIX", np.abs(vix) + 5)):
        for d, x in zip(dates, series):
            out.append({"date": d, "ticker": tk, "open": x, "high": x,
                        "low": x, "close": x, "volume": 0.0})
    return pd.DataFrame(out)


def test_new_factor_features_present_and_finite():
    df, dates = _panel()
    out, cols = build_features(df, _index(dates), n_jobs=1)
    for c in NEW_FEATURES:
        assert c in cols, f"{c} missing from feature set"
    # After the 252-day warmup the factor columns should be populated and finite
    # (z-scored + winsorized ⇒ bounded) for essentially the whole cross-section.
    recent = out[out["date"] >= dates[300]]
    for c in NEW_FEATURES:
        finite_frac = np.isfinite(recent[c].to_numpy()).mean()
        assert finite_frac > 0.9, f"{c} mostly NaN after warmup ({finite_frac:.2f})"


def test_features_have_no_lookahead():
    """Perturbing only strictly-future bars must leave earlier-date features
    unchanged — z-scoring couples names within a date, never across dates, and
    every new feature is backward-looking (shift / trailing-window only)."""
    df, dates = _panel()
    idx = _index(dates)
    base, cols = build_features(df, idx, n_jobs=1)

    cutoff = dates[320]
    pert = df.copy()
    fut = pert["date"] > cutoff
    pert.loc[fut, ["open", "high", "low", "close"]] *= 3.0
    after, _ = build_features(pert, idx, n_jobs=1)

    early_date = dates[250]  # comfortably before the cutoff
    b = base[base["date"] <= early_date].set_index(["ticker", "date"])[cols]
    a = after[after["date"] <= early_date].set_index(["ticker", "date"])[cols]
    b, a = b.sort_index(), a.sort_index()
    # Compare only where the baseline is finite (NaN != NaN otherwise).
    mask = np.isfinite(b.to_numpy())
    assert np.allclose(b.to_numpy()[mask], a.to_numpy()[mask], atol=1e-9), \
        "an earlier-date feature moved when only future bars changed → look-ahead"


def test_cs_zscore_is_winsorized():
    # One name is a wild 100σ-style outlier on a date; the cap must bound it.
    n = 50
    df = pd.DataFrame({
        "date": pd.Timestamp("2020-01-01"),
        "ticker": [f"T{i}" for i in range(n)],
        "x": np.r_[np.random.default_rng(0).normal(0, 1, n - 1), 1e6],
    })
    out = _cs_zscore(df.copy(), ["x"])
    assert out["x"].abs().max() <= _CS_ZSCORE_CLIP + 1e-9
    assert out["x"].iloc[-1] == pytest.approx(_CS_ZSCORE_CLIP)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
