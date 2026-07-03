"""Tests for the panel-derived market-breadth (plan §Phase 2.13) and
sector-relative-strength (plan §Phase 2.12) features
(docs/small-capital-strategy-plan.md §2): presence in the built feature set,
market-level breadth left un-z-scored, per-sector demeaning correctness, and
graceful degradation when the sector map is missing."""
import numpy as np
import pandas as pd

from src.features.engineer import (
    build_features, _add_breadth_features, _add_sector_features,
)

BREADTH = ["breadth_sma50", "breadth_sma200", "breadth_pos_5d",
           "breadth_new_high_21d", "breadth_sma50_chg_5d"]
SECTOR = ["sector_ret_5d", "sector_ret_21d",
          "rel_sector_mom_5d", "rel_sector_mom_21d"]


def _panel(n_days=300, n_tickers=8, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
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


def _sector_map(n_tickers=8):
    return {f"T{t}": ("Banks" if t % 2 == 0 else "IT") for t in range(n_tickers)}


def test_breadth_features_present_market_level_and_bounded():
    df, feature_cols = build_features(_panel()[0], index_df=None, n_jobs=1)
    for c in BREADTH:
        assert c in feature_cols, f"{c} missing from feature_cols"

    tail = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
    # market-level: identical for every ticker on a date …
    assert (tail.groupby("date")["breadth_sma50"].nunique() == 1).all()
    # … and NOT z-scored away: still a genuine 0..1 fraction
    assert tail["breadth_sma50"].between(0, 1).all()
    assert tail["breadth_sma200"].between(0, 1).all()
    assert tail["breadth_new_high_21d"].between(0, 1).all()


def test_breadth_uses_only_same_date_cross_section():
    """Truncating the future must not change past breadth values (no look-ahead)."""
    panel, dates = _panel()
    full = _add_breadth_features(_seed_min_features(panel))
    cut = _add_breadth_features(_seed_min_features(panel[panel["date"] <= dates[200]]))
    d = dates[150]
    a = full[full["date"] == d]["breadth_sma50"].iloc[0]
    b = cut[cut["date"] == d]["breadth_sma50"].iloc[0]
    assert np.isclose(a, b, equal_nan=True)


def _seed_min_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Minimal per-ticker inputs breadth needs (dist_sma*, ret_5d, price_pos_21d)."""
    panel = panel.sort_values(["ticker", "date"]).copy()
    g = panel.groupby("ticker")["close"]
    for n in (50, 200):
        panel[f"dist_sma{n}"] = g.transform(lambda c: c / c.rolling(n).mean() - 1)
    panel["ret_5d"] = g.transform(lambda c: np.log(c / c.shift(5)))
    lo = g.transform(lambda c: c.rolling(21).min())
    hi = g.transform(lambda c: c.rolling(21).max())
    panel["price_pos_21d"] = (panel["close"] - lo) / (hi - lo + 1e-9)
    return panel


def test_sector_features_demean_within_sector_date():
    panel, _ = _panel()
    panel = panel.sort_values(["ticker", "date"]).copy()
    for n in (5, 21):
        panel[f"cum_ret_{n}d"] = panel.groupby("ticker")["close"].transform(
            lambda c: c / c.shift(n) - 1)
    out = _add_sector_features(panel, sector_map=_sector_map())
    for c in SECTOR:
        assert c in out.columns

    tail = out.dropna(subset=["rel_sector_mom_21d"])
    # relative momentum sums to ~0 within each (date, sector) group
    grp_mean = tail.groupby(["date", "sector"])["rel_sector_mom_21d"].mean()
    assert np.allclose(grp_mean.values, 0.0, atol=1e-12)
    # sector_ret is common to all members of a sector on a date
    assert (tail.groupby(["date", "sector"])["sector_ret_21d"].nunique() == 1).all()


def test_sector_features_survive_full_pipeline_and_unknown_tickers():
    df, feature_cols = build_features(_panel()[0], index_df=None, n_jobs=1)
    # synthetic T0..T7 are not in config/sector_map.json → all "UNKNOWN":
    # demeaning happens within that one bucket, features exist and are finite
    for c in SECTOR:
        assert c in feature_cols
    assert "sector" not in feature_cols          # the string column is meta
    tail = df[df["date"] >= df["date"].max() - pd.Timedelta(days=10)]
    assert np.isfinite(tail["rel_sector_mom_21d"]).all()


def test_sector_features_noop_without_map():
    panel, _ = _panel(n_days=60)
    panel["cum_ret_5d"] = 0.0
    panel["cum_ret_21d"] = 0.0
    out = _add_sector_features(panel, sector_map={})
    assert "rel_sector_mom_21d" not in out.columns   # graceful no-op
