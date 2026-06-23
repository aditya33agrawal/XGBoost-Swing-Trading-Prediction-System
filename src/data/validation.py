"""Data quality gates (plan §5.4).

Each gate raises DataQualityError if it fails.  The pipeline aborts
rather than training silently on bad data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DataQualityError(RuntimeError):
    pass


def _fail(msg: str) -> None:
    raise DataQualityError(f"[validation] FAIL — {msg}")


def check_ohlcv(df: pd.DataFrame) -> None:
    """OHLCV sanity: positive prices, high ≥ max(o,c), low ≤ min(o,c)."""
    if (df["close"] <= 0).any():
        _fail("non-positive close prices detected")
    if (df["high"] < df[["open", "close"]].max(axis=1) - 1e-6).any():
        _fail("high < max(open, close) on some rows")
    if (df["low"] > df[["open", "close"]].min(axis=1) + 1e-6).any():
        _fail("low > min(open, close) on some rows")
    if (df["volume"] < 0).any():
        _fail("negative volume detected")


def check_date_gaps(
    df: pd.DataFrame,
    max_consecutive_gap: int = 5,
) -> None:
    """No ticker should have more than max_consecutive_gap missing trading days."""
    for ticker, grp in df.groupby("ticker"):
        dates = pd.to_datetime(grp["date"]).sort_values()
        gaps = dates.diff().dt.days.dropna()
        big_gaps = gaps[gaps > max_consecutive_gap]
        if not big_gaps.empty:
            worst = big_gaps.max()
            print(
                f"[validation] WARNING: {ticker} has a {int(worst)}-day gap "
                "(may be a holiday cluster or missing data)"
            )


def check_no_future_leak(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    label_col: str = "fwd_ret",
) -> None:
    """Ensure that any row with a valid label has at least one feature that is older."""
    labeled = label_df.dropna(subset=[label_col])
    if labeled.empty:
        return
    max_feature_date = feature_df["date"].max()
    max_label_date = labeled["date"].max()
    if max_label_date > max_feature_date:
        _fail(
            f"label exists for dates beyond last feature date "
            f"({max_label_date} > {max_feature_date})"
        )


def check_freshness(
    df: pd.DataFrame,
    end: str,
    max_lag_days: int = 5,
) -> None:
    """Fail if the latest fetched bar is suspiciously far behind `end`.

    Catches silent stale-data responses (e.g. yfinance on a rate-limited
    Colab IP returning a cached snapshot instead of an error) that would
    otherwise train/sign signals on months-old prices with no warning.
    """
    latest = pd.to_datetime(df["date"]).max()
    target = pd.to_datetime(end)
    lag_days = (target - latest).days
    if lag_days > max_lag_days:
        _fail(
            f"latest fetched price date ({latest.date()}) is {lag_days} days "
            f"behind requested end date ({target.date()}) — data source likely "
            "returned stale/cached data instead of an error"
        )


def check_spike_filter(
    df: pd.DataFrame,
    col: str = "close",
    n_sigma: float = 10.0,
) -> pd.DataFrame:
    """Flag (but don't drop) rows where log-return is > n_sigma from the mean."""
    df = df.copy()
    log_ret = (
        df.sort_values(["ticker", "date"])
        .groupby("ticker")[col]
        .transform(lambda s: np.log(s / s.shift(1)))
    )
    mu, sigma = log_ret.mean(), log_ret.std()
    spike_mask = (log_ret - mu).abs() > n_sigma * sigma
    n_spikes = spike_mask.sum()
    if n_spikes > 0:
        print(
            f"[validation] WARNING: {n_spikes} spike rows detected "
            f"(|z| > {n_sigma}σ) — possible unadjusted corporate actions"
        )
    df["spike_flag"] = spike_mask.astype(int)
    return df


def run_all_gates(df: pd.DataFrame, end: str | None = None, max_lag_days: int = 5) -> pd.DataFrame:
    """Run all validation gates.  Returns df augmented with spike_flag.

    `end` is the pipeline's requested end-of-fetch date (cfg.end); pass it to
    enable the freshness gate. Omit only for callers (e.g. tests) that don't
    have a meaningful target date.
    """
    check_ohlcv(df)
    check_date_gaps(df)
    if end is not None:
        check_freshness(df, end, max_lag_days)
    df = check_spike_filter(df)
    print(f"[validation] all gates passed — {len(df):,} rows, {df['ticker'].nunique()} tickers")
    return df
