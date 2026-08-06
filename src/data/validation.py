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


def check_index_freshness(
    price_df: pd.DataFrame,
    index_df: pd.DataFrame,
    index_tickers: tuple[str, ...] = ("^NSEI", "^INDIAVIX"),
    max_lag_days: int = 4,
) -> None:
    """Warn when an index/VIX feed lags the stock feed on its most recent bar.

    This is the upstream cause of the 2026-06-29 zero-signal incident: yfinance
    returned stock prices through the latest date but `^INDIAVIX` (and sometimes
    `^NSEI`) lagged a trading day, so the shared regime columns were NaN for the
    whole latest cross-section and `predict_latest`'s dropna wiped every signal.
    The feature layer now forward-fills these columns (no look-ahead), so this is
    a WARNING, not a hard FAIL — but a persistent lag means the latest regime
    value is stale and should be investigated.
    """
    if index_df is None or index_df.empty:
        print("[validation] WARNING: no index data — regime/VIX features unavailable")
        return
    stock_latest = pd.to_datetime(price_df["date"]).max()
    for tk in index_tickers:
        sub = index_df[index_df["ticker"] == tk]
        if sub.empty:
            print(f"[validation] WARNING: index feed {tk} missing entirely")
            continue
        idx_latest = pd.to_datetime(sub["date"]).max()
        lag = (stock_latest - idx_latest).days
        if lag > max_lag_days:
            print(
                f"[validation] WARNING: {tk} latest bar {idx_latest.date()} lags "
                f"stock data {stock_latest.date()} by {lag} days — regime features "
                "will be forward-filled (last-known value) for recent dates"
            )


def check_latest_bar_coverage(
    price_df: pd.DataFrame,
    max_stale_days: int = 4,
) -> list[str]:
    """Warn for tickers whose most-recent bar lags the feed's global latest date.

    Backlog item §2b.7. The per-ticker ``dropna`` in ``predict_latest`` silently
    drops any name that has no row on the latest date, so one ticker stuck a few
    days behind just vanishes from today's scoring with no trace. Surface those
    names up front; return the stale list so the caller can record/act on it.
    """
    if price_df is None or price_df.empty:
        return []
    dates = pd.to_datetime(price_df["date"])
    global_latest = dates.max()
    last_per_ticker = price_df.assign(_d=dates).groupby("ticker")["_d"].max()
    stale = last_per_ticker[(global_latest - last_per_ticker).dt.days > max_stale_days]
    stale_tickers = sorted(stale.index.tolist())
    if stale_tickers:
        preview = ", ".join(
            f"{t}({(global_latest - last_per_ticker[t]).days}d)" for t in stale_tickers[:8]
        )
        print(
            f"[validation] WARNING: {len(stale_tickers)} ticker(s) lag the latest "
            f"bar {global_latest.date()} by > {max_stale_days}d and may drop out of "
            f"today's scoring: {preview}{' …' if len(stale_tickers) > 8 else ''}"
        )
    return stale_tickers


def run_index_gates(index_df: pd.DataFrame) -> None:
    """OHLCV sanity + date-gap checks on the index/VIX feed (backlog §2b.6).

    Previously only the stock prices were validated; the index feed (which drives
    every shared regime/VIX feature) was fetched and used unvalidated. Runs the
    same OHLCV/date-gap gates per index ticker. Raises ``DataQualityError`` on
    genuine corruption (non-positive prices); gaps are warnings only.
    """
    if index_df is None or index_df.empty:
        print("[validation] WARNING: no index data to validate")
        return
    for tk, grp in index_df.groupby("ticker"):
        check_ohlcv(grp)
        check_date_gaps(grp)
    print(f"[validation] index gates passed — {index_df['ticker'].nunique()} index series")


def summarize_data_quality(
    price_df: pd.DataFrame,
    index_df: pd.DataFrame,
    stale_tickers: list[str] | None = None,
    index_tickers: tuple[str, ...] = ("^NSEI", "^INDIAVIX"),
) -> dict:
    """Compact per-run data-quality snapshot (backlog §2b.9).

    Persisted into run metadata so feed degradation (index lag, spike creep,
    tickers dropping out) shows up as a trend rather than a one-off log line.
    """
    dates = pd.to_datetime(price_df["date"])
    stock_latest = dates.max()
    index_lag = {}
    if index_df is not None and not index_df.empty:
        for tk in index_tickers:
            sub = index_df[index_df["ticker"] == tk]
            index_lag[tk] = (
                int((stock_latest - pd.to_datetime(sub["date"]).max()).days)
                if not sub.empty else None
            )
    n_spikes = int(price_df["spike_flag"].sum()) if "spike_flag" in price_df.columns else None
    summary = {
        "n_rows": int(len(price_df)),
        "n_tickers": int(price_df["ticker"].nunique()),
        "date_min": str(dates.min().date()),
        "date_max": str(stock_latest.date()),
        "index_lag_days": index_lag,
        "n_spike_rows": n_spikes,
        "n_stale_latest_tickers": len(stale_tickers or []),
        "stale_latest_tickers": list(stale_tickers or [])[:20],
    }
    print(
        f"[validation] data-quality: {summary['n_rows']:,} rows | "
        f"{summary['n_tickers']} tickers | index_lag={index_lag} | "
        f"spikes={n_spikes} | stale_latest={summary['n_stale_latest_tickers']}"
    )
    return summary


def check_spike_filter(
    df: pd.DataFrame,
    col: str = "close",
    n_sigma: float = 10.0,
    report_path: str | None = None,
    persistence_days: int = 5,
) -> pd.DataFrame:
    """Flag (but don't drop) rows where log-return is > n_sigma from the mean.

    If `report_path` is given and any rows are flagged, dump ticker/date/z-score
    for every flagged row to that CSV (overwritten each run) so the recurring
    "N spike rows detected" warning can actually be triaged instead of only
    being counted — see §2.5 of the 2026-07-25 weekly-retrain analysis.

    Each row also gets an automated `likely_cause` classification (doc §2.6,
    2026-08-06 plan): a genuine unadjusted corporate action (split/bonus)
    shifts the price to a new *permanent* level, while a vendor data glitch
    (bad print, decimal error) typically reverts within a few sessions. We
    approximate this by comparing the log-return of the jump itself to the
    log-return measured `persistence_days` sessions later relative to the
    pre-jump price — if most of the jump persists, it's classified
    "likely_corporate_action"; if it reverts, "likely_data_error"; if there
    isn't enough post-jump history yet, "insufficient_history_to_classify".
    This doesn't replace a human check but turns the CSV into an actual
    triage list instead of raw ticker/date rows nobody has opened in 4 runs.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])
    close_by_ticker = df.groupby("ticker")[col]
    log_ret = close_by_ticker.transform(lambda s: np.log(s / s.shift(1)))
    mu, sigma = log_ret.mean(), log_ret.std()
    z = (log_ret - mu) / sigma
    spike_mask = z.abs() > n_sigma
    n_spikes = spike_mask.sum()
    if n_spikes > 0:
        print(
            f"[validation] WARNING: {n_spikes} spike rows detected "
            f"(|z| > {n_sigma}σ) — possible unadjusted corporate actions"
        )
        if report_path:
            from pathlib import Path

            # Persistence check: jump log-return vs. the log-return measured
            # from just-before-the-jump to `persistence_days` sessions later.
            # A ratio near 1 means the level shift stuck (real corporate
            # action); a ratio near 0 (or opposite sign) means it reverted
            # (data glitch).
            pre_jump_price = close_by_ticker.transform(lambda s: s.shift(1))
            post_jump_price = close_by_ticker.transform(lambda s: s.shift(-persistence_days))
            persisted_log_ret = np.log(post_jump_price / pre_jump_price)
            persistence_ratio = persisted_log_ret / log_ret

            def _classify(ratio: float) -> str:
                if pd.isna(ratio):
                    return "insufficient_history_to_classify"
                if ratio >= 0.7:
                    return "likely_corporate_action"
                if ratio <= 0.2:
                    return "likely_data_error"
                return "ambiguous"

            report = (
                df.loc[spike_mask, ["ticker", "date", col]]
                .assign(
                    z_score=z[spike_mask].values,
                    log_return=log_ret[spike_mask].values,
                    persistence_ratio=persistence_ratio[spike_mask].values,
                )
                .sort_values(["ticker", "date"])
            )
            report["likely_cause"] = report["persistence_ratio"].apply(_classify)
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            report.to_csv(report_path, index=False)
            n_glitch = (report["likely_cause"] == "likely_data_error").sum()
            n_corp = (report["likely_cause"] == "likely_corporate_action").sum()
            print(
                f"[validation] spike rows written → {report_path} (triage list) — "
                f"auto-classified: {n_corp} likely corporate actions, "
                f"{n_glitch} likely data errors, "
                f"{len(report) - n_corp - n_glitch} ambiguous/insufficient history"
            )
    df["spike_flag"] = spike_mask.astype(int)
    return df


def run_all_gates(
    df: pd.DataFrame,
    end: str | None = None,
    max_lag_days: int = 5,
    spike_report_path: str | None = None,
) -> pd.DataFrame:
    """Run all validation gates.  Returns df augmented with spike_flag.

    `end` is the pipeline's requested end-of-fetch date (cfg.end); pass it to
    enable the freshness gate. Omit only for callers (e.g. tests) that don't
    have a meaningful target date.
    """
    check_ohlcv(df)
    check_date_gaps(df)
    if end is not None:
        check_freshness(df, end, max_lag_days)
    df = check_spike_filter(df, report_path=spike_report_path)
    print(f"[validation] all gates passed — {len(df):,} rows, {df['ticker'].nunique()} tickers")
    return df
