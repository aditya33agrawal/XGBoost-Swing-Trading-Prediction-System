"""Data ingestion layer.

Primary source: yfinance (.NS suffix for NSE stocks).
Provides OHLCV + corporate-action-adjusted prices.
Falls back to synthetic GBM data when network/yfinance is unavailable.

Usage:
    from src.data.ingestion import fetch_prices, UNIVERSE
    df = fetch_prices(UNIVERSE, "2015-01-01", "2024-12-31")
"""
from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Point-in-time Nifty 50 universe
# ---------------------------------------------------------------------------
# Current composition (as of June 2026).  For a production system, maintain
# a nifty50_membership(symbol, start_date, end_date) table to avoid
# survivorship bias (see plan §5.2).  This flat list is used as the starting
# universe; the storage layer checks membership per date.
UNIVERSE: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "GRASIM.NS", "HINDALCO.NS", "BAJAJFINSV.NS", "TECHM.NS",
    "INDUSINDBK.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BRITANNIA.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "APOLLOHOSP.NS",
    "TATACONSUM.NS", "BPCL.NS", "SBILIFE.NS", "HDFCLIFE.NS", "LTIM.NS",
    "SHRIRAMFIN.NS",
]


# ---------------------------------------------------------------------------
# Real data via yfinance
# ---------------------------------------------------------------------------
def _fetch_yfinance(
    tickers: list[str],
    start: str,
    end: str,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> Optional[pd.DataFrame]:
    """Download OHLCV from yfinance; returns None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,   # adjusts for splits/dividends
                threads=True,
            )
            if raw.empty:
                return None

            # yfinance ≥0.2 always returns MultiIndex (Price, Ticker) when
            # multiple tickers are passed.  Stack the Ticker level into a column.
            if isinstance(raw.columns, pd.MultiIndex):
                # Level 0 = price field ("Close" etc.), Level 1 = ticker symbol
                raw.columns.names = ["Price", "Ticker"]
                df = (
                    raw
                    .stack(level="Ticker", future_stack=True)
                    .reset_index()
                    .rename(columns={"Date": "date", "Ticker": "ticker",
                                     "Open": "open", "High": "high",
                                     "Low": "low", "Close": "close",
                                     "Volume": "volume"})
                )
                # Keep only needed columns (drop Dividends, Stock Splits if present)
                keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume"]
                        if c in df.columns]
                df = df[keep]
            else:
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df["ticker"] = tickers[0] if len(tickers) == 1 else "UNKNOWN"
                df = df.reset_index().rename(columns={"Date": "date"})

            df = df.rename(columns={"date": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]
            print(
                f"[ingestion] yfinance: {df['ticker'].nunique()} tickers, "
                f"{df['date'].min().date()} → {df['date'].max().date()}"
            )
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
            else:
                print(f"[ingestion] yfinance failed after {max_retries} attempts: {e}")
                return None
    return None


# ---------------------------------------------------------------------------
# Synthetic fallback (GBM, no predictive structure)
# ---------------------------------------------------------------------------
def _synthetic_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Geometric Brownian Motion prices — zero predictive signal by construction."""
    dates = pd.bdate_range(start, end)
    rows = []
    for ticker in tickers:
        mu = RNG.normal(0.0003, 0.0002)
        sigma = RNG.uniform(0.012, 0.025)
        rets = RNG.normal(mu, sigma, len(dates))
        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = close * np.exp(RNG.normal(0, 0.003, len(dates)))
        high = np.maximum(close, open_) * (1 + np.abs(RNG.normal(0, 0.005, len(dates))))
        low = np.minimum(close, open_) * (1 - np.abs(RNG.normal(0, 0.005, len(dates))))
        volume = RNG.integers(500_000, 5_000_000, len(dates)).astype(float)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    print(
        f"[ingestion] SYNTHETIC fallback: {df['ticker'].nunique()} tickers — "
        "expect ~0 Sharpe after costs (no real signal)"
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch OHLCV for `tickers` between `start` and `end`.

    Returns long-format DataFrame with columns:
        date, ticker, open, high, low, close, volume
    Prices are corporate-action-adjusted (yfinance auto_adjust=True).
    Falls back to synthetic data if yfinance is unavailable.
    """
    df = _fetch_yfinance(tickers, start, end)
    if df is None or df.empty:
        df = _synthetic_prices(tickers, start, end)

    # Ensure correct dtypes and sort
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def fetch_index_prices(start: str, end: str) -> pd.DataFrame:
    """Fetch Nifty 50 index (^NSEI) and India VIX (^INDIAVIX) for regime features."""
    index_tickers = ["^NSEI", "^INDIAVIX"]
    df = _fetch_yfinance(index_tickers, start, end)
    if df is None or df.empty:
        df = _synthetic_prices(index_tickers, start, end)
    return df
