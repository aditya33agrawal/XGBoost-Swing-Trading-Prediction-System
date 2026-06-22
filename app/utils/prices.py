"""Live/historical price fetching for the Streamlit app.

Used for:
  - unrealized P&L on open positions (Trade Desk)
  - resolving predictions against actual closes (Prediction Journal / Settings)
  - market-fill defaults when opening a manual trade
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_price(ticker: str) -> float | None:
    """Latest available close for `ticker`, or None on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("fetch_latest_price(%s) failed: %s", ticker, exc)
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Batch latest-close fetch. Returns {ticker: price} for tickers found."""
    out: dict[str, float] = {}
    if not tickers:
        return out
    try:
        import yfinance as yf
        data = yf.download(list(tickers), period="5d", group_by="ticker",
                            progress=False, threads=True)
        for t in tickers:
            try:
                col = data[t]["Close"] if len(tickers) > 1 else data["Close"]
                col = col.dropna()
                if not col.empty:
                    out[t] = float(col.iloc[-1])
            except Exception:
                continue
    except Exception as exc:
        logger.warning("fetch_latest_prices failed: %s", exc)
    return out


@st.cache_data(ttl=600, show_spinner=False)
def fetch_closes(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Historical closes for `tickers` between start/end (inclusive-ish).

    Returns DataFrame[ticker, date, close] — the exact shape
    src.tracking.outcome_tracker.resolve_outcomes expects as price_df.
    """
    if not tickers:
        return pd.DataFrame(columns=["ticker", "date", "close"])
    try:
        import yfinance as yf
        data = yf.download(list(tickers), start=start, end=end, group_by="ticker",
                            progress=False, threads=True)
        rows = []
        for t in tickers:
            try:
                sub = data[t] if len(tickers) > 1 else data
                sub = sub.dropna(subset=["Close"])
                for idx, r in sub.iterrows():
                    rows.append({"ticker": t, "date": idx, "close": float(r["Close"])})
            except Exception:
                continue
        return pd.DataFrame(rows)
    except Exception as exc:
        logger.warning("fetch_closes failed: %s", exc)
        return pd.DataFrame(columns=["ticker", "date", "close"])
