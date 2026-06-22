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


@st.cache_data(ttl=15, show_spinner=False)
def fetch_latest_price(ticker: str) -> float | None:
    """Current market price (CMP) for `ticker`, or None on failure.

    Every fill must happen at the actual CMP, not a stale daily close, so
    this tries the live last-traded price first (fast_info, intraday) and
    only falls back to the last daily close if that's unavailable (e.g.
    market closed or ticker delisted) — short 15s TTL keeps it fresh for
    the "Take Trade" button.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        try:
            last = t.fast_info["last_price"]
            if last and last > 0:
                return float(last)
        except Exception:
            pass
        hist = t.history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("fetch_latest_price(%s) failed: %s", ticker, exc)
        return None


@st.cache_data(ttl=15, show_spinner=False)
def fetch_latest_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Batch CMP fetch. Returns {ticker: price} for tickers found.

    Same live-first / daily-close-fallback logic as fetch_latest_price,
    just batched.
    """
    out: dict[str, float] = {}
    if not tickers:
        return out
    try:
        import yfinance as yf
        for t in tickers:
            try:
                last = yf.Ticker(t).fast_info["last_price"]
                if last and last > 0:
                    out[t] = float(last)
            except Exception:
                continue
        missing = [t for t in tickers if t not in out]
        if missing:
            data = yf.download(missing, period="5d", group_by="ticker",
                                progress=False, threads=True)
            for t in missing:
                try:
                    col = data[t]["Close"] if len(missing) > 1 else data["Close"]
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
