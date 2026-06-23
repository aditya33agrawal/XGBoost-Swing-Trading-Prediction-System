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
# Point-in-time Nifty 200 universe (Nifty 100 + Nifty Midcap 100)
# ---------------------------------------------------------------------------
# Current composition (as of June 2026), widened from the original Nifty 50
# list per docs/model-improvement-plan.md Phase 1.7 — more cross-sectional
# breadth (bigger long basket → less idiosyncratic drawdown) and reach into
# less-efficient mid-caps where IC tends to be structurally higher.
#
# NOTE — survivorship bias is NOT solved by this widening alone (plan §A2):
# this is still today's constituents applied across all history. A real fix
# needs a nifty200_membership(symbol, start_date, end_date) table sourced
# from NSE semi-annual reconstitution circulars (plan Phase 1.6, tracked in
# config/universe.json). A few very recent IPOs below (e.g. GROWW, LENSKART,
# SWIGGY, VMM, TATACAP) will simply have short/NaN history pre-listing —
# the existing dropna(feature_cols) handling already deals with that safely.
UNIVERSE: list[str] = [
    "360ONE.NS", "ABB.NS", "ABCAPITAL.NS", "ADANIENSOL.NS", "ADANIENT.NS",
    "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", "ALKEM.NS", "AMBUJACEM.NS",
    "APLAPOLLO.NS", "APOLLOHOSP.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTRAL.NS",
    "ATGL.NS", "AUBANK.NS", "AUROPHARMA.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS", "BAJAJHLDNG.NS", "BAJFINANCE.NS", "BANKBARODA.NS", "BANKINDIA.NS",
    "BDL.NS", "BEL.NS", "BHARATFORG.NS", "BHARTIARTL.NS", "BHEL.NS",
    "BIOCON.NS", "BLUESTARCO.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS",
    "BSE.NS", "CANBK.NS", "CGPOWER.NS", "CHOLAFIN.NS", "CIPLA.NS",
    "COALINDIA.NS", "COCHINSHIP.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS",
    "COROMANDEL.NS", "CUMMINSIND.NS", "DABUR.NS", "DIVISLAB.NS", "DIXON.NS",
    "DLF.NS", "DMART.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS", "GAIL.NS", "GET&D.NS",
    "GLENMARK.NS", "GMRAIRPORT.NS", "GODFRYPHLP.NS", "GODREJCP.NS", "GODREJPROP.NS",
    "GRASIM.NS", "GROWW.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS",
    "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDPETRO.NS", "HINDUNILVR.NS", "HINDZINC.NS", "HUDCO.NS", "HYUNDAI.NS",
    "ICICIBANK.NS", "ICICIGI.NS", "IDEA.NS", "IDFCFIRSTB.NS", "INDHOTEL.NS",
    "INDIANB.NS", "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS",
    "IOC.NS", "IRCTC.NS", "IREDA.NS", "IRFC.NS", "ITC.NS",
    "JINDALSTEL.NS", "JIOFIN.NS", "JSWENERGY.NS", "JSWSTEEL.NS", "JUBLFOOD.NS",
    "KALYANKJIL.NS", "KEI.NS", "KOTAKBANK.NS", "KPITTECH.NS", "LAURUSLABS.NS",
    "LENSKART.NS", "LICHSGFIN.NS", "LODHA.NS", "LT.NS", "LTF.NS",
    "LTM.NS", "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MANKIND.NS",
    "MARICO.NS", "MARUTI.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "MCDOWELL-N.NS",
    "MCX.NS", "MFSL.NS", "MOTHERSON.NS", "MOTILALOFS.NS", "MPHASIS.NS",
    "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NESTLEIND.NS",
    "NHPC.NS", "NMDC.NS", "NTPC.NS", "NYKAA.NS", "OBEROIRLTY.NS",
    "OFSS.NS", "OIL.NS", "ONGC.NS", "PAGEIND.NS", "PATANJALI.NS",
    "PAYTM.NS", "PERSISTENT.NS", "PFC.NS", "PHOENIXLTD.NS", "PIDILITIND.NS",
    "PIIND.NS", "PNB.NS", "POLICYBZR.NS", "POLYCAB.NS", "POWERGRID.NS",
    "POWERINDIA.NS", "PREMIERENE.NS", "PRESTIGE.NS", "RADICO.NS", "RECLTD.NS",
    "RELIANCE.NS", "RVNL.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS",
    "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SIEMENS.NS", "SOLARINDS.NS",
    "SRF.NS", "SUNPHARMA.NS", "SUPREMEIND.NS", "SUZLON.NS", "SWIGGY.NS",
    "TATACAP.NS", "TATACOMM.NS", "TATACONSUM.NS", "TATAELXSI.NS", "TATAINVEST.NS",
    "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TIINDIA.NS",
    "TITAN.NS", "TMPV.NS", "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS",
    "ULTRACEMCO.NS", "UNIONBANK.NS", "UPL.NS", "VBL.NS", "VEDL.NS",
    "VMM.NS", "VOLTAS.NS", "WAAREEENER.NS", "WIPRO.NS", "YESBANK.NS",
    "ZYDUSLIFE.NS",
]


# ---------------------------------------------------------------------------
# Real data via yfinance
# ---------------------------------------------------------------------------
_STALE_LAG_DAYS = 5   # matches src/data/validation.py's check_freshness default


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

            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]

            # Identify tickers that came back with no data and retry them individually
            loaded = set(df["ticker"].unique())
            missing = [t for t in tickers if t not in loaded]
            if missing:
                individual = _retry_tickers_individually(missing, start, end)
                if individual is not None and not individual.empty:
                    df = pd.concat([df, individual], ignore_index=True)
                    df = df.drop_duplicates(subset=["ticker", "date"])

            print(
                f"[ingestion] yfinance: {df['ticker'].nunique()} tickers, "
                f"{df['date'].min().date()} → {df['date'].max().date()}"
            )
            if missing:
                still_missing = [t for t in tickers if t not in set(df["ticker"].unique())]
                if still_missing:
                    print(f"[ingestion] WARNING: could not load {still_missing} — excluded from universe")

            # Yahoo occasionally serves a stale cached snapshot for bulk
            # multi-ticker calls (observed on rate-limited Colab IPs) — no
            # exception, just silently truncated dates. Retry before
            # accepting it, since this is usually transient.
            lag_days = (pd.Timestamp(end) - df["date"].max()).days
            if lag_days > _STALE_LAG_DAYS and attempt < max_retries - 1:
                print(
                    f"[ingestion] WARNING: latest bar ({df['date'].max().date()}) is "
                    f"{lag_days}d behind requested end ({end}) — looks like a stale "
                    f"cached response, retrying (attempt {attempt + 1}/{max_retries})…"
                )
                time.sleep(backoff ** attempt)
                continue
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
            else:
                print(f"[ingestion] yfinance failed after {max_retries} attempts: {e}")
                return None
    return None


def _retry_tickers_individually(
    tickers: list[str], start: str, end: str
) -> Optional[pd.DataFrame]:
    """Try downloading failed tickers one-by-one with alternate symbol variants."""
    import yfinance as yf

    # Some NSE tickers have known yfinance symbol differences, or were
    # recently renamed/relisted and the exact symbol is unconfirmed.
    ALIASES: dict[str, list[str]] = {
        "M&M.NS": ["M&M.NS", "MM.NS"],
        "GET&D.NS": ["GET&D.NS", "GVT&D.NS"],
        "LTF.NS": ["LTF.NS", "L&TFH.NS"],
        "MCDOWELL-N.NS": ["MCDOWELL-N.NS", "UNITDSPR.NS"],
        # LTIMindtree renamed its trading symbol LTIM -> LTM on 2026-02-27
        # (rebranded to "LTM Limited"); LTIM.NS no longer resolves.
        "LTM.NS": ["LTM.NS", "LTIM.NS"],
        # Tata Capital's NSE/yfinance ticker is TATACAP, not TATACAPITAL.
        "TATACAP.NS": ["TATACAP.NS", "TATACAPITAL.NS"],
    }

    # First-attempt failures here are often transient rate-limiting on Yahoo's
    # side (the bulk multi-ticker call above hammers the API), not genuine
    # delistings — so retry each candidate symbol a few times with backoff
    # before moving to the next alias / giving up on the ticker.
    max_attempts = 3
    backoff = 2.0

    frames = []
    for ticker in tickers:
        candidates = ALIASES.get(ticker, [ticker])
        for symbol in candidates:
            sub = None
            for attempt in range(max_attempts):
                try:
                    raw = yf.download(symbol, start=start, end=end,
                                      progress=False, auto_adjust=True)
                    if raw.empty:
                        raise ValueError("empty response")
                    raw = raw.reset_index()
                    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                    candidate_sub = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
                    candidate_sub.columns = ["date", "open", "high", "low", "close", "volume"]
                    candidate_sub["ticker"] = ticker  # use original symbol for consistency
                    candidate_sub["date"] = pd.to_datetime(candidate_sub["date"]).dt.normalize()
                    candidate_sub = candidate_sub.dropna(subset=["close"])
                    candidate_sub = candidate_sub[candidate_sub["close"] > 0]
                    if not candidate_sub.empty:
                        sub = candidate_sub
                        break
                except Exception:
                    pass
                if attempt < max_attempts - 1:
                    time.sleep(backoff ** attempt)
            if sub is not None:
                frames.append(sub)
                break

    return pd.concat(frames, ignore_index=True) if frames else None


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
