"""Feature engineering catalog (plan §6).

All features are computed strictly from data available at or before time t.
Features are per-stock time-series, then cross-sectionally z-scored within
each date so they are comparable across names and regime-robust.

Entry point:
    build_features(df, index_df=None) -> (df_with_features, feature_cols)

`df` must have columns: date, ticker, open, high, low, close, volume
`index_df` must have columns: date, ticker, close  (^NSEI and ^INDIAVIX)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    print("[features] pandas-ta not installed; using fallback implementations")


# ---------------------------------------------------------------------------
# Fallback implementations (used when pandas_ta is unavailable)
# ---------------------------------------------------------------------------
def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-9))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _macd(s: pd.Series, fast=12, slow=26, sig=9) -> pd.DataFrame:
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


def _bbands(s: pd.Series, n: int = 20) -> pd.DataFrame:
    mid = s.rolling(n).mean()
    std = s.rolling(n).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    pctb = (s - lower) / (upper - lower + 1e-9)
    width = (upper - lower) / (mid + 1e-9)
    return pd.DataFrame({"bb_pctb": pctb, "bb_width": width})


def _stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3
) -> pd.DataFrame:
    lo_n = low.rolling(k).min()
    hi_n = high.rolling(k).max()
    k_pct = 100 * (close - lo_n) / (hi_n - lo_n + 1e-9)
    d_pct = k_pct.rolling(d).mean()
    return pd.DataFrame({"stoch_k": k_pct, "stoch_d": d_pct})


# ---------------------------------------------------------------------------
# Per-ticker feature builder
# ---------------------------------------------------------------------------
def _features_for_ticker(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute all time-series features for a single ticker group."""
    grp = grp.sort_values("date").copy()
    c = grp["close"]
    h = grp["high"]
    l = grp["low"]
    v = grp["volume"]
    f = pd.DataFrame(index=grp.index)

    # --- price/return dynamics -------------------------------------------
    for n in (1, 2, 3, 5, 10, 21):
        f[f"ret_{n}d"] = np.log(c / c.shift(n))

    for n in (5, 10, 21, 63):
        f[f"cum_ret_{n}d"] = c / c.shift(n) - 1.0

    for n in (5, 10, 20, 50, 200):
        sma = c.rolling(n).mean()
        f[f"dist_sma{n}"] = c / (sma + 1e-9) - 1.0

    for n in (10, 21, 63):
        lo = c.rolling(n).min()
        hi = c.rolling(n).max()
        f[f"price_pos_{n}d"] = (c - lo) / (hi - lo + 1e-9)

    # --- trend/oscillators -----------------------------------------------
    if _HAS_TA:
        f["rsi_14"] = ta.rsi(c, length=14)
        stoch = ta.stoch(h, l, c)
        if stoch is not None and not stoch.empty:
            f["stoch_k"] = stoch.iloc[:, 0]
            f["stoch_d"] = stoch.iloc[:, 1]
        macd_df = ta.macd(c, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            f["macd"] = macd_df.iloc[:, 0]
            f["macd_signal"] = macd_df.iloc[:, 1]
            f["macd_hist"] = macd_df.iloc[:, 2]
        bb = ta.bbands(c, length=20)
        if bb is not None and not bb.empty:
            f["bb_pctb"] = bb.iloc[:, -1]
            f["bb_width"] = bb.iloc[:, 3] if bb.shape[1] > 3 else np.nan
        adx = ta.adx(h, l, c, length=14)
        if adx is not None and not adx.empty:
            f["adx"] = adx.iloc[:, 0]
    else:
        f["rsi_14"] = _rsi(c)
        stoch = _stoch(h, l, c)
        f[["stoch_k", "stoch_d"]] = stoch
        macd_df = _macd(c)
        f[["macd", "macd_signal", "macd_hist"]] = macd_df
        bb = _bbands(c)
        f[["bb_pctb", "bb_width"]] = bb

    # --- volatility -------------------------------------------------------
    atr = _atr(h, l, c, 14) if not _HAS_TA else (
        ta.atr(h, l, c, length=14) if _HAS_TA else _atr(h, l, c)
    )
    f["atr_norm"] = atr / (c + 1e-9)

    log_ret = np.log(c / c.shift(1))
    for n in (10, 21, 63):
        f[f"rvol_{n}d"] = log_ret.rolling(n).std()

    # Parkinson high-low volatility estimator (more efficient than close-to-close)
    f["park_vol_21"] = (
        (np.log(h / l) ** 2 / (4 * np.log(2))).rolling(21).mean() ** 0.5
    )

    # --- volume/microstructure ------------------------------------------
    v_mean = v.rolling(20).mean()
    v_std = v.rolling(20).std()
    f["vol_z20"] = (v - v_mean) / (v_std + 1e-9)

    obv = (np.sign(c.diff()) * v).cumsum()
    f["obv_z20"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)

    # VWAP deviation — use (open+high+low+close)/4 as intraday proxy when no VWAP
    typical = (grp["open"] + h + l + c) / 4
    vwap_proxy = (typical * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
    f["vwap_dev"] = c / (vwap_proxy + 1e-9) - 1.0

    # --- calendar features ----------------------------------------------
    dates = pd.to_datetime(grp["date"])
    f["day_of_week"] = dates.dt.dayofweek.values
    f["day_of_month"] = dates.dt.day.values
    f["month"] = dates.dt.month.values
    # Expiry week: NSE monthly F&O expiry is the last Thursday of each month
    f["is_expiry_week"] = (
        (dates.dt.dayofweek == 3) & (dates.dt.day >= 24)
    ).astype(int).values

    return f


# ---------------------------------------------------------------------------
# Cross-sectional z-score
# ---------------------------------------------------------------------------
def _cs_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score within each date across all tickers — eliminates level bias."""
    def _zscore(block):
        mu = block.mean()
        sigma = block.std()
        return (block - mu) / (sigma + 1e-9)

    df[cols] = df.groupby("date")[cols].transform(_zscore)
    return df


# ---------------------------------------------------------------------------
# Cross-sectional rank (0–1) — regime-robust alternative to z-score
# ---------------------------------------------------------------------------
def _cs_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    def _rank(block):
        return block.rank(pct=True)
    ranked = df.groupby("date")[cols].transform(_rank)
    ranked.columns = [f"{c}_rank" for c in cols]
    return pd.concat([df, ranked], axis=1)


# ---------------------------------------------------------------------------
# Market-regime / index features (attached per date, same for all tickers)
# ---------------------------------------------------------------------------
def _add_regime_features(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Nifty index + VIX features onto the stock frame."""
    nifty = index_df[index_df["ticker"] == "^NSEI"][["date", "close"]].copy()
    nifty = nifty.rename(columns={"close": "nifty_close"}).sort_values("date")
    vix = index_df[index_df["ticker"] == "^INDIAVIX"][["date", "close"]].copy()
    vix = vix.rename(columns={"close": "vix"}).sort_values("date")

    # Nifty trend features
    for n in (50, 200):
        nifty[f"nifty_dist_sma{n}"] = (
            nifty["nifty_close"] / nifty["nifty_close"].rolling(n).mean() - 1.0
        )
    nifty["nifty_ret_5d"] = np.log(
        nifty["nifty_close"] / nifty["nifty_close"].shift(5)
    )
    nifty["nifty_ret_21d"] = np.log(
        nifty["nifty_close"] / nifty["nifty_close"].shift(21)
    )

    vix["vix_change_5d"] = vix["vix"].pct_change(5)
    vix["vix_z20"] = (vix["vix"] - vix["vix"].rolling(20).mean()) / (
        vix["vix"].rolling(20).std() + 1e-9
    )

    idx = nifty.merge(vix, on="date", how="outer").sort_values("date")
    df = df.merge(idx.drop(columns=["nifty_close"]), on="date", how="left")
    return df


# ---------------------------------------------------------------------------
# Relative-strength features (stock vs index)
# ---------------------------------------------------------------------------
def _add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Beta and residual return vs Nifty.  Requires nifty_ret_5d already in df."""
    if "nifty_ret_5d" not in df.columns:
        return df

    df = df.sort_values(["ticker", "date"]).copy()
    stock_ret = df.groupby("ticker")["ret_5d"]

    # Rolling beta (63 days) via cov/var
    def _beta(grp):
        nifty = df.loc[grp.index, "nifty_ret_5d"]
        cov = grp.rolling(63).cov(nifty)
        var = nifty.rolling(63).var()
        return cov / (var + 1e-9)

    df["beta_63d"] = stock_ret.transform(lambda g: _beta(g))
    df["alpha_5d"] = df["ret_5d"] - df["beta_63d"] * df["nifty_ret_5d"]
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute all features; return (df_with_features, feature_col_names).

    Parameters
    ----------
    df : long-format OHLCV frame — columns: date, ticker, open, high, low, close, volume
    index_df : optional index frame (^NSEI, ^INDIAVIX) for regime features
    """
    df = df.sort_values(["ticker", "date"]).copy()

    # Per-ticker time-series features
    feat_frames = []
    for ticker, grp in df.groupby("ticker", sort=False):
        feat = _features_for_ticker(grp)
        feat_frames.append(feat)

    feat_df = pd.concat(feat_frames).reindex(df.index)
    df = pd.concat([df, feat_df], axis=1)

    # Regime features from index
    if index_df is not None and not index_df.empty:
        df = _add_regime_features(df, index_df)
        df = _add_relative_features(df)

    # Identify feature columns (everything that isn't OHLCV/meta)
    meta = {"date", "ticker", "open", "high", "low", "close", "volume", "spike_flag"}
    feature_cols = [c for c in df.columns if c not in meta]

    # Cross-sectional z-score (applied after joining index features)
    # Use only non-calendar features for z-scoring
    cs_cols = [
        c for c in feature_cols
        if c not in ("day_of_week", "day_of_month", "month", "is_expiry_week")
        and not c.startswith("vix")
        and not c.startswith("nifty")
    ]
    df = _cs_zscore(df, cs_cols)

    # Cross-sectional rank columns for key features
    rank_base = ["ret_5d", "cum_ret_21d", "rsi_14", "atr_norm", "vol_z20"]
    rank_base = [c for c in rank_base if c in df.columns]
    df = _cs_rank(df, rank_base)
    feature_cols = [c for c in df.columns if c not in meta]

    print(f"[features] {len(feature_cols)} features built for {df['ticker'].nunique()} tickers")
    return df, feature_cols
