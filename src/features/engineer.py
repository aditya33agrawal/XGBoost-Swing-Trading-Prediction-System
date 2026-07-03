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
from joblib import Parallel, delayed

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

    # --- classic cross-sectional equity factors --------------------------
    # These are the most-replicated OHLCV-derivable predictors of the
    # cross-section of returns and were absent (the existing momentum tops
    # out at 63d). They are the biggest new-signal source that needs no
    # external data (India/fundamental feeds are the other, out of scope).
    #
    # 12-1 and 6-1 momentum (Jegadeesh-Titman / Fama-French UMD): cumulative
    # return over the past year/half-year, *skipping the most recent month* to
    # avoid the well-documented short-term reversal that contaminates raw
    # trailing momentum. Uses only shifted (past) prices — no look-ahead.
    f["mom_12_1"] = np.log(c.shift(21) / c.shift(252))
    f["mom_6_1"] = np.log(c.shift(21) / c.shift(126))

    # 52-week-high proximity (George & Hwang 2004): how close the price is to
    # its trailing 1-year high. A distinct anchor from the short-window
    # price_pos features; nearness-to-high predicts continuation.
    hi_252 = c.rolling(252).max()
    f["pos_52w"] = c / (hi_252 + 1e-9)

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

    # Amihud (2002) illiquidity: mean |return|/dollar_volume — proxy for price
    # impact per unit traded. Illiquid stocks earn a premium cross-sectionally.
    # Scale by 1e6 before log1p so raw values (≈1e-10) land in a useful range.
    daily_dollar_vol = c * v
    f["amihud_21d"] = np.log1p(
        (log_ret.abs() / (daily_dollar_vol + 1e-9)).rolling(21).mean() * 1e6
    )

    # Risk-adjusted momentum: the 63-day cumulative return per unit of its own
    # realised volatility. Raw momentum loads heavily on high-vol names; scaling
    # by rvol gives a cleaner, more stationary momentum signal (a Sharpe-like
    # trend measure). cum_ret_63d / rvol_63d are both already computed above.
    f["sharpe_mom_63d"] = f["cum_ret_63d"] / (f["rvol_63d"] + 1e-9)

    # Return skewness (63d): negatively-skewed names command a premium
    # (crash-risk aversion); a genuinely different signal from level/vol.
    f["ret_skew_63d"] = log_ret.rolling(63).skew()

    # MAX effect (Bali-Cakici-Whitelaw 2011): the largest single-day return over
    # the past month proxies lottery-like demand and predicts *lower* future
    # returns cross-sectionally — a documented negative predictor the trees can
    # exploit once it is cross-sectionally ranked downstream.
    f["max_ret_21d"] = log_ret.rolling(21).max()

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

    # --- deepened volume signal (plan §Phase 2.9) -------------------------
    # vol_z20/obv_z20/vwap_dev above are all snapshot z-scores against a
    # 20-day window. These add: (a) volume *trend* (building interest, the
    # way price momentum captures price trend — a snapshot z-score can't see
    # this); (b) dollar volume / turnover (also a relative-liquidity rank
    # feature once cross-sectionally z-scored downstream); (c) volume-price
    # divergence (the classic "weak breakout" tell — new high on light volume).
    for n in (5, 21):
        f[f"volume_roc_{n}d"] = np.log((v + 1.0) / (v.shift(n) + 1.0))

    dollar_vol_20d = (c * v).rolling(20).mean()
    f["dollar_vol_20d"] = np.log1p(dollar_vol_20d)

    f["ret_vol_corr_21d"] = log_ret.rolling(21).corr(f["vol_z20"])
    new_high_21d = (c >= c.rolling(21).max())
    below_avg_vol = v < v_mean
    f["weak_breakout_21d"] = (new_high_21d & below_avg_vol).astype(int)

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
# Winsorization cap for cross-sectional z-scores. On early/thin dates a feature
# can be near-constant across names, so the (std + 1e-9) denominator explodes a
# tiny numerator into a huge z; a few |z|>>5 values then dominate every tree
# split. Clipping at ±5σ removes those pathologies (and genuine fat-tail
# outliers) without distorting the bulk of the distribution — a standard
# robustifier that reduces overfitting to single-name/day extremes.
_CS_ZSCORE_CLIP = 5.0


def _cs_zscore(df: pd.DataFrame, cols: list[str], clip: float = _CS_ZSCORE_CLIP) -> pd.DataFrame:
    """Z-score within each date across all tickers — eliminates level bias.

    Winsorized at ±``clip`` σ (see `_CS_ZSCORE_CLIP`) so a near-constant feature
    on a thin date can't produce runaway z-scores that dominate the model.
    """
    def _zscore(block):
        mu = block.mean()
        sigma = block.std()
        z = (block - mu) / (sigma + 1e-9)
        return z.clip(lower=-clip, upper=clip)

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
    nifty["nifty_ret_1d"] = np.log(
        nifty["nifty_close"] / nifty["nifty_close"].shift(1)
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
    idx_feat = idx.drop(columns=["nifty_close"])
    idx_cols = [c for c in idx_feat.columns if c != "date"]
    df = df.merge(idx_feat, on="date", how="left")

    # Forward-fill the shared index/VIX columns per ticker. yfinance frequently
    # lags ^INDIAVIX (and occasionally ^NSEI) by a trading day, so the most-recent
    # stock bar can have no matching index row. Without this, those columns are
    # NaN on the latest date for EVERY ticker, and predict_latest's
    # dropna(subset=feature_cols) drops the entire cross-section → zero signals.
    # ffill only propagates the last-known regime value (no look-ahead).
    df = df.sort_values(["ticker", "date"])
    df[idx_cols] = df.groupby("ticker")[idx_cols].ffill()
    return df


# ---------------------------------------------------------------------------
# Relative-strength features (stock vs index)
# ---------------------------------------------------------------------------
def _add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Beta and idiosyncratic-alpha features vs Nifty at multiple horizons.

    Alpha = stock_ret - beta * index_ret removes the market component from
    momentum, exposing the stock-specific signal.  Requires nifty_ret_5d.
    """
    if "nifty_ret_5d" not in df.columns:
        return df

    df = df.sort_values(["ticker", "date"]).copy()
    stock_ret = df.groupby("ticker")["ret_5d"]

    # Rolling beta (63 days) via cov/var of 5d log-returns
    def _beta(grp):
        nifty = df.loc[grp.index, "nifty_ret_5d"]
        cov = grp.rolling(63).cov(nifty)
        var = nifty.rolling(63).var()
        return cov / (var + 1e-9)

    df["beta_63d"] = stock_ret.transform(lambda g: _beta(g))

    # Idiosyncratic momentum: residual after stripping market beta
    df["alpha_5d"] = df["ret_5d"] - df["beta_63d"] * df["nifty_ret_5d"]
    if "nifty_ret_1d" in df.columns and "ret_1d" in df.columns:
        df["alpha_1d"] = df["ret_1d"] - df["beta_63d"] * df["nifty_ret_1d"]
        # Idiosyncratic volatility (Ang-Hodrick-Xing-Zhang 2006): the vol of the
        # market-residual daily return. The low-idio-vol anomaly is one of the
        # most robust cross-sectional effects; it needs the market residual, so
        # it lives here (not the per-ticker builder). Computed per ticker over a
        # 63-day trailing window — uses only past residuals, no look-ahead.
        df["idio_vol_63d"] = (
            df.groupby("ticker")["alpha_1d"]
            .transform(lambda s: s.rolling(63).std())
        )
    if "nifty_ret_21d" in df.columns and "ret_21d" in df.columns:
        df["alpha_21d"] = df["ret_21d"] - df["beta_63d"] * df["nifty_ret_21d"]
    return df


# ---------------------------------------------------------------------------
# Market breadth (plan §Phase 2.13) — computed from the panel itself, no
# external data. Same value for every ticker on a date (market context), so
# NOT cross-sectionally z-scored (mirrors the nifty_*/vix* treatment).
# Uses only same-date cross-sections of already-lagged per-ticker features —
# available at time t, no look-ahead.
# ---------------------------------------------------------------------------
def _add_breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date")
    breadth = pd.DataFrame({
        # % of universe above its own 50/200-SMA — the classic breadth gauges.
        # Breadth divergence (index up, breadth down) is a pre-drawdown tell
        # the single nifty_dist_sma200 distance cannot see.
        "breadth_sma50":  g["dist_sma50"].apply(lambda s: (s > 0).mean()),
        "breadth_sma200": g["dist_sma200"].apply(lambda s: (s > 0).mean()),
        # % with positive 5d return — short-horizon participation
        "breadth_pos_5d": g["ret_5d"].apply(lambda s: (s > 0).mean()),
        # % near their 21d high — thrust/exhaustion gauge
        "breadth_new_high_21d": g["price_pos_21d"].apply(lambda s: (s > 0.95).mean()),
    }).sort_index()
    # Breadth *momentum*: 5-session change — is participation building or rolling over?
    breadth["breadth_sma50_chg_5d"] = breadth["breadth_sma50"].diff(5)
    breadth = breadth.reset_index()
    return df.merge(breadth, on="date", how="left")


# ---------------------------------------------------------------------------
# Sector relative strength (plan §Phase 2.12) — needs only the static
# config/sector_map.json (ticker → sector), no external feed. Isolates
# stock-specific alpha from sector-wide moves, which the Nifty-only
# beta_63d/alpha_5d features cannot separate.
# ---------------------------------------------------------------------------
def _load_sector_map() -> dict[str, str]:
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "config" / "sector_map.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _add_sector_features(df: pd.DataFrame, sector_map: dict[str, str] | None = None) -> pd.DataFrame:
    sector_map = sector_map if sector_map is not None else _load_sector_map()
    if not sector_map:
        return df
    df = df.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("UNKNOWN")

    for n in (5, 21):
        col = f"cum_ret_{n}d"
        if col not in df.columns:
            continue
        sector_mean = df.groupby(["date", "sector"])[col].transform("mean")
        # Sector momentum (same for every member on a date). Once cross-
        # sectionally z-scored downstream it becomes a per-date sector-rotation
        # rank — which sector is leading right now.
        df[f"sector_ret_{n}d"] = sector_mean
        # Stock return net of its sector — pure within-sector stock picking.
        df[f"rel_sector_mom_{n}d"] = df[col] - sector_mean
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute all features; return (df_with_features, feature_col_names).

    Parameters
    ----------
    df : long-format OHLCV frame — columns: date, ticker, open, high, low, close, volume
    index_df : optional index frame (^NSEI, ^INDIAVIX) for regime features
    n_jobs : workers for the per-ticker feature loop (-1 = all cores). Each
        ticker's time series is independent, so this parallelizes cleanly.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    # Per-ticker time-series features (embarrassingly parallel across tickers)
    groups = [grp for _, grp in df.groupby("ticker", sort=False)]
    feat_frames = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_features_for_ticker)(grp) for grp in groups
    )

    feat_df = pd.concat(feat_frames).reindex(df.index)
    df = pd.concat([df, feat_df], axis=1)

    # Regime features from index
    if index_df is not None and not index_df.empty:
        df = _add_regime_features(df, index_df)
        df = _add_relative_features(df)

    # Market breadth (plan §2.13) + sector relative strength (plan §2.12) —
    # both derived from the panel itself (plus the static sector map), so they
    # are available in every run with no external data dependency.
    df = _add_breadth_features(df)
    df = _add_sector_features(df)

    # Identify feature columns (everything that isn't OHLCV/meta)
    meta = {"date", "ticker", "open", "high", "low", "close", "volume", "spike_flag", "sector"}
    feature_cols = [c for c in df.columns if c not in meta]

    # Cross-sectional z-score (applied after joining index features)
    # Use only non-calendar features for z-scoring. breadth_* are market-level
    # (identical for all tickers on a date) — z-scoring them would zero them out.
    cs_cols = [
        c for c in feature_cols
        if c not in ("day_of_week", "day_of_month", "month", "is_expiry_week")
        and not c.startswith("vix")
        and not c.startswith("nifty")
        and not c.startswith("breadth_")
    ]
    df = _cs_zscore(df, cs_cols)

    # Cross-sectional rank columns for key features
    rank_base = ["ret_5d", "cum_ret_21d", "rsi_14", "atr_norm", "vol_z20",
                 "dollar_vol_20d", "mom_12_1", "pos_52w"]
    rank_base = [c for c in rank_base if c in df.columns]
    df = _cs_rank(df, rank_base)
    feature_cols = [c for c in df.columns if c not in meta]

    print(f"[features] {len(feature_cols)} features built for {df['ticker'].nunique()} tickers")
    return df, feature_cols
