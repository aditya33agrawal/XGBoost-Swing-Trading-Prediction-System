"""
Cross-sectional swing-trading research pipeline for Indian equities (NSE).
Price-only data. Walk-forward backtest + live prediction.

WHAT THIS IS
------------
A research harness, NOT a money printer. It is built so that if there is no
real edge, the backtest will honestly show ~0 Sharpe after costs. That honesty
is the point: it lets you tell a real signal from an overfit one.

PIPELINE
--------
  data -> features -> cross-sectional normalisation -> forward-return label
       -> purged walk-forward training (HistGradientBoosting)
       -> cross-sectional long/short backtest with Indian transaction costs
       -> latest-date predictions (the actual trade signals)

DATA
----
Uses yfinance if installed + network is available (NSE tickers via the .NS
suffix). Otherwise falls back to synthetic geometric-Brownian-motion data so
the mechanics can be verified anywhere. Synthetic data has NO predictable
structure, so a near-zero post-cost Sharpe on it is the correct sanity result.

IMPORTANT INDIA-SPECIFIC NOTE ON SHORTING
-----------------------------------------
You CANNOT short-sell stocks in the cash/delivery segment overnight in India.
A positional short requires single-stock FUTURES (available for ~180 names) or
must be closed same day (intraday). So:
  * mode="long_short"  -> assumes you express shorts via stock futures.
  * mode="long_only"   -> realistic if you only trade the cash segment.
Default is long_only to stay honest about what a retail cash account can do.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)


# ----------------------------------------------------------------------------
# 1. UNIVERSE  (liquid Nifty-50 large caps; yfinance NSE tickers use ".NS")
# ----------------------------------------------------------------------------
UNIVERSE = [
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


# ----------------------------------------------------------------------------
# 2. CONFIG
# ----------------------------------------------------------------------------
@dataclass
class Config:
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    horizon: int = 10                 # swing holding period in trading days
    rebalance_every: int = 10         # rebalance frequency (match horizon)
    train_min_days: int = 504         # ~2y minimum before first prediction
    embargo: int = 10                 # purge gap = horizon, prevents leakage
    n_quantile: int = 5               # quintiles: long top, short bottom
    mode: str = "long_only"           # "long_only" or "long_short"
    cost_bps_per_side: float = 20.0   # round-trip ~40bps incl. STT+slippage
    feature_cols: list = field(default_factory=list)


# ----------------------------------------------------------------------------
# 3. DATA LAYER  (real via yfinance, else synthetic fallback)
# ----------------------------------------------------------------------------
def load_prices(tickers, start, end) -> pd.DataFrame:
    """Return long-format frame: [date, ticker, close]."""
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        df = close.stack().rename("close").reset_index()
        df.columns = ["date", "ticker", "close"]
        print(f"[data] Loaded REAL data from yfinance: {df.ticker.nunique()} tickers.")
        return df.dropna()
    except Exception as e:
        print(f"[data] yfinance unavailable ({type(e).__name__}); using SYNTHETIC data.")
        return _synthetic_prices(tickers, start, end)


def _synthetic_prices(tickers, start, end) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    rows = []
    for t in tickers:
        mu, sigma = RNG.normal(0.0003, 0.0002), RNG.uniform(0.012, 0.022)
        rets = RNG.normal(mu, sigma, len(dates))
        price = 100 * np.exp(np.cumsum(rets))
        rows.append(pd.DataFrame({"date": dates, "ticker": t, "close": price}))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------
# 4. FEATURES  (price-only, time-series then cross-sectional normalisation)
# ----------------------------------------------------------------------------
def _rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker")["close"]

    for h in (5, 10, 21, 63):                      # momentum over multiple horizons
        df[f"ret_{h}"] = g.pct_change(h)
    df["vol_21"] = g.pct_change().groupby(df["ticker"]).rolling(21).std().reset_index(0, drop=True)
    df["rsi_14"] = g.apply(lambda s: _rsi(s)).reset_index(0, drop=True)
    for w in (20, 50):                             # distance from moving average
        sma = g.transform(lambda s: s.rolling(w).mean())
        df[f"dist_sma{w}"] = df["close"] / sma - 1.0

    feats = [c for c in df.columns if c.startswith(("ret_", "vol_", "rsi_", "dist_"))]

    # cross-sectional z-score within each date -> comparable across price scales
    def _zscore(block):
        return (block - block.mean()) / (block.std() + 1e-9)
    df[feats] = df.groupby("date")[feats].transform(_zscore)
    return df, feats


# ----------------------------------------------------------------------------
# 5. LABEL  (forward H-day return; triple-barrier provided as an alternative)
# ----------------------------------------------------------------------------
def add_forward_return(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    df["fwd_ret"] = df.groupby("ticker")["close"].transform(
        lambda s: s.shift(-horizon) / s - 1.0
    )
    return df


def triple_barrier_label(df, horizon, vol_mult=1.5):
    """OPTIONAL alternative label: +1/-1/0 by which volatility barrier hits first."""
    df = df.sort_values(["ticker", "date"]).copy()
    out = []
    for _, grp in df.groupby("ticker"):
        c = grp["close"].to_numpy()
        daily_vol = pd.Series(c).pct_change().rolling(21).std().to_numpy()
        lab = np.zeros(len(c))
        for i in range(len(c) - horizon):
            up, dn = c[i] * (1 + vol_mult * daily_vol[i]), c[i] * (1 - vol_mult * daily_vol[i])
            window = c[i + 1: i + 1 + horizon]
            hit_up = np.argmax(window >= up) if (window >= up).any() else 1e9
            hit_dn = np.argmax(window <= dn) if (window <= dn).any() else 1e9
            lab[i] = 1 if hit_up < hit_dn else (-1 if hit_dn < hit_up else 0)
        g = grp.copy(); g["tb_label"] = lab
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# 6. PURGED WALK-FORWARD  (train on the past, predict the future, embargo gap)
# ----------------------------------------------------------------------------
def walk_forward(df, feats, cfg: Config) -> pd.DataFrame:
    df = df.dropna(subset=feats + ["fwd_ret"]).sort_values("date")
    dates = np.sort(df["date"].unique())
    preds = []
    start_idx = cfg.train_min_days
    while start_idx < len(dates):
        rebal_date = dates[start_idx]
        # purge: only train on samples whose label is fully realised before rebal
        cutoff = dates[max(0, start_idx - cfg.embargo)]
        train = df[df["date"] < cutoff]
        test = df[df["date"] == rebal_date]
        if len(train) < 200 or test.empty:
            start_idx += cfg.rebalance_every
            continue
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.03, max_depth=4,
            l2_regularization=1.0, random_state=42,
        )
        model.fit(train[feats], train["fwd_ret"])
        t = test.copy()
        t["pred"] = model.predict(test[feats])
        preds.append(t[["date", "ticker", "pred", "fwd_ret", "close"]])
        start_idx += cfg.rebalance_every
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


# ----------------------------------------------------------------------------
# 7. BACKTEST  (cross-sectional quantile portfolio + Indian transaction costs)
# ----------------------------------------------------------------------------
def backtest(preds: pd.DataFrame, cfg: Config) -> dict:
    if preds.empty:
        return {"error": "no predictions"}
    rt_cost = 2 * cfg.cost_bps_per_side / 1e4     # round-trip fraction
    period_rets, prev_long, prev_short = [], set(), set()

    for date, day in preds.groupby("date"):
        day = day.dropna(subset=["pred", "fwd_ret"])
        if len(day) < cfg.n_quantile * 2:
            continue
        day["q"] = pd.qcut(day["pred"].rank(method="first"), cfg.n_quantile, labels=False)
        longs = day[day["q"] == cfg.n_quantile - 1]
        long_ret = longs["fwd_ret"].mean()
        long_set = set(longs["ticker"])

        if cfg.mode == "long_short":
            shorts = day[day["q"] == 0]
            gross = long_ret - shorts["fwd_ret"].mean()
            short_set = set(shorts["ticker"])
        else:                                      # long_only
            gross = long_ret
            short_set = set()

        # turnover-based cost: fraction of book that changed since last rebalance
        turn_l = 1 - len(long_set & prev_long) / max(len(long_set), 1)
        turn_s = 1 - len(short_set & prev_short) / max(len(short_set), 1) if short_set else 0
        cost = rt_cost * (turn_l + turn_s) / (2 if cfg.mode == "long_short" else 1)
        period_rets.append(gross - cost)
        prev_long, prev_short = long_set, short_set

    r = pd.Series(period_rets)
    if r.empty:
        return {"error": "no tradable periods"}
    periods_per_year = 252 / cfg.rebalance_every
    equity = (1 + r).cumprod()
    cagr = equity.iloc[-1] ** (periods_per_year / len(r)) - 1
    sharpe = (r.mean() / (r.std() + 1e-9)) * np.sqrt(periods_per_year)
    dd = (equity / equity.cummax() - 1).min()
    return {
        "periods": len(r), "CAGR": cagr, "Sharpe": sharpe,
        "max_drawdown": dd, "hit_rate": (r > 0).mean(),
        "avg_period_ret": r.mean(), "final_equity": equity.iloc[-1],
        "equity_curve": equity,
    }


# ----------------------------------------------------------------------------
# 8. LIVE PREDICTION  (retrain on everything, score the most recent date)
# ----------------------------------------------------------------------------
def predict_latest(df, feats, cfg: Config, top_n=10) -> pd.DataFrame:
    df = df.dropna(subset=feats)
    latest = df["date"].max()
    train = df.dropna(subset=["fwd_ret"])
    train = train[train["date"] < latest]
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.03, max_depth=4,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(train[feats], train["fwd_ret"])
    today = df[df["date"] == latest].copy()
    today["pred"] = model.predict(today[feats])
    today = today.sort_values("pred", ascending=False)
    return today[["ticker", "pred"]].head(top_n).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 9. RUN
# ----------------------------------------------------------------------------
def main():
    cfg = Config()
    print("=" * 70)
    print("CROSS-SECTIONAL SWING PIPELINE  |  Indian equities  |  price-only")
    print("=" * 70)

    raw = load_prices(UNIVERSE, cfg.start, cfg.end)
    df, feats = build_features(raw)
    cfg.feature_cols = feats
    df = add_forward_return(df, cfg.horizon)
    print(f"[features] {len(feats)} features: {feats}")

    preds = walk_forward(df, feats, cfg)
    print(f"[walk-forward] generated predictions for {preds['date'].nunique() if not preds.empty else 0} rebalance dates")

    stats = backtest(preds, cfg)
    print("\n--- BACKTEST (mode={}) ---".format(cfg.mode))
    for k, v in stats.items():
        if k == "equity_curve":
            continue
        print(f"  {k:>16}: {v:.4f}" if isinstance(v, float) else f"  {k:>16}: {v}")

    print("\n--- LATEST SIGNALS (top long candidates) ---")
    try:
        print(predict_latest(df, feats, cfg).to_string(index=False))
    except Exception as e:
        print(f"  (prediction skipped: {e})")

    print("\nNOTE: on synthetic data, ~0 Sharpe after costs is the CORRECT result.")
    print("Plug in real yfinance data on your machine to test for a genuine edge.")


if __name__ == "__main__":
    main()
