# Technical Implementation Plan — XGBoost Swing-Trading Prediction System for Nifty 50

**Scope:** End-to-end system that ingests free Indian-market data, engineers features, trains and continuously retrains XGBoost models, and emits swing-horizon (multi-day) predictions for Nifty 50 constituents.

**Author's note on expectations (read this first):** Equity returns at a daily-to-weekly horizon have a very low signal-to-noise ratio. A well-built model on this problem typically achieves a directional accuracy in the **52–56%** range, not 70%+. Anything reporting much higher in backtest is almost always leaking future information or ignoring transaction costs. The engineering goal is therefore not "high accuracy" but a *small, stable, costed edge that survives walk-forward validation*. This document is an engineering specification, not investment advice.

---

## 1. Problem Framing

### 1.1 What we predict (do NOT predict raw price)

Predicting the absolute price level `P(t+h)` is the most common beginner mistake. Price is non-stationary (unit root); a regression on it is dominated by the trivial "persistence" baseline `P(t+h) ≈ P(t)`, producing a tiny RMSE that masks zero predictive skill. We instead target **stationary** quantities:

Choose one of three target formulations (the system supports all three behind a config flag):

1. **Forward log-return (regression)** — primary
   `y_reg = ln(P_{t+h} / P_t)` where `h` = swing horizon (default 5 trading days).
2. **Directional sign (binary classification)** — most robust
   `y_clf = 1 if forward return > threshold else 0`, threshold set to cover round-trip cost (≈ 0.3–0.5%).
3. **Triple-barrier label (classification)** — best for swing trades (López de Prado)
   Place an upper barrier (take-profit), lower barrier (stop-loss), and a vertical barrier (max holding period). Label = which barrier is hit first {+1, −1, 0}. Barriers are scaled by realized volatility (e.g., ±2·ATR) so the labels are *risk-adjusted* and comparable across stocks and regimes.

**Recommendation:** Ship classification (triple-barrier) as the production signal and run the return-regression in parallel for position sizing. Classification with calibrated probabilities is far more reliable than a point price forecast.

### 1.2 Horizon and universe

- **Horizon `h`:** 5 trading days (≈ 1 week), configurable {3, 5, 10}.
- **Universe:** Nifty 50 constituents *as they existed on each historical date* (point-in-time — see §5.2 on survivorship bias). The index is reconstituted semi-annually; using today's 50 names over history is a classic look-ahead leak.
- **Rebalance / prediction cadence:** generate signals end-of-day after market close (~15:40 IST), for entry next session.

---

## 2. Target / Label Engineering Detail

```python
import numpy as np
import pandas as pd

def forward_log_return(close: pd.Series, h: int) -> pd.Series:
    return np.log(close.shift(-h) / close)

def triple_barrier_labels(close, high, low, atr, h,
                          up_mult=2.0, dn_mult=2.0):
    """Return label {+1,-1,0} = first barrier touched within h bars."""
    n = len(close)
    labels = np.zeros(n)
    for i in range(n - h):
        upper = close.iloc[i] + up_mult * atr.iloc[i]
        lower = close.iloc[i] - dn_mult * atr.iloc[i]
        window_high = high.iloc[i+1:i+1+h]
        window_low  = low.iloc[i+1:i+1+h]
        t_up = window_high[window_high >= upper].index.min()
        t_dn = window_low[window_low <= lower].index.min()
        if pd.isna(t_up) and pd.isna(t_dn):
            labels[i] = 0                       # vertical barrier
        elif pd.isna(t_dn) or (not pd.isna(t_up) and t_up <= t_dn):
            labels[i] = 1
        else:
            labels[i] = -1
    return pd.Series(labels, index=close.index)
```

**Critical leakage rule:** the target uses *future* bars (`shift(-h)`). Therefore the **last `h` rows of every symbol have no valid label and must be dropped from training**. Failing to do this trains on rows whose labels were silently filled — a subtle leak.

**Label overlap / sample weighting:** consecutive labels share overlapping future windows, so they are not IID. Down-weight overlapping samples by *label uniqueness* (concurrency-based weights) and apply *time-decay weighting* so recent regimes count more. Both feed XGBoost via `sample_weight`.

---

## 3. Data Sources (Free, Indian Market)

We rely on free EOD OHLCV + corporate actions + basic fundamentals. Layer multiple sources for redundancy and cross-validation.

| Source | Access | Use | Notes / caveats |
|---|---|---|---|
| **Yahoo Finance** via `yfinance` (`.NS` suffix, e.g. `RELIANCE.NS`) | Python lib | Primary EOD OHLCV, split/dividend-adjusted, fundamentals | Free, deep history, but unofficial — schema can break; rate-limit and retry. |
| **NSE official** via `nsepython` / `nsemine` / `pnsea` / `nsetools` | Python libs scraping nseindia.com | Index constituents, delivery %, FII/DII, corporate actions, bhavcopy | Requires NSE cookie handling; scraping is fragile — cache aggressively, respect throttling. |
| **NSE Bhavcopy** (daily EOD CSV) | Direct download | Authoritative daily OHLCV + delivery volumes | Best ground-truth for EOD; archive every day to your own store. |
| **Stooq / GitHub EOD dumps** | CSV | Backup / cross-check | Validate against bhavcopy. |
| **ICICI Breeze API** (optional, free) | Official REST/WebSocket | 3 yrs of historical incl. F&O, streaming OHLC | Official & stable if you have an ICICI Direct account; **note SEBI mandates a static IP for API trading from 1 Apr 2026.** |
| **Fundamentals** (P/E, P/B, ROE, EPS, debt) | yfinance / NSE / screener-style scrapes | Cross-sectional + quality features | Point-in-time fundamentals are hard for free — see §5.3. |

**Market-structure facts to encode (current as of 2026):**
- Equity cash settlement is **T+1 (mandatory)**; **optional T+0 same-day** exists for the top 500 stocks (all Nifty 50 qualify). Your backtest entry/exit timing and capital-availability model should assume T+1 by default.
- NSE hours 09:15–15:30 IST; pre-open 09:00–09:15. Daily price bands / circuit limits apply — model can't assume fills beyond the band.
- Securities Transaction Tax (STT), stamp duty, exchange txn charges, SEBI fees, and 18% GST on charges all hit every trade — captured in the cost model (§11).

---

## 4. Data Ingestion & Storage Architecture

### 4.1 Ingestion pipeline (incremental, idempotent)

```
[Scheduler: Airflow/Prefect, daily 17:00 IST]
      │
      ├─ fetch_bhavcopy()            # authoritative EOD
      ├─ fetch_yfinance_eod()        # adjusted OHLCV + actions
      ├─ fetch_nse_meta()            # constituents, delivery%, FII/DII
      ├─ fetch_fundamentals()        # weekly cadence
      │
      ▼
[Raw landing zone]  (immutable, partitioned by date)  →  Parquet on S3/local
      │
      ▼
[Validation & reconciliation]  (Great Expectations checks)
      │
      ▼
[Curated layer]  point-in-time, adjusted, deduped  →  DuckDB / Postgres / TimescaleDB
      │
      ▼
[Feature store]  computed features keyed by (symbol, date)
```

Design principles: **idempotent upserts** keyed on `(symbol, date)`; **immutable raw layer** so you can always recompute; **incremental** — only pull the latest missing dates, but keep the ability to do full backfills.

### 4.2 Storage choices

- **Raw / curated:** Parquet (columnar, compressed) partitioned by `year/month`. For a 50-stock universe, total data is small (low GBs even with 20 yrs of EOD) — DuckDB over Parquet is ideal: zero-ops, fast analytical queries, SQL.
- **Operational metadata** (model runs, predictions, fills): Postgres.
- **If you later add intraday/tick:** TimescaleDB or ClickHouse.

### 4.3 Schema (curated `prices` table)

```
symbol TEXT, date DATE,
open, high, low, close, vwap, volume,           -- raw
adj_open, adj_high, adj_low, adj_close,         -- corporate-action adjusted
delivery_qty, delivery_pct,
adj_factor,                                      -- cumulative split/bonus/div factor
in_nifty50 BOOLEAN,                              -- point-in-time membership
PRIMARY KEY (symbol, date)
```

---

## 5. Data Quality, Adjustments & Bias Control

### 5.1 Corporate actions (must-do)

Splits, bonuses, and dividends create artificial gaps. **Train on back-adjusted series** (`adj_close`). Store the raw and the cumulative `adj_factor` so you can reconstruct either. yfinance provides adjusted prices, but **cross-check against NSE corporate-action records** — adjustment errors are a top cause of phantom signals.

### 5.2 Survivorship bias (most important & most overlooked)

The Nifty 50 changes composition over time (e.g., names get added/dropped at semi-annual reviews). If you train on the *current* 50 names over 15 years of history, you've silently filtered to "companies that survived and stayed in the index" — a massive upward bias.

**Fix:** build a **point-in-time constituent table** `nifty50_membership(symbol, start_date, end_date)`. At every historical date, the universe = whoever was actually in the index then, including names later removed. Source historical constituent changes from NSE index circulars / archives and maintain this table manually + scraped updates.

### 5.3 Point-in-time fundamentals

Fundamentals (EPS, P/E) are reported with a lag and *restated*. Using today's reported value for a past date is look-ahead leakage. With free data this is hard; mitigations: lag every fundamental by a conservative reporting delay (e.g., 45–90 days after quarter end), never use a value before its plausible publication date, and prefer slowly-changing ratios. If point-in-time fundamentals can't be sourced cleanly, **keep the model technical-only** rather than introducing leaked fundamentals.

### 5.4 Validation gates (Great Expectations)

- No negative/zero prices; `high ≥ max(open,close) ≥ min(open,close) ≥ low`.
- No date gaps beyond known NSE holidays (maintain an exchange calendar).
- Volume/delivery non-negative; flag > N-sigma jumps for manual review (often unadjusted splits).
- Cross-source reconciliation: |yfinance close − bhavcopy close| within tolerance.
- Halt the pipeline (don't silently train) if a gate fails.

---

## 6. Feature Engineering

All features computed **strictly from data available at or before time `t`** (no future bars). Build a versioned feature catalog; each feature is a pure function `f(history_up_to_t) -> value`.

### 6.1 Feature families

**Price/return dynamics**
- Lagged log returns: 1, 2, 3, 5, 10, 21 days.
- Cumulative returns over 5/10/21/63 days (momentum).
- Distance from moving averages: `close/SMA_n − 1` for n ∈ {5,10,20,50,200}.
- Price position in recent range: `(close − min_n) / (max_n − min_n)`.

**Trend / oscillators (use `pandas-ta` or `ta-lib`)**
- RSI(14), Stochastic %K/%D, MACD (12/26/9) line+signal+hist, ADX(14), Aroon, CCI.

**Volatility**
- ATR(14), normalized ATR (`ATR/close`).
- Rolling realized vol (std of returns) 10/21/63d.
- Bollinger Band width and %B.
- Parkinson / Garman-Klass volatility (use H/L/O/C — more efficient estimators).

**Volume / liquidity / microstructure proxies**
- Volume z-score vs 20d mean; OBV; volume-price-trend.
- **Delivery % and its trend** (NSE-specific, genuinely informative — high delivery = conviction).
- VWAP deviation: `close/vwap − 1`.

**Cross-sectional / relative (the alpha most retail bots miss)**
- Stock return minus Nifty index return (relative strength).
- Beta to Nifty (rolling 63d), residual return.
- Cross-sectional rank of momentum / RSI across the 50 names on each date (rank-normalize per day — makes features regime-robust).
- Sector relative strength (group by NSE sector).

**Market regime / context**
- India VIX level and change.
- Nifty trend state (above/below 50/200 SMA), Nifty breadth (% of constituents above their 50-SMA).
- FII/DII net flows (daily) and 5d trend.
- Calendar: day-of-week, day-of-month, expiry-week flag, pre/post-holiday flag, earnings-window flag.

### 6.2 Engineering rules

- **Stationarize:** prefer returns, ratios, z-scores, and ranks over raw levels. (Optionally fractional differentiation to retain memory while gaining stationarity.)
- **Per-day cross-sectional normalization** for features compared across stocks.
- **No global scaling fit on the whole dataset** — fit scalers on the training fold only (leakage). For tree models scaling is unnecessary anyway, which is one reason XGBoost suits this.
- **Handle NaNs explicitly** (early history of rolling windows). XGBoost handles missing natively via default-direction splits — prefer leaving NaN over imputing fabricated values.
- **Feature catalog versioning:** every feature has an ID + definition hash; the feature set used to train each model is logged so predictions are reproducible.

```python
import pandas_ta as ta

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, v = df.adj_close, df.adj_high, df.adj_low, df.volume
    for n in (1,2,3,5,10,21):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    for n in (5,10,20,50,200):
        f[f"sma_dist_{n}"] = c / c.rolling(n).mean() - 1
    f["rsi_14"]   = ta.rsi(c, 14)
    f["atr_norm"] = ta.atr(h, l, c, 14) / c
    macd = ta.macd(c, 12, 26, 9)
    f = f.join(macd)
    f["bb_pctb"]  = ta.bbands(c, 20).iloc[:, -1]
    f["vol_z20"]  = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["deliv_pct"]= df.delivery_pct
    f["rvol_21"]  = np.log(c/c.shift(1)).rolling(21).std()
    return f
```

---

## 7. Validation Methodology (the part that makes or breaks it)

**Never use random K-fold or `train_test_split(shuffle=True)` on time series.** It leaks the future into the past. Use time-ordered validation.

### 7.1 Walk-forward (rolling/expanding) validation

```
|--- train ---|gap|-- test --|
        |--- train ---|gap|-- test --|
                |--- train ---|gap|-- test --|
```

- **Expanding window** (train grows) for the default; **rolling window** (fixed length) as an A/B to test concept-drift sensitivity.
- **Gap / embargo** between train and test = at least `h` bars (the label horizon) so no test label overlaps train features. This is **purging + embargo** (López de Prado). Without it, the `h`-day forward labels straddle the boundary and leak.

### 7.2 Purged, embargoed K-fold (for hyperparameter search)

Use `PurgedKFold`: remove (purge) training samples whose label window overlaps the validation window, and add an embargo period after each validation block. This gives multiple folds for tuning without leakage.

```python
class PurgedWalkForward:
    def __init__(self, n_splits, embargo, label_h):
        self.n_splits, self.embargo, self.h = n_splits, embargo, label_h
    def split(self, t_index):
        folds = np.array_split(np.arange(len(t_index)), self.n_splits + 1)
        for k in range(1, len(folds)):
            test = folds[k]
            train_end = test[0] - self.h - self.embargo   # purge + embargo
            train = np.arange(0, max(train_end, 0))
            yield train, test
```

### 7.3 Cross-sectional panel handling

We train **one global model across all 50 stocks** (pooled panel), not 50 separate models — far more data, better generalization, and cross-sectional features become meaningful. Splits are by **date**, not by row, so all stocks on a given date go to the same fold together (otherwise stock A's future leaks via stock B).

### 7.4 The honest baselines you must beat

Report skill *relative to*: (a) buy-and-hold Nifty, (b) naive persistence (predict last return's sign), (c) a coin-flip after costs. A model that beats RMSE but not these is worthless.

---

## 8. XGBoost Model Specification

### 8.1 Objective

- Classification (triple-barrier / direction): `binary:logistic` (or `multi:softprob` for 3-class), eval `aucpr` / `logloss`.
- Regression (forward return): `reg:squarederror`, or **`reg:quantileerror`** to predict return quantiles (gives you a distribution → confidence + sizing), or Huber-like via `reg:pseudohubererror` for outlier robustness.

### 8.2 Hyperparameters (starting grid)

| Param | Start | Search range | Role |
|---|---|---|---|
| `n_estimators` | 400 | 200–2000 (early stop) | with `early_stopping_rounds=50` on the walk-forward val |
| `learning_rate` (`eta`) | 0.02 | 0.005–0.1 | low + early stopping = best generalization on noisy data |
| `max_depth` | 4 | 3–7 | **shallow** — deep trees overfit financial noise |
| `min_child_weight` | 5 | 1–50 | high → regularize, critical here |
| `subsample` | 0.8 | 0.5–1.0 | row bagging |
| `colsample_bytree` | 0.7 | 0.4–1.0 | feature bagging vs correlated TA features |
| `gamma` | 0.5 | 0–5 | min split loss, prunes noise splits |
| `reg_lambda` (L2) | 2.0 | 0–10 | regularization |
| `reg_alpha` (L1) | 0.5 | 0–5 | sparsity on many correlated features |
| `scale_pos_weight` | ratio | — | class imbalance (or use sample weights) |
| `tree_method` | `hist` | — | fast; `gpu_hist` if GPU |

**Philosophy:** on low-SNR data, *regularize hard and keep trees shallow*. The default XGBoost settings (depth 6, eta 0.3) will memorize noise.

### 8.3 Hyperparameter optimization

- **Optuna** with the `PurgedWalkForward` splitter as the CV object; objective = mean out-of-fold financial metric (e.g., information coefficient or cost-adjusted Sharpe of the signal), **not** raw accuracy.
- Use TPE sampler + median pruner. Budget 100–300 trials. Log all to MLflow.
- Guard against tuning overfit: hold out a final **untouched walk-forward test period** that the optimizer never sees.

### 8.4 Sample weighting

Pass `sample_weight = uniqueness_weight * time_decay_weight`. For imbalanced triple-barrier classes, prefer sample weights over `scale_pos_weight` so weighting and class balance compose cleanly.

### 8.5 Probability calibration

Tree probabilities are miscalibrated. Fit **isotonic / Platt calibration on a held-out validation fold** (time-ordered, not the test fold). Calibrated probabilities are essential because position sizing and the entry threshold depend on `P(up)` being meaningful.

### 8.6 Interpretability

- **SHAP** values for global + per-prediction attribution (sanity-check that the model uses sensible features, surface leakage via "too good" features).
- Permutation importance on the validation fold (not train).
- Caveat: don't naively drop "low-importance" features iteratively on the same data — that's selection leakage. Decide the feature set via a separate, earlier period.

---

## 9. Evaluation Metrics

**ML metrics:** for classification — AUC-PR, log-loss, calibration error, Matthews CC, directional accuracy. For regression — **Information Coefficient** (Spearman corr between predicted and realized forward return) is the canonical alpha metric; also RMSE/MAE but only versus the persistence baseline.

**Financial metrics (decisive):** translate predictions to a paper strategy and measure, *after costs*:
- Annualized return, volatility, **Sharpe** and **Sortino**, **max drawdown**, Calmar.
- Hit rate, average win/loss, profit factor.
- Turnover and cost drag (so you see how much edge the broker eats).
- **Deflated Sharpe Ratio** — adjusts for the number of strategy configurations you tried (multiple-testing). With heavy tuning, an unadjusted Sharpe of 1.5 can be statistically indistinguishable from luck.

All metrics reported **per walk-forward window** and aggregated, never as a single in-sample number.

---

## 10. Backtesting Engine

### 10.1 Engine choice

- **Vectorized** backtest (`vectorbt` or custom pandas) for fast research sweeps.
- **Event-driven** backtest (`backtrader` / custom) for the final, realistic run — handles T+1 capital, partial fills, position limits, order timing.

### 10.2 Realistic execution model

- **Entry** at next-day open or VWAP (not same-bar close — that's look-ahead). **Exit** per the triple-barrier or horizon.
- **Slippage** model: e.g., spread/2 + impact ∝ (order size / ADV). Nifty 50 names are liquid, but still budget 5–15 bps.
- **Circuit limits:** no fills beyond the daily price band.
- **T+1 settlement:** funds from a sale are available next day — model capital availability accordingly.

### 10.3 Indian cost model (per round trip) — encode all of these

```python
def indian_costs(buy_val, sell_val, intraday=False):
    # Equity delivery (swing) rates — verify current rates before production
    brokerage = 0.0                       # many discount brokers: ₹0 for delivery
    stt       = 0.001  * (buy_val + sell_val)      # 0.1% buy + 0.1% sell (delivery)
    exch_txn  = 0.0000297 * (buy_val + sell_val)   # NSE txn charge (approx)
    sebi_fee  = 0.000001  * (buy_val + sell_val)
    stamp     = 0.00015   * buy_val                # 0.015% on buy
    gst       = 0.18 * (brokerage + exch_txn + sebi_fee)
    dp_charge = 13.0                               # approx per scrip on sell
    return brokerage + stt + exch_txn + sebi_fee + stamp + gst + dp_charge
```
> Treat these rates as *parameterized* and refresh them from current broker/exchange schedules — STT and stamp rates change with budgets/regulation. The structure matters: STT + GST + stamp alone is often ~0.12% per round trip, which sets the *minimum edge* your signal threshold must clear.

### 10.4 Robustness checks

- Walk-forward out-of-sample only; no peeking.
- Sensitivity to costs (±50%), slippage, and entry timing.
- Regime breakdown: bull/bear/sideways, high/low VIX, COVID-crash stress.
- Parameter stability: small changes in `h`/threshold shouldn't flip Sharpe sign.
- Monte Carlo / block bootstrap of the trade sequence for confidence intervals on Sharpe.

---

## 11. Continuous Retraining Pipeline

### 11.1 Cadence

- **Data ingestion:** daily after close.
- **Feature recompute:** daily (incremental).
- **Inference:** daily (generate next-day signals).
- **Retraining:** scheduled **weekly** (e.g., Sunday) on the expanding window, *plus* event-triggered retrains (see drift below). Weekly balances drift-adaptation against overfitting to the latest noise; do not retrain daily on a tiny new slice.

### 11.2 Retraining workflow (automated)

```
1. Refresh curated data + run validation gates (abort on failure).
2. Recompute features for the full window; recompute labels (drop last h rows).
3. Re-fit hyperparameters periodically (e.g., monthly); reuse last-known-good HPs weekly.
4. Train candidate model on data up to T-embargo via walk-forward.
5. Evaluate candidate on the most recent out-of-sample window.
6. Champion/Challenger gate: promote ONLY if candidate beats current champion
   on cost-adjusted Sharpe / IC by a margin AND passes calibration + drift checks.
7. Register model + feature-set hash + data snapshot in the registry; tag 'staging'.
8. Shadow-run (paper) for N sessions; promote to 'production' if stable.
9. Roll back automatically if live metrics breach guardrails.
```

### 11.3 Drift detection (event triggers for off-cycle retrain)

- **Feature/data drift:** Population Stability Index (PSI) / KS test on feature distributions vs training reference. PSI > 0.2 → investigate; > 0.25 → retrain.
- **Concept drift:** rolling live IC / hit-rate vs backtest expectation; CUSUM or page-hinkley on the prediction-error stream.
- **Target drift:** realized volatility regime shift (e.g., VIX spike) → trigger retrain + widen barriers.
- **Performance guardrails:** if live rolling Sharpe < threshold or drawdown > limit, **demote model to flat/no-trade** and alert.

### 11.4 Windowing strategy

Default **expanding window** (more data) with **time-decay sample weights** so old regimes fade. Maintain a **rolling-window challenger** in parallel; if markets undergo a structural break, the rolling model often wins and is auto-promoted.

---

## 12. MLOps & Infrastructure

| Concern | Tool (free/OSS) | Purpose |
|---|---|---|
| Orchestration | **Airflow** or **Prefect** | schedule ingestion, features, train, infer; retries, alerting |
| Experiment tracking | **MLflow** | params, metrics, artifacts, every run reproducible |
| Model registry | **MLflow Model Registry** | staging→production, versioning, rollback |
| Feature store | **Feast** (or DuckDB tables) | consistent features train↔serve, point-in-time joins |
| Data validation | **Great Expectations** / **pandera** | quality gates in pipeline |
| Config | **Hydra** / pydantic-settings | all hyperparams, horizons, thresholds in versioned config |
| Containerization | **Docker** | reproducible env; pin all lib versions |
| CI/CD | **GitHub Actions** | unit + leakage tests, backtest regression, deploy |
| Monitoring | **Prometheus + Grafana** / Evidently | drift, latency, live-vs-backtest dashboards |
| Storage | Parquet + **DuckDB** + Postgres | as in §4 |

**Reproducibility non-negotiables:** every production prediction is traceable to {model version, feature-set hash, code git SHA, data snapshot, calibration object}. Pin library versions (XGBoost, pandas, numpy) — XGBoost split behavior can change across major versions.

### 12.1 Leakage test suite (CI gate)

Automated tests that *fail the build* if: (a) any feature correlates suspiciously high with the target, (b) a future timestamp appears in any training row's feature inputs, (c) scalers/encoders are fit on full data, (d) the last-`h`-rows label drop is missing, (e) walk-forward folds have train dates ≥ test dates.

---

## 13. Serving / Prediction Output

Daily batch job after market close produces, per stock:
```json
{
  "date": "2026-06-16", "symbol": "RELIANCE.NS",
  "model_version": "v2026.06.14", "feature_hash": "a3f9...",
  "prob_up": 0.61, "calibrated": true,
  "expected_return_5d": 0.018, "return_q10": -0.012, "return_q90": 0.041,
  "signal": "LONG", "conviction": 0.61,
  "suggested_weight": 0.04, "barriers": {"tp": 1.02·ATR, "sl": ...}
}
```
Persist to Postgres; expose via FastAPI / a simple dashboard. **Include the confidence band** (from quantile regression) — a point forecast without uncertainty is misleading on this problem.

---

## 14. Risk Management & Position Sizing

Prediction ≠ trading. The signal layer must be wrapped in risk logic:
- **Sizing:** fractional Kelly (e.g., ¼-Kelly) or volatility-targeting (size ∝ 1/σ) so each position contributes equal risk; cap per-name weight (e.g., ≤ 8%) and gross exposure.
- **Stops:** ATR-based stop-loss = the triple-barrier lower barrier; hard max-holding = vertical barrier.
- **Portfolio limits:** max N concurrent positions, sector caps, correlation cap.
- **Circuit breaker:** global kill-switch on drawdown breach → go flat.
- **Long-only vs long-short:** short selling in Indian cash market is intraday-only; multi-day shorts need F&O (futures) — out of scope unless you extend the universe to stock futures.

---

## 15. Tech Stack Summary

`Python 3.11+` · `xgboost` · `pandas` / `numpy` / `polars` · `pandas-ta` or `ta-lib` · `scikit-learn` · `optuna` · `mlflow` · `feast` · `great-expectations` · `duckdb` + `parquet` · `postgres` · `vectorbt` / `backtrader` · `airflow`/`prefect` · `shap` · `evidently` · `fastapi` · `docker` · `yfinance` / `nsepython` / `nsemine`.

---

## 16. Implementation Roadmap

**Phase 0 — Foundations (wk 1–2):** repo, env, Docker, config, exchange calendar, point-in-time Nifty 50 membership table.

**Phase 1 — Data (wk 2–4):** ingestion (bhavcopy + yfinance + NSE meta), raw→curated, corporate-action adjustment, validation gates, survivorship-safe universe. *Exit criterion: reproducible, validated curated dataset.*

**Phase 2 — Features & labels (wk 4–6):** feature catalog + tests, triple-barrier labels, sample weights, feature store. *Exit: leakage test suite green.*

**Phase 3 — Modeling (wk 6–9):** PurgedWalkForward, XGBoost baseline, Optuna tuning to IC/Sharpe objective, calibration, SHAP. *Exit: model beats all baselines OOS after costs.*

**Phase 4 — Backtest (wk 9–11):** event-driven engine, full Indian cost model, robustness/regime/MC tests, deflated Sharpe. *Exit: stable positive cost-adjusted Sharpe across regimes, or honest decision to stop.*

**Phase 5 — MLOps (wk 11–14):** Airflow DAGs, MLflow + registry, champion/challenger, drift monitors, dashboards, CI leakage gates.

**Phase 6 — Paper trading (wk 14+):** shadow/paper run for 1–3 months; compare live vs backtest IC and Sharpe; only then consider any capital — small, with strict risk limits.

---

## 17. Top Pitfalls (the ones that silently kill these projects)

1. **Look-ahead leakage** via features, labels, scalers, or random CV — the #1 reason backtests lie. Mitigated by purged walk-forward + the CI leakage suite.
2. **Survivorship bias** from a static current-constituent universe.
3. **Predicting raw price** and being fooled by persistence-driven low RMSE.
4. **Ignoring/under-modeling costs** — a 55%-accurate signal with high turnover loses money after STT+slippage.
5. **Overfitting hyperparameters** to one history → use deflated Sharpe + an untouched final OOS period.
6. **Concept drift** — a model trained on a bull regime fails in a crash; drift monitors + retraining + a flat-go kill-switch.
7. **Miscalibrated probabilities** breaking position sizing.
8. **Corporate-action errors** creating phantom jumps the model "learns."
9. **Treating overlapping labels as IID** — needs uniqueness weighting.
10. **Expecting too much.** A real, durable edge here is small. Engineering excellence buys you the *chance* of a modest, costed edge — and the discipline to know when there isn't one.

---

*This is an engineering specification for a research/prediction system, not financial advice or a recommendation to trade real capital. Validate every rate, regulation (SEBI/NSE), and data-source behavior against current official sources before any live use, and treat paper-trading results as the real test.*
