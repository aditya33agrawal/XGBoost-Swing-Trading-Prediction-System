# XGBoost Model Architecture

**Audience:** AI/ML engineers who want to understand the system end-to-end and reason about what to change and why.

---

## What This System Does

This is a **swing-trading signal generator** for NSE-listed equities (Nifty 50 + Nifty 200 universe). Every rebalance day (~every 5 trading days) it:

1. Fetches OHLCV prices for ~200 Indian stocks + Nifty index + India VIX
2. Engineers ~60+ cross-sectionally normalized features
3. Generates triple-barrier classification labels (or forward-return regression labels)
4. Searches for the best XGBoost hyperparameters using Optuna with purged walk-forward CV
5. Trains a bagged XGBoost ensemble across expanding walk-forward windows
6. Ranks stocks cross-sectionally by their predicted "probability of an up-move"
7. Long-only: buys the top quintile, stays flat in bear regimes (Nifty below 200-SMA)

The model is designed to be **information-coefficient (IC) maximizing**, not accuracy maximizing. The end goal is a cross-sectional ranking signal, not a binary yes/no predictor.

---

## System Flow (End to End)

```
Raw OHLCV prices (fetch_prices)
        │
        ▼
  Data Validation Gates
        │
        ▼
  Feature Engineering (build_features)
  ┌─────────────────────────────────────┐
  │ Per-ticker time-series features      │
  │ → cross-sectional z-score           │
  │ → cross-sectional rank features     │
  │ Index/regime features (Nifty, VIX)  │
  │ Relative-strength features (β, α)   │
  └─────────────────────────────────────┘
        │
        ▼
  Label Generation (add_labels)
  ┌──────────────────────────────────────┐
  │  Triple Barrier: tb_label ∈ {+1,-1,0}│
  │  OR Forward Return: fwd_ret          │
  └──────────────────────────────────────┘
        │
        ▼
  Sample Weights (uniqueness × time decay)
        │
        ▼
  Hyperparameter Optimization (Optuna, 50 trials)
  ┌──────────────────────────────────────────────────┐
  │  PurgedWalkForward CV (8 folds, purge+embargo)    │
  │  Objective: mean Spearman IC vs realized fwd_ret  │
  │  Sampler: TPE | Pruner: Median                    │
  └──────────────────────────────────────────────────┘
        │
        ▼
  Walk-Forward OOF Predictions (expanding window)
  ┌──────────────────────────────────────────────────┐
  │  For each rebalance step:                         │
  │    - Train bag of 3 XGBoost models               │
  │    - Early stopping (63-day val window)           │
  │    - Skip degenerate folds (best_iter=0)          │
  │    - Score next rebalance date's stocks           │
  └──────────────────────────────────────────────────┘
        │
        ▼
  Cost-Adjusted Backtest + OOF IC / Sharpe stats
        │
        ▼
  Final Model: Retrain on ALL labeled data (no early stopping)
  Probability Calibration (isotonic regression)
        │
        ▼
  Signal Generation: top quintile → LONG, rest → NEUTRAL
  Regime filter: if Nifty < SMA200 → all NEUTRAL
```

---

## Label Generation (`src/labels/targets.py`)

### Triple Barrier (Default, `label_type = "triple_barrier"`)

For each stock on each day `t`, three barriers are placed:

```
Upper barrier:  close[t] + barrier_up_mult × ATR14[t]   (take-profit)
Lower barrier:  close[t] - barrier_dn_mult × ATR14[t]   (stop-loss)
Vertical barrier:  t + horizon trading days              (time-limit)
```

The label is which barrier is hit first within the next `horizon` bars:
- `+1` → upper barrier hit first (bullish)
- `-1` → lower barrier hit first (bearish)
- `0`  → vertical barrier fires (indeterminate / time-out)

ATR14 is computed as an exponential moving average of True Range (span=14), not simple rolling. This makes the barriers adapt faster to recent volatility regime shifts.

**Critical**: The last `horizon` rows per ticker always have `NaN` labels (they look forward into the future). These are dropped before training.

**What the label distribution looks like**: With symmetric ±2 ATR barriers and a 5-day horizon, expect roughly 23% up, 23% down, 54% neutral. The model is therefore severely class-imbalanced, which is why `binary:logistic` with AUCPR (area under precision-recall curve) is used rather than accuracy or AUC-ROC.

### Forward Log Return (Alternative, `label_type = "fwd_ret"`)

```
fwd_ret[t] = ln(close[t + horizon] / close[t])
```

This is a regression target. The regressor (`reg:squarederror`) produces a predicted forward return instead of a probability. Less sharp on signal generation (harder to rank across stocks with different volatility regimes) but more interpretable.

### `fwd_ret` as the HPO Optimization Target

Regardless of whether you train a classifier or regressor, the **Optuna HPO objective always ranks predictions against the realized `fwd_ret`** (not the discrete `tb_label`). This is intentional: a 0/1 binary label collapses the cross-sectional variation that the IC (Spearman correlation) needs to distinguish. The Spearman IC against continuous forward returns measures the same thing the backtest cares about — "does a higher score correspond to a better stock?" — and produces a more stable optimization landscape.

---

## Feature Engineering (`src/features/engineer.py`)

All features are:
1. Computed strictly on data available at or before time `t` (no look-ahead)
2. Per-ticker time-series features (embarrassingly parallel, one process per ticker)
3. Cross-sectionally z-scored within each date (subtract date mean, divide by date std)
4. Key features also get a cross-sectional percentile rank (0–1) variant

The z-scoring step is critical: it removes the "level" bias where one stock always has higher RSI than another because of its volatility regime. Without it, the model learns "high-beta stocks have higher absolute returns" rather than "this stock is relatively strong vs its peers today."

### Feature Catalog

#### Price / Return Dynamics

| Feature | Formula | Purpose |
|---|---|---|
| `ret_Nd` (N=1,2,3,5,10,21) | `ln(close / close.shift(N))` | Raw momentum at multiple horizons |
| `cum_ret_Nd` (N=5,10,21,63) | `close / close.shift(N) - 1` | Level-change for longer windows |
| `dist_smaN` (N=5,10,20,50,200) | `close / SMA_N - 1` | Distance from moving average → mean-reversion or trend |
| `price_pos_Nd` (N=10,21,63) | `(close - min_N) / (max_N - min_N)` | Stochastic-style position within rolling range |

**Changing these**: `ret_Nd` features drive most of the momentum signal. Removing the shorter ones (`ret_1d`, `ret_2d`) reduces noise but also removes the microstructure signal. Adding very long horizons (`ret_252d`) would add a mean-reversion/value component.

#### Trend / Oscillators

| Feature | Formula | Purpose |
|---|---|---|
| `rsi_14` | EMA-based RSI, span=14 | Overbought/oversold |
| `stoch_k`, `stoch_d` | Stochastic %K and %D | Momentum within range |
| `macd`, `macd_signal`, `macd_hist` | EMA(12) − EMA(26), signal=EMA(9) | Trend direction and acceleration |
| `bb_pctb` | `(close − lower_band) / (upper − lower)` | Bollinger %B: position within band |
| `bb_width` | `(upper − lower) / mid` | Bollinger width: proxy for realized volatility |
| `adx` | Average Directional Index, length=14 | Trend strength (not direction) |

**Changing these**: These are all derived from price; they carry similar information to `ret_Nd` in a transformed form. The benefit is non-linearity that XGBoost can exploit with splits (e.g., RSI > 70 as a tree-split condition is more meaningful than a continuous return value). Removing them reduces feature count but the model adapts.

#### Volatility

| Feature | Formula | Purpose |
|---|---|---|
| `atr_norm` | `ATR14 / close` | Normalized daily range; input to barrier width |
| `rvol_Nd` (N=10,21,63) | `std(log_ret, N)` | Realized volatility at multiple horizons |
| `park_vol_21` | `sqrt(mean((log(H/L)² / (4·ln2)), 21))` | Parkinson estimator — more efficient than close-to-close, uses full daily range |

**Changing these**: Volatility features primarily help the model size positions (higher vol → smaller edge per unit of raw IC). The Parkinson estimator is more efficient than standard realized vol — it carries more information with fewer days of data. Adding GARCH or EWMA-based vol forecasts here could improve volatility targeting.

#### Volume / Microstructure

| Feature | Formula | Purpose |
|---|---|---|
| `vol_z20` | `(volume − SMA20_vol) / STD20_vol` | Volume anomaly: is today abnormally heavy? |
| `obv_z20` | Z-score of cumulative OBV | On-balance volume trend |
| `vwap_dev` | `close / VWAP_proxy20 − 1` | Price vs volume-weighted average: above/below institutional flow |
| `volume_roc_Nd` (N=5,21) | `log(vol+1 / vol.shift(N)+1)` | Volume trend: is interest building? |
| `dollar_vol_20d` | `log1p(mean(close × volume, 20))` | Liquidity proxy (rank-normalizes well cross-sectionally) |
| `ret_vol_corr_21d` | `corr(log_ret, vol_z20, 21)` | Positive = price-volume confirmation; negative = divergence |
| `weak_breakout_21d` | `1 if (new 21d high AND volume < 20d avg vol)` | Classic "false breakout" indicator |

**Changing these**: Volume features distinguish informed from uninformed price moves. A `ret_vol_corr_21d` near −1 means price is rising on declining volume (potential reversal). The `weak_breakout_21d` flag directly captures the "pump on thin volume" pattern. These are features more useful for longer holding periods.

#### Calendar Features

| Feature | Purpose |
|---|---|
| `day_of_week` (0=Mon, 4=Fri) | Monday effect, Friday positioning |
| `day_of_month` | Turn-of-month institutional rebalancing |
| `month` | Seasonal patterns (budget, earnings) |
| `is_expiry_week` | NSE F&O monthly expiry (last Thursday) → elevated volatility |

Note: Calendar features are **not** z-scored cross-sectionally (they're the same for all tickers on any given date, so z-scoring them produces zeros).

#### Market Regime / Index Features (require `index_df`)

| Feature | Formula | Purpose |
|---|---|---|
| `nifty_dist_sma50` | `Nifty / SMA50 − 1` | Short-term index trend |
| `nifty_dist_sma200` | `Nifty / SMA200 − 1` | Long-term index trend — also the regime filter trigger |
| `nifty_ret_5d` | `log(Nifty / Nifty.shift(5))` | Index momentum (short-term) |
| `nifty_ret_21d` | `log(Nifty / Nifty.shift(21))` | Index momentum (medium-term) |
| `vix` | India VIX level | Fear gauge |
| `vix_change_5d` | `VIX.pct_change(5)` | Rising VIX → increasing uncertainty |
| `vix_z20` | Z-score of VIX vs 20-day history | VIX relative to recent baseline |

**Not z-scored cross-sectionally** — they're identical for all tickers on a given date and represent the market context, not a per-stock signal.

#### Relative Strength (require Nifty present)

| Feature | Formula | Purpose |
|---|---|---|
| `beta_63d` | Rolling 63-day covariance(stock ret, Nifty ret) / variance(Nifty ret) | Market sensitivity |
| `alpha_5d` | `ret_5d − beta_63d × nifty_ret_5d` | Idiosyncratic return after removing market beta |

`alpha_5d` is the key feature here — a stock rising while the index is flat or falling is a strong signal. A stock rising because the market is up is less informative.

#### Cross-Sectional Rank Features (added for key features)

For `ret_5d`, `cum_ret_21d`, `rsi_14`, `atr_norm`, `vol_z20`, `dollar_vol_20d`:
- A `_rank` suffix version is added (percentile 0–1 within each date across all tickers)
- Provides the model an alternative, non-parametric representation of the same signal
- More robust to outliers than z-scores for extreme market events

---

## Sample Weighting (`src/labels/weights.py`)

Two schemes are composed multiplicatively:

```python
sample_weight = uniqueness_weight × time_decay_weight
(normalized so mean = 1)
```

### 1. Label Uniqueness Weights (López de Prado, AFML Ch. 4)

Triple-barrier labels look forward `horizon` days. A label at time `t` and another at `t+2` share 3 of their 5 forward days → they're not independent. Training on both equally double-counts the overlapping price path.

The weight for sample `i` is:
```
concurrency[i] = 1 + Σ_{d=1}^{h-1} (h - d)/h  (for each neighbor at distance d that exists)
uniqueness[i] = 1 / concurrency[i]
```

Higher concurrency → lower uniqueness → lower weight. This de-emphasizes samples that "see the same bars" as many of their neighbors.

**Effect of changing `horizon`**: Longer horizon → more overlap → lower average uniqueness weights → more regularization via weighting. The current 5-day horizon means each sample overlaps at most 4 others.

### 2. Time Decay Weights

```
weight[t] = exp(-ln(2) × age_days / halflife_days)
halflife_days = 252 (one year)
```

The most recent bar has weight 1.0. A bar from 252 days ago has weight 0.5. A bar from 504 days ago has weight 0.25. This makes the model prioritize recent regime behavior over distant history.

**Effect of changing `halflife_days`**:
- Smaller (e.g., 126) → model adapts faster to recent regimes, but higher variance and risk of overfitting to short-term noise
- Larger (e.g., 504) → more stable, slower to adapt, may miss regime shifts

---

## Hyperparameter Optimization (`src/models/trainer.py`)

### Optuna Setup

```python
study = optuna.create_study(
    direction="maximize",                        # maximize IC
    sampler=optuna.samplers.TPESampler(seed=42), # Tree-structured Parzen Estimator
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)
```

**TPE sampler**: Models the objective function as a density ratio of "good" vs "bad" parameter regions. Starts with random exploration, then focuses on promising regions. Better than grid search or random search for this many parameters.

**MedianPruner**: Stops unpromising trials early (after seeing 10 startup trials). If a trial's partial IC is below the median of completed trials at the same step, it's pruned. Saves compute without sacrificing exploration quality.

### Search Space

| Parameter | Range | Type | Notes |
|---|---|---|---|
| `n_estimators` | 200–1500 | int | Number of trees |
| `learning_rate` | 0.005–0.1 | float (log) | Shrinkage factor per tree |
| `max_depth` | 3–6 | int | Max tree depth |
| `min_child_weight` | 1–50 | int | Minimum sum of instance weights in a leaf |
| `subsample` | 0.5–1.0 | float | Row sampling fraction per tree |
| `colsample_bytree` | 0.4–1.0 | float | Column sampling fraction per tree |
| `gamma` | 0.0–5.0 | float | Minimum loss reduction for a split |
| `reg_lambda` | 0.1–10.0 | float (log) | L2 regularization on leaf weights |
| `reg_alpha` | 0.0–5.0 | float | L1 regularization on leaf weights |

### HPO Objective: Spearman IC vs Forward Return

```python
ic, _ = scipy.stats.spearmanr(preds, fwd_ret_on_val_fold)
fold_scores.append(ic)
return mean(fold_scores)  # across 8 CV folds
```

**Why Spearman IC (not AUCPR or RMSE)?**
- The backtest longs the top quintile — it cares about the *ranking* of stocks, not absolute calibration
- Spearman is rank-based, so it directly measures "does a higher score predict a better return?"
- AUCPR would only tell you "does a high score predict label=1?" — that's the wrong question if +1 labels don't cluster at the top of the forward return distribution perfectly
- Note: the model trains with `objective=binary:logistic` (optimizing log-loss) but is *selected* on Spearman IC. These can diverge: a model with good log-loss may rank stocks poorly if its probabilities are uncalibrated cross-sectionally.

### Parallelism During HPO

```python
n_parallel = min(cpu_count, max_workers, n_trials)
threads_per_trial = cpu_count // n_parallel
```

XGBoost's C++ tree-building releases Python's GIL, so multiple Optuna trials run in parallel using `ThreadPoolExecutor`. Each trial's XGBoost is capped to `threads_per_trial` threads to avoid oversubscribing the machine.

---

## XGBoost Model Parameters (Deep Dive)

### Default Parameters

```python
_BASE_PARAMS_CLF = {
    "objective":        "binary:logistic",  # P(label = +1)
    "eval_metric":      "aucpr",            # area under precision-recall
    "n_estimators":     400,
    "learning_rate":    0.02,
    "max_depth":        4,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "gamma":            0.5,
    "reg_lambda":       2.0,
    "reg_alpha":        0.5,
    "tree_method":      "hist",             # GPU-compatible histogram method
    "random_state":     42,
}
```

### Parameter-Level Explanation

---

**`objective = "binary:logistic"`**

Trains a binary classifier that outputs P(up-move). Labels are binarized: `+1 → 1`, `{-1, 0} → 0`. This means the model distinguishes "up" from "not up" — it doesn't separately model down vs neutral.

**What to change**: Switch to `"multi:softprob"` with `num_class=3` to model all three barrier outcomes separately. This would give you three probabilities (P_up, P_down, P_neutral) per stock. The ranking signal would then be `P_up − P_down` or `P_up / (1 − P_neutral)`. Potentially stronger signal but harder to calibrate and requires more data per class.

---

**`eval_metric = "aucpr"`**

AUCPR (Area Under Precision-Recall Curve) is used for early stopping, not for HPO. It's the right choice for imbalanced classes (~23% positives): ROC-AUC is misleadingly high when negatives dominate, whereas PR-AUC punishes the model for false positives.

**What to change**: Switching to `"auc"` (ROC-AUC) would make early stopping less precise on imbalanced data — avoid it. `"logloss"` is a valid alternative that measures calibration more directly.

---

**`n_estimators = 400`**

Number of boosting rounds (trees). The actual number used for any given model is determined by early stopping (`early_stopping_rounds = 50`) during walk-forward training — the model might stop at round 250 if the validation metric stops improving.

For the **final production model** (`predict_latest`), early stopping is disabled and the Optuna-chosen `n_estimators` is used directly (fixed number, no early stopping). This is because a small validation window (~21 days) causes early stopping to trigger at round 0 when the validation set is too small.

**What to change**: In HPO, the range is 200–1500. Allowing higher values captures more complex patterns but risks memorizing the training period. With `learning_rate = 0.02`, 1000+ trees is reasonable; with `learning_rate = 0.1`, 200–300 is usually enough.

---

**`learning_rate = 0.02`**

The shrinkage factor applied to each tree's contribution. Lower = more trees needed, more regularization effect. The current default of 0.02 is conservative for financial data.

**What to change**:
- `0.005–0.01`: Very stable, needs 1000+ trees, very slow training but best generalization
- `0.02`: Good balance (current default)
- `0.05–0.1`: Faster convergence, higher variance, works when data is large

There's a classic learning_rate × n_estimators trade-off: halving `learning_rate` typically requires doubling `n_estimators` for the same training error.

---

**`max_depth = 4`**

Maximum depth of each tree. Depth 4 allows up to 16 leaf nodes, capturing interactions between up to 4 features (feature A splits at top, then feature B given A, etc.).

**What to change**:
- `depth = 3`: Very regularized, avoids most interaction effects. Good baseline for noisy financial data.
- `depth = 4–5`: The current range. Captures 2-3 way interactions.
- `depth = 6+`: Risk of memorizing specific historical market events. Needs stronger L2/L1 regularization to compensate.

Financial data typically benefits from shallow trees (3–5) because the signal is weak and interactions are not stable across time.

---

**`min_child_weight = 5`**

Minimum sum of sample weights in a leaf node. With uniqueness+time-decay weights that are normalized to mean ~1, this roughly means "a leaf needs at least 5 samples to form." Higher = more conservative splits = more regularization.

**What to change**:
- `1–5`: More aggressive splitting, smaller leaves, more variance
- `5–20` (current default range): Moderate regularization
- `20–50` (in HPO): Heavy regularization, very coarse trees. Good when dataset is small or noisy.

With ~200 stocks × several years = hundreds of thousands of rows, mid-range (5–20) is appropriate. Raise this if you're overfit on early walk-forward folds.

---

**`subsample = 0.8`**

Row sampling: 80% of training rows are randomly selected to build each tree. Introduces stochasticity, reduces correlation between trees (same benefit as Random Forests), and slightly reduces overfitting.

**What to change**:
- `0.5`: High stochasticity, faster but noisier trees. Use when dataset is large.
- `0.8–1.0`: Less stochasticity. Current default is a good balance.
- `1.0`: No sampling — purely deterministic after training. Removes one source of variance.

---

**`colsample_bytree = 0.7`**

Column (feature) sampling: 70% of features are randomly selected per tree. Same idea as Random Forests' feature subsampling. With ~60 features, this uses ~42 per tree, which creates diversity without discarding too much information.

**What to change**:
- `0.4–0.5`: More aggressive subsampling. Forces trees to use different feature subsets → more diversity, less overfitting to any single feature. Good if features are correlated.
- `0.7` (current): Balanced.
- `1.0`: All features every tree — fast to reach best fit but higher correlation between trees → less variance reduction from ensembling.

There's also `colsample_bylevel` and `colsample_bynode` which subsample at split level instead of tree level — finer-grained but harder to tune.

---

**`gamma = 0.5`**

Minimum loss reduction required to make a split. A split is only accepted if it reduces the loss by at least `gamma`. Acts as a soft pruning mechanism.

**What to change**:
- `0.0`: No minimum — any split that reduces training loss is accepted. More splits, higher variance.
- `0.1–1.0` (current range): Light to moderate pruning.
- `2.0–5.0`: Heavy pruning. Trees become very shallow in practice (many potential splits pruned out). Use when overfitting is severe.

---

**`reg_lambda = 2.0` (L2)**

L2 regularization on leaf weight values. Higher values shrink leaf weights toward zero. The most reliable regularization parameter for XGBoost — always helps (unlike `gamma` which can occasionally hurt by blocking genuinely useful splits).

**What to change**:
- `0.1–1.0`: Light regularization.
- `1.0–3.0` (current default and common range): Moderate regularization.
- `5.0–10.0`: Heavy regularization. Leaf weights are very small, model is conservative. Good for sparse or noisy data.

For financial data where signal-to-noise is low, `reg_lambda = 2–5` is a sensible range.

---

**`reg_alpha = 0.5` (L1)**

L1 regularization on leaf weights. Encourages sparsity — some leaves get exactly zero weight, effectively pruning subtrees. Complementary to L2: L1 produces sparse solutions while L2 shrinks all weights smoothly.

**What to change**:
- `0.0`: No L1. Dense trees.
- `0.1–1.0` (current): Light sparsity pressure.
- `2.0–5.0`: Many leaves are zeroed out. Useful if you suspect many features are irrelevant and want the model to ignore them.

---

**`tree_method = "hist"`**

Uses the histogram-based approximate tree building algorithm. Bins continuous feature values into discrete buckets before finding splits. Faster than `exact` method, especially on large datasets and GPU. Required for GPU training (`device = "cuda"`).

**What to change**: Don't change this. `hist` is the modern default and is almost always better than `exact`. `approx` is an older variant with similar performance.

---

## Multi-Seed Bagging (`ensemble_size = 3`)

For each walk-forward rebalance step, instead of training one model, `ensemble_size` models are trained with identical data and hyperparameters but different random seeds:

```python
seed_params["random_state"] = base_seed + i  # 42, 43, 44
```

The final score for each stock is the **average predicted probability** across all bag members:

```python
preds = mean([m.predict_proba(X)[:, 1] for m in models])
```

**Why**: A single XGBoost fit on financial data has high run-to-run variance due to:
1. The subsample/colsample randomness in tree building
2. Early stopping triggering at different rounds depending on the random sample order

Averaging 3 seeds reduces this variance by ~40% (√3 reduction) without any additional data.

**What to change**:
- `ensemble_size = 1`: Single model, original behavior. Faster but higher variance.
- `ensemble_size = 3` (current): Good balance.
- `ensemble_size = 5–10`: Diminishing returns on variance reduction, linear cost increase.
- Cost scales linearly with `ensemble_size × walk_forward_steps`. On GPU, steps are parallelized via `max_parallel_fits`, so cost scales as `ceil(steps / max_parallel_fits) × ensemble_size`.

---

## Walk-Forward Training (`_walk_forward_predict` in runner.py)

### Why Walk-Forward (Not K-Fold)?

Standard K-fold uses future data to train on past data (data leakage). Walk-forward simulates actual deployment: train only on past, predict the next period.

### Expanding Window

```
Rebalance step 1:  train [day 0 → cutoff1]  →  predict day cutoff1+embargo+horizon
Rebalance step 2:  train [day 0 → cutoff2]  →  predict day cutoff2+embargo+horizon
...
```

The training window always starts from the beginning of history (day 0) and expands forward. This is different from a **sliding window** (fixed-length rolling window) which would forget early data. The choice matters: expanding window assumes market dynamics are stationary enough that 2015 data helps predict 2024. A sliding window would assume recent data is more representative.

### Purge + Embargo

Before each rebalance step, a gap is enforced between the training cutoff and the test date:

```
cutoff_date = dates[i - embargo - horizon]
```

- `horizon` bars purge: ensures no training label's forward window overlaps the test date
- `embargo` bars: additional buffer for microstructure effects (price impact, mean reversion)

Without this, training labels from day `t` that look forward to `t+5` would "see" the test day if the test day falls within those 5 days.

### Early Stopping Validation Window (`wf_es_val_days = 63`)

For each walk-forward step, a ~3-month window immediately before the purge gap is held out for early stopping:

```python
val_cutoff = dates[i - embargo - horizon - wf_es_val_days]
val_df = df[(df["date"] > val_cutoff) & (df["date"] <= cutoff_date)]
```

Early stopping halts training when the AUCPR on this validation set stops improving for `xgb_early_stopping = 50` rounds.

**Why 63 days (not 21)?** A 21-day window was tested and caused `best_iteration = 0` on most early walk-forward folds (2017–2020), because the window was too small and noisy for the low learning rate (0.007–0.02). The model would stop at round 0 and produce constant-prediction outputs (useless for cross-sectional ranking). 63 days gives a stable enough signal for early stopping to work correctly.

### Degenerate Fold Skip

After training, the code checks `model.best_iteration`:

```python
if all(bi is not None and bi == 0 for bi in best_iters):
    # All bag members stopped at round 0 → constant predictions → skip fold
```

A model with `best_iteration = 0` means it never improved over the base rate on the validation set. Its predictions are essentially a constant (the class prior), which produces zero cross-sectional dispersion — every stock gets the same score, quintile assignment becomes random noise, and the backtest IC for that fold is zero. Better to skip the fold entirely than add pure noise.

---

## Probability Calibration (`src/models/calibration.py`)

XGBoost's output probabilities are not well-calibrated in absolute terms (predicted P=0.7 doesn't mean 70% hit rate). For **cross-sectional ranking**, this doesn't matter (we only care about relative order). But for **reporting** and **risk sizing**, it does.

**Calibration procedure**:

1. Hold out the most recent 20% of the dataset as a final test set
2. Split that 20% in half: first half → fit calibrator, second half → measure calibration error
3. Train a temporary model on the 80% training data
4. Fit an isotonic regression (monotonic step function) mapping raw probabilities → calibrated probabilities

**Isotonic regression** is the right choice here (vs Platt scaling / temperature scaling) because:
- Non-parametric: doesn't assume the miscalibration is well-modeled by a sigmoid
- Monotone: preserves ranking (important since we use scores for ranking)
- No assumptions about the shape of the probability distribution

**Critical note**: The calibrator is NOT applied to the live signal generation (`predict_latest`). The calibrator was fitted on a model trained on 80% of the data, but the live model is retrained on 100% of the data — applying the calibrator to a different model's outputs maps all scores to near-constant values, collapsing the ranking. Raw probabilities from the live model are used for ranking instead.

---

## Signal Generation (`predict_latest`)

After retraining on all labeled data:

1. Score every stock in the universe on the latest available date
2. Sort by `prob_up` descending
3. Label top `1/n_quantile` fraction as `LONG`, rest as `NEUTRAL`
4. Apply regime filter: if `nifty_dist_sma200 < 0`, set all to `NEUTRAL`

### Why Quintile Ranking (Not Absolute Threshold)?

With base rate ~23% for label=+1 and a low-learning-rate model, calibrated probabilities almost never exceed 0.5. An absolute `prob > 0.5` threshold would produce zero signals every day. Cross-sectional ranking avoids this: "which stocks are relatively more likely to be up?" is a better question than "is this stock more than 50% likely to be up?"

### Regime Filter (`regime_filter = True`)

```python
if nifty_dist_sma200 < 0:  # Nifty is below its 200-day SMA
    all signals → NEUTRAL
```

This is a **hard risk overlay** that sits on top of the model signal. It's not learned by the model — it's a rule. When the market is in a bear regime (Nifty below 200-SMA), the historical hit rate for long signals collapses even if the model's relative rankings are correct, because market beta drowns out stock-specific alpha.

**What to change**: The current threshold is zero (below vs above 200-SMA). You could make this softer:
- Use `nifty_dist_sma200 < -0.05` (only filter if 5% below SMA, not just barely below)
- Add hysteresis (go flat when below, resume when 2% above)
- Replace with a VIX-based filter (`vix_z20 > 2.0 → go flat`)

---

## Conviction-Weighted Position Sizing

```python
conviction_weighted_sizing: bool = True
```

When enabled, stocks in the long basket are weighted by their score rank (higher-scoring stock gets a larger allocation) rather than equal-weighting. This improves the Calmar ratio (return per unit of max drawdown) by concentrating risk in the model's highest-conviction picks.

The implementation lives in `src/backtest/engine.py`. The backtest and live trading both use this logic consistently.

---

## Learning-to-Rank Mode (Flag-Gated, Default Off)

```python
ranker_enabled: bool = False
```

This is the **SOTA upgrade** that resolves the train/select divergence noted under `eval_metric` above: the system is IC-maximizing (it longs the top cross-sectional quantile), yet the default model trains `binary:logistic` and is only *selected* on Spearman IC. A model with good log-loss can still rank stocks poorly.

When enabled, the model trains XGBoost's **LambdaMART** ranker (`objective = "rank:ndcg"`) directly on the cross-sectional ordering, instead of P(up-move). This is the approach from Poh, Lim, Zohren & Roberts (2021), *"Building Cross-Sectional Systematic Strategies by Learning to Rank."*

### Query Group = Trading Day

```
qid = date   # one query group per rebalance date
```

The ranker forms its pairwise gradient *within each date* — exactly the comparison the strategy makes ("of the stocks tradeable today, which are the best?"). Rows must be grouped contiguously by `qid` before `fit`; `src/models/trainer.py:_sort_by_qid` does this.

### Graded Relevance from Forward Returns

```python
cross_sectional_relevance(df, bins=ranker_relevance_bins)  # → rank_rel ∈ {0 … bins-1}
```

Within each date, `fwd_ret` is bucketed into `ranker_relevance_bins` (default 8) integer relevance grades, higher = better forward return. This richer target replaces the binarized `{+1}` vs `{-1,0}` label. Grades are per-date and self-contained, so no information crosses the train/test boundary. Rows with NaN `fwd_ret` (the last `horizon` bars) get NaN grades and are dropped — same contract as the other labels.

### Top-k Focus

```python
"lambdarank_pair_method": "topk"
ranker_topk  # 0 = full NDCG; else ndcg@k
```

`topk` concentrates the pairwise gradient on the highest-relevance names — i.e. the top-quintile long basket the backtest actually trades. Set `ranker_topk = n_universe // n_quantile` to align the ranking metric with the basket size.

### How It Flows Through the Pipeline

- **HPO** (`tune_hyperparameters`, `task="ranking"`): each Optuna fold trains an `XGBRanker` with fold-local `qid`, but the objective is still the **same Spearman IC vs `fwd_ret`** — so tuned hyperparameters are directly comparable to the classifier baseline.
- **Walk-forward** (`_walk_forward_predict`, `train_xgb_bag_ranker`): identical bagging, degenerate-fold skip, and OOF IC / cost-adjusted backtest as the classifier path. The model is validated, not deployed blind.
- **Sample weights**: ranking weights one whole query group, not individual rows (an XGBoost requirement). The per-row uniqueness × time-decay weights are averaged per date by `_group_weights`, preserving the "recent dates count more" intent.
- **Live signal** (`predict_latest`): the ranking score is min-max rescaled into (0,1) and stored in the existing `prob_up` slot. Spearman IC is invariant to this monotonic rescale, so live IC tracking, drift, persistence, the paper trader, and the dashboard all work unchanged.

### Mutual Exclusion

Inert when `dynamic_horizon_enabled` is on (the quantile-surface path owns scoring there). Default off so the frozen `binary:logistic` baseline and its tests stay reproducible; `notebooks/colab_weekly.ipynb` flips it on via `USE_RANKER`.

---

## Dynamic Horizon (Flag-Gated, Default Off)

```python
dynamic_horizon_enabled: bool = False
```

When enabled, instead of a single 5-day classification model, the system trains a **quantile surface**: one XGBoost regressor per `(horizon, quantile_tau)` cell.

### Quantile Surface Architecture

```
horizon_grid = [5, 21, 63]  # 3 horizons
quantile_taus = [0.1, 0.5, 0.9]  # 3 quantiles per horizon
→ 9 XGBoost models per walk-forward step (vs 1 in the standard path)
```

Each model uses `objective = "reg:quantileerror"` (pinball loss). For a given `tau`:
- Model outputs a value `q` such that the realized forward return is below `q` with probability `tau`
- `q_0.1` = lower bound, `q_0.5` = median, `q_0.9` = upper bound of the return distribution

### Horizon Selection

For each stock, the optimal horizon `h*` is chosen by maximizing:

```
score = q90(h) - lambda_t × h
```

Where:
- `q90(h)` = predicted 90th percentile of the return distribution at horizon h
- `lambda_t × h` = time penalty (holding cost) — prefers shorter holds, all else equal
- `lambda_t = 0.0005` per day (small penalty)

This is stored in `q_star["q90"]` and represents "expected upside of the best-case return at this horizon."

### Dynamic Risk-Reward

Stop and target levels are derived from the quantile predictions:
- Stop: `entry − rr_k × |q10_star| × entry` (clamped to 0.8–3.0 ATR)
- Target: `entry + rr_k × q90_star × entry` (clamped to 1.0–6.0 ATR)

This makes the stop/target per-stock and per-market-condition, rather than the fixed `±2 ATR` in the standard path.

---

## Key Config Parameters and Their Effect

```python
@dataclass
class Config:
```

| Parameter | Default | What Changing It Does |
|---|---|---|
| `horizon` | 5 | Days to look forward for labels. Longer = less noise but more look-ahead bias risk. Affects barrier placement, label overlap, and embargo/purge sizing. |
| `barrier_up_mult` | 2.0 | ATR multiplier for upper take-profit barrier. Increase → fewer +1 labels (harder to hit), fewer longs signal but higher quality when they occur. |
| `barrier_dn_mult` | 2.0 | ATR multiplier for lower stop-loss barrier. Currently symmetric with `barrier_up_mult`. Making asymmetric (e.g., dn_mult=1.5, up_mult=2.5) changes the risk-reward of the label itself. |
| `signal_stop_atr_mult` | 1.5 | ATR multiplier for live trade stop-loss. Decoupled from label barriers — changing this only affects live trade sizing, not training. Tighter stop = lower loss per trade but more early exits. |
| `signal_target_atr_mult` | 3.0 | ATR multiplier for live trade take-profit. Current 3.0 vs 1.5 stop = 2:1 RR. Increase → fewer targets hit but larger wins when they do. |
| `train_min_days` | 504 | Minimum training rows before first prediction (~2 years). Increase → model sees more history before starting, fewer walk-forward steps, more stable early folds. |
| `embargo` | 5 | Extra buffer beyond horizon in the purge gap. Increase if you suspect autocorrelation in your features bleeds across the boundary. |
| `n_wf_splits` | 8 | Number of CV folds for Optuna HPO. More folds = better parameter estimates but more HPO compute. |
| `wf_es_val_days` | 63 | Early stopping validation window per walk-forward step. Too small → best_iter=0 degenerate folds. Too large → less training data per fold. |
| `rebalance_every` | 5 | Days between rebalance points. More frequent → more predictions (smoother equity curve) but more compute. Should match or be smaller than `horizon`. |
| `n_quantile` | 5 | Quintile for long selection: top 1/5th → LONG. Increase → fewer longs, higher precision but lower recall. Decrease → more longs, higher turnover. |
| `regime_filter` | True | Hard market-regime overlay. Disabling → longs even in bear markets (bad for drawdown). |
| `xgb_n_trials` | 50 | Optuna trials. More = better HPO but linear time cost. 30–50 is usually enough; 100+ has diminishing returns for this search space size. |
| `xgb_early_stopping` | 50 | Patience (rounds without improvement before stopping). Increase → models train longer, find the true best iteration better but slower. Decrease → faster but may stop too early. |
| `ensemble_size` | 3 | Multi-seed bag members per walk-forward step. Increase → lower variance, linear cost. 3–5 is the sweet spot. |
| `max_parallel_fits` | 8 | Concurrent XGBoost fits (Optuna trials + walk-forward steps). Set to `cpu_count` on CPU or `1–4` on GPU (overhead overlapping). Set to 1 for debugging. |
| `conviction_weighted_sizing` | True | Score-weighted position sizes. Disable → equal weight. Enabling typically improves Calmar ratio. |
| `ranker_enabled` | False | Train LambdaMART (`rank:ndcg`, qid=date) instead of `binary:logistic`. See *Learning-to-Rank Mode*. Default off keeps the frozen classifier baseline. |
| `ranker_objective` | `"rank:ndcg"` | Ranking loss. `rank:pairwise` is the alternative for multi-level relevance. |
| `ranker_relevance_bins` | 8 | Per-date graded relevance buckets from `fwd_ret`. More bins = finer ranking target; too many on thin days collapses toward a plain rank. |
| `ranker_topk` | 0 | 0 = full NDCG. Set to `n_universe // n_quantile` to focus the gradient on the top-quintile basket (`ndcg@k`). |

---

## Device Selection

```python
device: Literal["auto", "cuda", "cpu"] = "auto"
```

- `"auto"`: Detects GPU via `nvidia-smi -L`. Uses `cuda` if available, else `cpu`.
- `"cuda"`: Force GPU. Required for Colab training with T4/A100.
- `"cpu"`: Force CPU. Useful for local debugging or when GPU is being used for something else.

On GPU, XGBoost uses the `hist` method with CUDA kernels. The tree building itself runs on GPU but DMatrix construction and histogram binning still run on CPU — this is why `max_parallel_fits` still helps on GPU (overlapping the CPU-side overhead across concurrent fits).

---

## Where to Look for Each Concern

| Concern | File | Key Function / Class |
|---|---|---|
| Feature engineering | `src/features/engineer.py` | `build_features`, `_features_for_ticker` |
| Label generation | `src/labels/targets.py` | `triple_barrier_labels`, `forward_log_return` |
| Sample weighting | `src/labels/weights.py` | `sample_weights` |
| Model training | `src/models/trainer.py` | `train_xgb`, `train_xgb_bag`, `tune_hyperparameters` |
| CV splitter | `src/validation/walk_forward.py` | `PurgedWalkForward` |
| Probability calibration | `src/models/calibration.py` | `TimeOrderedCalibrator` |
| Walk-forward loop | `src/pipeline/runner.py` | `_walk_forward_predict` |
| Signal generation | `src/pipeline/runner.py` | `predict_latest` |
| Backtest engine | `src/backtest/engine.py` | `run_backtest` |
| All config knobs | `src/config.py` | `Config` dataclass |
| Dynamic horizon | `src/models/horizon_selection.py` | `select_horizon` |
| Quantile surface | `src/models/trainer.py` | `train_quantile_surface`, `predict_surface` |

---

## Common Diagnoses

**Low OOF IC (near zero)**:
- Check `best_iteration` distribution — if many folds hit 0, increase `wf_es_val_days`
- Check if features have leakage (`date` in feature cols, or any forward-looking computation)
- Try increasing `n_quantile` (look at top decile, not quintile)
- Check class imbalance — is the +1 base rate too low?

**High IC but poor backtest Sharpe**:
- Transaction costs eating the alpha — check `cost_bps_per_side`
- Turnover too high — increase `rebalance_every`
- Regime risk — check if losses cluster in bear periods, tune `regime_filter`

**Model overfitting (training IC >> OOF IC)**:
- Increase `min_child_weight`, `reg_lambda`, `reg_alpha`
- Decrease `max_depth`
- Increase `embargo`

**Model underfitting (flat OOF IC across all param combinations)**:
- Add new features (sector-relative momentum, earnings calendar, F&O OI data)
- Increase `max_depth`
- Try regression label (`label_type = "fwd_ret"`) instead of triple-barrier
- Check if the universe is too broad (signal diluted across uncorrelated stocks)

**Regime collapse (all predictions near base rate)**:
- Likely the `predict_latest` final model has `early_stopping_rounds` active — check `params_no_es`
- Check `best_iteration` on the final model
- Increase `wf_es_val_days` to fix degenerate early-stopping during walk-forward
