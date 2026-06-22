# Predictive Model Improvement Plan — SOTA Roadmap

**Baseline this plan improves on:** `reports/backtest_v20260622.md` — OOF IC 0.0193, dir-acc 54.5%, Sharpe 0.509, CAGR 5.4%, max_drawdown -24.74%, edge dies at 1.5-2x assumed costs.

**Read this first:** the engineering scaffolding in this repo (purged walk-forward, uniqueness/time-decay sample weights, an itemized Indian cost model, isotonic calibration, a model registry with champion/challenger promotion, PSI/CUSUM drift monitoring) is already more rigorous than most retail or even small-prop quant codebases. The problem is not sloppy engineering — most of `docs/swing-bot-plan.md`'s own spec is correctly implemented. The problem is that every component is currently working with **(1)** an unreliable measurement of its own performance, **(2)** a survivorship-biased and structurally over-efficient universe, **(3)** a generic, undifferentiated feature set, and **(4)** a single point-estimate model with no conviction-weighting. More Optuna trials on the same XGBoost over the same 52 technical features on the same 50 mega-caps will not produce a "drastic improvement" — new information and smarter use of the model's output will. This plan is organized accordingly.

---

## Implementation status (updated 2026-06-23)

**Done** (code merged, unit-tested locally; not yet trained/validated — training happens on Colab):
- Phase 0.1-0.3: `daily_information_coefficient`, `ic_information_ratio`, `deflated_sharpe_ratio`, `block_bootstrap_ci` added to `src/validation/metrics.py`, wired into `src/pipeline/runner.py` and `src/backtest/report.py`.
- Phase 1.7: universe widened 50 → 196 names (`src/data/ingestion.py:UNIVERSE`, `config/universe.json`) — Nifty 100 + Midcap 100 constituents as of June 2026.
- Phase 2.9: `volume_roc_5d/21d`, `dollar_vol_20d`, `ret_vol_corr_21d`, `weak_breakout_21d` added to `src/features/engineer.py`.
- Phase 3.16: multi-seed bagging (`train_xgb_bag`/`predict_bag` in `src/models/trainer.py`, `cfg.ensemble_size`, default 3) wired into the walk-forward OOF loop (`_walk_forward_predict`) — **not** wired into `predict_latest` (the live-signal path), which would require the model registry/bundle format to support a multi-model ensemble; deferred as a separate, riskier change.
- Phase 4.19: conviction-weighted basket sizing (`_conviction_weighted_return` in `src/backtest/engine.py`, `cfg.conviction_weighted_sizing`, default True) — replaces equal-weight quintile membership in the backtest only (not yet in live/paper position sizing, which still uses flat `position_size_pct`).

**Still open:**
- Phase 0.4-0.5 (CPCV, switch the promotion gate to IC-IR), Phase 1.6 (real point-in-time NSE reconstitution history — the universe widening above still does *not* fix survivorship bias), Phase 2.10-2.14 (delivery%, FII/DII, sector map, market breadth, F&O positioning — all need new data sourcing), Phase 3.15/3.17/3.18 (meta-labeling, market-neutral target, learned regime gate), Phase 4.20-4.21 (turnover hysteresis, turnover-penalized HPO objective), Phase 5 (ongoing validation discipline).
- A few Nifty 200 tickers are best-effort guesses for recent IPOs/renames (e.g. `GET&D.NS`, `LTF.NS`, `TATACAPITAL.NS`, `VMM.NS`, `GROWW.NS`) — `_retry_tickers_individually`'s existing graceful-degradation logging will flag (not crash on) any that don't resolve; check the `[ingestion] WARNING: could not load [...]` line on the next Colab run.

---

## A. Root-cause diagnosis (grounded in current code, not generic advice)

### A1 — You cannot fully trust the IC number being optimized
- `information_coefficient()` (`src/validation/metrics.py:16`) computes **one pooled Spearman correlation across every (date, ticker) OOF row**. This conflates cross-sectional stock-picking skill with time-series/market-level effects — a strong bull run inflates the pooled correlation even with zero genuine stock-picking skill.
- There is no per-period (daily) IC series, hence no **IC Information Ratio** (mean daily IC / std of daily IC) — the standard significance test for a cross-sectional signal (t-stat ≈ IC_IR × √N_periods).
- `_optuna_objective` (`src/models/trainer.py:162`, IC computed at line 217) runs the same pooled-Spearman objective per fold, averaged over 8 folds, for 50 trials, every retrain. With no **Deflated Sharpe Ratio** correction for that multiple-testing, the reported Sharpe (0.509) is optimistically biased by however many configurations were implicitly searched.
- Net effect: the run-to-run swings already on record (OOF IC 0.0139 → 0.0281 → 0.0308 → 0.0193 across the last four runs, per `reports/backtest_v20260622.md`'s own "Findings") are at least partly measurement noise. Until this is fixed, no later change in this plan is independently verifiable.

### A2 — Survivorship bias is still live
- `config/universe.json`: every one of the 50 members has `start_date: 2015-01-01, end_date: null`. `src/data/ingestion.py:31` (`UNIVERSE`) is the same static, present-day Nifty 50 list applied across all 11 backtested years.
- This is exactly the failure mode `docs/swing-bot-plan.md` §5.2 calls "the most important and most overlooked" bias: backtesting today's 50 winners as if they were always the universe. Names that were demoted, delisted, or replaced over 2015-2026 are invisible to the backtest, which silently inflates both the benchmark and the model's apparent stock-picking edge.

### A3 — The universe is small and already efficient
- 50 mega-cap, heavily-covered names are the most arbitraged slice of the Indian market — every domestic and FII quant desk already prices these efficiently. An IC of ~0.02 here is a believable result for this universe, not a bug. Wider, less efficient universes (Nifty 200/500 mid/small-caps) typically carry structurally higher raw IC because of lower analyst coverage and more liquidity/behaviorally-driven mispricing.

### A4 — The feature set has no India-specific informational edge
- `src/features/engineer.py` ships ~52 generic technicals: lagged/cumulative returns, SMA distance, RSI/MACD/Bollinger/ADX/Stochastic, ATR, Parkinson vol, volume z-score, OBV, VWAP deviation, plus index/VIX regime and calendar features. This is the same toolkit every public TA-based bot uses.
- `docs/swing-bot-plan.md` §6.1 explicitly named **delivery %**, **FII/DII flows**, and **sector relative strength** as "the alpha most retail bots miss." None of the three exist in the codebase today (verified: no `delivery`, `fii`, `dii`, `sector`, or `breadth` reference anywhere under `src/`).

### A5 — Single model, single seed, point-estimate signal
- One XGBoost model, refit at every walk-forward step (`src/pipeline/runner.py:106`, `_walk_forward_predict`), no second model family, no multi-seed bagging, no stacking. Combined with A1's noisy IC objective, the production model is one noisy draw from a wide distribution of "equally valid" XGBoost fits — a likely contributor to the IC instability in A1.
- No **meta-labeling** layer (a secondary model that predicts whether the primary model's call is actually trustworthy, à la López de Prado). Every top-quintile name today gets identical equal weight regardless of how confident the model actually is.

### A6 — The label is not market-neutralized
- `src/labels/targets.py`'s `tb_label`/`fwd_ret` are raw absolute forward returns, not residualized against the market. The model is implicitly asked to predict both "will the market go up" and "will this stock beat its peers," with no way to separate the two in its own loss.
- This is almost certainly why a separate, hand-built regime overlay (`nifty_dist_sma200 < 0` → flat, `src/backtest/engine.py:60-67`) had to be bolted on externally to cut max drawdown — the model itself was never trained to manage market-timing risk, because nothing in the training objective rewarded it for that.

### A7 — Position construction is binary, not conviction-weighted
- `src/backtest/engine.py:71-79`: every rebalance, the top quintile (~10 of 50 names) is bought equal-weight, full stop. A name barely inside the top quintile gets the same weight as the most confident name in the book.
- Consistent with the observed hit_rate 48.5% / profit_factor 1.21 — many marginal, low-conviction bets diluting the few good ones — and with a -24.7% drawdown that's large relative to a 5.4% CAGR (Calmar 0.218).

### A8 — Turnover-awareness stops at the cost model, never reaches the objective
- The Indian cost model (`src/backtest/costs.py`) is itemized correctly, and the sensitivity table already shows edge dying at 1.5-2x costs. But nothing upstream — not the Optuna objective, not the quintile-reassignment logic — is penalized for turnover. The book's full quintile membership is recomputed every 5 days with no hysteresis, so names that bounce around the quintile boundary churn the book for no real signal gain.

### A9 — Validation is a single walk-forward path
- One sequence of folds (`PurgedWalkForward`, `src/validation/walk_forward.py`) produces one Sharpe number. There is no Combinatorial Purged CV (CPCV) to see the *distribution* of Sharpe across many resampled paths, and no block-bootstrap confidence interval on the backtest itself. The -24.7% max drawdown could be a median outcome or a 10th-percentile bad path — today there's no way to tell.

---

## B. The plan, by phase

### Phase 0 — Fix the ruler before changing anything else (1-3 days, no new data)
Nothing here adds edge. It lets you tell real edge from noise, which every later phase depends on to be verifiable.

1. **Daily IC + IC-IR + significance.** Add `daily_information_coefficient()` and `ic_information_ratio()` to `src/validation/metrics.py`: group OOF preds by date, Spearman-correlate cross-sectionally per date, report the mean and t-stat (`mean/std × √n_days`). Report both old pooled IC (continuity) and new daily IC/IC-IR in `reports/backtest_*.md`, next to the existing computation at `src/pipeline/runner.py:510-512`.
2. **Deflated Sharpe Ratio**, parameterized by Optuna trial count (`cfg.xgb_n_trials`) as the multiple-testing count. Report alongside raw Sharpe in every backtest report.
3. **Block-bootstrap CI** on Sharpe/CAGR/max_drawdown — resample `stats["period_returns"]` in overlapping blocks (~20-40 periods) ~1000x, report 5th/50th/95th percentile, not a point estimate. New `src/validation/robustness.py`.
4. **Combinatorial Purged CV (CPCV)** for both HPO and the final backtest — generates N paths from combinatorially-held-out purged groups instead of one. Bigger lift than 1-3; pull into Phase 5 if it doesn't fit here.
5. Switch `compare_models`/`should_retrain` (`src/models/improvement.py`) to gate on the new daily IC-IR (or a confidence-interval-aware rule), not the raw pooled `oof_ic` they read today.

*Expected effect:* no metric improves. Every later phase becomes falsifiable instead of another noise draw.

### Phase 1 — Kill survivorship bias, widen the universe (3-7 days, data work)
6. **Real point-in-time membership.** Replace `config/universe.json`'s all-`null` end_dates with actual Nifty 50 reconstitution history (semi-annual NSE index circulars). Filter the trainable universe per-date from this table instead of the static `UNIVERSE` list (`src/data/ingestion.py:31`).
7. **Expand the candidate universe to Nifty 200** (or Nifty Next 50 + Nifty 50). Going from 50→200 names per date takes the long quintile basket from ~10 to ~40 names (materially better diversification, lower idiosyncratic drawdown) and reaches into less-efficient mid-caps where IC tends to be structurally higher. Add an ADV/liquidity filter (e.g., 20-day average traded value floor) so position sizing stays realistic as smaller names enter.
8. Keep a Nifty-50-only run as a labeled comparison (`label="nifty50_baseline"`) so any Sharpe/IC change is attributable to universe breadth specifically, not conflated with other phases.

*Expected effect:* primarily targets max_drawdown (diversification) and IC (less efficient names). Likely the single highest-leverage lever that needs no new exotic data.

### Phase 2 — Deeper use of existing data, plus India-specific alpha features (new data sources except where noted)
9. **Deepen the existing volume signal (no new data needed — ship alongside Phase 0).** Today's only volume features are `vol_z20`, `obv_z20`, and `vwap_dev` (`src/features/engineer.py:153-161`) — all snapshot z-scores against a 20-day window, and only `vol_z20` gets a cross-sectional rank (`_cs_rank`, `engineer.py:301`). Add: **(a) volume rate-of-change** (`volume_roc_5d`/`21d` = `log(v/v.shift(n))`) — captures *building* interest the way price momentum captures price trend, which a single 20-day z-score snapshot doesn't; **(b) dollar volume / turnover** (`(close*volume).rolling(20).mean()`, z-scored + cross-sectionally ranked) — also doubles as the ADV/liquidity filter Phase 1.7 needs once the universe widens into smaller names; **(c) volume-price divergence** (e.g. rolling correlation of return vs. volume, or a "new N-day price high on below-average volume" flag — the classic weak-breakout tell, only partially captured by `obv_z20` today).
10. **Delivery percentage** (NSE bhavcopy — free, official, daily). Add `delivery_pct`, `delivery_pct_zscore_20d`, `delivery_pct_trend` in `src/features/engineer.py`. The spec (`docs/swing-bot-plan.md` §6.1) already flagged this as the highest-value India-specific signal and it has never been implemented. High delivery % on an up day signals real (not intraday-leveraged/speculative) conviction — something with no equivalent in standard US-style technicals.
11. **FII/DII daily net flows** (NSE/SEBI provisional daily figures). Add `fii_net_5d`, `dii_net_5d` next to the existing VIX/index regime features in `_add_regime_features` (`src/features/engineer.py:204`). Well-documented short-horizon correlation with Nifty direction.
12. **Sector relative strength.** Build `config/sector_map.json` (NSE sector per symbol) and add sector-demeaned return/momentum features — isolates stock-specific alpha from sector-wide moves, which the current Nifty-only relative features (`beta_63d`/`alpha_5d`, `src/features/engineer.py:251-252`) don't capture.
13. **Market breadth** — % of universe above its own 50/200-SMA per date. A more robust regime signal than one index's distance from its own SMA, and a natural fit once Phase 1 widens the universe.
14. **F&O positioning data — the highest-conviction "out of the box" lever.** Given this sits alongside a Bank Nifty options system, the path to NSE F&O data likely already exists: index/stock futures basis (cost-of-carry — rich/cheap near-term direction signal), Put-Call Ratio (OI and volume basis), and OI buildup classification (long/short buildup from price+OI co-movement). This is free, daily, official data that's materially harder for a generic/non-Indian quant pipeline to bother integrating — genuine differentiation, not another oscillator. Start with index-level PCR/basis as regime features (cheap integration); extend to per-stock F&O OI for the ~190 F&O-enabled names if it proves out.

*Expected effect:* item 9 is a free improvement on data already in hand — do it first. Items 10-14 target IC directly via genuinely new information sources, at the cost of new data-acquisition work.

### Phase 3 — Model architecture: stop trusting one point estimate (1-2 weeks)
15. **Meta-labeling.** Keep XGBoost as the *primary* direction/magnitude model. Train a *secondary* binary model — same walk-forward discipline — whose target is "was the primary model's top-quintile call actually correct," using the primary's score plus uncertainty features (prediction margin, regime, recent realized vol) as inputs. Use its probability as a continuous position-sizing weight instead of equal-weighting the quintile (directly fixes A5/A7). New stage in `src/models/trainer.py` + a sizing path in `src/backtest/engine.py`.
16. **Ensemble for stability, not just accuracy.** Bag 5-10 XGBoost seeds per walk-forward step (cheap: same data, different `random_state`) and average scores before ranking; optionally add one structurally different model (LightGBM, or a plain cross-sectional Ridge/ElasticNet on the same features) and blend. Directly targets the run-to-run IC instability in A1/A5 — averaging independent noisy estimators reduces variance even when no individual one is better.
17. **Market-neutral auxiliary target.** Add `fwd_ret_demeaned = fwd_ret - cross_sectional_mean(fwd_ret, same date)` in `src/labels/targets.py`. Track IC against this in parallel with raw `fwd_ret` IC. Separates pure stock-selection skill from market-timing exposure (A6) — tells you how much of today's Sharpe is "picking good stocks" versus "being long through a secular Nifty bull run."
18. **Learned regime gate.** Replace the single hardcoded `nifty_dist_sma200 < 0` rule (`src/backtest/engine.py:60-67`) with a small classifier (logistic/shallow tree) over VIX level/change, breadth (Phase 2.13), and FII flow (Phase 2.11) predicting near-term drawdown risk — keep the hard SMA200 rule underneath as a non-negotiable backstop. Turns A6's externally-bolted-on fix into something tunable and independently backtestable.

### Phase 4 — Portfolio construction & cost/turnover (3-5 days)
19. **Confidence-weighted sizing**, replacing equal-weight quintile membership (`src/backtest/engine.py:77-79`) — size ∝ meta-label conviction (3.15) and/or inverse recent volatility, capped per name. Directly targets Calmar/drawdown (A7).
20. **Turnover hysteresis band.** Exit a held name only below, e.g., the 60th percentile (not just outside the top quintile); enter a new name only above the 85th. Damps the boundary-churn cost in A8 without needing new signal.
21. **Penalize turnover inside the HPO objective** itself — e.g. `objective = mean_fold_IC − λ·turnover` — instead of measuring turnover only after the fact. Small change to `_optuna_objective` (`src/models/trainer.py:162`).

### Phase 5 — Validation rigor, so "drastic improvement" is trustworthy when you see it (ongoing)
22. Re-run Phase 0's CPCV/bootstrap/deflated-Sharpe machinery after **every** phase above, not only at the end — each phase should report a confidence interval, not a point estimate.
23. Add a leakage/regression test per new feature family (volume, delivery%, FII/DII, sector, F&O) to `tests/test_leakage.py` — same discipline already applied to the current feature set.
24. Keep the frozen v20260622 numbers (IC 0.0193, Sharpe 0.509, CAGR 5.4%, maxDD -24.74%) as a permanent comparison row in every future backtest report, so every change is reported as a delta against a fixed baseline, never in isolation.

---

## C. Priority — what to do before the *next* retrain if time is limited

These four are grounded in your actual current bottlenecks (A1, A2/A3, A5/A7, A6) rather than generic best practice, and don't require sourcing new data:

1. **Phase 0.1–0.2** (daily IC/IC-IR + deflated Sharpe) — without this, "improvement" isn't measurable at all.
2. **Phase 2.9** (deepen the existing volume features — volume ROC, dollar-volume/turnover, volume-price divergence) — zero data-acquisition cost, ship it in the same pass as Phase 0.
3. **Phase 1.7** (widen universe to Nifty 200 + real point-in-time membership) — single highest-leverage lever; touches both IC (less-efficient names) and drawdown (diversification), no exotic data required.
4. **Phase 3.16** (multi-seed ensemble) — cheap, mechanical, directly attacks the IC instability already observed across four runs.
5. **Phase 4.19** (confidence-weighted sizing via meta-labeling) — directly attacks weak Calmar/drawdown without touching the underlying alpha signal.

Delivery%, FII/DII, sector, and F&O positioning (Phase 2, items 10-14) are real edge but are a data-engineering project, not a retrain config change — sequence them as the follow-up phase once 1-5 establish a trustworthy, reproducible measurement baseline.

---

## D. What not to expect

`docs/swing-bot-plan.md`'s own framing is worth restating, because "best model in the world" and "drastic improvement" are in tension with it: durable daily/weekly equity edge tops out around 52-56% directional accuracy and IC in the 0.03-0.06 range for liquid large-caps. Current dir-acc is 54.5% — already in-range. "Best in the world" here means the most disciplined, best-measured, hardest-to-arbitrage edge available at this universe and horizon — not 70%+ accuracy or a Sharpe of 3. Any future backtest showing dramatically more than that should be treated as a leakage bug to hunt down, not a result to ship.

---

## Appendix — minor inconsistency noticed while reading the code
`config/strategy.yaml` defines `top_k: 10` and `prob_threshold: 0.5` under "decision / ranking," but `Config.from_yaml` (`src/config.py:112-152`) never maps either key into the `Config` dataclass, and live signal selection already moved to quintile-based selection per the 2026-06-21 fix. These two YAML keys are currently dead config — either wire them in if still meaningful, or delete them so the file doesn't imply behavior that doesn't exist.
