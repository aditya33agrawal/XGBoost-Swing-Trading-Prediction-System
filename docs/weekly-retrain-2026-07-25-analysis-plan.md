# Weekly retrain 2026-07-25 — output analysis & improvement plan

Source: `colab_weekly.ipynb` run on 2026-07-25 (Cells 4–9) + `outputs/model_runs.json`,
`outputs/feature_importance_v20260725.json`, `outputs/signals_20260725.json`,
`outputs/portfolio.json`. Ranker mode (`USE_RANKER=True`, `rank:ndcg`/LambdaMART).

## 1. What happened this run

| Run | oof_ic (pooled) | oof_dir_acc | bt_sharpe | bt_cagr | bt_max_dd | deployed |
|---|---|---|---|---|---|---|
| v20260621 | 0.0308 | 54.7% | 0.555 | 5.95% | -18.3% | no |
| v20260719 | -0.0343 | 47.7% | 0.956 | 18.82% | -43.0% | yes (was current) |
| v20260725 | -0.0010 | 47.5% | 1.159 | 23.23% | -47.5% | **yes (new)** |

- OOF daily IC-IR = 0.125, t-stat = 2.70 (n=469 days) — statistically non-zero but
  small; deflated Sharpe P(true Sharpe>0| 50 trials) = 0.891.
- Model was deployed only because `-0.0010 > -0.0343 + 0.005` (relative-improvement
  gate) — not because it demonstrated a real edge.
- **Both this week's live-signal cells (5 and 8) produced 0 LONG / all-NEUTRAL**
  across 88 and 193 scored tickers respectively. Paper trader had nothing to plan.
- 72 rows flagged as >10σ price spikes (possible unadjusted corporate actions),
  every run, never resolved.
- Every Supabase write/read this run failed (`[Errno -2] Name or service not
  known`) — DB layer effectively dark; run only persisted to local JSON/Drive.
- Walk-forward took 15,443s (~4.3h) for 469 folds; most folds show
  `best_iters` landing at 1400–1455, i.e. pinned near the Optuna search-space
  ceiling (`n_estimators` ∈ [200, 1500]) rather than converging naturally.

## 2. Diagnosis, ranked by impact

### 2.1 Zero tradeable signals is the most urgent problem (silent, not a bug)
`src/pipeline/runner.py:557-560` applies `regime_filter` (index below 200-SMA →
force all `NEUTRAL`) as a hard on/off switch *after* ranking, with no log line
explaining it fired. Two weeks running, 0/193 LONG. The notebook output makes
this look like a broken model rather than an intentional risk-off overlay —
there is no way to tell the two apart from the printed summary alone.

- **Fix:** log `regime_filter suppressed N candidate LONGs (nifty_dist_sma200=X)`
  whenever it trips, both in `predict_latest`/`predict_latest_surface` and in
  Cell 9's summary. Consider a graduated response (halve `n_long` / raise the
  rank-percentile cutoff) instead of a full kill switch, so a genuinely
  risk-off month doesn't produce total silence for weeks at a stretch.

### 2.2 The "edge" is marginal and unstable across retrains
Pooled OOF IC has been +0.031 → -0.034 → -0.001 across the last three runs —
sign-flipping, not converging. Directional accuracy is *below* 50% in the two
most recent runs. The only consistently positive number is the day-wise mean
IC-IR (0.125, t=2.70), which is a much noisier, higher-variance statistic than
it looks with only 469 (highly overlapping, horizon=5) observations.

- **Fix:** track a rolling multi-run IC/IC-IR trend (not just this week vs
  last) before trusting any single retrain's number. `compute_weekly_ic_series`
  already exists (Cell 4) — surface it in Cell 6/9 as a decision input, not
  just a display.
- Investigate why pooled IC and daily-mean IC diverge so much — check whether
  a handful of high-volatility names/days dominate the pooled correlation and
  distort it relative to the (equal-weighted-per-day) mean IC.

### 2.3 Promotion gate has no absolute floor
`compare_models` (`src/models/improvement.py`) only checks
`new_ic - current_ic >= 0.005`. It just deployed a model with IC ≈ 0 because
the previous one was IC = -0.034. Two back-to-back models with no real edge
can ping-pong through this gate forever without ever being *good*.

- **Fix:** add an absolute floor (e.g. require `new_ic > 0` AND deflated Sharpe
  `P(true Sharpe>0) > 0.9`, or similar) before deployment, alongside the
  existing relative-improvement check. Otherwise "improved" and "worth trading"
  are conflated.

### 2.4 Optuna's `n_estimators` ceiling is routinely binding
`src/models/trainer.py:573` — `suggest_int("n_estimators", 200, 1500)` — and
the walk-forward log shows `best_iters` clustering at 1400–1455 in most late
folds (e.g. fold 460: `[1438, 1090, 1453]`). When early stopping keeps
choosing near-max, the search space is truncating the true optimum, and the
model is both slower to train (4.3h for this run) and possibly still
underfit in the folds that hit the ceiling.

- **Fix:** raise the upper bound (e.g. to 2500–3000) and/or lower
  `learning_rate`'s floor so the same effective model can be reached with a
  smaller ceiling; watch for CV objective time cost. Also worth logging how
  often `best_iter` hits the ceiling as an explicit HPO health metric.

### 2.5 Unresolved data-quality warning: 72 spike rows every run
`[validation] WARNING: 72 spike rows detected (|z| > 10.0σ)` fires identically
run over run — it's flagged but apparently never actioned (not filtered,
adjusted, or whitelisted). If these are real unadjusted splits/bonuses in the
196-ticker universe, they inject label/feature noise directly into training.

- **Fix:** dump the 72 spike rows (ticker + date) to a report once, triage
  which are genuine corporate actions vs data errors, and either patch the
  adjustment or add the tickers to an exclusion/adjustment list. This is a
  one-time data-cleaning pass, not a recurring judgment call.

### 2.6 Supabase is fully dark this run
Every `fetch_rows`/`upsert_rows` call failed with a DNS error
(`Name or service not known`). This means: no IC-trend history in Cell 4, no
`predictions`/`model_runs`/`feature_importance`/`paper_trades`/
`account_ledger` sync, and the Streamlit dashboard referenced in Cell 9's
summary is reading stale or empty data. This may be a transient Colab
networking issue, but it happened for the *entire* run (both training and
fast-signal cells), so it's worth confirming it isn't a persistent
misconfiguration (e.g. stale/rotated `SUPABASE_URL`).

- **Fix:** verify Supabase project URL/secret are current; consider having
  Cell 1 fail loudly (not just print a warning) if the DB is unreachable,
  since silent JSON-only mode means the dashboard and the 4-week rolling IC
  gate (which drives the emergency-retrain check) are running on stale data.

### 2.7 Dynamic risk-reward is built but not switched on
`risk_reward` is a flat `2.00` for every signal in `signals_20260725.json` —
the dynamic-horizon/RR system from `dynamic_horizon_rr_implementation.md` is
still flag-gated off (`dynamic_horizon_enabled=False` in this run's `Config`).
This is a known, already-scoped lever, not a new finding — listed here only
to prioritize it against the items above.

## 3. Prioritized next steps

1. **(Quick, high value)** Log regime-filter suppressions explicitly; surface
   "N LONGs suppressed by regime filter" in the notebook summary so a
   zero-signal week is self-explanatory. (§2.1)
2. **(Quick)** Add an absolute-IC floor to `compare_models` promotion gate so
   near-zero models stop auto-deploying just for being "less bad." (§2.3)
3. **(Medium)** Dump and triage the 72 recurring spike rows; fix or exclude
   the underlying tickers. (§2.5)
4. **(Medium)** Widen the `n_estimators` Optuna bound and add a
   ceiling-hit-rate diagnostic to the walk-forward log. (§2.4)
5. **(Medium)** Confirm/fix Supabase connectivity from Colab; make the
   fallback loud instead of silent. (§2.6)
6. **(Larger, already scoped)** Turn on `dynamic_horizon_enabled` in a
   controlled A/B run and compare against this week's fixed-RR baseline.
   (§2.7, see `dynamic-horizon-rr-plan.md`)
7. **(Ongoing)** Track the multi-run IC trend (not single-run) as the primary
   signal-quality dashboard before trusting any one retrain. (§2.2)

## 4. What's *not* broken

- Feature importance is diffuse across ~50 momentum/volatility/breadth
  features with no single dominant feature — no obvious leakage.
- The walk-forward/backtest/cost-sensitivity machinery, deflated Sharpe, and
  paper-trading ledger are all running end-to-end correctly and producing
  self-consistent numbers.
- Data ingestion (196 tickers, 497k rows) and feature engineering (80
  features) both completed cleanly.
