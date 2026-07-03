# Weekly Retrain — Bug Fixes & Improvement Plan (2026-06-29)

Triggered by the Colab weekly retrain output for `v20260629`. This doc records
(1) the bugs/anomalies found in that run and what was fixed, and (2) a focused
plan to make the model better, building on `docs/model-improvement-plan.md`
(do not duplicate its Phase 0–5 roadmap — this is the delta for *this* run).

---

## 1. What went wrong this run, and the fixes

### 1.1 ✅ FIXED — Zero signals generated (the headline bug)
**Symptom:** Both the full run and the `--fast-signals` run printed
`TOP SIGNALS (today) → (no signals generated)` and
`Next-week signals: 0 total | 0 LONG`, yet the model still deployed.

**Root cause:** `predict_latest()` (and `predict_latest_surface()`) score the
latest price date via
`latest_df = df_full[df_full["date"] == latest_date].dropna(subset=feature_cols)`.
`feature_cols` includes the **shared index/VIX regime columns**
(`nifty_dist_sma50/200`, `nifty_ret_1d/5d/21d`, `vix`, `vix_change_5d`,
`vix_z20`). yfinance frequently lags `^INDIAVIX` (and occasionally `^NSEI`) by a
trading day, so on the most-recent stock bar (2026-06-25) those columns were
`NaN` **for every ticker**. `dropna` then dropped the entire cross-section →
`latest_df` empty → empty signal frame. The walk-forward OOF loop was unaffected
because it only scores historical dates where the index data exists, which is why
the bug was invisible in backtest metrics but killed live signals.

**Fix (`src/features/engineer.py`, `_add_regime_features`):** forward-fill the
shared index/VIX columns per ticker after the merge. ffill only propagates the
last-known regime value (no look-ahead), so a missing latest index bar reuses the
prior day's regime instead of nuking the cross-section.

**Defense/observability (`src/pipeline/runner.py`, `predict_latest`):** when the
latest cross-section is wiped, log exactly which feature columns were all-NaN on
the latest date instead of silently returning empty.

**Test:** `tests/test_latest_signals.py::test_latest_bar_survives_missing_vix`
reproduces a missing latest `^INDIAVIX` bar and asserts the cross-section
survives. Full suite: **64 passed**.

**Guards added (this session):**
- `scripts/weekly_retrain.py` now raises a loud `DATA ALARM` when the week's
  signal frame is empty — the regime overlay only ever produces all-NEUTRAL
  (a populated frame), so an *empty* frame is definitionally a data bug, not a
  flat-market decision.
- `src/data/validation.py::check_index_freshness` (wired into Phase 1b) warns
  when `^NSEI`/`^INDIAVIX` lags the stock feed — the upstream cause — so the
  alignment problem is visible at ingestion, before it can wipe signals.

### 1.2 ✅ FIXED (already, in working tree) — Delisted/renamed tickers
`MCDOWELL-N.NS` and `GET&D.NS` 404 on yfinance (renamed to `UNITDSPR.NS` and
`GVT&D.NS` after corporate actions). The Colab log showing the 2 failed downloads
ran the *old committed code*; the working tree already renames both in
`src/data/ingestion.py:UNIVERSE` and `config/universe.json`, and drops the stale
alias entries. Pushing the current branch makes the next run clean.

### 1.3 ⚠️ Benign — "72 spike rows (|z|>10σ)" warning
72 rows out of 493,725 (0.015%) over 11 years × 196 names. Prices are fetched
with `auto_adjust=True`, and spikes are tagged via `spike_flag` (excluded from
`feature_cols`), not used as features. These are overwhelmingly genuine large
moves (2020 crash, circuit days, earnings gaps), not unadjusted actions. No fix
needed; leave the warning as a data-quality tripwire.

### 1.4 ✅ FIXED — Misleading "synthetic data" footer
The completion footer claimed results were on "synthetic/low-signal data" on
every run, including real-data runs — confusing. Reworded to describe a
near-efficient large-cap universe.

### 1.5 ℹ️ Expected — "Model health: IC is NaN"
First deployment with 0 resolved outcomes; rolling live IC needs ≥1 resolved
horizon of predictions. Self-resolves once `outcomes` accumulate. Not a bug.

---

## 2. The real problem: the model has ~no edge this run

These are not crashes — they are the modelling signal, and they regressed vs
prior runs:

| Metric (OOF) | This run (`v20260629`, ranker) | Prior runs (clf/reg) |
|---|---|---|
| Pooled IC | **0.0005** | 0.014–0.031 |
| Daily IC mean / IC-IR / t-stat | 0.0031 / 0.019 / **0.39** | — |
| Directional accuracy | **0.4752** (below 50%) | ~0.51–0.52 |
| Deflated Sharpe P(>0) | **0.629** (want ≥0.95) | — |
| Max drawdown | **-45.79%** (Calmar 0.391) | -24.7% |
| HPO best CV IC | 0.0212 | — |

Two things stand out and should drive the next retrain:

### 2.1 The Learning-to-Rank switch coincides with the IC collapse
`ranker_enabled` is on in the notebook (memory `ltr_ranker_2026_06_28`). This is
the first ranker run, and pooled OOF IC dropped an order of magnitude (0.02–0.03
→ 0.0005) with **below-random** directional accuracy. The walk-forward
`best_iters` are wildly unstable across folds (e.g. `[996,984,917]` vs
`[2,6,6]`), i.e. the ranker is not converging to a stable solution.

**Action — A/B the objective before trusting the ranker.** Run two retrains with
identical data/universe:
- (a) `ranker_enabled=False` → regression on `fwd_ret` (the objective Optuna
  already tunes against), and
- (b) `ranker_enabled=True` (current).
Compare **daily IC-IR and t-stat** (not pooled IC). Keep the ranker only if it
wins on IC-IR. If regression wins, revert the notebook flag.

### 2.2 Massive CV→OOF generalization gap = overfit HPO
HPO best CV IC = 0.0212 but walk-forward OOF pooled IC = 0.0005. The search is
selecting configs that fit the CV folds and don't generalize. The recent trainer
diff widened `n_estimators` to **3000** for ranking and bumped
`lambdarank_num_pair_per_sample` 8→16 — both *increase* capacity/variance and
likely worsened this gap.

**Actions:**
1. ✅ DONE — Reverted the ranking `n_estimators` Optuna ceiling 3000 → 1500
   (uniform for all tasks, with CV early stopping) and reverted
   `lambdarank_num_pair_per_sample` 16 → 8 (`trainer.py`, all 3 sites: base
   params, Optuna search space, final params).
2. ✅ DONE — The Optuna objective is now the **mean daily IC-IR** (information
   ratio of the pooled per-date cross-sectional ICs), not pooled IC
   (`_optuna_objective`, `trainer.py`). The promotion gate uses the same IC-IR
   significance floor (see §2.3). → still open: `model-improvement-plan.md`
   Phase 0.4-0.5 deeper CV.
3. ✅ DONE — `runner.py` prints a `CV→OOF IC-IR gap` line each training run
   (CV best IC-IR vs walk-forward OOF IC-IR, same scale) and raises an
   `OVERFIT ALARM` when OOF retains < 50% of CV; persisted as
   `cv_best_ic_ir` / `cv_oof_ic_ir_gap` in stats.

### 2.3 ✅ FIXED — The deploy gate was too permissive
`Min improvement req: 0.0050`, but the model deployed with OOF IC `0.0005`
purely because there was no incumbent ("No deployed model → deploy").

**Fix (`src/registry/promotion.py::evaluate_promotion`):** added absolute
quality floors enforced on *every* promotion, including the first model:
`MIN_IC_T_STAT = 2.0` (IC-IR t-stat) and `MIN_DSR = 0.95` (Deflated Sharpe).
A model below either floor cannot auto-deploy even with no champion. Floors are
only enforced when the metric was actually measured (absent ⇒ not blocked, so
fast-signals runs are unaffected). `weekly_retrain.py` now passes
`oof_ic_t_stat` and `deflated_sharpe` into the challenger dict.
Tests: `test_registry.py::test_quality_floor_blocks_first_model`,
`::test_quality_floor_blocks_against_champion`.

---

## 2b. Data-bug cleanup — status & backlog

The zero-signal incident was a *data* bug, not a model bug. Hardening the data
layer so a silent feed problem can never again silently zero out signals:

**Done this session:**
1. ✅ **Index/VIX ffill** so a missing latest index bar can't wipe the
   cross-section (`engineer.py`).
2. ✅ **Index-freshness gate** (`validation.py::check_index_freshness`, Phase 1b)
   — warns when `^NSEI`/`^INDIAVIX` lags the stock feed, or is missing entirely.
3. ✅ **predict_latest all-NaN diagnostic** — names the offending columns when a
   cross-section is dropped, instead of silently returning empty.
4. ✅ **Empty-signal DATA ALARM** in `weekly_retrain.py` — empty ≠ flat market.
5. ✅ **Delisted-ticker renames** (`UNITDSPR`, `GVT&D`) in `ingestion.py` +
   `universe.json` (working tree).

**Backlog:**
6. ✅ DONE — `validation.run_index_gates(index_df)` runs OHLCV sanity + date-gap
   checks on the index/VIX feed (wired into Phase 1b); freshness already covered
   by `check_index_freshness`.
7. ✅ DONE — `validation.check_latest_bar_coverage` warns any ticker whose latest
   bar lags the feed by > N days (the names the per-ticker `dropna` would
   silently drop from today's scoring); returns the stale list for the summary.
8. ✅ DONE (folded into 9) — spike count + per-ticker context surfaced in the
   data-quality snapshot; standalone periodic eyeball audit still a manual task.
9. ✅ DONE — `validation.summarize_data_quality` persists a per-run snapshot
   (rows, tickers, date range, index lag per series, #spikes, #stale-latest
   tickers) into `stats["data_quality"]` → run metadata, so feed degradation
   trends instead of vanishing into one log line.
10. **Survivorship bias** — the universe is still today's constituents applied
    to all history → `model-improvement-plan.md` A2 / Phase 1.6. STILL OPEN:
    needs real point-in-time NSE reconstitution history (external data); the
    largest remaining *data correctness* issue, inflating every backtest metric.

## 3. Prioritized roadmap (next 1–3 retrains)

Ordered by expected impact ÷ effort. Items marked → reference the deeper writeup
already in `docs/model-improvement-plan.md`.

**Tier 1 — do before the next retrain (no new data):**
1. Push the working-tree fixes (ticker renames + 0-signal ffill) so the next run
   is clean and actually emits signals. *(code ready)*
2. ⏳ A/B ranker vs regression on **daily IC-IR** (§2.1) — process action, must
   run on Colab. Infra is now ready: both modes optimise & report daily IC-IR,
   so the comparison is apples-to-apples. Pick the winner.
3. ✅ DONE — Reverted ranking `n_estimators` ceiling to 1500 and
   `lambdarank_num_pair_per_sample` 16 → 8 (§2.2).
4. ✅ DONE — Optuna objective switched to **mean daily IC-IR**; promotion gate
   gained absolute IC-IR t-stat ≥ 2 / Deflated Sharpe ≥ 0.95 floors (§2.2, §2.3).
5. ✅ DONE — 0-signal DATA ALARM (§1.1) and large-CV→OOF-gap OVERFIT ALARM
   (§2.2.3) guards added; the promotion floors block insignificant first-model
   deploys (§2.3).

**Tier 2 — structural edge (some new data / model work):**
6. ✅ DONE (flag-gated, default OFF) — **Market-neutral label**:
   `cfg.market_neutral_label` trains/tunes on the index-residual forward return
   (`residualize_fwd_ret`, scoped to the regression path), while reported IC and
   backtest P&L stay on REAL returns so metrics stay honest. Inert by default;
   flip on and A/B on Colab before trusting. → `model-improvement-plan.md` A6 /
   Phase 3.17. Targets the -45.79% drawdown the regime overlay only patches.
7. **India-specific features** — delivery %, FII/DII flows, sector relative
   strength, market breadth → `model-improvement-plan.md` A4 / Phase 2.10-2.14.
   STILL OPEN: needs external NSE/exchange data feeds not yet ingested. Most
   likely source of *new* IC; the current features are generic technicals.

   ✅ DONE (the subset needing **no external data**) — **classic cross-sectional
   equity factors** added to `features/engineer.py` (auto-picked-up via
   `feature_cols`): 12-1 & 6-1 momentum (`mom_12_1`, `mom_6_1`, skipping the last
   month), 52-week-high proximity (`pos_52w`), idiosyncratic vol (`idio_vol_63d`,
   market-residual), risk-adjusted momentum (`sharpe_mom_63d`), 63d return skew
   (`ret_skew_63d`), and the MAX/lottery effect (`max_ret_21d`). These are the
   most-replicated OHLCV-derivable predictors and were absent (momentum topped
   out at 63d). Also **winsorized the cross-sectional z-scores at ±5σ**
   (`_cs_zscore`) so a near-constant feature on a thin date can't produce runaway
   z-scores that dominate the trees — a mild anti-overfit robustifier. Tests:
   `tests/test_factor_features.py` (presence, finiteness, full-pipeline
   no-look-ahead, winsorization cap). **NOT yet validated for IC lift — needs a
   Colab A/B (features on vs off) on daily IC-IR before trusting.** The India/FII
   feeds above remain the open, external-data half of this item.
8. ✅ DONE — **Multi-seed bagging in the live `predict_latest` path**:
   `train_xgb_bag_no_es` / `train_xgb_bag_ranker_no_es` average `cfg.ensemble_size`
   seeds (n_seeds=1 ⇒ original single fit), matching the walk-forward OOF loop →
   `model-improvement-plan.md` Phase 3.16. Cuts the single-draw variance behind
   the run-to-run IC swings.

**Tier 3 — validation rigor (so "better" is trustworthy):**
9. Combinatorial Purged CV + block-bootstrap CI on Sharpe/maxDD
   → `model-improvement-plan.md` Phase 0.4 / 5. STILL OPEN (block-bootstrap CI
   already exists via `block_bootstrap_ci`; CPCV is the larger remaining piece).
   The -45.79% maxDD with no CI could be a median or a tail path.
10. Fix survivorship bias with real point-in-time NSE reconstitution history
    → `model-improvement-plan.md` A2 / Phase 1.6. STILL OPEN: needs external
    PIT constituent data; the Nifty200 widening did *not* fix it.

**Realistic expectation:** on a near-efficient large/mid-cap universe, a genuine,
*stable* daily IC-IR around 0.05–0.10 (t-stat > 2) and Deflated Sharpe > 0.95 is
the target. A pooled IC of 0.02–0.03 is already plausible for this universe; the
job is making it stable and significant, not chasing a big headline IC.
