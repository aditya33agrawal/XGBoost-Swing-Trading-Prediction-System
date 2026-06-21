# Implementation Plan — Nifty Swing Bot with Weekly Colab Retraining & Mistake-Learning Loop

> Companion to `nifty50_swing_bot_plan.md` (data, features, validation, costs). This document is the **build plan**: the runtime architecture, the weekly Google Colab training job, model persistence, and the feedback loop that retrains the model on its own resolved predictions so it learns from mistakes.

---

## 0. Decisions locked in

- **Market:** Equity **cash/delivery** (no leverage), long-only swing, 3–10 day holds, universe = ~100–200 most-liquid NSE names (Nifty 100/200 subset), point-in-time membership.
- **Prediction:** **Triple-barrier directional classification** → per-stock calibrated `P(take-profit hit before stop-loss within H days)`, plus a parallel quantile return-regression for sizing. The prediction *is* the trade spec (entry, TP, SL, max-hold).
- **Decision = cross-sectional ranking:** trade the top-K names whose calibrated probability clears the cost-adjusted threshold.
- **Compliance (live now):** personal-account algo, < 10 orders/sec ⇒ no exchange registration; broker handles Algo-ID tagging. Requires SEBI-compliant broker API, static IP, OAuth + 2FA, daily session re-auth.

---

## 1. Three-layer architecture

The system splits compute by cadence and constraints. **Colab is ephemeral and can't legally place orders or hold a static IP all day**, so it is the *training brain* only. A small always-on VM is the *hands* that trade. Google Drive is the *shared memory* both read/write.

```
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 3 — SHARED STATE  (Google Drive, or GCS)                        │
│   /data_lake      curated EOD prices + features (Parquet, by date)    │
│   /ledger         prediction–outcome ledger (the "memory")            │
│   /registry       versioned model bundles + manifests                 │
│   /config         universe, thresholds, hyperparams (versioned)       │
│   /reports        weekly eval + drift reports                         │
└───────────────▲───────────────────────────────────────▲──────────────┘
                │ read/write                              │ read/write
┌───────────────┴───────────────┐         ┌───────────────┴──────────────┐
│ LAYER 1 — DAILY (always-on VM) │         │ LAYER 2 — WEEKLY (Colab)      │
│ static IP, broker API, 2FA     │         │ Google Colab notebook         │
│  • ingest EOD (post-close)     │         │  • pull data_lake + ledger    │
│  • load latest PROD model      │         │  • resolve outcomes, relabel  │
│  • score → rank → risk engine  │         │  • retrain challenger         │
│  • place buy / SL / TP orders  │         │  • champion vs challenger     │
│  • log every prediction→ledger │         │  • save bundle → registry     │
│  • monitor, kill-switch        │         │  • write eval/drift report    │
└────────────────────────────────┘         └───────────────────────────────┘
        runs daily ~15:40 IST                   runs weekly (e.g. Sun)
```

**Why this split works:** training is weekly and tiny (XGBoost on EOD data for ~200 names is seconds–minutes on CPU; no GPU needed), so a once-a-week Colab run is sufficient. Inference/trading must be reliable, IP-pinned, and broker-routed → that lives on the VM, never on Colab.

---

## 2. Google Drive layout (the shared memory)

```
gdrive/swingbot/
├── config/
│   ├── universe.json              # point-in-time tradable symbols + membership windows
│   ├── strategy.yaml              # H, barriers, K, thresholds, risk caps
│   └── hyperparams.json           # last good XGBoost params
├── data_lake/
│   ├── prices/year=2026/month=06/*.parquet     # curated, adjusted OHLCV+delivery
│   └── features/asof=2026-06-19.parquet         # daily feature snapshots
├── ledger/
│   └── predictions.parquet        # APPEND-ONLY prediction→outcome record
├── registry/
│   ├── prod/manifest.json         # pointer to current production bundle
│   ├── bundles/model_2026-06-15/  # model.json, calibrator.pkl, manifest.json, metrics.json
│   └── bundles/model_2026-06-22/
└── reports/
    └── weekly_2026-06-22.html
```

Drive is fine to start (free, mounts in Colab, syncs to the VM via `rclone`). Swap `/data_lake` and `/registry` to **GCS/S3** when you scale — same layout.

---

## 3. The prediction–outcome ledger (this is what enables learning)

Every prediction the bot makes is written immediately, **before** the outcome is known. The horizon (`H` days) later, the outcome resolves and the same row is updated. This append-only ledger is the ground truth the weekly job learns from.

Schema (`ledger/predictions.parquet`):

```
pred_id            # uuid
asof_date          # date features were computed (decision date)
symbol
model_version      # which bundle produced this
features_hash      # exact feature-set fingerprint
prob_tp_first      # calibrated P(TP before SL within H)   [the prediction]
pred_return_q50    # quantile-reg point estimate (sizing)
entry_planned      # intended entry, TP, SL, max_hold_date
traded             # bool: did the bot actually take it?
fill_price, size   # if traded
--- resolved later (after H days) ---
realized_label     # +1 TP-first / -1 SL-first / 0 vertical   [the truth]
realized_return_H
exit_price, pnl, costs
resolved           # bool
error              # |prob_tp_first - (realized_label==1)|    [mistake magnitude]
```

`error` and `realized_label` are exactly the "predictive mistakes" the retraining loop consumes.

---

## 4. Daily loop (Layer 1, VM) — abbreviated

Post-close each trading day:

1. Ingest EOD (broker API + bhavcopy), validate, append to `data_lake/prices`.
2. Recompute features for today → `data_lake/features`.
3. **Resolve due predictions:** any ledger row whose `max_hold_date <= today` gets its `realized_label`, return, pnl, and `error` filled in.
4. Load **prod** model bundle from `registry/prod`. Score universe → calibrated probabilities → cross-sectional rank.
5. Risk engine gates orders (per-trade risk, caps, kill-switch, regime filter).
6. Place buys + exchange-resident SL and TP brackets via broker API.
7. **Append new predictions to the ledger** (resolved=false).
8. Manage open positions, reconcile fills, alert.

(Full feature list, cost model, and risk rules are in the companion plan.)

---

## 5. Weekly Colab training job (Layer 2) — the core of this request

A single notebook, run weekly. Treat each numbered block as one or more cells.

### 5.1 Notebook skeleton

```python
# ── Cell 1: mount + deps ───────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/swingbot'
!pip -q install xgboost optuna pyarrow scikit-learn pandas pandas-ta

# ── Cell 2: load shared state ──────────────────────────────────────────
import pandas as pd, json, glob
cfg     = json.load(open(f'{ROOT}/config/strategy.yaml'))   # via yaml.safe_load in practice
prices  = pd.read_parquet(f'{ROOT}/data_lake/prices')
ledger  = pd.read_parquet(f'{ROOT}/ledger/predictions.parquet')
H       = cfg['horizon']

# ── Cell 3: resolve + assemble training set (learn from mistakes) ──────
# Only rows old enough to have a settled outcome are labelable.
train_df = build_training_frame(prices, asof_cutoff=today - H)   # features X
labels   = triple_barrier_labels(...)                            # y from realized future
weights  = uniqueness_weight(labels) * time_decay(asof) * error_weight(ledger)  # §6

# ── Cell 4: walk-forward CV + (monthly) Optuna tune ───────────────────
splits = PurgedWalkForward(n_splits=6, embargo=H, label_h=H)
params = json.load(open(f'{ROOT}/config/hyperparams.json'))      # reuse weekly
# every 4th week: re-run Optuna to refresh params, then persist

# ── Cell 5: train challenger + calibrate ──────────────────────────────
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
booster = xgb.XGBClassifier(**params)
booster.fit(X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
calib = IsotonicRegression(out_of_bounds='clip').fit(
            booster.predict_proba(X_val)[:,1], y_val)

# ── Cell 6: evaluate on most-recent OOS window ────────────────────────
metrics = evaluate(booster, calib, X_oos, y_oos, costs=cfg['costs'])
#  -> IC, hit-rate, calibration error, cost-adjusted Sharpe of the signal

# ── Cell 7: champion/challenger gate ──────────────────────────────────
champ = load_bundle(f'{ROOT}/registry/prod')
promote = (metrics['sharpe_net'] > champ.metrics['sharpe_net'] + MARGIN
           and metrics['calib_error'] < TOL
           and not drift_alarm)

# ── Cell 8: persist bundle + manifest; flip prod pointer if promoted ──
bundle_dir = save_bundle(ROOT, booster, calib, params, metrics, features_used)
if promote:
    set_prod_pointer(ROOT, bundle_dir)         # shadow first in practice
write_weekly_report(ROOT, metrics, drift_stats)
```

### 5.2 What runs weekly vs monthly

- **Weekly:** resolve outcomes, rebuild labels, retrain on expanded data, recalibrate, evaluate, champion/challenger, persist.
- **Monthly:** re-run Optuna hyperparameter search (expensive; weekly reuse of last-good params avoids overfitting to one week of new data).
- **Never:** retrain daily (chases noise) or retrain on any window overlapping the period about to be traded (leakage).

---

## 6. Learning from predictive mistakes — the mechanism

Three reinforcing mechanisms, in priority order:

**(a) Expanded-window retraining (primary).** Each week the freshly *resolved* outcomes — including every name the model got wrong — are appended to the training set. Retraining on the expanding window with **time-decay weights** means the model continually re-fits to the recent regime, automatically absorbing where it has been failing. This is the main, principled "learn from mistakes."

**(b) Error-aware sample weighting (secondary, guarded).** Optionally up-weight recently mis-predicted samples so the model attends to its blind spots:
```python
def error_weight(ledger, cap=2.0, lookback_days=90):
    recent = ledger[ledger.resolved & (ledger.asof_date > today-lookback_days)]
    # map symbol/regime error rate → weight in [1, cap]; CAPPED to avoid chasing noise
    return 1.0 + (cap-1.0) * recent_error_rate_normalized
```
Cap the multiplier (≤2×) — uncapped error-chasing overfits to noise and outliers. Use this only if it improves OOS metrics, not by default.

**(c) Continuous recalibration.** Refit the isotonic calibrator weekly on the most recent resolved outcomes so `prob_tp_first` stays honest as regimes shift — miscalibration is itself a correctable "mistake" that breaks sizing/thresholds.

**Error analysis report (weekly):** segment `error` by sector, volatility regime, day-of-week, and feature buckets. Persistent error clusters are a signal to add a feature — the human-in-the-loop part of learning. Don't auto-add features; review the report.

**Guardrail:** every "learned" model must still clear the champion/challenger gate (§7) on *out-of-sample* data. Learning from mistakes only ships if it demonstrably improves performance — otherwise the champion stays.

---

## 7. Model persistence — bundle + manifest

A model is never a single file. Save a **bundle** so any prediction is fully reproducible.

```
registry/bundles/model_2026-06-22/
├── model.json          # xgboost booster (use save_model, NOT pickle — version-safe)
├── calibrator.pkl      # isotonic calibrator
├── quantile_reg.json   # sizing model
├── features.json       # ordered feature list + per-feature definition hashes
├── manifest.json       # see below
└── metrics.json        # OOS IC, hit-rate, calib error, net Sharpe, drawdown
```

`manifest.json`:
```json
{
  "model_version": "2026-06-22",
  "created_utc": "2026-06-22T08:14:00Z",
  "train_window": {"start": "2010-01-01", "end": "2026-06-15"},
  "embargo_days": 5, "horizon_days": 5,
  "universe_asof": "config/universe.json@<hash>",
  "features_hash": "a3f9c1...",
  "hyperparams": { "...": "..." },
  "xgboost_version": "3.x", "code_git_sha": "<sha>",
  "promoted": true, "champion_replaced": "2026-06-15",
  "metrics": {"ic": 0.041, "hit_rate": 0.546, "sharpe_net": 1.1, "calib_err": 0.03}
}
```

Save / load helpers:
```python
def save_bundle(root, booster, calib, params, metrics, features):
    d = f"{root}/registry/bundles/model_{TODAY}"
    os.makedirs(d, exist_ok=True)
    booster.save_model(f"{d}/model.json")            # portable, version-robust
    joblib.dump(calib, f"{d}/calibrator.pkl")
    json.dump(features, open(f"{d}/features.json","w"))
    json.dump(manifest(params, metrics, features), open(f"{d}/manifest.json","w"))
    return d

def set_prod_pointer(root, bundle_dir):              # atomic promote
    json.dump({"prod_bundle": bundle_dir, "ts": now()},
              open(f"{root}/registry/prod/manifest.json","w"))
```

**Rules:** use XGBoost's native `save_model('*.json')`, *not* pickle (pickle breaks across library versions — a classic production failure). Pin `xgboost` version in the manifest. Keep the last N bundles for instant rollback. The VM's daily loop reads `registry/prod/manifest.json` to know which bundle to load.

---

## 8. Champion / challenger + rollback

- New model trains as **challenger**, evaluated on an OOS window the optimizer never saw.
- Promote to **prod** only if it beats champion on cost-adjusted Sharpe by a margin AND passes calibration + drift checks.
- **Shadow period:** run challenger in parallel (predict, log to ledger, don't trade) for a few sessions; promote only if live shadow tracks backtest.
- **Auto-rollback:** if live prod metrics breach guardrails (drawdown, hit-rate collapse), flip `prod` pointer back to the previous bundle and demote the bot to flat.

---

## 9. Drift monitoring & off-cycle retrain

Computed in the weekly job and logged to `/reports`:
- **Feature drift:** PSI / KS of each feature vs the training reference; PSI > 0.25 ⇒ retrain trigger.
- **Concept drift:** rolling live IC / hit-rate from the ledger vs backtest expectation; CUSUM on the prediction-error stream.
- **Calibration drift:** reliability curve from recent resolved outcomes.
- A sharp breach triggers an **off-cycle** Colab run or auto-demotes prod to flat until reviewed.

---

## 10. Automating the weekly Colab run (honest options)

Colab free tier **cannot be cleanly scheduled headless** (sessions are interactive, time-limited, and disconnect when idle). Realistic choices, best to worst for reliability:

1. **Manual weekly run (start here).** Weekly cadence makes this perfectly viable — open the notebook every Sunday, Run-all, done in minutes. Zero infra. Recommended for Phases A–B.
2. **`papermill` + a trigger.** Parameterize the notebook; execute it programmatically. Still needs something to launch it.
3. **Move training to GitHub Actions (recommended once live).** Convert the notebook to a `train.py` and run it on a **GitHub Actions scheduled workflow** (free cron, reliable, headless). It reads/writes the *same* Drive/GCS layout. This is the robust production answer — "Colab for development, Actions for the recurring job." Keep the Colab notebook as the interactive research/debug surface.
4. **Cloud Scheduler → Cloud Run / Cloud Function** if you want it fully on GCP alongside GCS storage.

> Recommendation: develop and validate the training notebook on **Colab**; for the recurring weekly job in production, run the *same code* via **GitHub Actions cron** for reliability. Both persist to the identical Drive/GCS registry, so nothing else changes.

---

## 11. Phased rollout with go/no-go gates

| Phase | What | Gate to advance |
|---|---|---|
| **A — Backtest** | walk-forward, full Indian cost model, survivorship-safe | positive *deflated*, cost-adjusted Sharpe across multiple regimes; stable to small param changes |
| **B — Paper (2–3 mo)** | exact prod code path, simulated/broker-sandbox fills, ledger live | live IC/hit-rate within band of backtest; zero ops failures; SL/TP + kill-switch verified |
| **C — Live tiny (₹25–50k)** | real fills, same code | 2–3 mo live tracking paper; drawdown within tolerance; no unexplained behavior |
| **D — Scale** | step capital up gradually | edge survives larger size; scale across *more names* before *bigger per name* |

Weekly Colab retraining + the ledger feedback loop run through **all** phases — the model is learning from resolved outcomes from day one of paper trading.

---

## 12. Tech stack

`Python 3.11` · `xgboost` (native JSON save) · `scikit-learn` (isotonic calibration) · `optuna` · `pandas`/`polars` · `pandas-ta` · `pyarrow`/Parquet · **Google Colab** (dev/training) + **Google Drive** (registry & data lake) → GCS/S3 at scale · **GitHub Actions** (recurring weekly job) · broker API (Kite Connect / Upstox / Dhan) · `rclone` (VM↔Drive sync) · Telegram/email alerts · `great-expectations` (data gates).

---

## 13. Build sequence (checklist)

1. Drive layout + `config/` files + universe (point-in-time).
2. Data ingestion + validation → `data_lake` (companion plan §3–5).
3. Feature + triple-barrier label functions; leakage test suite.
4. **Colab training notebook** (§5) producing a saved bundle + manifest.
5. Backtest harness (walk-forward, costs) — Phase A gate.
6. VM daily loop: load prod bundle → score → rank → risk engine → broker orders → **ledger logging** → resolve outcomes.
7. Champion/challenger + rollback + drift report.
8. Paper trade 2–3 months; weekly retrain learning from the ledger.
9. Port weekly job to GitHub Actions cron.
10. Live tiny → scale, gated.

---

## 14. Pitfalls specific to this setup

- **Pickle'd models break across `xgboost` versions** → always `save_model('*.json')` + pin version.
- **Colab is ephemeral** → never store the only copy of data/models on the Colab VM; everything lives in Drive/GCS.
- **Relabeling leakage** → only label rows older than `H`; never retrain on the about-to-be-traded window.
- **Error-chasing** → cap error-weights; require OOS improvement before promotion.
- **Ledger gaps** → if the daily loop misses resolving outcomes, the weekly learning degrades silently; monitor ledger completeness.
- **Colab can't trade** → it has no static IP / broker session; keep all order placement on the compliant VM.

---

*Engineering plan only — not investment advice or a promise of returns. SEBI's own data shows the large majority of retail algo/F&O traders lose money; the backtest-and-paper gauntlet exists to tell you honestly whether you have an edge before risking real capital, and "no edge, don't deploy" is a valid outcome. Verify current SEBI/broker/tax specifics against official sources before going live.*
