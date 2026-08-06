# Weekly retrain 2026-08-06 — output analysis & improvement plan

Source: `colab_weekly.ipynb` run on 2026-08-06 (Cells 1–9) +
`/Users/adityaagrawal/Downloads/2026-08-06/candidates/v20260806/` (registry
bundle manifests/metrics for v20260806, v20260725, v20260719). Ranker mode
(`USE_RANKER=True`, `rank:ndcg`/LambdaMART). This run is the first to exercise
the fixes from `docs/weekly-retrain-2026-07-25-analysis-plan.md` (items 1-5,7
implemented per `weekly_retrain_2026_07_25_implementation` memory).

**Status as of this update:** 6 weeks of retraining against an original
8-week target. Revised target: get to a defensible paper-trading state within
**2 more retrain cycles** (2026-08-09 and 2026-08-16). §0 below reframes the
plan around that deadline — read it first.

## 0. Six-week retrospective & 2-week path to paper trading

**What's actually been fixed vs. what's still guessing.** The last three
weeks of work (2026-06-29, 2026-07-25, 2026-08-06) have all been
*infrastructure* fixes: logging, promotion gates, diagnostics, trend
surfacing. Every one of them worked correctly when tested this run (§4). None
of them have made the model itself better — pooled OOF IC is still
sign-flipping (`+0.031, -0.034, -0.001, -0.025`) and nobody has yet dug into
*why*. That's the gap between "6 weeks of retraining" and "no model good
enough to trust yet."

**New finding this pass: two of the numbers driving that pessimistic
read are themselves measured wrong**, not just noisy (§2.2, §2.3 below):

- `directional_accuracy` (printed as "OOF Directional Accuracy" and stored as
  `oof_dir_acc`) is computed as `sign(y_pred) == sign(y_true)`
  (`src/validation/metrics.py:25-29`). For the ranker, `y_pred` is a raw
  LambdaMART relevance score with no meaningful zero-crossing — its sign is
  an artifact of tree structure, not a predicted market direction. All three
  ranker runs report an eerily tight 47.5-47.7%, which looks less like "the
  model is anti-predictive" and more like this metric measuring something
  arbitrary for LTR mode. **This has been silently misreported as evidence
  of model failure for three straight runs.**
- The promotion gate's *relative* champion-vs-challenger check
  (`src/registry/promotion.py:89-107`) is supposed to prefer a Sharpe-margin
  comparison and only fall back to pooled-IC comparison when Sharpe isn't
  available. But Cell 6 of the notebook builds
  `champion={"ic": current_ic}` — it never populates the champion's
  `sharpe_net`, because `load_current_model_ic()` only ever returns pooled
  IC (`src/models/improvement.py:84-122`). So the Sharpe-margin branch is
  **dead code** — every week's decision has silently fallen through to the
  noisier pooled-IC fallback, the exact metric the code's own docstring
  (`src/validation/metrics.py:35-38`) warns "conflates cross-sectional
  stock-picking skill with time-series/market-level effects."

Put together: the headline story ("model quality trending down") is built
partly on a metric that may not mean anything (dir_acc) and a gate that's
comparing on the noisiest available axis (pooled IC) instead of the one it
was designed to prefer (Sharpe) or the one the model is actually trained
against (daily cross-sectional IC-IR, which has stayed *positive* — 0.125
t=2.70, then 0.051 t=1.11 — across the two runs it's been measured for).
**Before writing off six weeks of retrains as "no edge," fix the measurement
and re-read the same historical runs.** This is code-only work, doesn't need
a Colab retrain, and can be done today.

### 2-week execution plan

**This week, before the 2026-08-09 run (code-only, no retrain needed):**
1. Fix or clearly disable `oof_dir_acc` reporting for ranker mode (§2.2) —
   don't let a meaningless number keep reading as "model is worse than a
   coin flip."
2. Extend `load_current_model_ic` (or add a sibling function) to return the
   champion's full metrics dict — `sharpe_net`, `ic_t_stat`, `deflated_sharpe`
   — not just pooled `ic`, and wire it into Cell 6's `champion=` argument so
   the gate's primary comparison is Sharpe-margin, as designed (§2.3).
3. Fix `SUPABASE_URL` DNS resolution (§2.4) — needed so the multi-run trend
   and dashboard are reading live data for the final two decision-making
   weeks, not stale JSON fallback.
4. Triage the 72 recurring spike rows in `outputs/spike_rows_report.csv`
   (§2.6) — if they're genuine unadjusted corporate actions, they're
   injecting noise into every one of the last 4 runs' training data and
   metrics, which would itself partially explain instability.

**2026-08-09 run:** first retrain under the corrected metrics/gate. Compare
the corrected Sharpe-margin decision against what the old pooled-IC-margin
decision would have said for the same challenger — if they disagree, that's
direct confirmation the old gate was mis-deciding.

**2026-08-16 run:** second and last retrain inside the 2-week window.

**Go/no-go for paper trading, decided after the 2026-08-16 run:**
- **Go, at current size:** if either of the last two runs clears the
  existing absolute floors (IC-IR t-stat ≥ 2.0, deflated Sharpe ≥ 0.95) *and*
  the corrected champion comparison agrees — treat that model as the
  standing champion and let paper trading (already running via
  `paper_trade=True` in Cells 5/8) continue at current sizing
  (`max_positions=10`, ₹1,000,000 capital).
- **Go, at reduced size:** if daily IC-IR stays positive with t-stat in the
  1.0-2.0 range (i.e. "weak but real edge" territory, matching this run and
  2026-07-25) but never clears 2.0 — don't keep chasing a higher bar
  indefinitely. Ship the best-available model for paper trading but cut
  `max_positions` and/or per-name budget so a wrong signal costs less, and
  treat the next several weeks of *actual* paper-trade outcomes (resolved
  via Cell 4, not backtest OOF) as the real validation data.
- **No-go / escalate:** only if daily IC-IR turns negative or t-stat drops
  below ~1.0 on *both* remaining runs — that would be the first real
  evidence (as opposed to a pooled-IC artifact) that the ranker approach
  itself isn't working, and would justify an A/B run with `USE_RANKER=False`
  (the frozen classifier baseline already built into Cell 1) before
  extending the timeline further.

This makes "paper trading" a scheduled decision with explicit criteria at a
fixed date, instead of an open-ended bar that moves every time a retrain
comes back mixed.

## 1. What happened this run

| Run | oof_ic (pooled) | oof_dir_acc* | oof daily IC-IR (t-stat) | bt_sharpe | bt_cagr | deployed |
|---|---|---|---|---|---|---|
| v20260621 | 0.0308 | 54.7% | — | 0.555 | 5.95% | no |
| v20260719 | -0.0343 | 47.7% | — | 0.956 | 18.82% | no (superseded) |
| v20260725 | -0.0010 | 47.5% | 0.125 (2.70) | 1.159 | 23.23% | yes (was current) |
| v20260806 | -0.0253 | 47.5% | 0.051 (1.11) | 0.98 | 19.5% | **no — rejected** |

\* See §0/§2.2 — `oof_dir_acc` is likely not a meaningful metric for the
ranker (sign of a raw `rank:ndcg` score vs. sign of actual return). Treat
the tight 47.5-47.7% clustering across all three ranker runs as suspect, not
as three independent confirmations of "worse than random."

- Multi-run IC trend (Cell 6, pooled IC): `+0.0308 -> -0.0343 -> -0.0010 ->
  -0.0253 (1/4 positive)`. Daily IC-IR trend (only measured for the last two
  runs so far): `0.125 (t=2.70) -> 0.051 (t=1.11)` — down, but both positive,
  a materially different story than the pooled-IC trend line alone.
- **Promotion gate rejected the challenger** on the absolute floors (IC-IR
  t-stat 1.11 < 2.0; deflated Sharpe 0.759 < 0.95) — those floors are correct
  and daily-metric-based, so this part of the decision stands regardless of
  §0's findings. The *relative* margin check ("IC -0.0253 < champion
  -0.0010+0.005") is the part that used the wrong (pooled, Sharpe-less) axis;
  it happened not to change the outcome this run since the floors already
  failed, but it will matter on a run where the floors pass and the relative
  check is the tiebreaker.
- Zero LONG signals again this week, self-explanatory via the regime-filter
  log line: `regime_filter suppressed 31 candidate LONGs
  (nifty_dist_sma200=-0.0059)` + Cell 9 banner (§2.1 fix from last week,
  confirmed working).
- 72 spike rows again, dumped to `outputs/spike_rows_report.csv`, still not
  triaged — same count as 2026-07-25.
- HPO ceiling-hit diagnostic: `320/1413 bag fits (22.6%) landed at/near the
  search-space ceiling (n_estimators=514)` — but the *optimal* n_estimators
  this run (514) is well under the 1500 Optuna cap; folds are hitting their
  own early-stopped optimum near ~500, not straining the search bound. No
  action needed (§2.5).
- Supabase: client instantiates (`Supabase connected` prints) but **every**
  `fetch_rows`/`upsert_rows` call failed with `[Errno -2] Name or service not
  known` — a DNS failure, confirmed persistent across two full runs now
  (§2.4).

## 2. Diagnosis, ranked by impact

### 2.1 (superseded by §0) Model quality trending down
Originally the top-ranked concern from the previous pass at this doc. Still
true at face value, but §0 above found two concrete measurement problems
(dir_acc metric validity, promotion-gate axis) that need fixing *before*
concluding the model itself is the problem. Keeping the historical framing
here for continuity; treat §0's plan as the current priority order.

### 2.2 `oof_dir_acc` is likely not a valid metric for ranker mode
`directional_accuracy()` (`src/validation/metrics.py:25-29`) computes
`mean(sign(y_pred) == sign(y_true))`. For `binary:logistic`, `y_pred` is a
P(up) probability with a natural 0.5 (or after centering, 0) threshold, so
this is meaningful. For `rank:ndcg`/LambdaMART, `y_pred` is a raw relevance
margin trained only to preserve *within-day ordering* — XGBoost gives no
guarantee about where that score sits relative to zero, and nothing in the
training objective ties its sign to predicted return direction. All three
ranker runs land at 47.5-47.7%, suspiciously tighter than the pooled-IC
numbers' spread — consistent with this being close to some fixed artifact
(e.g. the base rate of `sign(raw_score)==+1` combined with the base rate of
positive `fwd_ret`) rather than three independent measurements of skill.

- **Fix:** for `ranker_enabled=True` runs, either (a) stop computing/printing
  `oof_dir_acc` and rely on daily IC-IR (which *is* scale/threshold
  invariant — Spearman correlation doesn't care about the raw score's sign
  or magnitude) as the directional-quality metric, or (b) redefine a
  ranker-appropriate version: e.g. per day, does the ticker predicted #1
  actually land in the top half of realized returns that day (a "top-1
  hit-rate" or similar rank-based accuracy). Do not keep reporting the
  current number unlabeled — it has been read as "model failure" for three
  runs running.

### 2.3 Promotion gate's Sharpe-margin path is dead code
`evaluate_promotion()` prefers comparing `sharpe_net` when both challenger
and champion have it, falling back to pooled `ic` only if Sharpe is missing
(`src/registry/promotion.py:96-107`). Cell 6 always calls it with
`champion=None if math.isnan(current_ic) else {"ic": current_ic}` — champion
never has `sharpe_net`, `ic_t_stat`, or `deflated_sharpe`, because
`load_current_model_ic()` (`src/models/improvement.py:84-122`) only reads and
returns the pooled `oof_ic` field. Every week so far, the relative check has
silently used the pooled-IC fallback — the metric the code's own comment
(`src/validation/metrics.py:35-38`) says conflates stock-picking skill with
market-level drift. This didn't change this run's outcome (the absolute
floors already failed) but will on a run where the floors pass and the
relative check breaks the tie.

- **Fix:** extend `load_current_model_ic` (or add
  `load_current_model_metrics`) to pull the deployed model's full metrics row
  (`sharpe_net`, `ic_t_stat`, `deflated_sharpe`, `ic`) from `model_runs`
  Supabase/JSON — these are already stored per-run (see e.g.
  `registry/bundles/model_v20260725/metrics.json`), just not read back out.
  Wire the full dict into Cell 6's `champion=` argument so the gate compares
  Sharpe-margin first, as it was designed to.

### 2.4 Supabase connectivity is still 100% broken — two weeks straight, exact same error
`[Errno -2] Name or service not known` is a DNS lookup failure, not
transient (100% of ~8 calls across ~3h wall-clock, two runs running).
Consequences: multi-run IC trend and champion-metrics lookups (§2.3's fix)
both degrade to local-JSON fallback, and the Streamlit dashboard is stale.

- **Fix:** verify `SUPABASE_URL` in Colab Secrets against the live Supabase
  project (a paused/rotated free-tier project is the most common cause of a
  dead hostname). Test with
  `!python -c "import socket; socket.gethostbyname('<host>')"` in a scratch
  Colab cell before the 2026-08-09 run.

### 2.5 n_estimators ceiling — no action, diagnostic confirms prior decision
This run's optimal `n_estimators=514` is well under the 1500 cap; the 22.6%
ceiling-hit rate reflects folds converging near their own ~500-513 optimum,
not the search bound being starved. Leave as-is per the 2026-06-29 decision.

### 2.6 Spike rows and dynamic RR — still open, unchanged from last week
- 72 spike rows, un-triaged, `outputs/spike_rows_report.csv` exists but
  nobody has opened it. Given the 2-week deadline, this now doubles as a
  possible explanation for retrain instability, not just a hygiene item —
  worth doing this week (§0).
- `dynamic_horizon_enabled` still `False`, not evaluated. Out of scope for
  the 2-week paper-trading push — revisit after go/no-go.

## 3. Prioritized next steps

1. **(This week, code-only)** Fix/relabel `oof_dir_acc` for ranker mode
   (§2.2).
2. **(This week, code-only)** Wire the champion's full metrics (not just
   pooled IC) into the promotion gate so the Sharpe-margin comparison
   actually runs (§2.3).
3. **(This week, operational)** Fix `SUPABASE_URL` DNS resolution (§2.4).
4. **(This week, data)** Triage the 72 spike rows — genuine corporate
   actions vs. data errors (§2.6).
5. **(2026-08-09 run)** Retrain under the corrected metrics/gate; compare
   old-vs-new gate decision on the same challenger.
6. **(2026-08-16 run)** Second and final retrain inside the 2-week window.
7. **(2026-08-16, decision)** Apply the go/no-go criteria in §0 and commit to
   a paper-trading configuration (full size, reduced size, or — only if both
   remaining runs show negative/sub-1.0-t-stat daily IC-IR — escalate to a
   classifier-mode A/B before extending further).
8. **(Deferred past the 2-week window)** `dynamic_horizon_enabled` A/B run.

## 4. What's *not* broken

- All five infrastructure fixes from the 2026-07-25 plan (regime-filter
  logging, IC-floor promotion gate, spike-row report, HPO ceiling diagnostic,
  loud-Supabase-failure warnings) verified working correctly in this run.
- The promotion gate's **absolute floors** did exactly what they were built
  to do: reject a statistically-insignificant challenger. (It's the
  *relative* comparison path that needs the §2.3 fix — the floors are fine.)
- Walk-forward/backtest/paper-trading machinery ran end-to-end cleanly (471
  folds, 7955s ≈ 2.2h — down from 4.3h two runs ago, consistent with the
  `max_parallel_fits` change already in place).
- Data ingestion (196 tickers, 499k rows) and feature engineering (80
  features) both completed cleanly, same as last run.
- Paper trading is already live and mechanically working (Cells 5/8,
  `paper_trade=True`) — 6 closed trades, 50% win rate, +0.4% return this run.
  The gap to close in the next 2 weeks is *confidence in the signal*, not
  standing up paper trading itself, which already exists.
