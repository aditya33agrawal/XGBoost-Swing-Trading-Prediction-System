# Dynamic Horizon & Risk-Reward Plan — Learning *When* to Hold and *How Much* to Risk

**Goal.** Today the model answers exactly one question: *"is this name in the bullish tail of the cross-section right now?"* It then hands that name to a fixed, hand-set holding period (5 trading days) and a fixed risk-reward (1.5×ATR stop / 3.0×ATR target). This plan adds two more learned outputs so the model also answers:

1. **How long should I hold this position to maximise the expected pay-off?** (a horizon from a few days to several months, chosen *per name, per date*).
2. **Where do I put the stop and the target?** (a *dynamic* risk-reward ratio, chosen per name from the model's own view of the up/down skew, instead of one global 2:1).

**Thesis.** Selection skill (the IC) and *exit* skill (horizon + RR) are different sources of edge. A name can be a genuine top-quintile pick and still be the wrong trade if its edge needs 40 days to play out and you exit at day 5, or if its realistic downside is 1.2×ATR and you risk 1.5×. Letting the model size *time* and *risk* to the opportunity attacks the two weakest numbers in the current backtest — **Calmar 0.218** and **max-drawdown −24.7% against a 5.4% CAGR** — without needing the directional IC to improve at all. It also directly cuts turnover (positions that need months stop being churned every 5 days) and improves capital efficiency (capital isn't parked in a name whose edge already decayed).

**Read `docs/model-improvement-plan.md` first.** This plan *depends* on that plan's **Phase 0** (daily IC-IR, deflated Sharpe, block-bootstrap CI). Without a trustworthy ruler, a dynamic-horizon/RR change will look like an improvement on noise. Do Phase 0 before any of this.

---

## 0. What the code does today (grounded, not generic)

| Concern | Where it lives | Current behaviour |
|---|---|---|
| Horizon | `cfg.horizon=5` (`config.py:27`) | One global constant. Drives the label (`forward_log_return`/`triple_barrier_labels`, `targets.py:19,34`), the purge/embargo (`runner.py:147,164`), the backtest exit (every name held exactly `horizon` days, implicit in the `fwd_ret` column), and the signal metadata (`signals.py:101`). **The model never predicts it.** |
| Risk-reward | `signal_stop_atr_mult=1.5`, `signal_target_atr_mult=3.0` (`config.py:36-37`) | A fixed ≈2:1 applied to *every* signal (`signals.py:89-97`). Same multiples in a calm mega-cap and a volatile mid-cap. **The model never predicts it.** |
| Label | `triple_barrier_labels` (`targets.py:34`) | Symmetric ±2×ATR barriers, vertical barrier at `h=5`. Produces a single `{+1,-1,0}` (or scalar `fwd_ret`) target. |
| Model output | `train_xgb`/`train_xgb_bag` (`trainer.py:125,164`) | A **single scalar score** per (date, ticker). No horizon head, no risk head, no distribution. |
| Backtest | `run_backtest` (`engine.py:41`) | Synchronous quintile rebalance every `rebalance_every=5` days; books the realised `fwd_ret` of the top quintile. **Never simulates a stop or a target** — so a dynamic RR is currently *invisible* to the backtest. |
| Exit (paper/live) | `PaperPortfolio.update` (`paper_trader.py:327-361`) | The *only* place stop/target/expiry actually fire: `price>=target → "target"`, `price<=stop → "stop"`, `days>=horizon_days → "expired"`. Already per-trade, so it can consume per-name horizon/stop/target with almost no change. |

Two structural facts fall out of this table and shape the whole plan:

- **A1.** The label, the loss, and the backtest are all welded to a single horizon. Predicting horizon is not a feature you bolt on — it requires labels at *multiple* horizons and a model output that spans them.
- **A2.** The backtest evaluates *selection* (`fwd_ret` of a quintile) but **not exits**. The live trader evaluates *exits* (stop/target/expiry) but isn't what you tune against. They already disagree. Dynamic RR widens that gap to the point where the current `run_backtest` literally cannot score the change — an **event-driven backtest that mirrors `paper_trader.update` becomes a hard dependency**, not a nice-to-have (Phase 4).

---

## 1. The core reframing

Stop thinking of the model output as a *point* (one score at one horizon) and start thinking of it as a **conditional return surface**: for this name, on this date, what does the distribution of forward return look like as a function of holding horizon?

If you can estimate, per (date, ticker):

```
        horizon h →   5d    10d   21d   42d   63d   126d
  q10 (downside)    -2.1% -2.8% -3.9% -5.1% -6.0% -8.2%
  q50 (median)      +0.4% +0.9% +1.8% +2.9% +3.1% +2.7%   ← edge peaks at 63d then decays
  q90 (upside)      +3.0% +4.4% +6.5% +9.1% +10.0% +11.5%
```

then **all three decisions fall out of one object**:

- **Selection score** = the risk-adjusted edge at the chosen horizon (replaces today's raw `pred`), still ranked cross-sectionally into quintiles.
- **Optimal horizon** `h*` = the horizon that maximises predicted risk-adjusted edge net of time-scaled cost — literally the peak of the predicted Sharpe term structure (here ≈63d).
- **Dynamic RR** at `h*`: stop ∝ predicted downside `|q10_{h*}|`, target ∝ predicted upside `q90_{h*}`. The ratio `q90/|q10|` *is* the risk-reward, and it's now asymmetry-aware and regime-aware instead of a hardcoded 2.0.

This is the **recommended primary design (Design A)** because one coherent, learnable object yields selection + horizon + RR simultaneously, it's fully consistent with the repo's existing `fwd_ret` regression mode and its López-de-Prado discipline, and every piece is independently inspectable (you can plot the predicted surface for any name and sanity-check it).

### Alternative framings (know them, don't lead with them)

- **Design B — direct multi-task heads.** One model, three outputs: the usual score, a regression onto the *realised* optimal horizon, and a regression onto the realised optimal barrier width. *Problem:* the horizon/barrier targets are themselves look-ahead argmaxes — noisy, discontinuous, and easy to overfit. Useful as a **cheap cross-check** on Design A's `h*`, not as the primary.
- **Design C — first-passage / survival.** Model the hazard of hitting the up-barrier vs the down-barrier over time; horizon emerges from the survival curve. The most theoretically correct treatment of "time to outcome," but heavier to build and validate. Hold as a Phase-7 upgrade if Design A's horizon selection proves valuable but coarse.
- **Design D — meta-labelling a barrier grid.** Sweep `(up_mult, dn_mult, h)` historically, label the config that maximised pay-off, train a model to pick it. The most *direct* "learn the RR," and the **highest overfitting risk in the whole plan** — selecting the historically best barrier per regime is a textbook backtest overfit. Only ever as a *constrained refinement* on top of Design A, under the same purged walk-forward, never standalone.

---

## 2. Phased plan

### Phase 0 — Prerequisites (do not skip)
- Land **`model-improvement-plan.md` Phase 0** (daily IC-IR, deflated Sharpe, block-bootstrap CI, `robustness.py`). Every phase below must report a confidence interval, because horizon/RR changes are exactly the kind of thing that produces a flattering point estimate on one walk-forward path.
- Add a **frozen baseline row** to the report harness: the current fixed-horizon/fixed-RR numbers (IC 0.0193, Sharpe 0.509, CAGR 5.4%, maxDD −24.74%). Every change here is reported as a delta against it.

### Phase 1 — Multi-horizon labels (the foundation; no new data, but real costs)
1. **Horizon grid.** `cfg.horizon_grid = [5, 10, 21, 42, 63, 126]` (≈1wk → 6mo). Keep `cfg.horizon=5` as the legacy/default so nothing breaks.
2. **Multi-horizon forward returns.** Extend `targets.py`: `forward_log_return_grid(df, grid)` → adds `fwd_ret_5, fwd_ret_10, …, fwd_ret_126`. Vectorised; the last `max(grid)=126` rows per ticker are unlabeled.
3. **Multi-horizon triple-barrier (for Design C/D and for diagnostics).** Generalise `triple_barrier_labels` to record, per row, the realised first-passage time and which barrier hit, at the widest horizon — cheaper than re-running per `h`.
4. **Pay the leakage cost honestly.** The unlabeled tail grows from 5 to **126 rows per ticker**, and purge/embargo (`runner.py:147,164`) must use `max(grid)`, not `cfg.horizon`. This materially shrinks the trainable tail and widens the walk-forward gaps — budget for it; it is the single biggest hidden cost of long horizons.
5. **Sample-weight uniqueness becomes essential.** Overlapping 126-day labels have *massive* concurrency; the existing uniqueness × time-decay weights (`labels/weights.py`) must be computed against the *per-row* horizon, or long-horizon rows will dominate the loss through sheer overlap. Add a leakage/uniqueness test for the multi-horizon case to `tests/test_leakage.py`.

### Phase 2 — The return-surface model & horizon selection
6. **Quantile heads across the grid.** Train, per horizon `h`, three quantile regressors at τ∈{0.1, 0.5, 0.9} using XGBoost `reg:quantileerror` (xgboost ≥2.0, already the repo's version) under the *existing* `PurgedWalkForward` discipline and multi-seed bagging (`train_xgb_bag`). Output per (date, ticker): `q10_h, q50_h, q90_h` for every `h`. (Start with a coarse grid — {5, 21, 63} — to keep the training cost sane, widen later.)
7. **Horizon selection rule.** Add `select_horizon(surface, cfg)`:
   `h* = argmax_h [ q50_h − λ_t·cost(h) ] / (q90_h − q10_h)` — predicted median edge, net of a horizon-scaled cost/time-decay penalty `λ_t`, per unit of predicted spread. Expose `λ_t` as config; default it so the rule reduces to "max predicted Sharpe." Keep a hard `h_max` cap.
8. **Selection score** for the cross-sectional quintile becomes the risk-adjusted edge at `h*`, not the raw point score. Rank, quintile, and the conviction-weighting (`engine.py:_conviction_weighted_return`) all continue to work unchanged on the new score.
9. **Diagnostic, not just an output.** Log the *distribution* of chosen `h*` across names/dates. If everything collapses to one horizon, the grid or the penalty is wrong; if it's pure noise, the horizon signal isn't real — both are findings you want before trusting it.

### Phase 3 — Dynamic risk-reward from the predicted surface
10. **Derive stop/target from the predicted distribution at `h*`**, replacing the fixed multipliers in `enrich_signals` (`signals.py:89-97`):
    - `stop = entry − k·|q10_{h*}|·entry` (downside the model actually expects),
    - `target = entry + k·q90_{h*}·entry` (upside it actually expects),
    - `risk_reward = q90_{h*} / |q10_{h*}|` — now genuinely dynamic.
11. **Translate to ATR multiples for execution continuity** and to keep the paper trader's ATR-native plumbing intact: `stop_atr_mult = k·|q10_{h*}|·entry / atr14`, same for target. Clamp both to sane floors/ceilings (e.g. stop ∈ [0.8, 3]×ATR) so a degenerate quantile prediction can't place an absurd stop.
12. **Keep the fixed multipliers as a fallback** when the surface is unavailable or clamped — `getattr(cfg, ...)` already does this gracefully in `signals.py:89-90`; preserve that pattern.
13. **Resist Design D unless Phase 2/4 prove the surface is calibrated.** A dynamic RR is only as trustworthy as the `q10/q90` calibration — verify the quantiles are calibrated (realised 10%/90% exceedance rates match) *before* trading on them. Add this to the report.

### Phase 4 — The event-driven backtest (the critical dependency)
14. **Build `src/backtest/event_engine.py`** — a position-book simulator that mirrors `paper_trader.update` exactly: open on signal at conviction-weighted size and predicted horizon/stop/target; each subsequent bar, close on whichever fires first — `high>=target → target`, `low<=stop → stop`, `days_held>=h* → expired`; mark-to-market daily. This is what finally lets the backtest *see* a stop/target at all, and it makes the backtest and the live trader share one exit semantics (they diverge today — A2).
15. **Asynchronous, overlapping positions.** Variable `h*` breaks the synchronous "period return" abstraction in `run_backtest` (`engine.py:72-133`) — names no longer share a common holding window. The event engine holds an arbitrary book across time and respects `max_positions`/capital, exactly like the portfolio.
16. **Staged delivery to de-risk the rewrite.** First ship **horizon-bucketed sleeves**: discretise `h*` into the grid buckets and run the *existing* vectorised `run_backtest` once per bucket against that bucket's `fwd_ret_h` column, then combine equity curves by capital allocation. This validates the horizon signal with minimal new code *before* committing to the full event-driven engine. The event engine remains necessary to evaluate dynamic *RR* (sleeves still can't see stop/target), but sleeves let you separate "does variable horizon help?" from "does variable RR help?".
17. **Re-run the full Phase-0 robustness suite** (CPCV paths, block-bootstrap CI, deflated Sharpe) on the event engine — a path-dependent stop/target/expiry simulator has more degrees of freedom to overfit than a quintile mean, so it needs *more* robustness scrutiny, not less.

### Phase 5 — Live / paper wiring (small, mostly mechanical)
18. **`enrich_signals`** emits per-name `horizon_days = h*` and the dynamic `stop_loss`/`target_price` (Phase 3) instead of `cfg.horizon` and the fixed multipliers.
19. **`paper_trader`** already consumes per-trade `horizon_days`/`stop_loss`/`target_price` (`Trade` dataclass, `paper_trader.py:46-73`; exit logic `327-361`) — **no structural change needed**, it just receives varied values. Confirm `outcome_tracker`/`resolve_outcomes` (`runner.py:394`) resolves against the *per-trade* horizon, not the global `cfg.horizon`.
20. **Registry/bundle** must persist the horizon grid, the quantile models (now multiple boosters), and the selection/RR config so a bundle is reproducible — extend `registry/bundle.py`'s manifest. This is the one place the multi-model output meaningfully complicates the existing single-booster bundle format (same deferral noted for ensembles in `model-improvement-plan.md` Phase 3.16).

### Phase 6 — Validation & guardrails
21. **Quantile calibration report** (Phase 3.13) per horizon, every run.
22. **`h*` stability** across seeds and across CPCV paths — an unstable horizon choice is a red flag, same logic as the IC-instability concern in the main plan.
23. **Ablations, reported as deltas vs the frozen baseline:** (a) fixed h / fixed RR [baseline], (b) dynamic h only, (c) dynamic RR only, (d) both. If (d) doesn't beat the best of (b)/(c) outside the CI, the interaction isn't real.
24. **Leakage test per new label family** (multi-horizon fwd-ret, first-passage time) in `tests/test_leakage.py`.

---

## 3. Priority — what to do first if time is limited

1. **Phase 0 prerequisites** — non-negotiable; without the ruler nothing here is measurable.
2. **Phase 1 multi-horizon labels + the honest leakage/uniqueness cost** — the foundation everything else stands on; also immediately diagnostic (just plotting `fwd_ret_h` IC by horizon tells you whether a horizon signal even exists before you build any of the machinery).
3. **Phase 2 coarse-grid quantile model + horizon selection** + **Phase 4.16 bucketed-sleeve backtest** — the cheapest end-to-end loop that answers "does variable horizon help?" without the full engine rewrite.
4. **Phase 4.14 event-driven engine** — the gate that unlocks evaluating dynamic **RR** at all; do it once the horizon signal has earned the investment.
5. **Phase 3 dynamic RR from the calibrated surface** — last, because it's worthless until the quantiles are verified calibrated (Phase 3.13) on the event engine.

---

## 4. What not to expect (and where the risk is)

- **This is mostly a Calmar/turnover/capital-efficiency play, not a directional-accuracy play.** Don't expect dir-acc to move — expect drawdown to shrink, holding periods to match the edge's actual decay, and turnover (hence cost drag, the thing that kills the edge at 1.5–2× costs in the current report) to fall. If raw IC jumps, suspect leakage from the longer-horizon labels, not alpha.
- **Long horizons fight the data budget.** A 126-day label burns 126 rows of tail per ticker and widens every purge gap. On ~11 years of data that's a real reduction in effective training size — the grid's top end must *earn* its place, not be assumed.
- **The biggest overfitting surface in the repo.** A path-dependent stop/target/expiry engine plus a per-name barrier choice is exactly where backtests lie. Design D is the sharpest edge of this; even Design A's RR is only as good as its quantile calibration. Every dynamic-exit result must clear the deflated-Sharpe / CPCV bar from Phase 0, and any large jump should be hunted as a bug, per `model-improvement-plan.md` §D.
- **Backtest/live parity is a feature, not overhead.** The single highest-value structural by-product here is that the event engine (Phase 4) makes the backtest and the paper trader finally simulate the *same* exits — which is also what makes any reported improvement believable.

---

## Appendix — concrete code touch-points

| File | Change |
|---|---|
| `config.py` | `horizon_grid`, `quantile_taus`, horizon-selection `λ_t`, `h_max`, stop/target ATR clamps, RR `k`. |
| `labels/targets.py` | `forward_log_return_grid`; generalise `triple_barrier_labels` to record first-passage time. |
| `labels/weights.py` | uniqueness/concurrency against per-row horizon. |
| `models/trainer.py` | quantile heads (`reg:quantileerror`) across the grid, under existing WF + bagging. |
| `pipeline/runner.py` | `select_horizon`; purge/embargo use `max(grid)`; multi-head predict; resolve outcomes per-trade horizon. |
| `trading/signals.py` | derive `horizon_days`, `stop_loss`, `target_price`, `risk_reward` from the predicted surface (replaces `signals.py:89-97`). |
| `backtest/engine.py` | bucketed-sleeve mode (Phase 4.16). |
| `backtest/event_engine.py` | **new** — event-driven, overlapping-position simulator mirroring `paper_trader.update`. |
| `registry/bundle.py` | persist grid + multiple quantile boosters + selection/RR config. |
| `validation/` | quantile-calibration + `h*`-stability diagnostics. |
| `tests/test_leakage.py` | multi-horizon label leakage + uniqueness tests. |
