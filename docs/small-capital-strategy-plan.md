# Small-Capital Strategy Plan — ₹20k Live-Ready Model (2026-07-03)

**Goal (user):** train the best-possible weekly prediction model on the Nifty 200
universe, paper trade it for ~2 months with a ₹20,000 account profile, compare
predicted vs. actual outcomes every week so the system self-corrects, and only
then deploy real capital. Target ambition is ~35% CAGR with low drawdown.

**Honest expectation setting (read first):** the repo's own validated numbers
(`docs/model-improvement-plan.md` §D, `reports/backtest_v*.md`) put a durable
weekly-horizon equity edge at 52–56% directional accuracy and daily IC-IR
0.05–0.10. A *verified* 35% CAGR at low drawdown from long-only weekly swing
trading on liquid Indian equities would be world-class; treat any backtest
showing it as a leakage bug until proven otherwise. The plan below maximises
the chance of a real, costed, compounding edge — the 2-month paper window is
exactly the right way to find out what the true number is. Do **not** skip it.

---

## 1. Why ₹20k changes the strategy itself (not just the settings)

The pipeline was built around a ₹10L account longing the whole top quintile
(~40 Nifty 200 names). At ₹20k that strategy cannot exist:

| Constraint | ₹10L / 40 names | ₹20k / 5 names |
|---|---|---|
| Per-position size | ~₹25,000 | ~₹3,800 |
| Flat DP charge (₹15.93/scrip/sell) | 0.06% | **0.42%** |
| Full round-trip cost (STT+stamp+exch+GST+DP+2×10bps slip) | ~0.30% | **~0.85–0.95%** |
| Weekly rebalance → annual cost drag at 100% turnover | ~15% | **~45%** |
| Affordable universe | all 200 names | names with price ≤ ₹3,800 (MRF, Page, Bosch etc. excluded) |
| Idiosyncratic risk | diversified across 40 | concentrated in 5 |

Consequences implemented in code:

1. **Capital-aware cost model** (`src/backtest/costs.py::per_position_cost_fraction`,
   `cfg.capital_aware_costs=True`): every backtest and sensitivity table now
   bleeds at the ₹3.8k-position cost fraction, not a hypothetical ₹1L trade.
   The model's edge bar is the *real* one.
2. **Top-K concentrated book** (`cfg.top_k_positions`): the backtest longs only
   the K highest-conviction names (K = `max_positions` = 5), so the backtest
   finally simulates the strategy the account can actually hold. The quintile
   backtest remains available (top_k_positions=0) as the research baseline.
3. **Turnover hysteresis** (`cfg.hysteresis_enabled`, plan §Phase 4.20): a held
   name is kept while its score-rank percentile stays ≥ `exit_rank_pct` (0.60);
   new names enter only above `entry_rank_pct` (0.80). At ~0.9% per avoided
   round trip this is worth several % CAGR per year at ₹20k sizing. Applied in
   the backtest AND in live position management (expiring positions whose
   ticker still ranks above the exit band are rolled, not sold-and-rebought).
4. **Integer-share trade planner** (`src/trading/trade_planner.py`): converts
   LONG signals into an executable plan — whole shares within the per-position
   budget, affordability filter (price ≤ budget), itemised entry/exit charges,
   breakeven move %, and a cost-vs-target edge screen that drops names whose
   expected move can't clear costs. Output persisted to
   `outputs/trade_plan_latest.{json,csv}` for download from Colab.
5. **Hands-off paper trading** (`cfg.auto_open_signals`): the weekly retrain
   opens the planned trades in the paper portfolio automatically (exits were
   already automatic), so the 2-month paper run needs zero manual clicks and
   every cost is in the ledger.

## 2. New model inputs (no external data feeds required)

Added to `src/features/engineer.py`, auto-picked-up by `feature_cols`:

- **Market breadth** (plan §Phase 2.13): % of the Nifty 200 universe above its
  own 50/200-SMA, % with positive 5d return, % near 21d highs, and the 5-day
  change in breadth. A more robust regime signal than the single
  `nifty_dist_sma200` distance — breadth divergence (index up, breadth down)
  is a classic pre-drawdown tell the hard SMA filter can't see.
- **Sector relative strength** (plan §Phase 2.12): a hand-built
  `config/sector_map.json` (~21 sectors across the 196 names) enables
  `sector_ret_21d` (sector momentum — sector rotation signal once
  cross-sectionally ranked) and `rel_sector_mom_5d/21d` (stock return minus
  sector mean — isolates stock-specific alpha from sector-wide moves, which
  `beta_63d`/`alpha_5d` vs the index alone cannot).

Both are same-date cross-sectional aggregates of data already in the panel —
no look-ahead, no new feeds, leakage-tested.

Still open (external-data backlog, highest expected new-IC per
`model-improvement-plan.md` Phase 2): delivery %, FII/DII flows, F&O
basis/PCR/OI. These need NSE bhavcopy/scraper ingestion work.

## 3. The weekly self-correction loop (predict → compare → adapt)

Already largely built; this plan wires it into one artifact per week:

1. **Predict** — Sunday Colab run scores the latest bar, saves per-ticker
   predictions (prob_up, entry, stop, target, horizon) to Supabase + JSON.
2. **Resolve** — next Sunday, `resolve_outcomes()` fetches realized prices and
   marks every matured prediction correct/incorrect with its actual return.
3. **Compare** — NEW `src/tracking/review.py` writes
   `reports/weekly_review_<date>.md`: last week's predicted-vs-actual table,
   rolling 4/8-week live IC and directional accuracy vs the backtest's OOF IC,
   paper-portfolio equity/cost drag, and a verdict line (ON TRACK / DEGRADING /
   EMERGENCY). Download this from Colab each week — it is the paper-trading
   scorecard.
4. **Adapt** — three mechanisms, in increasing severity:
   - every retrain refits on the newest data with time-decay weights
     (half-life 252d) so the model continuously tilts toward the live regime;
   - the champion/challenger gate (`registry/promotion.py`) only deploys a new
     model when it beats the incumbent's rolling live IC, with absolute floors
     (IC t-stat ≥ 2, deflated Sharpe ≥ 0.95) so a lucky week can't deploy noise;
   - drift monitors (PSI/CUSUM) + `should_retrain` trigger emergency retrains,
     and the regime overlay goes flat when Nifty < 200-SMA.

## 4. Run configuration (what the weekly Colab run should use)

```
python scripts/weekly_retrain.py            # defaults now: ₹20k, 5 positions
# equivalent explicit form:
#   --capital 20000 --max-positions 5 --trials 50
```
Weekly retrain now sets: `top_k_positions=5`, `hysteresis_enabled=True`,
`capital_aware_costs=True`, `auto_open_signals=True` (opt out with
`--no-auto-trade`).

Model A/B still owed on Colab (unchanged from
`docs/weekly-retrain-fixes-2026-06-29.md` Tier 1): ranker vs regression on
daily IC-IR — the 06-29 ranker run regressed IC badly; run both and keep the
winner. The new small-capital backtest applies to either.

## 5. Milestones for the 2-month paper window

- **Week 0 (now):** this plan's code merged; first ₹20k-profile retrain on
  Colab; verify trade plan has ≥3 affordable names and breakeven < 1.2%.
- **Weeks 1–4:** weekly runs, zero manual intervention. Success = live rolling
  IC > 0 and within CI of backtest OOF IC; paper cost drag matches the
  planner's estimates (±20%).
- **Weeks 4–8:** A/B the open levers one at a time (ranker vs regression,
  market_neutral_label, breadth/sector features on vs off) — one change per
  retrain, judged on daily IC-IR only.
- **Go/no-go (end of month 2):** deploy real ₹20k only if: live 8-week IC > 0
  with t-stat trending up, paper equity above costs, max drawdown inside the
  backtest CI, and the weekly review shows no unexplained live-vs-backtest gap.
  Otherwise: extend paper, or conclude the edge isn't there yet — that verdict
  is the system working, not failing.

## 6. Definition of done for this implementation

- [x] Cost model: DP ₹15.93, `per_position_cost_fraction`, capital-aware backtest
- [x] Backtest: top-K book + hysteresis, cfg-driven, legacy path preserved
- [x] Trade planner: integer shares, affordability, breakeven screen, persisted
- [x] Paper trader: auto-open from plan, hysteresis-aware expiry roll
- [x] Features: breadth + sector relative strength + sector_map.json
- [x] Weekly review report generator, wired into weekly_retrain
- [x] Config/YAML defaults moved to the ₹20k profile
- [x] Unit tests for all of the above; full suite green locally
- [ ] First Colab retrain with the new profile (user-run; training is Colab-only)
- [ ] Ranker-vs-regression A/B decision recorded in a follow-up doc
