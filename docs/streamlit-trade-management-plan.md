# Plan — Streamlit Trade-Management & Prediction-Log Console

**Status:** Proposed
**Date:** 2026-06-22
**Owner:** adityaagrawal
**Scope:** Make the Streamlit app the day-to-day cockpit for (1) acting on model predictions as paper trades, (2) logging/managing those trades, and (3) auditing predicted-vs-actual outcomes that feed the next weekly Colab run. Also fix the broken env-var / Supabase wiring so the app stops silently running in JSON-fallback mode.

> Companion to `Baisc structure.md` (feedback-loop architecture) and `bot-implementaion-plan.md` (Colab weekly retrain). This doc is the **UI build plan**.

---

## 0. Goals (what "done" looks like)

1. **Env/Supabase actually connects.** Local and Streamlit-Cloud both read credentials and show a green "Supabase" badge. No more silent JSON fallback when `.env` is present.
2. **One-click paper trading from predictions.** From the Signals page, open a paper trade on any signal (auto-sized, pre-filled entry/stop/target) without typing prices by hand.
3. **Trades are first-class and durable.** Every open/close/edit writes to Supabase (source of truth) so trades survive Streamlit-Cloud restarts and are visible from Colab. Local JSON stays as a mirror/fallback.
4. **A real Prediction Journal.** A page that shows, per prediction: what the model said, what actually happened (ground truth), whether it was correct, and lets you export the resolved set for the next weekly run.
5. **Outcome resolution works from the UI.** The "Resolve Outcomes" button actually fetches prices and settles predictions (today it passes `price_df=None` and resolves nothing).

---

## 1. Current state & problems (with file references)

### 1.1 The env-var bug — root cause found
`load_dotenv()` is **never called anywhere in the repo** (`grep load_dotenv` → 0 hits), yet `python-dotenv` is in both `requirements.txt` files. Every consumer reads credentials via `os.getenv(...)` / `st.secrets`:
- `app/streamlit_app.py:33` `_get_secret()` → `st.secrets.get(key, os.getenv(...))`
- `app/utils/data_loader.py:30` `_get_client()` → same
- `src/config.py:79-85` → `os.getenv("SUPABASE_URL"/"SUPABASE_KEY"/...)`

Locally there is no Streamlit `secrets.toml`, so `st.secrets` is empty, and because nothing loads `.env` into the process, `os.getenv("SUPABASE_URL")` returns `""`. Result: `is_connected = False` → the app shows the amber "JSON fallback" badge and never talks to Supabase even though `.env` holds valid `SUPABASE_URL` / `SUPABASE_KEY`.

**Fix:** a single shared bootstrap that calls `load_dotenv()` before any secret read, plus a committed `.streamlit/secrets.toml` template for Cloud. (Details in §3.)

### 1.2 Writes never reach Supabase and die on Cloud restart
All UI mutations go **only** to local `outputs/portfolio.json`:
- `app/pages/5_Settings.py:76-100` (Close Position) and `:118-154` (Add Position) read/write `outputs/portfolio.json` directly.

On Streamlit Cloud the filesystem is ephemeral — these edits vanish on every restart/redeploy and are invisible to Colab. The data layer (`data_loader.py`) can *read* Supabase but there is **no write path** from the app. `prediction_journal.sync_paper_trades()` exists but is only called by the pipeline, not the UI.

**Fix:** add Supabase write helpers and route all UI mutations through them, dual-writing to JSON locally. (§4.)

### 1.3 Trade management is clunky and disconnected from signals
- Add/close lives in a **Settings** page, requires manually typing entry/stop/target/shares.
- There is **no path from a signal to a trade**. The Signals page (`app/pages/1_Signals.py`) is read-only; the user re-keys numbers into Settings.
- Open positions show no live/unrealized P&L (no current-price fetch).

**Fix:** new **Trade Desk** page + per-row "Trade this signal" actions on the Signals page. (§5, §6.)

### 1.4 No prediction log / ground-truth view
Outcome data exists end-to-end — `predictions_journal.json` rows gain `outcome_resolved`, `actual_close`, `actual_fwd_ret`, `hit_target`, `hit_stop`, `is_correct`, `exit_reason` after `resolve_outcomes()` (`src/tracking/outcome_tracker.py:190-217`), and there's an `outcomes` table. But the UI only surfaces **aggregate** IC/accuracy (`app/pages/3_Performance.py`). There is no per-prediction "predicted vs actual" ledger to eyeball what the model got right/wrong or to export for the next run.

**Fix:** new **Prediction Journal** page. (§7.)

### 1.5 "Resolve Outcomes Now" is a no-op
`app/pages/5_Settings.py:39` calls `resolve_outcomes(price_df=None, ...)`. With `price_df=None`, `_build_price_lookup()` returns `{}` (`outcome_tracker.py:51`) and nothing resolves. The button reports success while doing nothing.

**Fix:** fetch closes via yfinance for the unresolved (ticker, date) set before resolving. (§8.)

### 1.6 Auth/RLS caveat (must decide early)
Writing to Supabase from the app requires either the **secret** key or RLS policies that permit the **publishable** key to insert/update `paper_trades`/`predictions`/`outcomes`. The read-only publishable key alone will silently fail writes (our helpers swallow errors → looks like another "nothing happened" bug). Decision needed in §2.

---

## 2. Key decisions (resolve before building)

| # | Decision | Recommendation |
|---|----------|----------------|
| D1 | **Source of truth for trades** | **Supabase** `paper_trades`; local `portfolio.json` becomes a mirror/offline fallback. |
| D2 | **How the app authenticates for writes** | Use `SUPABASE_SECRET_KEY` in Streamlit Cloud **Secrets** (not committed), OR add RLS insert/update policies for the publishable key on `paper_trades`, `predictions`, `outcomes`. Recommend **secret key in Cloud secrets** for a private single-user app — simplest, and the dashboard isn't public. |
| D3 | **Live price source for unrealized P&L / resolution** | `yfinance` (already a dep), cached via `st.cache_data(ttl=...)`. Acceptable for paper trading. |
| D4 | **Position sizing default** | Reuse pipeline config (`capital`, `position_size`, `max_positions`). Pre-fill shares = `floor(capital * position_size / entry)`. |
| D5 | **Manual ground-truth override** | Allow, but tag `resolution_source = 'manual'` so auto-resolution never clobbers a hand-entered actual. |

I'll surface D1/D2 as the only blocking choices; the rest have sane defaults.

---

## 3. Workstream A — Fix env vars & Supabase connectivity

**A1. Shared secrets bootstrap.** New `app/utils/env.py`:
- `load_dotenv()` from repo root (idempotent, called once at import).
- `get_secret(key, *aliases, default="")` that checks `st.secrets` then `os.getenv` across the known aliases (`SUPABASE_KEY` / `SUPABASE_SECRET_KEY` / `SUPABASE_PUBLISHABLE_KEY`).
- Import this at the **top** of `streamlit_app.py` and `data_loader.py`; replace the ad-hoc `_get_secret`/`_get_client` secret reads with it. This removes the three near-duplicate try/except blocks.

**A2. Commit a `.streamlit/secrets.toml` template** (`.streamlit/secrets.toml.example`, real file gitignored):
```toml
SUPABASE_URL = "https://<ref>.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_xxx"   # write-capable (see D2)
# SUPABASE_PUBLISHABLE_KEY = "sb_publishable_xxx"  # read-only alt
```

**A3. Connection diagnostics panel** (Settings → "Database"): a **Test Connection** button that runs a `select count(*)` against each table (`predictions`, `paper_trades`, `outcomes`, `model_runs`) and prints row counts + which key type is in use + whether writes are permitted (attempt a no-op upsert to a `health` row). Turns "is it connected?" from a guess into a fact. Replaces the static masked-key display at `Settings.py:161-178`.

**A4. Normalize key naming.** `.env` currently has `SUPABASE_KEY`; `.env.example` advertises `SUPABASE_PUBLISHABLE_KEY`/`SUPABASE_SECRET_KEY`. Document both in one place and make `get_secret` accept all. (No code in `src/` needs to change — `config.py:82-85` already handles the aliases.)

**Acceptance:** with only `.env` present (no `secrets.toml`), `streamlit run app/streamlit_app.py` shows the green "Supabase" badge and Settings → Test Connection returns row counts.

---

## 4. Workstream B — Supabase write layer

New `app/utils/writer.py` (thin, error-surfacing — unlike the silent helpers in `supabase_client.py`, these **return (ok, message)** so the UI can show failures):
- `open_trade(trade: dict) -> (ok, msg)` → upsert `paper_trades` (on_conflict `trade_id`) **and** mirror into `portfolio.json`.
- `close_trade(trade_id, exit_price, exit_reason) -> (ok, msg)` → update row, recompute `pnl`/`pnl_pct`, bump portfolio `cash`.
- `edit_trade(trade_id, **fields)` → update stop/target/shares/notes.
- `log_manual_prediction(row)` → insert into `predictions` (for trades taken outside a model run).
- `upsert_outcome(row)` → manual ground-truth entry with `resolution_source='manual'`.

All functions reuse `src/db/supabase_client.upsert_rows`/`fetch_rows` for the Supabase side and keep the JSON mirror in sync so local dev and Cloud agree. After any write, call `data_loader.refresh_all()` so the UI reflects it.

**Schema additions** (document in a new `docs/supabase_schema.sql` and run once in the Supabase SQL editor):
- `paper_trades`: add `notes text`, `opened_via text` (`signal` | `manual`), `current_price numeric`, `unrealized_pnl numeric` (nullable, UI-updated).
- `predictions`: confirm columns used by resolution exist (`outcome_resolved bool`, `outcome_date`, `actual_close`, `actual_fwd_ret`, `hit_target`, `hit_stop`, `label_direction`).
- `outcomes`: add `resolution_source text default 'auto'`.

---

## 5. Workstream C — Signals page → action surface

Rework `app/pages/1_Signals.py` so each LONG signal is **actionable**:
- Replace the static styled table with `st.data_editor` (or per-row expanders) exposing a **"Trade this signal"** button. Pre-fills entry = `entry_price`, stop = `stop_loss`, target = `target_price`, shares = sized from config; user can tweak then confirm → `writer.open_trade(...)`.
- **Bulk action:** "Open all LONG signals (top N by prob_up)" with a confirm dialog and a capital/`max_positions` guardrail.
- **Already-traded indicator:** mark signals that already have an open `paper_trades` row for that ticker+signal_date (avoid double-entry).
- Keep the confidence bar chart and CSV download.

---

## 6. Workstream D — New "Trade Desk" page (`app/pages/2_Trade_Desk.py`)

Consolidates trade management out of Settings into a purpose-built blotter. (Renumber existing pages; keep Settings for config only.)
- **Open positions** with **live unrealized P&L**: fetch current price via cached yfinance (§D3), show `entry → last`, `unreal P&L ₹/%`, distance-to-stop / distance-to-target as progress bars, days-held vs `horizon_days`, and a "⚠ stop breached / 🎯 target hit / ⏰ expired" flag.
- **Row actions:** Close @ market (prefill last price), Close @ custom price, Edit stop/target, Add note.
- **Bulk:** "Auto-close all that hit stop/target/expiry" (uses the same logic as `PaperPortfolio.update`, but driven from the UI and written to Supabase).
- **Closed trades** table with P&L coloring (move from current Portfolio page).
- Keep the existing portfolio KPIs (value, cash, return, win rate) and equity curve at the top.

The current `2_Portfolio.py` becomes the analytics half (equity curve, exit-reason/PnL distributions, stats) or is merged into the Trade Desk — decide during build to avoid an empty page.

---

## 7. Workstream E — New "Prediction Journal" page (`app/pages/_Prediction_Journal.py`)

The "what did we predict vs what actually happened" ledger the user asked for.
- **One row per prediction** joining `predictions` (+ resolved fields) / `outcomes`:
  `signal_date · ticker · signal · prob_up · entry · stop · target · horizon · → status (pending/resolved) · actual_close · actual_fwd_ret · hit_target · hit_stop · is_correct · exit_reason · model_version`.
- **Status split:** "Pending resolution" (horizon not elapsed) vs "Resolved (ground truth in)". Pending rows show a countdown ("resolves in N trading days").
- **Filters:** week, ticker, model_version, correct/incorrect, signal type.
- **Scorecard strip:** resolved count, hit rate, mean `actual_fwd_ret` for LONG vs NEUTRAL, IC over the visible window (reuse `compute_recent_ic`).
- **Calibration mini-view:** reuse the prob_up-vs-actual scatter from Performance, scoped to the filter.
- **Export for next run:** "⬇ Export resolved outcomes (CSV/JSON)" producing exactly the ledger Colab consumes, so the feedback loop is one click. Also a "Copy Supabase query" helper.
- **Manual ground-truth entry/override** (D5): for a pending/odd row, enter the actual close → `writer.upsert_outcome(..., resolution_source='manual')`.

This page is the audit/trust surface; it reads from Supabase first, JSON fallback otherwise (same pattern as `data_loader`).

---

## 8. Workstream F — Make outcome resolution work from the UI

Fix `Settings.py` "Resolve Outcomes Now" (and add the same control to the Prediction Journal):
1. Load unresolved predictions (Supabase or `predictions_journal.json`).
2. Compute the set of `(ticker, target_date)` needed and **download closes via yfinance** (cached), building the `price_df` that `resolve_outcomes` expects (`ticker`, `date`, `close`).
3. Call `resolve_outcomes(price_df=that_df, supabase_client=sb, horizon_days=...)`.
4. Report N resolved + refresh.

New helper `app/utils/prices.py::fetch_closes(tickers, start, end) -> price_df` (cached). This also powers the Trade Desk live P&L (§6).

**Note:** the current resolver uses close-only barrier checks (`outcome_tracker.py:180-185`) — acceptable for now; a high/low intraday-barrier upgrade is out of scope but worth a backlog ticket.

---

## 9. Workstream G — Polish & guardrails (value-adds from my side)

- **Home dashboard upgrade** (`streamlit_app.py`): replace the static info cards with live tiles — # open positions, unrealized P&L, # predictions pending resolution, latest weekly IC, model-health color — each linking to its page. A real landing pad.
- **"What do I do this week?" checklist** on Home: e.g. "3 LONG signals ungtraded", "5 predictions ready to resolve", "Sunday retrain not yet run". Turns the app into a workflow guide for the 2-month paper-trading phase.
- **Stale-data banner:** if `signal_date` is older than ~7 days, warn that signals are stale (missed weekly run).
- **Optimistic UI + error surfacing:** because writes can fail on RLS/key issues, always show the `(ok, msg)` from the writer; never a silent success.
- **Idempotency:** `trade_id = f"{ticker}_{signal_date}"` for signal-driven trades to prevent duplicates on double-click.
- **Tests:** `tests/test_writer.py` (open→close→pnl math, JSON/Supabase mirror parity using a fake client) and `tests/test_prices.py` (fetch_closes shape). Extend the existing 18-test suite.

---

## 9b. Workstream H — Real Indian-market charges & account ledger

Make every paper trade behave like a real NSE delivery trade: charges deducted on entry **and** exit, P&L shown **net of costs**, and a running account-transaction statement like a broker contract note + funds ledger.

### H1. Reuse the existing cost engine
`src/backtest/costs.py::indian_round_trip_cost()` already models the full Indian schedule (STT 0.1%/side, exchange txn ~0.00297%, SEBI ~0.0001%, stamp 0.015% on buy, GST 18% on brokerage+exch+SEBI, DP ₹13/scrip on sell, brokerage ₹0 for delivery). We **split it per leg** instead of round-trip so the UI can charge at entry and at exit separately. New `app/utils/charges.py`:
- `buy_charges(value) -> dict` → `{stt, exchange, sebi, stamp, gst, brokerage, total}` for the **buy** leg (stamp + STT-buy + exch + sebi + GST; no DP).
- `sell_charges(value) -> dict` → the **sell** leg (STT-sell + exch + sebi + GST + DP ₹13; no stamp).
- A small `breakdown` helper returns the line items so the UI can render a **contract-note-style table**.
- Rates pulled from `config/strategy.yaml` (new `costs:` block) so a Union-Budget change is a one-line edit, not a code change. Keep `src/backtest/costs.py` defaults as the single source of the rates (import them into `charges.py` — don't duplicate constants).
- Add **slippage** on fills (config `slippage_bps`, default 10) so entry/exit prices are nudged against you, like real execution.

### H2. Charge trades on entry and exit (real cash accounting)
- **On open** (`writer.open_trade`): `gross_cost = shares*entry`; `entry_charges = buy_charges(gross_cost).total`; `cash -= (gross_cost + entry_charges)`. Store `entry_charges`, `entry_slippage` on the trade. **Break-even price** = `entry*(1+...) ` computed and shown so the user knows the price the stock must clear just to cover costs.
- **On close** (`writer.close_trade`): `proceeds = shares*exit`; `exit_charges = sell_charges(proceeds).total`; `cash += (proceeds - exit_charges)`; **net P&L** = `proceeds - exit_charges - gross_cost - entry_charges`; `pnl_pct` computed on net. Replaces the cost-free math currently in `Settings.py:90-94`.
- Open-position **unrealized P&L** (Trade Desk §6) is shown **net of the round-trip estimate** so it doesn't look artificially green.

### H3. Account-transactions / funds ledger (new)
A broker-style **funds statement**: every cash movement is an immutable ledger entry. New table `account_ledger` (+ JSON mirror `outputs/ledger.json`):
`ts · type (BUY|SELL|CHARGE|DEPOSIT|WITHDRAWAL|OPENING_BALANCE) · trade_id · ticker · qty · price · amount (signed) · running_balance · note`.
- `writer` appends ledger rows on every open/close (one BUY/SELL row + itemized CHARGE rows, or a single net row with an expandable breakdown).
- Seed with an `OPENING_BALANCE` = `initial_capital` and support **manual Deposit/Withdrawal** entries so the account funds are real and reconcilable.
- **Reconciliation invariant** (assert in tests): `running_balance == initial_capital + Σ deposits − Σ withdrawals + Σ realized_net_pnl − cash_locked_in_open_positions`.

### H4. UI surfaces
- **Trade ticket / contract note:** when opening from Signals (§5), show a pre-trade preview — qty, gross value, itemized charges, slippage, net debit, and break-even price — before "Confirm". Same on close (estimated charges + net credit + net P&L).
- **New "Account" page** (`app/pages/_Account.py`): funds ledger table (filter by type/date), current cash vs deployed vs total equity, total charges paid to date (a real drag metric), and a "charges as % of turnover" stat. Export CSV (statement).
- **Portfolio KPIs** updated to show **Total charges paid** and **Net realized P&L** (gross vs net side by side) so the cost drag the backtest warns about (`APPROX_RT_COST_FRACTION`, ~0.21%+slippage) is visible in live paper trading.

### H5. Schema additions (fold into `docs/supabase_schema.sql`)
- `paper_trades`: add `entry_charges numeric`, `exit_charges numeric`, `entry_slippage numeric`, `breakeven_price numeric`, `gross_pnl numeric`, `net_pnl numeric`.
- New `account_ledger` table as in H3.

### H6. Tests
`tests/test_charges.py` — buy/sell legs sum to `indian_round_trip_cost` within tolerance; STT/stamp asymmetry correct; DP only on sell. `tests/test_ledger.py` — reconciliation invariant after a randomized open/close sequence.

> **Realism caveat to document in-app:** charges use current (2026) NSE/SEBI/budget rates from config; these change with Union Budgets. This is paper trading — fills assume liquid Nifty-50 execution at close ± slippage, not true intraday microstructure.

---

## 10. Proposed file changes (summary)

**New**
- `app/utils/env.py` — dotenv bootstrap + `get_secret`.
- `app/utils/writer.py` — Supabase+JSON write helpers returning `(ok, msg)`.
- `app/utils/prices.py` — cached yfinance close fetcher.
- `app/utils/charges.py` — per-leg Indian charges (buy/sell) + slippage + break-even, reusing `src/backtest/costs.py` rates.
- `app/pages/2_Trade_Desk.py` — blotter with live P&L + actions.
- `app/pages/_Prediction_Journal.py` — predicted-vs-actual ledger + export.
- `app/pages/_Account.py` — funds ledger / contract notes / charges-paid statement.
- `.streamlit/secrets.toml.example` — Cloud secrets template.
- `docs/supabase_schema.sql` — canonical schema incl. new columns.

**Modified**
- `app/streamlit_app.py` — bootstrap import, live home tiles, weekly checklist, fixed badge.
- `app/utils/data_loader.py` — use `env.get_secret`; add `load_prediction_journal()`, `load_open_trades_with_live_pnl()`.
- `app/pages/1_Signals.py` — per-row "Trade this signal" + bulk open.
- `app/pages/2_Portfolio.py` — merge into / cross-link Trade Desk (analytics half).
- `app/pages/5_Settings.py` — config + diagnostics only; real outcome resolution; remove ad-hoc portfolio.json mutation (moves to writer).
- `app/requirements.txt` — add `yfinance` (used by app now), `python-dotenv` already implied.

**No changes needed** in the trained-model / pipeline core — this is a UI + glue effort.

---

## 11. Phasing (incremental, each shippable)

1. **Phase 1 — Connectivity (½ day).** A1–A4 env bootstrap + diagnostics. *Unblocks everything; verifies writes are even possible (D2).*
2. **Phase 2 — Write layer + charges/ledger + Trade Desk (2–3 days).** B + H + D. Trades become durable in Supabase, charged like real NSE delivery, with a funds ledger and live net P&L. *(Build H's charge math into the writer before Trade Desk so P&L is net from day one.)*
3. **Phase 3 — Signals→trade actions (½ day).** C. One-click from prediction to paper trade.
4. **Phase 4 — Prediction Journal + resolution fix (1–1.5 days).** E + F. The ground-truth audit + export loop.
5. **Phase 5 — Polish & tests (½ day).** G.

Phase 1 must land first (and D2 answered) or Phases 2/4 writes will fail silently.

---

## 12. Open questions for the user

1. **D2 — write auth:** OK to put the Supabase **secret** key in Streamlit Cloud Secrets, or do you prefer to add RLS policies and keep the publishable key? (Affects whether trade logging from the UI works at all.)
2. **D1 confirm:** Supabase as source of truth, `portfolio.json` demoted to a mirror — good?
3. **Page layout:** keep separate **Portfolio (analytics)** and **Trade Desk (actions)** pages, or merge into one?
4. Is the existing Supabase schema already created with all the `predictions`/`outcomes` columns the resolver writes, or do we need to run `docs/supabase_schema.sql` to add them?
