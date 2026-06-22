-- Supabase schema migration for the Streamlit trade-management console.
-- Companion to docs/streamlit-trade-management-plan.md.
--
-- Run this ONCE in the Supabase SQL editor (Project → SQL Editor → New query).
-- All statements are idempotent (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS),
-- safe to re-run. The base tables (predictions, paper_trades, outcomes,
-- model_runs, feature_importance) are assumed to already exist — this file
-- only ADDS the columns/tables the real-money trade-management UI needs.

-- ============================================================================
-- 1. paper_trades — real-money charge/slippage/notes columns
-- ============================================================================
alter table paper_trades add column if not exists entry_charges   numeric;
alter table paper_trades add column if not exists exit_charges    numeric;
alter table paper_trades add column if not exists entry_slippage  numeric;
alter table paper_trades add column if not exists exit_slippage   numeric;
alter table paper_trades add column if not exists breakeven_price numeric;
alter table paper_trades add column if not exists gross_pnl       numeric;
alter table paper_trades add column if not exists gross_pnl_pct   numeric;
alter table paper_trades add column if not exists opened_via      text default 'signal';
alter table paper_trades add column if not exists notes           text default '';

-- ============================================================================
-- 2. outcomes — manual ground-truth override tagging
-- ============================================================================
alter table outcomes add column if not exists resolution_source text default 'auto';

-- ============================================================================
-- 3. account_ledger — broker-style funds statement (new table)
-- ============================================================================
create table if not exists account_ledger (
    id              uuid primary key,
    run_id          text,
    ts              timestamptz not null default now(),
    type            text not null,         -- BUY|SELL|CHARGE|DEPOSIT|WITHDRAWAL|OPENING_BALANCE
    trade_id        text,
    ticker          text,
    qty             numeric,
    price           numeric,
    amount          numeric not null,      -- signed INR movement
    running_balance numeric not null,
    note            text,
    created_at      timestamptz not null default now()
);

create index if not exists account_ledger_ts_idx       on account_ledger (ts desc);
create index if not exists account_ledger_trade_id_idx  on account_ledger (trade_id);

-- Row Level Security: enable + allow the app's key full access on these
-- two objects. If you are using the SECRET key for the Streamlit app
-- (recommended for a private single-user app — see plan §2 D2), RLS can
-- stay permissive since the secret key bypasses RLS by default in Supabase.
-- If instead you are using the PUBLISHABLE key for writes, uncomment and
-- adapt the policies below.

-- alter table account_ledger enable row level security;
-- create policy "allow all on account_ledger" on account_ledger
--   for all using (true) with check (true);

-- alter table paper_trades enable row level security;
-- create policy "allow all on paper_trades" on paper_trades
--   for all using (true) with check (true);
