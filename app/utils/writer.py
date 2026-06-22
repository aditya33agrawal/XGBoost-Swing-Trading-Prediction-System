"""Write layer for the Streamlit app — the only place UI mutations happen.

Every function returns (ok: bool, msg: str) so the UI can surface failures
instead of silently doing nothing (the old Settings.py bug). Every mutation:
  1. Goes through PaperPortfolio (real Indian charges + slippage + ledger —
     see src/trading/paper_trader.py) so paper trading behaves like a real
     NSE delivery account.
  2. Is persisted to the local outputs/portfolio.json mirror.
  3. Is upserted to Supabase (paper_trades + account_ledger) when a client
     is configured — Supabase is the source of truth; JSON is the offline
     fallback (see docs/streamlit-trade-management-plan.md D1).
  4. Clears the data_loader cache so the UI reflects the change immediately.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

from app.utils.env import get_supabase_url, get_supabase_key
from src.backtest.costs import buy_leg_cost, sell_leg_cost
from src.db.supabase_client import get_supabase_client
from src.trading.paper_trader import PaperPortfolio
from src.tracking.prediction_journal import sync_paper_trades, sync_ledger

logger = logging.getLogger(__name__)

PORTFOLIO_PATH = "outputs/portfolio.json"
_UI_RUN_ID = "ui"


def _client():
    return get_supabase_client(get_supabase_url(), get_supabase_key())


def _persist(portfolio: PaperPortfolio) -> tuple[bool, str]:
    """Save to JSON, sync to Supabase, refresh the UI cache."""
    try:
        portfolio.save(PORTFOLIO_PATH)
    except Exception as exc:
        return False, f"Local save failed: {exc}"

    client = _client()
    sb_msg = ""
    if client is None:
        sb_msg = " (Supabase not configured — saved locally only)"
    else:
        try:
            n_trades = sync_paper_trades(portfolio, _UI_RUN_ID, client)
            n_ledger = sync_ledger(portfolio, _UI_RUN_ID, client)
            sb_msg = f" · synced {n_trades} trade(s), {n_ledger} ledger row(s) to Supabase"
        except Exception as exc:
            sb_msg = f" (Supabase sync failed: {exc})"

    try:
        from app.utils.data_loader import refresh_all
        refresh_all()
    except Exception:
        pass

    return True, sb_msg


# ---------------------------------------------------------------------------
# Trade preview (no side effects) — for the contract-note confirmation UI
# ---------------------------------------------------------------------------
def preview_buy(ref_price: float, shares: int, slippage_bps: float = 10.0) -> dict:
    fill_price = ref_price * (1 + slippage_bps / 1e4)
    gross_cost = fill_price * shares
    charges = buy_leg_cost(gross_cost)
    est_exit_charges = sell_leg_cost(gross_cost)["total"]
    breakeven = (gross_cost + charges["total"] + est_exit_charges) / max(shares, 1)
    return {
        "fill_price": round(fill_price, 2),
        "gross_cost": round(gross_cost, 2),
        "charges": charges,
        "total_debit": round(gross_cost + charges["total"], 2),
        "breakeven_price": round(breakeven, 2),
        "slippage_amount": round((fill_price - ref_price) * shares, 2),
    }


def preview_sell(ref_price: float, shares: int, slippage_bps: float = 10.0) -> dict:
    fill_price = ref_price * (1 - slippage_bps / 1e4)
    proceeds = fill_price * shares
    charges = sell_leg_cost(proceeds)
    return {
        "fill_price": round(fill_price, 2),
        "proceeds": round(proceeds, 2),
        "charges": charges,
        "net_credit": round(proceeds - charges["total"], 2),
        "slippage_amount": round((ref_price - fill_price) * shares, 2),
    }


# ---------------------------------------------------------------------------
# Open / close / edit trades
# ---------------------------------------------------------------------------
def get_trade_price(ticker: str, fallback: Optional[float] = None) -> tuple[float, str]:
    """Live Yahoo price to actually trade at, with a labeled fallback.

    Every buy/sell should fill at the current market price, not the
    (potentially stale) price recorded when the signal was generated.
    Returns (price, source) where source is "live" or "fallback".
    """
    from app.utils.prices import fetch_latest_price
    live = fetch_latest_price(ticker)
    if live is not None and live > 0:
        return float(live), "live"
    if fallback is not None:
        return float(fallback), "fallback"
    raise ValueError(f"Could not fetch a live price for {ticker} and no fallback was given")


def open_trade(
    ticker: str,
    ref_price: float,
    shares: int,
    stop_loss: float,
    target_price: float,
    horizon_days: int = 5,
    prob_up: float = 0.5,
    opened_via: str = "manual",
    notes: str = "",
) -> tuple[bool, str]:
    try:
        fill_ref_price, price_source = get_trade_price(ticker, fallback=ref_price)
    except ValueError as exc:
        return False, str(exc)

    portfolio = PaperPortfolio.load(PORTFOLIO_PATH)
    try:
        trade = portfolio.open_manual(
            ticker, fill_ref_price, shares, stop_loss, target_price,
            horizon_days, prob_up, opened_via=opened_via, notes=notes,
        )
    except ValueError as exc:
        return False, str(exc)

    ok, sb_msg = _persist(portfolio)
    if not ok:
        return False, sb_msg
    price_note = "" if price_source == "live" else " ⚠️ live price unavailable, used signal price"
    return True, (
        f"Opened {ticker} × {shares} @ ₹{trade.entry_price:,.2f} "
        f"(charges ₹{trade.entry_charges:,.2f}, breakeven ₹{trade.breakeven_price:,.2f})"
        f"{price_note}{sb_msg}"
    )


def close_trade(
    trade_id: str,
    exit_price: Optional[float] = None,
    reason: str = "manual",
) -> tuple[bool, str]:
    portfolio = PaperPortfolio.load(PORTFOLIO_PATH)
    target = next((t for t in portfolio.open_trades if t.trade_id == trade_id), None)
    if target is None:
        return False, f"No open trade with id {trade_id}"

    if exit_price is None:
        from app.utils.prices import fetch_latest_price
        exit_price = fetch_latest_price(target.ticker)
        if exit_price is None:
            return False, f"Could not fetch a live price for {target.ticker} — supply an exit price manually"

    try:
        trade = portfolio.close_manual(trade_id, float(exit_price), reason=reason)
    except ValueError as exc:
        return False, str(exc)

    ok, sb_msg = _persist(portfolio)
    if not ok:
        return False, sb_msg
    sign = "+" if (trade.pnl or 0) >= 0 else ""
    return True, (
        f"Closed {trade.ticker} @ ₹{trade.exit_price:,.2f} — "
        f"net P&L {sign}₹{trade.pnl:,.2f} ({sign}{trade.pnl_pct:.1f}%){sb_msg}"
    )


def edit_trade(trade_id: str, **fields) -> tuple[bool, str]:
    portfolio = PaperPortfolio.load(PORTFOLIO_PATH)
    try:
        portfolio.edit_trade(trade_id, **fields)
    except ValueError as exc:
        return False, str(exc)
    ok, sb_msg = _persist(portfolio)
    if not ok:
        return False, sb_msg
    return True, f"Updated trade {trade_id}{sb_msg}"


def auto_close_expired(price_df: pd.DataFrame) -> tuple[bool, str]:
    """Close every open trade that has hit stop/target/expiry, using price_df."""
    portfolio = PaperPortfolio.load(PORTFOLIO_PATH)
    closed = portfolio.update(price_df)
    if not closed:
        return True, "No positions hit stop/target/expiry."
    ok, sb_msg = _persist(portfolio)
    if not ok:
        return False, sb_msg
    names = ", ".join(f"{t.ticker} ({t.exit_reason})" for t in closed)
    return True, f"Auto-closed {len(closed)} position(s): {names}{sb_msg}"


# ---------------------------------------------------------------------------
# Funds ledger — deposit / withdraw
# ---------------------------------------------------------------------------
def deposit(amount: float, note: str = "") -> tuple[bool, str]:
    portfolio = PaperPortfolio.load(PORTFOLIO_PATH)
    portfolio.deposit(amount, note)
    ok, sb_msg = _persist(portfolio)
    if not ok:
        return False, sb_msg
    return True, f"Deposited ₹{amount:,.2f}{sb_msg}"


def withdraw(amount: float, note: str = "") -> tuple[bool, str]:
    portfolio = PaperPortfolio.load(PORTFOLIO_PATH)
    try:
        portfolio.withdraw(amount, note)
    except ValueError as exc:
        return False, str(exc)
    ok, sb_msg = _persist(portfolio)
    if not ok:
        return False, sb_msg
    return True, f"Withdrew ₹{amount:,.2f}{sb_msg}"


# ---------------------------------------------------------------------------
# Manual ground-truth / prediction overrides (Prediction Journal page)
# ---------------------------------------------------------------------------
def upsert_outcome(row: dict) -> tuple[bool, str]:
    from src.db.supabase_client import upsert_rows
    row = {**row, "resolution_source": "manual"}
    client = _client()
    ok = upsert_rows(client, "outcomes", [row], on_conflict="prediction_id") if client else False

    import json
    from pathlib import Path
    fpath = Path("outputs/outcomes.json")
    try:
        existing = json.loads(fpath.read_text()) if fpath.exists() else []
        existing = [o for o in existing if o.get("prediction_id") != row.get("prediction_id")]
        existing.append(row)
        fpath.write_text(json.dumps(existing, indent=2, default=str))
    except Exception as exc:
        return ok, f"JSON fallback write failed: {exc}"

    try:
        from app.utils.data_loader import refresh_all
        refresh_all()
    except Exception:
        pass
    return True, "Outcome saved" + ("" if client else " (JSON fallback — Supabase not configured)")
