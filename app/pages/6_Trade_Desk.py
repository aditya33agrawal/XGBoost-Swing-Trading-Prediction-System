"""Page 6: Trade Desk — manage open positions with live, net-of-cost P&L."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import date

import streamlit as st
import pandas as pd

from app.utils.data_loader import (
    load_open_trades_with_live_pnl, load_paper_trades, load_portfolio_meta, refresh_all,
)
from app.utils import writer
from app.utils.prices import fetch_latest_price
from src.trading.market_hours import is_market_open, market_status_message

st.set_page_config(page_title="Trade Desk", page_icon="🎯", layout="wide")
st.title("🎯 Trade Desk")
st.caption("Live unrealized P&L is shown **net** of entry charges and an estimated exit-charge drag — "
           "what you'd actually keep if you closed right now.")
st.caption(market_status_message())
st.caption("Open positions only ever close automatically when **stop-loss**, **target**, or "
           "**horizon expiry** is hit — those are the orders you placed when you opened the trade. "
           "Everything else here (Close @ Market / Custom) is your manual call.")

if st.button("⟳ Refresh"):
    refresh_all()
    st.rerun()

open_t = load_open_trades_with_live_pnl()
all_trades = load_paper_trades()
closed_t = all_trades[all_trades["status"] == "closed"] if not all_trades.empty else pd.DataFrame()
meta = load_portfolio_meta()

st.divider()

# ── Open positions ────────────────────────────────────────────────────────────
st.subheader(f"Open Positions ({len(open_t)})")

if open_t.empty:
    st.info("No open positions. Go to **Signals** to open one from a model prediction.")
else:
    for _, row in open_t.iterrows():
        ticker = row["ticker"]
        entry  = float(row["entry_price"])
        cp     = row.get("current_price")
        u_pnl  = row.get("unrealized_pnl")
        u_pct  = row.get("unrealized_pnl_pct")
        stop   = float(row.get("stop_loss", 0) or 0)
        target = float(row.get("target_price", 0) or 0)

        label = f"{ticker}  ·  entry ₹{entry:,.2f}"
        if u_pct is not None:
            sign = "+" if u_pct >= 0 else ""
            label += f"  ·  {sign}{u_pct:.1f}% unrl."
        with st.expander(label, expanded=False):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Entry", f"₹{entry:,.2f}")
            c2.metric("Now", f"₹{cp:,.2f}" if cp else "—")
            c3.metric("Unrl P&L (net)", f"₹{u_pnl:,.2f}" if u_pnl is not None else "—",
                      delta=f"{u_pct:+.1f}%" if u_pct is not None else None)
            c4.metric("Stop", f"₹{stop:,.2f}")
            c5.metric("Target", f"₹{target:,.2f}")

            if cp and target > entry:
                progress = max(0.0, min(1.0, (cp - stop) / max(target - stop, 1e-9)))
                st.progress(progress, text=f"Stop ₹{stop:,.0f} ───── Now ₹{cp:,.2f} ───── Target ₹{target:,.0f}")

            days_held = (date.today() - pd.to_datetime(row.get("entry_date")).date()).days \
                if row.get("entry_date") else None
            horizon = int(row.get("horizon_days", 5) or 5)
            if days_held is not None:
                flag = ""
                if cp and target and cp >= target:
                    flag = "🎯 Target hit — close now"
                elif cp and stop and cp <= stop:
                    flag = "⚠️ Stop breached — close now"
                elif days_held >= horizon:
                    flag = "⏰ Horizon expired — close now"
                st.caption(f"Days held: {days_held} / {horizon}" + (f"  ·  **{flag}**" if flag else ""))

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("Close @ Market", key=f"closemkt_{row['trade_id']}"):
                    ok, msg = writer.close_trade(row["trade_id"], exit_price=None, reason="manual")
                    (st.success if ok else st.error)(msg)
                    st.rerun()
            with col_b:
                custom_price = st.number_input("Custom exit ₹", min_value=0.0, step=0.5,
                                                value=float(cp) if cp else entry,
                                                key=f"customprice_{row['trade_id']}")
                if st.button("Close @ Custom", key=f"closecustom_{row['trade_id']}"):
                    ok, msg = writer.close_trade(row["trade_id"], exit_price=custom_price, reason="manual")
                    (st.success if ok else st.error)(msg)
                    st.rerun()
            with col_c:
                new_stop = st.number_input("Edit stop ₹", value=stop, step=0.5, key=f"editstop_{row['trade_id']}")
                new_target = st.number_input("Edit target ₹", value=target, step=0.5, key=f"edittarget_{row['trade_id']}")
                if st.button("Save Edits", key=f"saveedit_{row['trade_id']}"):
                    ok, msg = writer.edit_trade(row["trade_id"], stop_loss=new_stop, target_price=new_target)
                    (st.success if ok else st.error)(msg)
                    st.rerun()

    st.divider()
    if st.button("🤖 Auto-close all that hit stop/target/expiry"):
        tickers = tuple(open_t["ticker"].unique().tolist())
        from app.utils.prices import fetch_latest_prices
        live = fetch_latest_prices(tickers)
        price_rows = [{"ticker": t, "date": pd.Timestamp.today(), "close": p} for t, p in live.items()]
        if not price_rows:
            st.warning("Could not fetch live prices.")
        else:
            price_df = pd.DataFrame(price_rows)
            ok, msg = writer.auto_close_expired(price_df)
            (st.success if ok else st.error)(msg)
            st.rerun()

st.divider()

# ── Closed trades ─────────────────────────────────────────────────────────────
st.subheader(f"Closed Trades ({len(closed_t)})")
if closed_t.empty:
    st.info("No closed trades yet.")
else:
    disp_cols = [c for c in [
        "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
        "gross_pnl", "entry_charges", "exit_charges", "pnl", "pnl_pct", "exit_reason",
    ] if c in closed_t.columns]
    closed_sorted = closed_t[disp_cols].sort_values("exit_date", ascending=False)

    def _color_pnl(val):
        if not isinstance(val, (int, float)):
            return ""
        return "color: #22c55e" if val > 0 else "color: #ef4444"

    fmt = {c: "₹{:,.2f}" for c in ["entry_price", "exit_price", "gross_pnl", "entry_charges", "exit_charges"]
           if c in disp_cols}
    fmt.update({"pnl": "₹{:+,.2f}", "pnl_pct": "{:+.2f}%"})

    st.dataframe(
        closed_sorted.style.format(fmt).map(_color_pnl, subset=[c for c in ["pnl", "pnl_pct"] if c in disp_cols]),
        width="stretch",
        height=350,
    )

    if {"pnl", "gross_pnl"}.issubset(closed_t.columns):
        total_gross = closed_t["gross_pnl"].sum()
        total_net   = closed_t["pnl"].sum()
        total_charges = (closed_t.get("entry_charges", 0).sum() + closed_t.get("exit_charges", 0).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Gross P&L", f"₹{total_gross:+,.0f}")
        c2.metric("Total Net P&L", f"₹{total_net:+,.0f}")
        c3.metric("Total Charges Paid (closed)", f"₹{total_charges:,.0f}")
