"""Page 8: Account — broker-style funds statement (real money accounting)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from datetime import date

from app.utils.data_loader import load_account_ledger, load_paper_trades, load_portfolio_meta, refresh_all
from app.utils import writer

st.set_page_config(page_title="Account", page_icon="🧾", layout="wide")
st.title("🧾 Account — Funds Statement")
st.caption("Every cash movement in the paper account: BUY/SELL fills, itemized CHARGE rows "
           "(STT, exchange, SEBI, stamp, GST, DP), deposits and withdrawals.")

if st.button("⟳ Refresh"):
    refresh_all()
    st.rerun()

ledger = load_account_ledger()
trades = load_paper_trades()
meta   = load_portfolio_meta()

cash = float(meta.get("cash", meta.get("initial_capital", 1_000_000)))
initial_capital = float(meta.get("initial_capital", 1_000_000))

open_t = trades[trades["status"] == "open"] if not trades.empty else pd.DataFrame()
deployed = sum(float(r.get("shares", 0)) * float(r.get("entry_price", 0)) for _, r in open_t.iterrows()) \
    if not open_t.empty else 0.0

total_charges = 0.0
if not ledger.empty and "type" in ledger.columns:
    total_charges = -ledger[ledger["type"] == "CHARGE"]["amount"].sum()

turnover = 0.0
if not ledger.empty and "type" in ledger.columns:
    turnover = ledger[ledger["type"].isin(["BUY", "SELL"])]["amount"].abs().sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cash", f"₹{cash:,.0f}")
c2.metric("Deployed (open positions)", f"₹{deployed:,.0f}")
c3.metric("Total Equity", f"₹{cash + deployed:,.0f}")
c4.metric("Total Charges Paid", f"₹{total_charges:,.0f}")
c5.metric("Charges / Turnover", f"{(total_charges / turnover * 100):.3f}%" if turnover else "—")

st.divider()

# ── Deposit / Withdraw ────────────────────────────────────────────────────────
st.subheader("Manage Funds")
dc1, dc2, dc3 = st.columns(3)
with dc1:
    amount = st.number_input("Amount (₹)", min_value=0.0, step=1000.0)
with dc2:
    note = st.text_input("Note", value="")
with dc3:
    st.write("")
    st.write("")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Deposit", type="primary", disabled=amount <= 0):
            ok, msg = writer.deposit(amount, note)
            (st.success if ok else st.error)(msg)
            st.rerun()
    with b2:
        if st.button("Withdraw", disabled=amount <= 0):
            ok, msg = writer.withdraw(amount, note)
            (st.success if ok else st.error)(msg)
            st.rerun()

st.divider()

# ── Ledger table ───────────────────────────────────────────────────────────────
st.subheader("Funds Ledger")

if ledger.empty:
    st.info("No ledger entries yet.")
else:
    fc1, fc2 = st.columns(2)
    with fc1:
        type_filter = st.multiselect("Type", sorted(ledger["type"].unique().tolist()),
                                      default=sorted(ledger["type"].unique().tolist()))
    with fc2:
        ticker_filter = st.multiselect("Ticker", sorted([t for t in ledger["ticker"].unique().tolist() if t]))

    view = ledger[ledger["type"].isin(type_filter)]
    if ticker_filter:
        view = view[view["ticker"].isin(ticker_filter)]

    def _color_amount(val):
        if not isinstance(val, (int, float)):
            return ""
        return "color: #22c55e" if val > 0 else ("color: #ef4444" if val < 0 else "")

    disp_cols = [c for c in ["ts", "type", "ticker", "qty", "price", "amount", "running_balance", "note"]
                 if c in view.columns]
    st.dataframe(
        view[disp_cols].style.format({
            "price": "₹{:,.2f}", "amount": "₹{:+,.2f}", "running_balance": "₹{:,.2f}",
        }).map(_color_amount, subset=["amount"]),
        width="stretch",
        height=450,
    )

    st.download_button(
        "⬇ Download Statement (CSV)",
        data=view.to_csv(index=False),
        file_name=f"account_statement_{date.today()}.csv",
        mime="text/csv",
    )
