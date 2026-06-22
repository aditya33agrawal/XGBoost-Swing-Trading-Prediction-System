"""Page 2: Paper Trading Portfolio."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from datetime import date

from app.utils.data_loader import load_paper_trades, load_portfolio_meta, refresh_all
from app.utils.charts import equity_curve_chart, exit_reason_chart, pnl_distribution_chart

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
st.title("💼 Paper Trading Portfolio")
st.caption("Analytics overview. To open, close, or edit positions, go to **🎯 Trade Desk**.")

col_r, col_link = st.columns([1, 5])
with col_r:
    if st.button("⟳ Refresh"):
        refresh_all()
        st.rerun()
with col_link:
    st.markdown("🎯 [Manage positions in Trade Desk](Trade_Desk)")

# ── Load data ────────────────────────────────────────────────────────────────
trades = load_paper_trades()
meta   = load_portfolio_meta()

initial_capital = float(meta.get("initial_capital", 1_000_000))
cash            = float(meta.get("cash", initial_capital))

# ── Portfolio metrics ─────────────────────────────────────────────────────────
open_t   = trades[trades["status"] == "open"]   if not trades.empty else pd.DataFrame()
closed_t = trades[trades["status"] == "closed"] if not trades.empty else pd.DataFrame()

total_pnl   = closed_t["pnl"].sum()    if not closed_t.empty else 0.0
invested    = sum(
    float(r.get("shares", 0)) * float(r.get("entry_price", 0))
    for _, r in open_t.iterrows()
) if not open_t.empty else 0.0
port_value  = cash + invested
total_ret   = (port_value / initial_capital - 1) * 100
n_win       = int((closed_t["pnl"] > 0).sum()) if not closed_t.empty else 0
win_rate    = n_win / max(len(closed_t), 1) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portfolio Value",  f"₹{port_value:,.0f}")
c2.metric("Cash",             f"₹{cash:,.0f}")
c3.metric("Total Return",     f"{total_ret:+.1f}%",
          delta=f"₹{total_pnl:+,.0f}")
c4.metric("Open Positions",   len(open_t))
c5.metric("Win Rate",         f"{win_rate:.0f}%",
          delta=f"{n_win} / {len(closed_t)} trades")

# ── Gross vs net (cost drag) ──────────────────────────────────────────────────
if not closed_t.empty and "gross_pnl" in closed_t.columns:
    total_gross = closed_t["gross_pnl"].sum()
    total_charges = closed_t.get("entry_charges", pd.Series(dtype=float)).sum() + \
                    closed_t.get("exit_charges", pd.Series(dtype=float)).sum()
    g1, g2, g3 = st.columns(3)
    g1.metric("Gross P&L (price only)", f"₹{total_gross:+,.0f}")
    g2.metric("Net P&L (after charges)", f"₹{total_pnl:+,.0f}")
    g3.metric("Cost Drag", f"₹{total_charges:,.0f}",
              delta=f"{(total_charges/abs(total_gross)*100 if total_gross else 0):.1f}% of gross")

st.divider()

# ── Equity curve ──────────────────────────────────────────────────────────────
st.plotly_chart(equity_curve_chart(trades, initial_capital), use_container_width=True)

# ── Open positions ────────────────────────────────────────────────────────────
st.subheader("Open Positions")
if open_t.empty:
    st.info("No open positions.")
else:
    disp_cols = [c for c in ["ticker","entry_date","entry_price","shares",
                              "stop_loss","target_price","prob_up"] if c in open_t.columns]
    st.dataframe(
        open_t[disp_cols].style.format({
            "entry_price":  "₹{:,.2f}",
            "stop_loss":    "₹{:,.2f}",
            "target_price": "₹{:,.2f}",
            "prob_up":      "{:.1%}",
        }),
        use_container_width=True,
    )

# ── Closed trades ─────────────────────────────────────────────────────────────
st.subheader("Closed Trades")
if closed_t.empty:
    st.info("No closed trades yet.")
else:
    disp_cols = [c for c in ["ticker","entry_date","exit_date","entry_price",
                              "exit_price","pnl","pnl_pct","exit_reason"] if c in closed_t.columns]
    closed_sorted = closed_t[disp_cols].sort_values("exit_date", ascending=False)

    def _color_pnl(val):
        if not isinstance(val, (int, float)):
            return ""
        return "color: #22c55e" if val > 0 else "color: #ef4444"

    st.dataframe(
        closed_sorted.style.format({
            "entry_price": "₹{:,.2f}",
            "exit_price":  "₹{:,.2f}",
            "pnl":         "₹{:+,.0f}",
            "pnl_pct":     "{:+.1f}%",
        }).applymap(_color_pnl, subset=["pnl","pnl_pct"]),
        use_container_width=True,
        height=300,
    )

    # ── Charts row ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(exit_reason_chart(trades), use_container_width=True)
    with col2:
        st.plotly_chart(pnl_distribution_chart(trades), use_container_width=True)

# ── Summary stats ─────────────────────────────────────────────────────────────
if not closed_t.empty and "pnl_pct" in closed_t.columns:
    st.subheader("Statistics")
    pnls = closed_t["pnl_pct"].dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg P&L per trade",  f"{pnls.mean():+.2f}%")
    c2.metric("Best trade",         f"{pnls.max():+.2f}%")
    c3.metric("Worst trade",        f"{pnls.min():+.2f}%")
    c4.metric("Profit factor",
              f"{closed_t[closed_t['pnl']>0]['pnl'].sum() / max(abs(closed_t[closed_t['pnl']<0]['pnl'].sum()), 1):.2f}x")
