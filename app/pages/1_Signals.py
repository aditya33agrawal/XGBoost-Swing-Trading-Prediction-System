"""Page 1: Today's Signals — and trade them straight into the paper account."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

from app.utils.data_loader import load_signals, load_paper_trades, load_portfolio_meta, refresh_all
from app.utils import writer
from app.utils.prices import fetch_latest_prices

st.set_page_config(page_title="Signals", page_icon="📡", layout="wide")
st.title("📡 Today's Signals")

# ── Refresh ──────────────────────────────────────────────────────────────────
col_r, col_date = st.columns([1, 5])
with col_r:
    if st.button("⟳ Refresh"):
        refresh_all()
        st.rerun()

# ── Load data ────────────────────────────────────────────────────────────────
df = load_signals()

if df.empty:
    st.info("No signals yet. Run the pipeline to generate predictions.")
    st.stop()

signal_date = df["signal_date"].iloc[0] if "signal_date" in df.columns else "unknown"
with col_date:
    st.caption(f"Signal date: **{signal_date}** · Horizon: **{df['horizon_days'].iloc[0] if 'horizon_days' in df.columns else '?'} days**")

trades_df = load_paper_trades()
open_tickers = set(
    trades_df[trades_df["status"] == "open"]["ticker"].tolist()
) if not trades_df.empty else set()

meta = load_portfolio_meta()
cash = float(meta.get("cash", meta.get("initial_capital", 1_000_000)))
position_size_pct = float(meta.get("position_size_pct", 0.05))

# ── Metric cards ─────────────────────────────────────────────────────────────
long_df    = df[df["signal"] == "LONG"] if "signal" in df.columns else pd.DataFrame()
neutral_df = df[df["signal"] == "NEUTRAL"] if "signal" in df.columns else pd.DataFrame()
avg_prob   = long_df["prob_up"].mean() if ("prob_up" in long_df.columns and not long_df.empty) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Signals",     len(df))
c2.metric("LONG Signals",      len(long_df))
c3.metric("NEUTRAL",           len(neutral_df))
c4.metric("Avg Confidence",    f"{avg_prob:.1%}" if avg_prob else "—")

st.divider()

# ── Signal table ─────────────────────────────────────────────────────────────
display_cols = [c for c in [
    "ticker", "signal", "prob_up",
    "entry_price", "stop_loss", "target_price",
    "risk_reward", "atr14",
] if c in df.columns]

view = df[display_cols].copy()

def _color_signal(val):
    if val == "LONG":
        return "background-color: rgba(34,197,94,0.2); color: #22c55e; font-weight:bold"
    return "color: #6b7280"

def _color_prob(val):
    if not isinstance(val, float):
        return ""
    if val >= 0.65:
        return "color: #22c55e; font-weight:bold"
    if val >= 0.55:
        return "color: #f59e0b"
    return "color: #6b7280"

styled = (
    view.style
    .format({
        "prob_up":      "{:.1%}",
        "entry_price":  "₹{:,.2f}",
        "stop_loss":    "₹{:,.2f}",
        "target_price": "₹{:,.2f}",
        "risk_reward":  "{:.2f}x",
        "atr14":        "₹{:,.2f}",
    })
    .applymap(_color_signal, subset=["signal"])
    .applymap(_color_prob,   subset=["prob_up"])
)

st.dataframe(styled, use_container_width=True, height=350)

# ── Probability chart ─────────────────────────────────────────────────────────
if not long_df.empty and "prob_up" in long_df.columns:
    import plotly.express as px
    fig = px.bar(
        long_df.sort_values("prob_up", ascending=True),
        x="prob_up", y="ticker", orientation="h",
        color="prob_up",
        color_continuous_scale=["#f59e0b", "#22c55e"],
        range_color=[0.5, 1.0],
        labels={"prob_up": "Confidence (prob_up)", "ticker": ""},
        title="LONG Signal Confidence",
    )
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="#e2e8f0", coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "⬇ Download CSV",
    data=df.to_csv(index=False),
    file_name=f"signals_{signal_date}.csv",
    mime="text/csv",
)

st.divider()

# ── Trade actions ──────────────────────────────────────────────────────────────
st.subheader("🎯 Trade These Signals")

if long_df.empty:
    st.info("No LONG signals to trade today.")
else:
    # Live prices for every LONG ticker — these (not the stale signal entry_price)
    # are what the order will actually fill at; writer.open_trade re-fetches the
    # same way so the preview below matches the real fill.
    live_prices = fetch_latest_prices(tuple(long_df["ticker"].tolist()))

    # Bulk open
    with st.expander("⚡ Bulk: open all LONG signals", expanded=False):
        top_n = st.slider("Open top N by confidence (skips already-open tickers)",
                           1, len(long_df), min(5, len(long_df)))
        if st.button("Open Top N", type="primary"):
            candidates = long_df.sort_values("prob_up", ascending=False).head(top_n)
            results = []
            for _, row in candidates.iterrows():
                ticker = row["ticker"]
                if ticker in open_tickers:
                    results.append(f"⏭️ {ticker} — already open, skipped")
                    continue
                signal_entry = float(row["entry_price"])
                live = live_prices.get(ticker)
                entry = float(live) if live else signal_entry
                shares = max(1, int(cash * position_size_pct // entry))
                ok, msg = writer.open_trade(
                    ticker, entry, shares,
                    stop_loss=float(row.get("stop_loss", entry * 0.97)),
                    target_price=float(row.get("target_price", entry * 1.04)),
                    horizon_days=int(row.get("horizon_days", 5)),
                    prob_up=float(row.get("prob_up", 0.5)),
                    opened_via="signal",
                )
                results.append(f"{'✅' if ok else '❌'} {ticker} — {msg}")
            for r in results:
                st.write(r)
            st.rerun()

    # Per-row trade
    for _, row in long_df.sort_values("prob_up", ascending=False).iterrows():
        ticker = row["ticker"]
        signal_entry = float(row["entry_price"])
        live = live_prices.get(ticker)
        entry = float(live) if live else signal_entry
        already_open = ticker in open_tickers

        with st.expander(
            f"{'🟢 OPEN ' if not already_open else '✅ ALREADY OPEN — '}{ticker}  ·  "
            f"prob_up {row.get('prob_up', 0):.1%}  ·  live ₹{entry:,.2f}"
            f"{'' if live else ' (live price unavailable — using signal price)'}",
            expanded=False,
        ):
            if live:
                st.caption(f"Signal price (when generated): ₹{signal_entry:,.2f}  ·  "
                           f"Live price (Yahoo, now): ₹{entry:,.2f}")
            else:
                st.warning(f"Could not fetch a live price for {ticker} — falling back to the signal price.")
            default_shares = max(1, int(cash * position_size_pct // entry))
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                shares = st.number_input("Shares", min_value=1, value=default_shares,
                                          step=1, key=f"shares_{ticker}")
            with col_b:
                stop = st.number_input("Stop Loss (₹)", value=float(row.get("stop_loss", entry * 0.97)),
                                        step=0.5, key=f"stop_{ticker}")
            with col_c:
                target = st.number_input("Target (₹)", value=float(row.get("target_price", entry * 1.04)),
                                          step=0.5, key=f"target_{ticker}")

            preview = writer.preview_buy(entry, int(shares))
            st.markdown("**Contract note preview**")
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("Fill price", f"₹{preview['fill_price']:,.2f}")
            pcol2.metric("Charges", f"₹{preview['charges']['total']:,.2f}")
            pcol3.metric("Total debit", f"₹{preview['total_debit']:,.2f}")
            pcol4.metric("Break-even", f"₹{preview['breakeven_price']:,.2f}")

            with st.popover("Charge breakdown"):
                st.json(preview["charges"])

            if st.button(f"Confirm — Open {ticker}", key=f"confirm_{ticker}", type="primary"):
                ok, msg = writer.open_trade(
                    ticker, entry, int(shares), float(stop), float(target),
                    horizon_days=int(row.get("horizon_days", 5)),
                    prob_up=float(row.get("prob_up", 0.5)),
                    opened_via="signal",
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()
