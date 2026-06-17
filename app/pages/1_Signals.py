"""Page 1: Today's Signals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from app.utils.data_loader import load_signals, refresh_all

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

st.dataframe(styled, use_container_width=True, height=400)

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

# ── Download ─────────────────────────────────────────────────────────────────
st.download_button(
    "⬇ Download CSV",
    data=df.to_csv(index=False),
    file_name=f"signals_{signal_date}.csv",
    mime="text/csv",
)
