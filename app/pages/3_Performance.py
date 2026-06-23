"""Page 3: Prediction Performance — IC, accuracy, hit rates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
import streamlit as st
import pandas as pd

from app.utils.data_loader import load_weekly_ic, load_outcomes, load_model_runs, refresh_all
from app.utils.charts import ic_trend_chart, dir_accuracy_chart

st.set_page_config(page_title="Performance", page_icon="📈", layout="wide")
st.title("📈 Prediction Performance")

col_r, col_w = st.columns([1, 5])
with col_r:
    if st.button("⟳ Refresh"):
        refresh_all()
        st.rerun()
with col_w:
    n_weeks = st.slider("Weeks to display", min_value=4, max_value=26, value=12, step=2)

# ── Load ─────────────────────────────────────────────────────────────────────
ic_df    = load_weekly_ic(n_weeks=n_weeks)
outcomes = load_outcomes(n_weeks=n_weeks)
runs     = load_model_runs()

# ── Summary metrics ───────────────────────────────────────────────────────────
if not ic_df.empty:
    recent_ic   = ic_df["ic"].dropna().iloc[-1] if len(ic_df) > 0 else float("nan")
    avg_ic      = ic_df["ic"].dropna().mean()
    avg_dir_acc = ic_df["dir_accuracy"].dropna().mean() if "dir_accuracy" in ic_df.columns else float("nan")
    total_obs   = int(ic_df["n_obs"].sum()) if "n_obs" in ic_df.columns else 0
else:
    recent_ic = avg_ic = avg_dir_acc = float("nan")
    total_obs = 0

def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.4f}" if not pct else f"{v:.1%}"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Weekly IC",    _fmt(recent_ic))
c2.metric("Avg IC (period)",     _fmt(avg_ic))
c3.metric("Avg Dir Accuracy",    _fmt(avg_dir_acc, pct=True))
c4.metric("Resolved Predictions", f"{total_obs:,}")

st.divider()

# ── IC + accuracy charts ──────────────────────────────────────────────────────
if not ic_df.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(ic_trend_chart(ic_df), width="stretch")
    with col2:
        st.plotly_chart(dir_accuracy_chart(ic_df), width="stretch")
else:
    st.info("No resolved predictions yet. Predictions resolve after the horizon period has passed.")

# ── IC table ──────────────────────────────────────────────────────────────────
if not ic_df.empty:
    with st.expander("Weekly IC table"):
        disp = ic_df.copy()
        if "ic" in disp.columns:
            disp["ic"] = disp["ic"].apply(lambda v: f"{v:.4f}" if v is not None else "—")
        if "dir_accuracy" in disp.columns:
            disp["dir_accuracy"] = disp["dir_accuracy"].apply(lambda v: f"{v:.1%}" if v is not None else "—")
        st.dataframe(disp, width="stretch")

# ── Scatter: prob_up vs actual return ─────────────────────────────────────────
if not outcomes.empty and "prob_up" in outcomes.columns and "actual_fwd_ret" in outcomes.columns:
    st.subheader("Signal Confidence vs Actual Return")
    import plotly.express as px
    plot_df = outcomes.dropna(subset=["prob_up", "actual_fwd_ret"]).sample(
        min(500, len(outcomes)), random_state=42
    )
    fig = px.scatter(
        plot_df, x="prob_up", y="actual_fwd_ret",
        color="is_correct" if "is_correct" in plot_df.columns else None,
        color_discrete_map={True: "#22c55e", False: "#ef4444"},
        labels={"prob_up": "Model Confidence (prob_up)", "actual_fwd_ret": "Actual Log Return"},
        title="Calibration: Confidence vs Realised Return",
        opacity=0.5,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="#6b7280", opacity=0.5)
    fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="#e2e8f0")
    st.plotly_chart(fig, width="stretch")

# ── Model run history ─────────────────────────────────────────────────────────
if not runs.empty:
    st.subheader("Model Run History")
    disp_cols = [c for c in ["run_id","run_date","oof_ic","bt_sharpe",
                              "bt_cagr","bt_max_drawdown","is_deployed","n_trials"] if c in runs.columns]
    st.dataframe(
        runs[disp_cols].style.format({
            "oof_ic":        lambda v: f"{v:.4f}" if v is not None else "—",
            "bt_sharpe":     lambda v: f"{v:.3f}" if v is not None else "—",
            "bt_cagr":       lambda v: f"{v:.1%}"  if v is not None else "—",
            "bt_max_drawdown": lambda v: f"{v:.1%}" if v is not None else "—",
        }),
        width="stretch",
    )
