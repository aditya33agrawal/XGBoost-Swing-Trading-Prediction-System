"""Page 7: Prediction Journal — what the model predicted vs. what actually happened.

This is the ground-truth ledger that feeds the next weekly Colab run: every
prediction, its resolution status, and (once resolved) the actual outcome.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
from datetime import date, timedelta

import streamlit as st
import pandas as pd

from app.utils.data_loader import load_prediction_journal, refresh_all
from app.utils import writer

st.set_page_config(page_title="Prediction Journal", page_icon="📓", layout="wide")
st.title("📓 Prediction Journal")
st.caption("One row per prediction — predicted (prob_up, entry/stop/target) vs. ground truth (actual close, "
           "actual return, hit/miss) once the horizon has elapsed.")

col_r, col_w = st.columns([1, 5])
with col_r:
    if st.button("⟳ Refresh"):
        refresh_all()
        st.rerun()
with col_w:
    n_weeks = st.slider("Weeks to display", 4, 26, 12, 2)

df = load_prediction_journal(n_weeks=n_weeks)

if df.empty:
    st.info("No predictions logged yet. Run the pipeline or open a trade from Signals.")
    st.stop()

pending  = df[~df["outcome_resolved"].astype(bool)]
resolved = df[df["outcome_resolved"].astype(bool)]

# ── Resolve outcomes now ──────────────────────────────────────────────────────
st.subheader("Resolve Outcomes")
horizon_default = int(df["horizon_days"].mode().iloc[0]) if "horizon_days" in df.columns and not df.empty else 5
horizon = st.number_input("Horizon (trading days) used to find resolvable predictions",
                           min_value=1, value=horizon_default, step=1)

if st.button("🔄 Resolve Outcomes Now", type="primary"):
    with st.spinner("Fetching closes and resolving …"):
        try:
            from app.utils.prices import fetch_closes
            from src.db.supabase_client import get_supabase_client
            from src.tracking.outcome_tracker import resolve_outcomes
            from app.utils.env import get_supabase_url, get_supabase_key

            unresolved_tickers = tuple(sorted(pending["ticker"].unique().tolist())) if not pending.empty else ()
            if not unresolved_tickers:
                st.info("No unresolved predictions to settle.")
            else:
                min_date = pending["signal_date"].min()
                start = (pd.to_datetime(min_date) - timedelta(days=2)).strftime("%Y-%m-%d")
                end = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")
                price_df = fetch_closes(unresolved_tickers, start, end)

                sb = get_supabase_client(get_supabase_url(), get_supabase_key())
                n = resolve_outcomes(price_df=price_df, supabase_client=sb, horizon_days=int(horizon))
                st.success(f"Resolved {n} prediction(s).")
                refresh_all()
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# ── Scorecard ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Predictions", len(df))
c2.metric("Pending Resolution", len(pending))
c3.metric("Resolved", len(resolved))
if not resolved.empty and "is_correct" in resolved.columns:
    hit_rate = resolved["is_correct"].mean()
    c4.metric("Hit Rate", f"{hit_rate:.1%}")
elif not resolved.empty and "actual_fwd_ret" in resolved.columns and "signal" in resolved.columns:
    long_correct = resolved[(resolved["signal"] == "LONG") & (resolved["actual_fwd_ret"] > 0)]
    hit_rate = len(long_correct) / max(len(resolved[resolved["signal"] == "LONG"]), 1)
    c4.metric("Hit Rate (LONG)", f"{hit_rate:.1%}")
else:
    c4.metric("Hit Rate", "—")

if not resolved.empty and "actual_fwd_ret" in resolved.columns and "signal" in resolved.columns:
    long_ret = resolved[resolved["signal"] == "LONG"]["actual_fwd_ret"].mean()
    neutral_ret = resolved[resolved["signal"] == "NEUTRAL"]["actual_fwd_ret"].mean() \
        if (resolved["signal"] == "NEUTRAL").any() else float("nan")
    c5, c6 = st.columns(2)
    c5.metric("Mean fwd ret — LONG", f"{long_ret:.4f}" if not math.isnan(long_ret) else "—")
    c6.metric("Mean fwd ret — NEUTRAL", f"{neutral_ret:.4f}" if not math.isnan(neutral_ret) else "—")

st.divider()

# ── Filters ────────────────────────────────────────────────────────────────────
st.subheader("Journal")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    status_filter = st.selectbox("Status", ["All", "Pending", "Resolved"])
with fc2:
    tickers = ["All"] + sorted(df["ticker"].unique().tolist())
    ticker_filter = st.selectbox("Ticker", tickers)
with fc3:
    signal_filter = st.selectbox("Signal", ["All"] + sorted(df["signal"].dropna().unique().tolist())) \
        if "signal" in df.columns else "All"
with fc4:
    correct_filter = st.selectbox("Correctness", ["All", "Correct", "Incorrect"]) \
        if "is_correct" in df.columns else "All"

view = df.copy()
if status_filter == "Pending":
    view = view[~view["outcome_resolved"].astype(bool)]
elif status_filter == "Resolved":
    view = view[view["outcome_resolved"].astype(bool)]
if ticker_filter != "All":
    view = view[view["ticker"] == ticker_filter]
if signal_filter != "All" and "signal" in view.columns:
    view = view[view["signal"] == signal_filter]
if correct_filter != "All" and "is_correct" in view.columns:
    view = view[view["is_correct"] == (correct_filter == "Correct")]

display_cols = [c for c in [
    "signal_date", "ticker", "signal", "prob_up", "entry_price", "stop_loss", "target_price",
    "horizon_days", "outcome_resolved", "outcome_date", "actual_close", "actual_fwd_ret",
    "hit_target", "hit_stop", "is_correct", "model_version",
] if c in view.columns]

st.dataframe(view[display_cols], use_container_width=True, height=420)

# ── Calibration scatter ───────────────────────────────────────────────────────
if not resolved.empty and {"prob_up", "actual_fwd_ret"}.issubset(resolved.columns):
    st.subheader("Confidence vs. Actual Return")
    import plotly.express as px
    plot_df = resolved.dropna(subset=["prob_up", "actual_fwd_ret"])
    fig = px.scatter(
        plot_df, x="prob_up", y="actual_fwd_ret",
        color="is_correct" if "is_correct" in plot_df.columns else None,
        color_discrete_map={True: "#22c55e", False: "#ef4444"},
        labels={"prob_up": "Model Confidence", "actual_fwd_ret": "Actual Log Return"},
        opacity=0.6,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", opacity=0.5)
    fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font_color="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Manual ground-truth override ──────────────────────────────────────────────
with st.expander("✏️ Manual ground-truth override (pending/odd row)"):
    if pending.empty:
        st.info("No pending predictions to override.")
    else:
        idx_options = pending.apply(lambda r: f"{r['ticker']} · {r['signal_date']} (id={r.get('id', '?')})", axis=1).tolist()
        choice = st.selectbox("Prediction", idx_options)
        sel_row = pending.iloc[idx_options.index(choice)]
        actual_close = st.number_input("Actual close (₹)", min_value=0.0, step=0.5,
                                        value=float(sel_row.get("entry_price", 0) or 0))
        if st.button("Save Ground Truth"):
            entry = float(sel_row.get("entry_price", 0) or 0)
            fwd_ret = math.log(actual_close / entry) if entry > 0 else 0.0
            row = {
                "prediction_id": sel_row.get("id"),
                "run_id": sel_row.get("run_id", ""),
                "signal_date": sel_row.get("signal_date"),
                "outcome_date": str(date.today()),
                "ticker": sel_row.get("ticker"),
                "prob_up": sel_row.get("prob_up"),
                "actual_fwd_ret": round(fwd_ret, 6),
                "label_direction": 1 if fwd_ret > 0 else -1,
                "is_correct": (sel_row.get("signal") == "LONG" and fwd_ret > 0),
                "exit_reason": "manual_override",
            }
            ok, msg = writer.upsert_outcome(row)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()

# ── Export ─────────────────────────────────────────────────────────────────────
st.subheader("Export for next weekly run")
ec1, ec2 = st.columns(2)
with ec1:
    st.download_button(
        "⬇ Export resolved outcomes (CSV)",
        data=resolved.to_csv(index=False) if not resolved.empty else "",
        file_name=f"resolved_outcomes_{date.today()}.csv",
        mime="text/csv",
        disabled=resolved.empty,
    )
with ec2:
    st.download_button(
        "⬇ Export resolved outcomes (JSON)",
        data=resolved.to_json(orient="records", indent=2) if not resolved.empty else "",
        file_name=f"resolved_outcomes_{date.today()}.json",
        mime="application/json",
        disabled=resolved.empty,
    )
