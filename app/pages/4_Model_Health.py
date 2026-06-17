"""Page 4: Model Health — IC trend, feature importance, drift signals."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
import streamlit as st
import pandas as pd

from app.utils.data_loader import load_weekly_ic, load_feature_importance, load_model_runs, refresh_all
from app.utils.charts import ic_trend_chart, feature_importance_chart

st.set_page_config(page_title="Model Health", page_icon="🩺", layout="wide")
st.title("🩺 Model Health")

if st.button("⟳ Refresh"):
    refresh_all()
    st.rerun()

# ── Load ─────────────────────────────────────────────────────────────────────
ic_df = load_weekly_ic(n_weeks=12)
runs  = load_model_runs(limit=10)

# ── Health banner ─────────────────────────────────────────────────────────────
def _health_status(ic_df: pd.DataFrame) -> tuple[str, str, str]:
    """Returns (status, color, message)."""
    if ic_df.empty:
        return "UNKNOWN", "#6b7280", "No resolved predictions yet."
    recent = ic_df["ic"].dropna().tail(3)
    if recent.empty:
        return "UNKNOWN", "#6b7280", "Insufficient data."
    last_ic   = recent.iloc[-1]
    n_bad     = int((recent < 0.0).sum())
    n_warn    = int((recent < 0.01).sum())
    if n_bad >= 2:
        return "CRITICAL", "#ef4444", f"IC has been negative for {n_bad} consecutive weeks ({last_ic:.4f}). Consider emergency retrain."
    if last_ic < 0.0:
        return "WARNING", "#f59e0b", f"Latest IC is negative ({last_ic:.4f}). Monitor closely."
    if n_warn >= 2:
        return "CAUTION", "#f59e0b", f"IC below target for {n_warn} weeks. Model may be weakening."
    return "HEALTHY", "#22c55e", f"IC is positive ({last_ic:.4f}). Model performing as expected."

status, color, msg = _health_status(ic_df)

st.markdown(
    f"""<div style="padding:16px;border-radius:8px;border-left:4px solid {color};
    background:rgba(30,41,59,0.8);margin-bottom:16px">
    <span style="color:{color};font-weight:bold;font-size:18px">⬤ {status}</span>
    <span style="color:#e2e8f0;margin-left:12px">{msg}</span>
    </div>""",
    unsafe_allow_html=True,
)

# ── IC trend chart ────────────────────────────────────────────────────────────
st.plotly_chart(ic_trend_chart(ic_df), use_container_width=True)

st.divider()

# ── Feature importance ────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Feature Importance")
    deployed = runs[runs.get("is_deployed", False) == True] if not runs.empty and "is_deployed" in runs.columns else pd.DataFrame()
    run_id_to_load = str(deployed.iloc[0]["run_id"]) if not deployed.empty else "latest"
    fi_df = load_feature_importance(run_id=run_id_to_load)
    if not fi_df.empty:
        st.plotly_chart(feature_importance_chart(fi_df, top_n=20), use_container_width=True)
    else:
        st.info("Feature importance not yet available. Requires a full training run.")

with col2:
    st.subheader("Model Versions")
    if not runs.empty:
        disp_cols = [c for c in ["run_id","run_date","oof_ic","bt_sharpe","is_deployed"] if c in runs.columns]
        def _color_deployed(val):
            return "background-color: rgba(34,197,94,0.2)" if val else ""
        st.dataframe(
            runs[disp_cols].style
            .format({"oof_ic": lambda v: f"{v:.4f}" if v is not None else "—",
                     "bt_sharpe": lambda v: f"{v:.3f}" if v is not None else "—"})
            .applymap(_color_deployed, subset=["is_deployed"]),
            use_container_width=True,
            height=350,
        )
    else:
        st.info("No model runs recorded yet.")

# ── IC health details ─────────────────────────────────────────────────────────
if not ic_df.empty:
    with st.expander("IC Health Details"):
        thresholds = {"target": 0.01, "warn": 0.00, "critical": -0.01}
        for label, thresh in thresholds.items():
            count = int((ic_df["ic"].dropna() < thresh).sum())
            icon  = "🔴" if label == "critical" else ("🟡" if label == "warn" else "🟢")
            st.write(f"{icon} Weeks below {label} threshold ({thresh:.2f}): **{count}** of {len(ic_df)}")

        consecutive_bad = 0
        for ic in ic_df["ic"].dropna().tolist()[::-1]:
            if ic < 0:
                consecutive_bad += 1
            else:
                break
        if consecutive_bad > 0:
            st.warning(f"⚠️ {consecutive_bad} consecutive week(s) with negative IC")

# ── Recommendations ───────────────────────────────────────────────────────────
st.subheader("Recommendations")
recs = []
if status in ("CRITICAL", "WARNING"):
    recs.append("🔴 **Run emergency retrain** (`python scripts/weekly_retrain.py --trials 100`)")
    recs.append("🔴 Check for **data quality issues** — run `python run_pipeline.py --fast-signals` and inspect validation output")
if status == "CAUTION":
    recs.append("🟡 **Increase Optuna trials** next Sunday (set `TRIALS = 100` in colab_weekly.ipynb)")
    recs.append("🟡 Review feature importance — top feature shifts may indicate **regime change**")
if status == "HEALTHY":
    recs.append("🟢 Continue weekly retraining as scheduled")
    recs.append("🟢 Consider expanding to **long/short** mode once 6+ weeks of positive IC confirmed")

if not ic_df.empty and len(ic_df) < 4:
    recs.append("ℹ️ **Insufficient data** for reliable IC estimates — need at least 4 resolved weeks")

for r in recs:
    st.markdown(r)
