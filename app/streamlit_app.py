"""Main Streamlit entry point — session state, sidebar, global styles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, datetime

import streamlit as st

from app.utils.env import get_supabase_url, get_supabase_key

st.set_page_config(
    page_title="Swing Trader Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark-theme override ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Body / sidebar */
    [data-testid="stAppViewContainer"] { background-color: #0f172a; }
    [data-testid="stSidebar"]          { background-color: #1e293b; }
    /* Metric cards */
    [data-testid="metric-container"]   { background: #1e293b; border-radius: 8px; padding: 12px; }
    /* Tables */
    .stDataFrame                       { border-radius: 8px; overflow: hidden; }
    /* Headings */
    h1, h2, h3                         { color: #f1f5f9; }
    /* Divider */
    hr                                 { border-color: #334155; }
</style>
""", unsafe_allow_html=True)

# ── Session-state defaults ─────────────────────────────────────────────────────
if "supabase_url" not in st.session_state:
    st.session_state["supabase_url"] = get_supabase_url()
if "supabase_key" not in st.session_state:
    st.session_state["supabase_key"] = get_supabase_key()

is_connected = bool(st.session_state.get("supabase_url") and st.session_state.get("supabase_key"))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
        width=60,
    )
    st.title("Swing Trader")
    st.caption("Nifty 50 · XGBoost · Auto-improving")
    st.divider()

    badge_color  = "#22c55e" if is_connected else "#f59e0b"
    badge_label  = "Supabase" if is_connected else "JSON fallback"
    st.markdown(
        f'<span style="background:{badge_color};color:#fff;border-radius:4px;'
        f'padding:3px 8px;font-size:12px;font-weight:bold">⬤ {badge_label}</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("""
**Pages**
- 📡 [Signals](Signals) — this week's picks
- 💼 [Portfolio](Portfolio) — equity curve & analytics
- 📈 [Performance](Performance) — IC, accuracy
- 🩺 [Model_Health](Model_Health) — drift, features
- ⚙️ [Settings](Settings) — config, diagnostics
- 🎯 [Trade_Desk](Trade_Desk) — open/close/edit trades
- 📓 [Prediction_Journal](Prediction_Journal) — predicted vs actual
- 🧾 [Account](Account) — funds ledger / statement
""")

    st.divider()
    st.caption(
        "Retrain every **Sunday** via Colab.  \n"
        "Predictions resolve after **5 trading days**.  \n"
        "Paper-trade for **2 months** before going live."
    )

# ── Home page ─────────────────────────────────────────────────────────────────
st.title("📊 Swing Trader Dashboard")

# ── Live workflow tiles ────────────────────────────────────────────────────────
try:
    from app.utils.data_loader import (
        load_signals, load_paper_trades, load_prediction_journal,
        load_weekly_ic, load_portfolio_meta,
    )

    signals_df = load_signals()
    trades_df  = load_paper_trades()
    journal_df = load_prediction_journal(n_weeks=8)
    ic_df      = load_weekly_ic(n_weeks=4)
    meta       = load_portfolio_meta()

    open_t = trades_df[trades_df["status"] == "open"] if not trades_df.empty else trades_df
    n_open = len(open_t)

    pending_n = 0
    if not journal_df.empty and "outcome_resolved" in journal_df.columns:
        pending_n = int((~journal_df["outcome_resolved"].astype(bool)).sum())

    latest_ic = None
    if not ic_df.empty:
        ic_series = ic_df["ic"].dropna()
        latest_ic = ic_series.iloc[-1] if len(ic_series) else None

    long_n = 0
    signal_date_str = None
    if not signals_df.empty:
        long_n = int((signals_df["signal"] == "LONG").sum()) if "signal" in signals_df.columns else 0
        signal_date_str = str(signals_df["signal_date"].iloc[0]) if "signal_date" in signals_df.columns else None

    cash = float(meta.get("cash", meta.get("initial_capital", 1_000_000)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open Positions", n_open, delta=f"₹{cash:,.0f} cash")
    c2.metric("LONG Signals (latest)", long_n, delta=signal_date_str or "—")
    c3.metric("Pending Resolutions", pending_n)
    c4.metric("Latest Weekly IC", f"{latest_ic:.4f}" if latest_ic is not None else "—")

    st.divider()

    # ── "What do I do this week?" checklist ───────────────────────────────────
    st.subheader("This Week's Checklist")
    checklist = []
    if long_n > 0:
        checklist.append(f"📡 **{long_n} LONG signal(s)** ready to trade — go to Signals.")
    if pending_n > 0:
        checklist.append(f"🔄 **{pending_n} prediction(s)** ready to resolve — go to Prediction Journal.")
    if signal_date_str:
        try:
            sd = datetime.strptime(signal_date_str[:10], "%Y-%m-%d").date()
            days_old = (date.today() - sd).days
            if days_old > 7:
                checklist.append(f"⚠️ Signals are **{days_old} days old** — looks like the weekly Colab run was missed.")
        except Exception:
            pass
    if not checklist:
        checklist.append("✅ Nothing urgent — all caught up.")
    for item in checklist:
        st.markdown(f"- {item}")

except Exception as exc:
    st.info(f"Live tiles unavailable yet ({exc}). Run the pipeline to generate data.")

st.divider()

st.markdown(
    "Use the **sidebar** to navigate. Start with **📡 Signals** to act on this week's predictions, "
    "or **🎯 Trade Desk** to manage open positions."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
**📡 Signals → 🎯 Trade Desk**
Open a real (paper) position straight from a signal — pre-filled entry/stop/target,
charged real NSE delivery costs (STT, GST, DP charge) and slippage.
""")

with col2:
    st.info("""
**📓 Prediction Journal**
Every prediction vs. its ground-truth outcome — pending vs resolved,
hit/miss, exportable for the next weekly Colab run.
""")

with col3:
    st.info("""
**🧾 Account**
A broker-style funds statement — every BUY/SELL/CHARGE/DEPOSIT/WITHDRAWAL,
running balance, and total charges paid to date.
""")

st.divider()

# ── Quick-start notice ────────────────────────────────────────────────────────
if not is_connected:
    st.warning(
        "**Supabase not configured** — running in JSON fallback mode.  \n"
        "Set `SUPABASE_URL` and `SUPABASE_KEY` (or `SUPABASE_SECRET_KEY`) in `.env` (local) or "
        "Streamlit Cloud Secrets (deployed) to enable live DB sync."
    )
else:
    st.success("Connected to Supabase. All data syncs automatically after each pipeline run and every trade action.")
