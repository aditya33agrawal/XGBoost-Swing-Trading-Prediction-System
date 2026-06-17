"""Main Streamlit entry point — session state, sidebar, global styles."""
import os
import streamlit as st

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
    st.session_state["supabase_url"] = st.secrets.get(
        "SUPABASE_URL", os.getenv("SUPABASE_URL", "")
    )
if "supabase_key" not in st.session_state:
    st.session_state["supabase_key"] = st.secrets.get(
        "SUPABASE_KEY", os.getenv("SUPABASE_KEY", "")
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
        width=60,
    )
    st.title("Swing Trader")
    st.caption("Nifty 50 · XGBoost · Auto-improving")
    st.divider()

    # DB status badge
    is_connected = bool(st.session_state.get("supabase_url"))
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
- 💼 [Portfolio](Portfolio) — open / closed trades
- 📈 [Performance](Performance) — IC, accuracy
- 🩺 [Model Health](Model_Health) — drift, features
- ⚙️ [Settings](Settings) — controls, config
""")

    st.divider()
    st.caption(
        "Retrain every **Sunday** via Colab.  \n"
        "Predictions resolve after **5 trading days**.  \n"
        "Paper-trade for **2 months** before going live."
    )

# ── Home page ─────────────────────────────────────────────────────────────────
st.title("📊 Swing Trader Dashboard")
st.markdown(
    "Welcome! Use the **sidebar** to navigate between pages.  \n"
    "Start with **📡 Signals** to see this week's predictions."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
**📡 Signals**
This week's LONG signals with entry, stop-loss, and target levels.
Confidence scored by the XGBoost model (prob_up).
""")

with col2:
    st.info("""
**💼 Portfolio**
Paper trading account — equity curve, open positions, closed trade history,
win rate, and P&L breakdown.
""")

with col3:
    st.info("""
**🩺 Model Health**
IC trend, feature importance, model version history.
GREEN = IC > 0.01, AMBER = IC 0–0.01, RED = IC < 0.
""")

st.divider()

# ── Quick-start notice ────────────────────────────────────────────────────────
if not is_connected:
    st.warning(
        "**Supabase not configured** — running in JSON fallback mode.  \n"
        "Set `SUPABASE_URL` and `SUPABASE_KEY` in `.env` (local) or "
        "Streamlit Cloud Secrets (deployed) to enable live DB sync."
    )
else:
    st.success("Connected to Supabase. All data syncs automatically after each pipeline run.")
