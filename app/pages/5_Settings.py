"""Page 5: Settings — manual position management, triggers, config display."""
import os
import json
from datetime import date, datetime

import streamlit as st
import pandas as pd

from app.utils.data_loader import load_paper_trades, load_portfolio_meta, refresh_all

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings & Controls")

# ── Manual controls ───────────────────────────────────────────────────────────
st.subheader("Manual Controls")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⟳ Refresh All Data", use_container_width=True):
        refresh_all()
        st.rerun()

with col2:
    if st.button("🔄 Resolve Outcomes Now", use_container_width=True):
        with st.spinner("Resolving outcomes…"):
            try:
                import yfinance as yf
                from src.db.supabase_client import get_supabase_client
                from src.tracking.outcome_tracker import resolve_outcomes

                url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
                key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
                sb  = get_supabase_client(url, key)
                n   = resolve_outcomes(price_df=None, supabase_client=sb)
                st.success(f"Resolved {n} predictions.")
                refresh_all()
            except Exception as e:
                st.error(f"Error: {e}")

with col3:
    if st.button("📊 Generate Signals (Fast)", use_container_width=True):
        st.info("Run this command in your terminal:\n```\npython run_pipeline.py --fast-signals\n```")

st.divider()

# ── Manual position management ────────────────────────────────────────────────
st.subheader("Manual Position Management")

trades = load_paper_trades()
open_t = trades[trades["status"] == "open"] if not trades.empty else pd.DataFrame()

tab1, tab2 = st.tabs(["Close Position", "Add Position"])

with tab1:
    if open_t.empty:
        st.info("No open positions to close.")
    else:
        tickers = open_t["ticker"].tolist() if "ticker" in open_t.columns else []
        ticker_to_close = st.selectbox("Select position to close", tickers)

        if ticker_to_close:
            row = open_t[open_t["ticker"] == ticker_to_close].iloc[0]
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Entry price:** ₹{row.get('entry_price', '?'):,.2f}")
                st.write(f"**Shares:** {row.get('shares', '?')}")
            with col_b:
                exit_price  = st.number_input("Exit Price (₹)", min_value=0.0, step=0.5)
                exit_reason = st.selectbox("Exit Reason", ["manual", "target", "stop", "expired"])

            if st.button("Close Position", type="primary"):
                try:
                    portfolio_path = os.path.join("outputs", "portfolio.json")
                    if not os.path.exists(portfolio_path):
                        st.error("portfolio.json not found.")
                    else:
                        with open(portfolio_path) as f:
                            port = json.load(f)
                        for t in port.get("trades", []):
                            if t["ticker"] == ticker_to_close and t["status"] == "open":
                                t["status"]      = "closed"
                                t["exit_date"]   = str(date.today())
                                t["exit_price"]  = exit_price
                                t["exit_reason"] = exit_reason
                                shares    = float(t.get("shares", 0))
                                entry     = float(t.get("entry_price", 0))
                                t["pnl"]     = round((exit_price - entry) * shares, 2)
                                t["pnl_pct"] = round((exit_price / entry - 1) * 100, 2) if entry else 0
                                port["cash"]  = round(float(port.get("cash", 0)) + exit_price * shares, 2)
                                break
                        with open(portfolio_path, "w") as f:
                            json.dump(port, f, indent=2)
                        st.success(f"Closed {ticker_to_close} at ₹{exit_price:,.2f}")
                        refresh_all()
                        st.rerun()
                except Exception as e:
                    st.error(f"Error closing position: {e}")

with tab2:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        new_ticker     = st.text_input("Ticker (e.g. RELIANCE.NS)")
        new_entry      = st.number_input("Entry Price (₹)", min_value=0.0, step=0.5)
        new_shares     = st.number_input("Shares", min_value=1, step=1, value=10)
    with col_b:
        new_stop       = st.number_input("Stop Loss (₹)", min_value=0.0, step=0.5)
        new_target     = st.number_input("Target Price (₹)", min_value=0.0, step=0.5)
        new_horizon    = st.number_input("Horizon (days)", min_value=1, value=5, step=1)
    with col_c:
        new_prob       = st.slider("Model Confidence (prob_up)", 0.0, 1.0, 0.5, 0.01)
        new_entry_date = st.date_input("Entry Date", value=date.today())

    if st.button("Add Position", type="primary"):
        if not new_ticker or new_entry <= 0:
            st.warning("Ticker and entry price are required.")
        else:
            try:
                portfolio_path = os.path.join("outputs", "portfolio.json")
                port = {"trades": [], "cash": 1_000_000, "initial_capital": 1_000_000}
                if os.path.exists(portfolio_path):
                    with open(portfolio_path) as f:
                        port = json.load(f)

                trade_id = f"{new_ticker}_{new_entry_date}_{datetime.now().strftime('%H%M%S')}"
                trade = {
                    "trade_id":    trade_id,
                    "ticker":      new_ticker.upper(),
                    "entry_date":  str(new_entry_date),
                    "entry_price": new_entry,
                    "shares":      int(new_shares),
                    "signal":      "LONG",
                    "stop_loss":   new_stop,
                    "target_price":new_target,
                    "horizon_days":int(new_horizon),
                    "prob_up":     new_prob,
                    "status":      "open",
                    "run_id":      "manual",
                }
                port.setdefault("trades", []).append(trade)
                cost = new_entry * int(new_shares)
                port["cash"] = round(float(port.get("cash", 0)) - cost, 2)

                with open(portfolio_path, "w") as f:
                    json.dump(port, f, indent=2)
                st.success(f"Added {new_ticker.upper()} position (cost ₹{cost:,.0f})")
                refresh_all()
                st.rerun()
            except Exception as e:
                st.error(f"Error adding position: {e}")

st.divider()

# ── Configuration display ─────────────────────────────────────────────────────
st.subheader("Current Configuration")

def _mask(val: str) -> str:
    if not val:
        return "— (not set)"
    return val[:8] + "…" + val[-4:] if len(val) > 12 else "****"

supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Database**")
    st.write(f"Supabase URL: `{_mask(supabase_url)}`")
    st.write(f"Supabase Key: `{_mask(supabase_key)}`")
    st.write(f"Status: {'✅ Connected' if supabase_url else '⚠️ JSON fallback mode'}")

with col2:
    st.markdown("**Output paths**")
    st.write(f"Signals: `outputs/signals_latest.csv`")
    st.write(f"Portfolio: `outputs/portfolio.json`")
    st.write(f"Model runs: `outputs/model_runs.json`")

st.divider()

# ── Quick reference ───────────────────────────────────────────────────────────
with st.expander("CLI Quick Reference"):
    st.code("""
# Generate signals only (fast, no retrain)
python run_pipeline.py --fast-signals

# Full weekly retrain + deploy
python scripts/weekly_retrain.py

# Paper trade without retrain
python run_pipeline.py --fast-signals --capital 1000000 --position-size 0.05

# Skip paper trading
python run_pipeline.py --fast-signals --no-paper-trade

# Use specific portfolio file
python run_pipeline.py --fast-signals --portfolio path/to/portfolio.json
""", language="bash")

with st.expander("Supabase Setup"):
    st.markdown("""
**1. Create account** at [supabase.com](https://supabase.com) (free tier)

**2. Run the SQL schema** in the Supabase dashboard SQL editor (see `README.md` or plan doc)

**3. Set environment variables** in `.env`:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

**4. For Streamlit Cloud**: App Settings → Secrets:
```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```
""")

with st.expander("Colab Weekly Workflow"):
    st.markdown("""
**Every Sunday:**
1. Open `notebooks/colab_weekly.ipynb` in Google Colab
2. Set secrets in Colab: `SUPABASE_URL`, `SUPABASE_KEY`
3. Run all cells (GPU runtime recommended)
4. Cell 4 resolves last week's outcomes + shows IC trend
5. Cell 5 retrains model (50 Optuna trials, ~30 min on GPU)
6. Cell 6–7 deploys new model if IC improved by ≥ 0.005
7. Cell 8 generates next week's signals → pushed to Supabase
8. This Streamlit app refreshes automatically from new data
""")
