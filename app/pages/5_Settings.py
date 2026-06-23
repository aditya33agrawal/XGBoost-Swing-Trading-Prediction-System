"""Page 5: Settings — connection diagnostics, config display, CLI reference.

Manual position management has moved to 🎯 Trade Desk; outcome resolution
has moved to 📓 Prediction Journal (both now do the real thing — this page
used to call resolve_outcomes(price_df=None, ...) which silently resolved
nothing).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.utils.data_loader import refresh_all
from app.utils.env import get_supabase_url, get_supabase_key

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings & Diagnostics")

# ── Manual controls ───────────────────────────────────────────────────────────
st.subheader("Manual Controls")
col1, col2 = st.columns(2)
with col1:
    if st.button("⟳ Refresh All Data", width="stretch"):
        refresh_all()
        st.rerun()
with col2:
    st.markdown("📡 [Go trade a signal](Signals)")

st.divider()

# ── Danger zone ───────────────────────────────────────────────────────────────
st.subheader("⚠️ Danger Zone")
with st.expander("Reset paper portfolio"):
    st.warning("Wipes every open/closed trade and ledger row, locally and in Supabase, "
               "and restarts the account at the initial capital below. Cannot be undone.")
    reset_capital = st.number_input("Initial capital (₹)", min_value=1.0, value=1_000_000.0, step=10_000.0)
    confirm = st.checkbox("I understand this deletes all paper-trading history")
    if st.button("Reset Portfolio", type="primary", disabled=not confirm):
        from app.utils import writer
        ok, msg = writer.reset_portfolio(initial_capital=reset_capital)
        (st.success if ok else st.error)(msg)
        st.rerun()

st.divider()

# ── Connection diagnostics ────────────────────────────────────────────────────
st.subheader("Database Connection")

url = get_supabase_url()
key = get_supabase_key()


def _mask(val: str) -> str:
    if not val:
        return "— (not set)"
    return val[:8] + "…" + val[-4:] if len(val) > 12 else "****"


col1, col2 = st.columns(2)
with col1:
    st.write(f"Supabase URL: `{_mask(url)}`")
    st.write(f"Supabase Key: `{_mask(key)}`")
    st.write(f"Status: {'✅ Configured' if (url and key) else '⚠️ JSON fallback mode'}")
with col2:
    st.write("**Output paths**")
    st.write("Signals: `outputs/signals_latest.csv`")
    st.write("Portfolio: `outputs/portfolio.json`")
    st.write("Ledger: `outputs/ledger.json`")
    st.write("Model runs: `outputs/model_runs.json`")

if st.button("🔍 Test Connection", type="primary"):
    if not (url and key):
        st.error("No URL/key configured — set SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_KEY) "
                 "in .env (local) or Streamlit Cloud Secrets.")
    else:
        from src.db.supabase_client import get_supabase_client, fetch_rows, upsert_rows
        client = get_supabase_client(url, key)
        if client is None:
            st.error("Could not create a Supabase client — check the URL/key and that the `supabase` "
                      "package is installed.")
        else:
            st.success("Client created. Checking tables…")
            results = []
            for table in ["predictions", "paper_trades", "outcomes", "model_runs",
                          "feature_importance", "account_ledger"]:
                try:
                    rows = fetch_rows(client, table, limit=1)
                    n = len(rows)
                    results.append((table, "✅ reachable", f"sample rows: {n}"))
                except Exception as exc:
                    results.append((table, "❌ error", str(exc)))
            for table, status, detail in results:
                st.write(f"**{table}** — {status} — {detail}")

            st.write("**Write check** (no-op upsert + delete on a health probe row):")
            probe_id = "00000000-0000-0000-0000-000000000000"
            ok = upsert_rows(client, "account_ledger", [{
                "id": probe_id, "ts": "2000-01-01T00:00:00", "type": "OPENING_BALANCE",
                "amount": 0, "running_balance": 0, "note": "connection probe",
            }], on_conflict="id")
            if ok:
                st.success("✅ Writes are permitted with this key.")
                try:
                    client.table("account_ledger").delete().eq("id", probe_id).execute()
                except Exception:
                    pass
            else:
                st.error("❌ Write failed — this key cannot write (check RLS policies or use the "
                          "secret key). Trade logging from the app will not persist to Supabase "
                          "until this is fixed.")

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

**2. Run `docs/supabase_schema.sql`** once in the Supabase SQL editor — adds the
real-money charge columns to `paper_trades` and creates `account_ledger`.

**3. Set environment variables** in `.env`:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SECRET_KEY=sb_secret_xxx
```

**4. For Streamlit Cloud**: App Settings → Secrets:
```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_xxx"
```

Use the **secret** key (not the publishable/anon key) so the app can write
trades and ledger rows — see Test Connection above to verify.
""")

with st.expander("Colab Weekly Workflow"):
    st.markdown("""
**Every Sunday:**
1. Open `notebooks/colab_weekly.ipynb` in Google Colab
2. Set secrets in Colab: `SUPABASE_URL`, `SUPABASE_SECRET_KEY`
3. Run all cells (GPU runtime recommended)
4. Cell 4 resolves last week's outcomes + shows IC trend
5. Cell 5 retrains model (50 Optuna trials, ~30 min on GPU)
6. Cell 6–7 deploys new model if IC improved by ≥ 0.005
7. Cell 8 generates next week's signals → pushed to Supabase
8. This Streamlit app refreshes automatically from new data — go to
   **📡 Signals** to trade them and **📓 Prediction Journal** to resolve
   last week's outcomes.
""")
