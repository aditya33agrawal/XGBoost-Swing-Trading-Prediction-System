"""Page 1: Today's Signals — and trade them straight into the paper account."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

from app.utils.data_loader import load_signals, load_paper_trades, load_portfolio_meta, refresh_all
from app.utils import writer
from app.utils.prices import fetch_latest_prices
from src.trading.market_hours import is_market_open, market_status_message

st.set_page_config(page_title="Signals", page_icon="📡", layout="wide")
st.title("📡 Today's Signals")

market_open = is_market_open()
if market_open:
    st.info(market_status_message())
else:
    st.warning(market_status_message())
if not market_open:
    st.caption("Signals are still shown below — trading just opens the **Take Trade** "
               "buttons only during market hours so every fill is a real CMP.")

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

# ── Regime detection ─────────────────────────────────────────────────────────
# The pipeline always marks the top cross-sectional quintile as LONG. The ONLY
# way a full, non-empty universe comes back with zero LONGs is the risk-off
# regime overlay (index below its 200-day SMA) wiping every long to NEUTRAL.
# So "no LONGs on a populated board" deterministically means: staying flat is
# the strategy today — not a broken model. Surface that instead of a blank page.
long_df    = df[df["signal"] == "LONG"] if "signal" in df.columns else pd.DataFrame()
neutral_df = df[df["signal"] == "NEUTRAL"] if "signal" in df.columns else pd.DataFrame()
risk_off   = (not df.empty) and ("signal" in df.columns) and long_df.empty

# prob_up is a *relative* cross-sectional score (min-maxed rank in ranker mode,
# raw class-prob otherwise) — NOT a standalone probability. 1.0 just means
# "top-ranked name today"; it does not imply a tradeable long. The top quintile
# is the basket the strategy would buy in a risk-on regime.
top_q = df.nlargest(max(1, len(df) // 5), "prob_up") if ("prob_up" in df.columns and not df.empty) else df

# ── Regime banner ────────────────────────────────────────────────────────────
if risk_off:
    st.error(
        "🛑 **Risk-off regime — all longs suppressed.**  The index is below its "
        "200-day SMA, so the strategy's risk overlay flattens every signal to "
        "NEUTRAL and stays in cash. This is by design: the backtest only earns "
        "its edge going long in risk-on regimes. The scores below still rank the "
        "universe — they're the names that *would* be bought once the regime "
        "turns risk-on — but nothing is tradeable today.",
        icon="🛑",
    )
elif long_df.empty:
    st.info("No signals available yet. Run the pipeline to generate predictions.")

# ── Metric cards ─────────────────────────────────────────────────────────────
avg_top = top_q["prob_up"].mean() if ("prob_up" in top_q.columns and not top_q.empty) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Universe Scored", len(df))
c2.metric("LONG Signals",    len(long_df), help="Top-quintile names, after the regime overlay")
c3.metric("Regime",          "🔴 Risk-off" if risk_off else "🟢 Risk-on",
          help="Risk-off = index below its 200-day SMA → longs suppressed")
c4.metric("Top-Quintile Score", f"{avg_top:.1%}" if avg_top else "—",
          help="Mean model score of the top 20% — the basket that trades in risk-on")

st.divider()

# ── Signal table ─────────────────────────────────────────────────────────────
display_cols = [c for c in [
    "ticker", "signal", "prob_up",
    "entry_price", "stop_loss", "target_price",
    "risk_reward", "atr14",
] if c in df.columns]

view = df[display_cols].copy()
# The pipeline now surfaces the full scored universe (~200 names), so order the
# table by confidence — LONG basket (top quintile) first, then NEUTRAL — rather
# than whatever order the DB returned them in.
if "prob_up" in view.columns:
    view = view.sort_values("prob_up", ascending=False).reset_index(drop=True)

# On a risk-off day nothing is LONG, so colour by the *rank* the strategy uses
# (top quintile = the basket it would buy) rather than by an absolute prob cutoff
# that no min-maxed score ever clears. This keeps the board informative when the
# regime overlay has flattened every signal.
q_cut = view["prob_up"].quantile(0.8) if ("prob_up" in view.columns and not view.empty) else 1.0

def _color_signal(val):
    if val == "LONG":
        return "background-color: rgba(34,197,94,0.2); color: #22c55e; font-weight:bold"
    return "color: #94a3b8"

def _color_prob(val):
    if not isinstance(val, (float, int)):
        return ""
    if val >= q_cut:
        return "color: #22c55e; font-weight:bold"      # top quintile — the long basket
    if val >= view["prob_up"].median():
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
    .map(_color_signal, subset=["signal"])
    .map(_color_prob,   subset=["prob_up"])
)

st.dataframe(
    styled, width="stretch", height=350,
    column_config={
        "prob_up": st.column_config.TextColumn(
            "Model Score",
            help="Relative cross-sectional rank (0–100%). NOT a standalone "
                 "probability — the top 20% is the basket the strategy longs in "
                 "a risk-on regime. Green = top quintile.",
        ),
        "risk_reward": st.column_config.TextColumn(
            "R:R", help="Reward-to-risk of the ATR-based target vs stop"),
        "atr14": st.column_config.TextColumn(
            "ATR14", help="14-day Average True Range (₹) — volatility unit for stops/targets"),
    },
)
st.caption(
    f"Green rows = top quintile (score ≥ {q_cut:.0%}) — the long basket. "
    "In a risk-on regime these carry a **LONG** signal; today the regime overlay "
    "has set them to NEUTRAL." if risk_off else
    f"Green rows = top quintile (score ≥ {q_cut:.0%}) — the LONG basket."
)

# ── Score chart ───────────────────────────────────────────────────────────────
# Always chart the top-ranked names — the LONG basket when risk-on, or the
# model's would-be picks when the regime overlay has suppressed everything — so
# the page is never blank on a risk-off day.
chart_df = long_df if not long_df.empty else top_q
if not chart_df.empty and "prob_up" in chart_df.columns:
    import plotly.express as px
    plot_df = chart_df.sort_values("prob_up", ascending=True).tail(20)
    lo = float(plot_df["prob_up"].min())
    fig = px.bar(
        plot_df,
        x="prob_up", y="ticker", orientation="h",
        color="prob_up",
        color_continuous_scale=["#f59e0b", "#22c55e"],
        range_color=[lo, 1.0],
        labels={"prob_up": "Model score (relative rank)", "ticker": ""},
        title="LONG Basket — Model Score" if not long_df.empty
              else "Top-Ranked Names (suppressed — risk-off regime)",
    )
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="#e2e8f0", coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width="stretch")

st.download_button(
    "⬇ Download CSV",
    data=df.to_csv(index=False),
    file_name=f"signals_{signal_date}.csv",
    mime="text/csv",
)

st.divider()

# ── Trade actions ──────────────────────────────────────────────────────────────
st.subheader("🎯 Take a Trade")
st.caption("Every trade fills at CMP, fetched the moment you click — never the (older) price "
           "shown in the signal table above. Nothing here ever opens by itself.")

if long_df.empty:
    if risk_off:
        st.warning(
            "🛑 **Risk-off regime — no trades today.** The strategy stays in cash "
            "while the index is below its 200-day SMA. The scored names above are "
            "a watchlist for when the regime flips back to risk-on, not entries.",
            icon="🛑",
        )
    else:
        st.info("No LONG signals to trade today.")
else:
    # CMP for every LONG ticker, fetched fresh (short TTL) right before display —
    # writer.open_trade re-fetches again at click time so the fill always matches.
    # Some pipeline runs can insert duplicate rows for the same ticker/date —
    # keep only the highest-confidence row per ticker so widget keys stay unique.
    trade_candidates = long_df.sort_values("prob_up", ascending=False).drop_duplicates(
        subset="ticker", keep="first"
    )
    cmp_prices = fetch_latest_prices(tuple(trade_candidates["ticker"].tolist()))

    for _, row in trade_candidates.iterrows():
        ticker = row["ticker"]
        cmp = cmp_prices.get(ticker)
        already_open = ticker in open_tickers
        has_cmp = cmp is not None

        label = f"{'✅ ALREADY OPEN — ' if already_open else ''}{ticker}  ·  prob_up {row.get('prob_up', 0):.1%}"
        label += f"  ·  CMP ₹{cmp:,.2f}" if has_cmp else "  ·  CMP unavailable"

        with st.expander(label, expanded=False):
            if not has_cmp:
                st.warning(f"Could not fetch a live CMP for {ticker} right now — try refreshing.")
                continue
            entry = float(cmp)
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
            pcol1.metric("Fill price (CMP)", f"₹{preview['fill_price']:,.2f}")
            pcol2.metric("Charges", f"₹{preview['charges']['total']:,.2f}")
            pcol3.metric("Total debit", f"₹{preview['total_debit']:,.2f}")
            pcol4.metric("Break-even", f"₹{preview['breakeven_price']:,.2f}")

            with st.popover("Charge breakdown"):
                st.json(preview["charges"])

            if not market_open:
                st.button(f"Take Trade — {ticker}", key=f"confirm_{ticker}",
                          type="primary", disabled=True)
                st.caption(market_status_message())
                continue

            if st.button(f"Take Trade — {ticker}", key=f"confirm_{ticker}", type="primary"):
                ok, msg = writer.open_trade(
                    ticker, int(shares), float(stop), float(target),
                    horizon_days=int(row.get("horizon_days", 5)),
                    prob_up=float(row.get("prob_up", 0.5)),
                    opened_via="signal",
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
                st.rerun()
