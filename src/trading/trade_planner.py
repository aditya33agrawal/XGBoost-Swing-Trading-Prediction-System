"""Small-capital trade planner (docs/small-capital-strategy-plan.md §1.4).

Converts the weekly signal frame into an *executable* plan for a real small
account: whole shares only, per-position budget, affordability filter,
itemised Indian delivery charges, breakeven move, and an edge screen that
drops names whose expected move to target cannot clear round-trip costs.

The planner is pure (no I/O side effects in build_trade_plan) so it is unit
testable; save_trade_plan persists the artifact the user downloads from Colab.

Actions emitted per row:
  BUY  — open this position (shares, value, charges, breakeven all computed)
  HOLD — already held and still ranks above cfg.exit_rank_pct → roll, don't
         pay a needless ~0.9% round trip (turnover hysteresis, plan §4.20)
  SELL — held but now ranks below the exit band → close at next session
  SKIP — LONG signal that failed affordability or the edge screen (kept in
         the frame with a note so the decision is auditable)
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.costs import buy_leg_cost, sell_leg_cost

logger = logging.getLogger(__name__)


def _score_col(signals: pd.DataFrame) -> str | None:
    for c in ("prob_up", "pred_return", "score", "pred"):
        if c in signals.columns:
            return c
    return None


def _round_trip_charges(value: float, slippage_bps: float) -> float:
    """Estimated total INR cost of a full round trip on `value`, including
    slippage both ways and the flat DP charge on the sell."""
    buy = buy_leg_cost(value)["total"]
    sell = sell_leg_cost(value)["total"]
    slip = 2 * slippage_bps / 1e4 * value
    return buy + sell + slip


def build_trade_plan(
    signals: pd.DataFrame,
    cfg,
    cash: float | None = None,
    held_tickers: set[str] | frozenset[str] | None = None,
) -> pd.DataFrame:
    """Build the executable weekly plan from the scored signal frame.

    Parameters
    ----------
    signals      : enriched signal frame (ticker, signal, score col,
                   entry_price, stop_loss, target_price, horizon_days)
    cfg          : Config — uses initial_capital, position_size_pct,
                   max_positions, slippage_bps, min_edge_ratio,
                   hysteresis_enabled, entry_rank_pct, exit_rank_pct
    cash         : available cash; defaults to initial_capital
    held_tickers : currently open paper/live positions
    """
    cols = ["ticker", "action", "score", "rank_pct", "entry_price", "shares",
            "position_value", "est_charges", "breakeven_move_pct",
            "target_move_pct", "edge_ratio", "stop_loss", "target_price",
            "horizon_days", "note"]
    if signals.empty:
        return pd.DataFrame(columns=cols)

    score_col = _score_col(signals)
    if score_col is None or "entry_price" not in signals.columns:
        logger.warning("trade_planner: signal frame missing score/entry_price — no plan")
        return pd.DataFrame(columns=cols)

    held = set(held_tickers or ())
    cash = float(cash if cash is not None else cfg.initial_capital)
    slippage_bps = float(getattr(cfg, "slippage_bps", 10.0))
    min_edge_ratio = float(getattr(cfg, "min_edge_ratio", 2.0))
    hysteresis = bool(getattr(cfg, "hysteresis_enabled", False))
    exit_pct = float(getattr(cfg, "exit_rank_pct", 0.60))
    entry_pct = float(getattr(cfg, "entry_rank_pct", 0.80))
    budget = float(cfg.initial_capital) * float(cfg.position_size_pct)

    df = signals.copy()
    df["rank_pct"] = df[score_col].rank(pct=True)
    rows: list[dict] = []

    # --- held names first: HOLD (roll) vs SELL ---------------------------
    keep: set[str] = set()
    for tkr in sorted(held):
        row = df[df["ticker"] == tkr]
        if row.empty:
            continue    # no score today (data gap) — leave position to its stop/target
        r = row.iloc[0]
        stays = (r["rank_pct"] >= exit_pct) if hysteresis else (r.get("signal") == "LONG")
        if stays:
            keep.add(tkr)
        rows.append({
            "ticker": tkr, "action": "HOLD" if stays else "SELL",
            "score": float(r[score_col]), "rank_pct": round(float(r["rank_pct"]), 3),
            "entry_price": float(r["entry_price"]), "shares": 0,
            "position_value": 0.0, "est_charges": 0.0,
            "breakeven_move_pct": np.nan, "target_move_pct": np.nan,
            "edge_ratio": np.nan,
            "stop_loss": float(r.get("stop_loss", np.nan)),
            "target_price": float(r.get("target_price", np.nan)),
            "horizon_days": int(r.get("horizon_days", cfg.horizon)),
            "note": ("rank %.0f%% ≥ exit band %.0f%% — roll, save the round trip"
                     % (r["rank_pct"] * 100, exit_pct * 100)) if stays else
                    ("rank %.0f%% < exit band %.0f%% — close"
                     % (r["rank_pct"] * 100, exit_pct * 100)),
        })

    # --- new entries ------------------------------------------------------
    free_slots = max(0, int(cfg.max_positions) - len(keep))
    candidates = df[(df.get("signal") == "LONG") & (~df["ticker"].isin(held))]
    if hysteresis:
        candidates = candidates[candidates["rank_pct"] >= entry_pct]
    candidates = candidates.sort_values(score_col, ascending=False)

    n_bought = 0
    for _, r in candidates.iterrows():
        price = float(r["entry_price"])
        if price <= 0 or not math.isfinite(price):
            continue

        shares = int(budget // price)
        value = shares * price
        note = ""
        action = "BUY"

        if shares < 1:
            action, note = "SKIP", f"unaffordable: 1 share ₹{price:,.0f} > budget ₹{budget:,.0f}"
        else:
            charges = _round_trip_charges(value, slippage_bps)
            breakeven = charges / value
            target = float(r.get("target_price", np.nan))
            target_move = (target - price) / price if math.isfinite(target) and target > 0 else np.nan
            edge_ratio = target_move / breakeven if breakeven > 0 and math.isfinite(target_move) else np.nan

            if math.isfinite(edge_ratio) and edge_ratio < min_edge_ratio:
                action = "SKIP"
                note = (f"edge too thin: target move {target_move:.2%} < "
                        f"{min_edge_ratio:.1f}× breakeven {breakeven:.2%}")
            elif n_bought >= free_slots:
                action, note = "SKIP", "no free position slot"
            elif value + charges / 2 > cash:
                action, note = "SKIP", f"insufficient cash ₹{cash:,.0f} for ₹{value:,.0f}"

        if action == "BUY":
            charges = _round_trip_charges(value, slippage_bps)
            breakeven = charges / value
            target = float(r.get("target_price", np.nan))
            target_move = (target - price) / price if math.isfinite(target) and target > 0 else np.nan
            edge_ratio = target_move / breakeven if breakeven > 0 and math.isfinite(target_move) else np.nan
            cash -= value + buy_leg_cost(value)["total"]
            n_bought += 1
            rows.append({
                "ticker": r["ticker"], "action": "BUY",
                "score": float(r[score_col]), "rank_pct": round(float(r["rank_pct"]), 3),
                "entry_price": price, "shares": shares,
                "position_value": round(value, 2),
                "est_charges": round(charges, 2),
                "breakeven_move_pct": round(breakeven * 100, 3),
                "target_move_pct": round(target_move * 100, 3) if math.isfinite(target_move) else np.nan,
                "edge_ratio": round(edge_ratio, 2) if math.isfinite(edge_ratio) else np.nan,
                "stop_loss": float(r.get("stop_loss", np.nan)),
                "target_price": float(r.get("target_price", np.nan)),
                "horizon_days": int(r.get("horizon_days", cfg.horizon)),
                "note": f"₹{value:,.0f} ({shares} sh), RT cost ₹{charges:,.0f} = {breakeven:.2%}",
            })
        else:
            rows.append({
                "ticker": r["ticker"], "action": action,
                "score": float(r[score_col]), "rank_pct": round(float(r["rank_pct"]), 3),
                "entry_price": price, "shares": 0,
                "position_value": 0.0, "est_charges": 0.0,
                "breakeven_move_pct": np.nan, "target_move_pct": np.nan,
                "edge_ratio": np.nan,
                "stop_loss": float(r.get("stop_loss", np.nan)),
                "target_price": float(r.get("target_price", np.nan)),
                "horizon_days": int(r.get("horizon_days", cfg.horizon)),
                "note": note,
            })

    plan = pd.DataFrame(rows, columns=cols)
    n_buy = int((plan["action"] == "BUY").sum())
    logger.info("Trade plan: %d BUY / %d HOLD / %d SELL / %d SKIP (budget ₹%.0f/position)",
                n_buy, int((plan["action"] == "HOLD").sum()),
                int((plan["action"] == "SELL").sum()),
                int((plan["action"] == "SKIP").sum()), budget)
    return plan


# ---------------------------------------------------------------------------
# Persistence + display
# ---------------------------------------------------------------------------
def save_trade_plan(plan: pd.DataFrame, output_dir: str = "outputs") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    json_path = out / f"trade_plan_{date_str}.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "n_buy": int((plan.get("action", pd.Series(dtype=str)) == "BUY").sum()),
        "plan": plan.replace({np.nan: None}).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    plan.to_csv(out / "trade_plan_latest.csv", index=False)
    logger.info("Trade plan saved → %s", json_path.name)
    return json_path


def print_trade_plan(plan: pd.DataFrame) -> None:
    w = 110
    print(f"\n{'─' * w}")
    print("  TRADE PLAN (₹-sized, whole shares, costs included)")
    print(f"{'─' * w}")
    if plan.empty:
        print("  (no plan — no signals)")
        print(f"{'─' * w}")
        return
    hdr = (f"  {'Ticker':<16}{'Action':<7}{'Px':>9}{'Shares':>7}{'Value ₹':>10}"
           f"{'RT cost ₹':>10}{'B/E %':>7}{'Tgt %':>7}{'Edge×':>6}  Note")
    print(hdr)
    print(f"  {'─' * (w - 4)}")
    for _, r in plan.iterrows():
        be = f"{r['breakeven_move_pct']:.2f}" if pd.notna(r["breakeven_move_pct"]) else "-"
        tm = f"{r['target_move_pct']:.2f}" if pd.notna(r["target_move_pct"]) else "-"
        er = f"{r['edge_ratio']:.1f}" if pd.notna(r["edge_ratio"]) else "-"
        print(f"  {r['ticker']:<16}{r['action']:<7}{r['entry_price']:>9.2f}"
              f"{int(r['shares']):>7}{r['position_value']:>10,.0f}"
              f"{r['est_charges']:>10,.0f}{be:>7}{tm:>7}{er:>6}  {r['note']}")
    print(f"{'─' * w}")
