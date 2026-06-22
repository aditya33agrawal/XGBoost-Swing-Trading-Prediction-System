"""Indian transaction cost model (plan §10.3).

All rates are parameterised — update from current NSE/SEBI schedules
before any live use.  STT and stamp duty rates change with Union Budgets.

Round-trip cost for equity delivery (swing) trading:
  STT:         0.1% on buy + 0.1% on sell  = 0.2% of value traded
  Exchange:    ~0.00297% each side
  SEBI fee:    ~0.0001% each side
  Stamp duty:  0.015% on buy side only
  GST:         18% on (brokerage + exch + SEBI)
  DP charge:   ~₹13 per scrip on sell (flat)
  Brokerage:   ₹0 for delivery at most discount brokers

Total ≈ 0.21% per round trip before DP charge + slippage.
Add ~5–15 bps slippage for Nifty 50 names (liquid).

This module is the single source of truth for cost *rates*. Both the
vectorised backtest (round-trip) and the live/paper trade accounting
(per-leg, in src/trading/paper_trader.py) import from here so a
Union-Budget rate change is a one-line edit.
"""
from __future__ import annotations

# Default rates — current as of the 2026 schedule. Update here only.
DEFAULT_BROKERAGE_RATE = 0.0        # delivery is free at most discount brokers
DEFAULT_STT_RATE       = 0.001      # 0.1% each side for delivery
DEFAULT_EXCH_TXN_RATE  = 0.0000297
DEFAULT_SEBI_RATE      = 0.000001
DEFAULT_STAMP_RATE     = 0.00015    # 0.015% on buy side only
DEFAULT_GST_RATE       = 0.18       # on (brokerage + exch + SEBI)
DEFAULT_DP_CHARGE      = 13.0       # per scrip on sell, INR, flat


def buy_leg_cost(
    value: float,
    brokerage_rate: float = DEFAULT_BROKERAGE_RATE,
    stt_rate: float = DEFAULT_STT_RATE,
    exch_txn_rate: float = DEFAULT_EXCH_TXN_RATE,
    sebi_rate: float = DEFAULT_SEBI_RATE,
    stamp_rate: float = DEFAULT_STAMP_RATE,
    gst_rate: float = DEFAULT_GST_RATE,
) -> dict:
    """Itemised charges for the BUY leg of a delivery trade (no DP charge).

    Returns a dict with each line item plus 'total', in INR.
    """
    brokerage = brokerage_rate * value
    stt = stt_rate * value
    exch = exch_txn_rate * value
    sebi = sebi_rate * value
    stamp = stamp_rate * value
    gst = gst_rate * (brokerage + exch + sebi)
    total = brokerage + stt + exch + sebi + stamp + gst
    return {
        "brokerage": brokerage,
        "stt": stt,
        "exchange": exch,
        "sebi": sebi,
        "stamp": stamp,
        "gst": gst,
        "dp_charge": 0.0,
        "total": total,
    }


def sell_leg_cost(
    value: float,
    brokerage_rate: float = DEFAULT_BROKERAGE_RATE,
    stt_rate: float = DEFAULT_STT_RATE,
    exch_txn_rate: float = DEFAULT_EXCH_TXN_RATE,
    sebi_rate: float = DEFAULT_SEBI_RATE,
    gst_rate: float = DEFAULT_GST_RATE,
    dp_charge: float = DEFAULT_DP_CHARGE,
) -> dict:
    """Itemised charges for the SELL leg of a delivery trade (incl. DP charge, no stamp)."""
    brokerage = brokerage_rate * value
    stt = stt_rate * value
    exch = exch_txn_rate * value
    sebi = sebi_rate * value
    gst = gst_rate * (brokerage + exch + sebi)
    total = brokerage + stt + exch + sebi + gst + dp_charge
    return {
        "brokerage": brokerage,
        "stt": stt,
        "exchange": exch,
        "sebi": sebi,
        "stamp": 0.0,
        "gst": gst,
        "dp_charge": dp_charge,
        "total": total,
    }


def indian_round_trip_cost(
    buy_value: float,
    sell_value: float,
    brokerage_rate: float = DEFAULT_BROKERAGE_RATE,
    stt_rate: float = DEFAULT_STT_RATE,
    exch_txn_rate: float = DEFAULT_EXCH_TXN_RATE,
    sebi_rate: float = DEFAULT_SEBI_RATE,
    stamp_rate: float = DEFAULT_STAMP_RATE,
    gst_rate: float = DEFAULT_GST_RATE,
    dp_charge: float = DEFAULT_DP_CHARGE,
) -> float:
    """Total cost in INR for a single round-trip trade (buy leg + sell leg)."""
    buy = buy_leg_cost(
        buy_value, brokerage_rate, stt_rate, exch_txn_rate, sebi_rate, stamp_rate, gst_rate,
    )
    sell = sell_leg_cost(
        sell_value, brokerage_rate, stt_rate, exch_txn_rate, sebi_rate, gst_rate, dp_charge,
    )
    return buy["total"] + sell["total"]


def cost_fraction(
    trade_value: float,
    slippage_bps: float = 10.0,
) -> float:
    """Fractional cost (0–1) for a round-trip trade of given INR value.

    Approximates cost / trade_value.  For portfolio backtesting, multiply
    by position value to get cost drag per rebalance.
    """
    rt = indian_round_trip_cost(trade_value, trade_value)
    slip = 2 * slippage_bps / 1e4 * trade_value
    return (rt + slip) / (trade_value + 1e-9)


# Convenience constant: approximate round-trip cost as fraction of value
# for a typical ₹100k trade in a Nifty 50 name.
APPROX_RT_COST_FRACTION = cost_fraction(100_000)
