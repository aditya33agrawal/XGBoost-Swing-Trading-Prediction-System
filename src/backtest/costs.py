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
"""
from __future__ import annotations


def indian_round_trip_cost(
    buy_value: float,
    sell_value: float,
    brokerage_rate: float = 0.0,
    stt_rate: float = 0.001,          # 0.1% each side for delivery
    exch_txn_rate: float = 0.0000297,
    sebi_rate: float = 0.000001,
    stamp_rate: float = 0.00015,      # 0.015% on buy
    gst_rate: float = 0.18,
    dp_charge: float = 13.0,          # per scrip on sell, INR
) -> float:
    """Total cost in INR for a single round-trip trade."""
    brokerage = brokerage_rate * (buy_value + sell_value)
    stt = stt_rate * (buy_value + sell_value)
    exch = exch_txn_rate * (buy_value + sell_value)
    sebi = sebi_rate * (buy_value + sell_value)
    stamp = stamp_rate * buy_value
    gst = gst_rate * (brokerage + exch + sebi)
    total = brokerage + stt + exch + sebi + stamp + gst + dp_charge
    return total


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
