"""Per-leg Indian charge math must reconcile with the round-trip cost model."""
import pytest

from src.backtest.costs import buy_leg_cost, sell_leg_cost, indian_round_trip_cost


def test_buy_sell_legs_sum_to_round_trip():
    value = 100_000.0
    buy = buy_leg_cost(value)
    sell = sell_leg_cost(value)
    rt = indian_round_trip_cost(value, value)
    assert abs((buy["total"] + sell["total"]) - rt) < 1e-6


def test_stamp_only_on_buy():
    value = 50_000.0
    buy = buy_leg_cost(value)
    sell = sell_leg_cost(value)
    assert buy["stamp"] > 0
    assert sell["stamp"] == 0


def test_dp_charge_only_on_sell():
    value = 50_000.0
    buy = buy_leg_cost(value)
    sell = sell_leg_cost(value)
    assert buy["dp_charge"] == 0
    assert sell["dp_charge"] > 0


def test_charges_scale_with_value():
    small = buy_leg_cost(10_000.0)
    large = buy_leg_cost(100_000.0)
    assert large["total"] > small["total"]


def test_zero_value_gives_zero_or_flat_charges():
    buy = buy_leg_cost(0.0)
    assert buy["stt"] == 0
    assert buy["stamp"] == 0
    sell = sell_leg_cost(0.0)
    # DP charge is a flat fee regardless of trade value
    assert sell["dp_charge"] > 0
