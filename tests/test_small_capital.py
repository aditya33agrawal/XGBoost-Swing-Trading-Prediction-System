"""Tests for the small-capital (₹20k) strategy layer
(docs/small-capital-strategy-plan.md):

  * capital-aware cost model — the flat DP charge must dominate small positions
  * top-K concentrated backtest book + turnover hysteresis
  * integer-share trade planner (affordability, edge screen, HOLD/SELL bands)
  * paper-portfolio auto-open from the plan + hysteresis expiry roll
"""
import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.backtest.costs import (
    cost_fraction, per_position_cost_fraction, DEFAULT_DP_CHARGE,
)
from src.backtest.engine import run_backtest, base_rt_cost_fraction, _select_long_basket
from src.trading.trade_planner import build_trade_plan
from src.trading.paper_trader import PaperPortfolio


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def test_small_position_cost_fraction_dominated_by_dp_charge():
    small = cost_fraction(4_000)
    large = cost_fraction(100_000)
    # flat DP charge alone is ~0.40% of a ₹4k position
    assert DEFAULT_DP_CHARGE / 4_000 > 0.0035
    # proportional costs (STT+slippage ≈0.42%) are size-invariant; the flat DP
    # charge nearly doubles the small trade: ~0.82% at ₹4k vs ~0.44% at ₹1L
    assert small > 1.7 * large
    assert 0.006 < small < 0.015        # ~0.8% RT — the honest ₹20k number
    assert 0.002 < large < 0.005


def test_per_position_cost_fraction_uses_true_position_size():
    f_20k = per_position_cost_fraction(20_000, 0.19)
    assert f_20k == pytest.approx(cost_fraction(3_800), rel=1e-9)


def test_backtest_base_cost_honours_capital_profile():
    cfg = Config(initial_capital=20_000, position_size_pct=0.19,
                 capital_aware_costs=True)
    legacy = Config(capital_aware_costs=False)
    assert base_rt_cost_fraction(cfg) == pytest.approx(cost_fraction(3_800))
    assert base_rt_cost_fraction(legacy) == pytest.approx(cost_fraction(100_000))


# ---------------------------------------------------------------------------
# Backtest: top-K book + hysteresis
# ---------------------------------------------------------------------------
def _pred_frame(n_dates=30, n_tickers=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for t in range(n_tickers):
            rows.append({
                "date": d, "ticker": f"T{t:02d}",
                "pred": rng.normal(), "fwd_ret": rng.normal(0.001, 0.02),
            })
    return pd.DataFrame(rows)


def test_top_k_basket_size():
    day = _pred_frame(n_dates=1).groupby("date").get_group(
        pd.Timestamp("2024-01-01"))
    cfg = Config(top_k_positions=5, hysteresis_enabled=False)
    basket = _select_long_basket(day, cfg, prev_long=set())
    assert len(basket) == 5
    # exactly the 5 highest scores
    assert set(basket["ticker"]) == set(day.nlargest(5, "pred")["ticker"])


def test_hysteresis_keeps_held_names_above_exit_band():
    day = _pred_frame(n_dates=1).groupby("date").get_group(
        pd.Timestamp("2024-01-01")).copy()
    day["rank_tmp"] = day["pred"].rank(pct=True)
    # a held name at ~70th percentile: outside top-5 but above the 60% exit band
    held_name = day[(day["rank_tmp"] > 0.65) & (day["rank_tmp"] < 0.75)].iloc[0]["ticker"]
    cfg = Config(top_k_positions=5, hysteresis_enabled=True,
                 entry_rank_pct=0.80, exit_rank_pct=0.60)
    basket = _select_long_basket(day, cfg, prev_long={held_name})
    assert held_name in set(basket["ticker"])          # rolled, not churned
    assert len(basket) == 5

    # …but a held name below the exit band is dropped
    weak_name = day[day["rank_tmp"] < 0.3].iloc[0]["ticker"]
    basket2 = _select_long_basket(day, cfg, prev_long={weak_name})
    assert weak_name not in set(basket2["ticker"])


def test_hysteresis_reduces_cost_drag_on_noisy_scores():
    preds = _pred_frame(n_dates=40, n_tickers=25, seed=3)
    base = dict(top_k_positions=5, regime_filter=False,
                initial_capital=20_000, position_size_pct=0.19,
                capital_aware_costs=True, rebalance_every=5, n_quantile=5)
    churn = run_backtest(preds, Config(hysteresis_enabled=False, **base))
    calm = run_backtest(preds, Config(hysteresis_enabled=True, **base))
    assert "error" not in churn and "error" not in calm
    # same random returns, fewer forced round trips → gross-of-alpha the
    # hysteresis book must not pay MORE in turnover cost
    # (compare realized mean period return; alpha is zero by construction so
    # the difference is pure cost drag)
    assert calm["avg_period_ret"] >= churn["avg_period_ret"] - 1e-9


def test_backtest_quintile_mode_unchanged_when_top_k_zero():
    preds = _pred_frame()
    cfg = Config(top_k_positions=0, hysteresis_enabled=False, regime_filter=False)
    stats = run_backtest(preds, cfg)
    assert "error" not in stats
    assert stats["n_periods"] > 0


# ---------------------------------------------------------------------------
# Trade planner
# ---------------------------------------------------------------------------
def _signals(prices: dict[str, float]) -> pd.DataFrame:
    rows = []
    n = len(prices)
    for i, (tkr, px) in enumerate(prices.items()):
        prob = 0.9 - i * (0.8 / max(n - 1, 1))   # descending conviction
        rows.append({
            "ticker": tkr, "signal": "LONG", "prob_up": prob,
            "entry_price": px, "stop_loss": px * 0.95,
            "target_price": px * 1.10, "horizon_days": 5,
        })
    return pd.DataFrame(rows)


def _cfg20k(**over):
    defaults = dict(initial_capital=20_000, position_size_pct=0.19,
                    max_positions=5, hysteresis_enabled=False, min_edge_ratio=2.0)
    defaults.update(over)
    return Config(**defaults)


def test_plan_integer_shares_within_budget():
    plan = build_trade_plan(_signals({"AAA.NS": 750.0}), _cfg20k(), cash=20_000)
    buy = plan[plan["action"] == "BUY"].iloc[0]
    assert buy["shares"] == 5                       # floor(3800 / 750)
    assert buy["position_value"] == pytest.approx(3_750)
    assert buy["est_charges"] > 20                  # includes flat DP + slippage
    assert buy["breakeven_move_pct"] > 0.5          # ~0.85% at this size


def test_plan_skips_unaffordable_share():
    # one share of MRF-like stock >> per-position budget
    plan = build_trade_plan(_signals({"MRF.NS": 140_000.0, "BBB.NS": 500.0}),
                            _cfg20k(), cash=20_000)
    mrf = plan[plan["ticker"] == "MRF.NS"].iloc[0]
    assert mrf["action"] == "SKIP" and "unaffordable" in mrf["note"]
    assert (plan[plan["ticker"] == "BBB.NS"]["action"] == "BUY").all()


def test_plan_edge_screen_drops_thin_trades():
    # target only +0.8% above entry → cannot clear 2× the ~0.9% breakeven
    sigs = _signals({"CCC.NS": 500.0})
    sigs["target_price"] = sigs["entry_price"] * 1.008
    plan = build_trade_plan(sigs, _cfg20k(), cash=20_000)
    row = plan.iloc[0]
    assert row["action"] == "SKIP" and "edge too thin" in row["note"]


def test_plan_respects_max_positions_and_cash():
    prices = {f"S{i}.NS": 400.0 + i for i in range(10)}
    plan = build_trade_plan(_signals(prices), _cfg20k(), cash=20_000)
    assert (plan["action"] == "BUY").sum() == 5     # max_positions cap
    total_spend = plan.loc[plan["action"] == "BUY", "position_value"].sum()
    assert total_spend <= 20_000


def test_plan_hold_and_sell_bands_for_held_names():
    prices = {f"S{i}.NS": 500.0 for i in range(10)}
    cfg = _cfg20k(hysteresis_enabled=True, entry_rank_pct=0.8, exit_rank_pct=0.6)
    # S0 has the top prob (rank 1.0 → HOLD); S9 the lowest (rank 0.1 → SELL)
    plan = build_trade_plan(_signals(prices), cfg, cash=10_000,
                            held_tickers={"S0.NS", "S9.NS"})
    assert plan.loc[plan["ticker"] == "S0.NS", "action"].iloc[0] == "HOLD"
    assert plan.loc[plan["ticker"] == "S9.NS", "action"].iloc[0] == "SELL"


# ---------------------------------------------------------------------------
# Paper portfolio: auto-open + expiry roll
# ---------------------------------------------------------------------------
def test_open_from_plan_executes_buys_with_cost_accounting():
    pf = PaperPortfolio(initial_capital=20_000, position_size_pct=0.19,
                        max_positions=5)
    plan = build_trade_plan(_signals({"AAA.NS": 750.0, "BBB.NS": 380.0}),
                            _cfg20k(), cash=pf.cash)
    opened = pf.open_from_plan(plan, entry_date="2026-07-03")
    assert len(opened) == 2
    assert all(t.entry_charges > 0 for t in opened)
    spent = sum(t.shares * t.entry_price + t.entry_charges for t in opened)
    assert pf.cash == pytest.approx(20_000 - spent, abs=1.0)
    # idempotent: re-running the same plan opens nothing new
    assert pf.open_from_plan(plan, entry_date="2026-07-03") == []


def test_update_rolls_expired_position_when_in_keep_set():
    pf = PaperPortfolio(initial_capital=20_000, position_size_pct=0.19,
                        max_positions=5)
    t = pf.open_manual("AAA.NS", ref_price=500.0, shares=7, stop_loss=450.0,
                       target_price=600.0, horizon_days=5,
                       entry_date="2026-06-20")
    price_df = pd.DataFrame({
        "date": [pd.Timestamp("2026-07-03")],
        "ticker": ["AAA.NS"], "close": [505.0],
    })
    closed = pf.update(price_df, keep_tickers={"AAA.NS"})
    assert closed == []                       # rolled, not expired
    assert t.horizon_days > 5 and "rolled" in t.notes
    # a week later, without the keep set, the rolled horizon expires normally
    later = price_df.assign(date=[pd.Timestamp("2026-07-15")])
    closed2 = pf.update(later)
    assert len(closed2) == 1 and closed2[0].exit_reason == "expired"
