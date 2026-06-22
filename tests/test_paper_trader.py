"""PaperPortfolio real-money accounting: charges, slippage, and the funds
ledger must reconcile to the penny (within float rounding)."""
import pandas as pd
import pytest

from src.trading.paper_trader import PaperPortfolio


def _reconciles(pf: PaperPortfolio) -> bool:
    if not pf.ledger:
        return pf.cash == pf.initial_capital
    return abs(pf.cash - pf.ledger[-1]["running_balance"]) < 0.01


def test_opening_balance_seeded():
    pf = PaperPortfolio(initial_capital=500_000)
    assert len(pf.ledger) == 1
    assert pf.ledger[0]["type"] == "OPENING_BALANCE"
    assert pf.ledger[0]["amount"] == 500_000
    assert _reconciles(pf)


def test_open_manual_charges_cash_and_appends_ledger():
    pf = PaperPortfolio(initial_capital=1_000_000)
    cash_before = pf.cash
    trade = pf.open_manual("RELIANCE.NS", ref_price=2500.0, shares=10,
                            stop_loss=2400, target_price=2700)
    assert trade.entry_charges > 0
    assert trade.entry_price > 2500.0          # slippage moves fill against buyer
    assert pf.cash < cash_before
    assert len(pf.ledger) == 3                  # OPENING_BALANCE, BUY, CHARGE
    assert _reconciles(pf)


def test_close_manual_net_pnl_includes_all_costs():
    pf = PaperPortfolio(initial_capital=1_000_000)
    trade = pf.open_manual("RELIANCE.NS", ref_price=2500.0, shares=10,
                            stop_loss=2400, target_price=2700)
    closed = pf.close_manual(trade.trade_id, ref_price=2600.0, reason="manual")

    gross_cost = closed.shares * closed.entry_price
    expected_net = closed.gross_pnl - closed.exit_charges - closed.entry_charges
    assert abs(closed.pnl - expected_net) < 0.01
    assert closed.exit_price < 2600.0            # slippage moves fill against seller
    assert _reconciles(pf)


def test_insufficient_cash_raises():
    pf = PaperPortfolio(initial_capital=1_000.0)
    with pytest.raises(ValueError):
        pf.open_manual("RELIANCE.NS", ref_price=2500.0, shares=100,
                        stop_loss=2400, target_price=2700)


def test_breakeven_price_above_entry():
    pf = PaperPortfolio(initial_capital=1_000_000)
    trade = pf.open_manual("RELIANCE.NS", ref_price=2500.0, shares=10,
                            stop_loss=2400, target_price=2700)
    assert trade.breakeven_price > trade.entry_price


def test_deposit_withdraw_reconcile():
    pf = PaperPortfolio(initial_capital=1_000_000)
    pf.deposit(50_000, note="top-up")
    assert pf.cash == 1_050_000
    pf.withdraw(20_000, note="profit-take")
    assert pf.cash == 1_030_000
    assert _reconciles(pf)
    with pytest.raises(ValueError):
        pf.withdraw(10_000_000)


def test_save_load_roundtrip(tmp_path):
    pf = PaperPortfolio(initial_capital=1_000_000)
    trade = pf.open_manual("RELIANCE.NS", ref_price=2500.0, shares=10,
                            stop_loss=2400, target_price=2700)
    pf.close_manual(trade.trade_id, ref_price=2600.0, reason="manual")

    path = tmp_path / "portfolio.json"
    pf.save(str(path))
    pf2 = PaperPortfolio.load(str(path))

    assert abs(pf2.cash - pf.cash) < 1e-6
    assert len(pf2.trades) == len(pf.trades)
    assert len(pf2.ledger) == len(pf.ledger)
    assert _reconciles(pf2)


def test_auto_close_on_target_charges_correctly():
    pf = PaperPortfolio(initial_capital=1_000_000)
    pf.open_manual("RELIANCE.NS", ref_price=2500.0, shares=10,
                    stop_loss=2400, target_price=2600)
    price_df = pd.DataFrame([
        {"ticker": "RELIANCE.NS", "date": pd.Timestamp("2026-06-25"), "close": 2650.0},
    ])
    closed = pf.update(price_df)
    assert len(closed) == 1
    assert closed[0].exit_reason == "target"
    assert closed[0].exit_charges > 0
    assert _reconciles(pf)
