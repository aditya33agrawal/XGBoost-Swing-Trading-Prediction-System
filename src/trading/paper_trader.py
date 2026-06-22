"""Paper-trading portfolio tracker.

Tracks open and closed positions driven by model signals (or opened
manually from the UI). Every fill is charged real Indian delivery-trade
costs (STT, exchange, SEBI, stamp, GST, DP charge — see
src/backtest/costs.py) plus a configurable slippage, so paper P&L behaves
like a real NSE account rather than a frictionless backtest. Every cash
movement is recorded in self.ledger as an immutable, broker-style funds
statement (BUY / SELL / CHARGE / DEPOSIT / WITHDRAWAL / OPENING_BALANCE).

Typical flow
------------
    portfolio = PaperPortfolio.load(cfg.portfolio_path)
    portfolio.update(price_df)                  # close expired / stop / target
    portfolio.add_signals(signals, price_df, cfg)   # open new LONG positions
    portfolio.print_summary(price_df)
    portfolio.save(cfg.portfolio_path)
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.costs import buy_leg_cost, sell_leg_cost

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    trade_id:     str
    ticker:       str
    entry_date:   str
    entry_price:  float          # actual fill price (post-slippage)
    shares:       int
    signal:       str            # "LONG"
    stop_loss:    float
    target_price: float
    horizon_days: int
    prob_up:      float
    status:       str = "open"   # open | closed
    exit_date:    Optional[str]   = None
    exit_price:   Optional[float] = None   # actual fill price (post-slippage)
    exit_reason:  Optional[str]   = None   # target | stop | expired | manual
    pnl:          Optional[float] = None   # NET of all charges + slippage
    pnl_pct:      Optional[float] = None   # net pnl / gross entry cost, %
    # --- real-money accounting ---------------------------------------
    entry_charges:   float = 0.0           # STT+exch+SEBI+stamp+GST on entry
    exit_charges:    Optional[float] = None
    entry_slippage:  float = 0.0           # INR lost to slippage on entry fill
    exit_slippage:   Optional[float] = None
    breakeven_price: float = 0.0           # exit price needed to net P&L = 0
    gross_pnl:       Optional[float] = None  # price-only P&L (slippage incl., charges excl.)
    gross_pnl_pct:   Optional[float] = None
    opened_via:      str = "signal"        # signal | manual
    notes:           str = ""

    # ------------------------------------------------------------------
    def unrealised_pnl(self, current_price: float) -> float:
        """Net unrealised P&L: price move minus entry charges and an
        estimated exit-charge drag (so it doesn't look artificially green)."""
        gross = self.shares * (current_price - self.entry_price)
        est_exit_charges = sell_leg_cost(self.shares * current_price)["total"]
        return gross - self.entry_charges - est_exit_charges

    def unrealised_pnl_pct(self, current_price: float) -> float:
        gross_cost = self.shares * self.entry_price
        if gross_cost == 0:
            return 0.0
        return self.unrealised_pnl(current_price) / gross_cost

    def days_held(self, as_of: str) -> int:
        try:
            e = datetime.strptime(self.entry_date, _DATE_FMT).date()
            a = datetime.strptime(as_of,           _DATE_FMT).date()
            return (a - e).days
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
class PaperPortfolio:
    """Paper-trading portfolio with automatic stop / target / expiry logic
    and real Indian delivery-trade cost accounting.

    Parameters
    ----------
    initial_capital   : starting cash in INR
    position_size_pct : fraction of *current* portfolio value allocated per position
    max_positions     : maximum concurrent open positions
    slippage_bps       : one-way slippage in basis points applied against the
                          trader on every fill (entry fills higher, exit fills lower)
    """

    def __init__(
        self,
        initial_capital:   float = 1_000_000,
        position_size_pct: float = 0.05,
        max_positions:     int   = 10,
        slippage_bps:      float = 10.0,
    ) -> None:
        self.initial_capital   = initial_capital
        self.cash              = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions     = max_positions
        self.slippage_bps      = slippage_bps
        self.trades:  list[Trade] = []
        self.ledger:  list[dict]  = []
        self._seed_opening_balance()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def open_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.status == "open"]

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.status == "closed"]

    @property
    def total_charges_paid(self) -> float:
        entry = sum(t.entry_charges or 0.0 for t in self.trades)
        exitc = sum(t.exit_charges or 0.0 for t in self.closed_trades)
        return round(entry + exitc, 2)

    def _latest_prices(self, price_df: pd.DataFrame) -> dict[str, float]:
        max_date = price_df["date"].max()
        return (
            price_df[price_df["date"] == max_date]
            .set_index("ticker")["close"]
            .to_dict()
        )

    def _as_of_str(self, price_df: pd.DataFrame) -> str:
        d = price_df["date"].max()
        return str(d.date()) if hasattr(d, "date") else str(d)

    def portfolio_value(self, price_df: pd.DataFrame) -> float:
        lp = self._latest_prices(price_df)
        invested = sum(t.shares * lp.get(t.ticker, t.entry_price) for t in self.open_trades)
        return self.cash + invested

    # ------------------------------------------------------------------
    # Ledger
    # ------------------------------------------------------------------
    def _append_ledger(
        self, entry_type: str, amount: float,
        trade_id: str = "", ticker: str = "",
        qty: float = 0, price: float = 0.0, note: str = "",
    ) -> dict:
        """Append an immutable funds-ledger row. Caller must apply `amount`
        to self.cash *before* calling this so running_balance is accurate."""
        row = {
            "id":              str(uuid.uuid4()),
            "ts":              datetime.now().isoformat(),
            "type":            entry_type,   # BUY|SELL|CHARGE|DEPOSIT|WITHDRAWAL|OPENING_BALANCE
            "trade_id":        trade_id,
            "ticker":          ticker,
            "qty":             qty,
            "price":           round(price, 2),
            "amount":          round(amount, 2),
            "running_balance": round(self.cash, 2),
            "note":            note,
        }
        self.ledger.append(row)
        return row

    def _seed_opening_balance(self) -> None:
        if not self.ledger:
            self._append_ledger("OPENING_BALANCE", self.initial_capital, note="Account opened")

    def deposit(self, amount: float, note: str = "") -> dict:
        self.cash += amount
        return self._append_ledger("DEPOSIT", amount, note=note or "Manual deposit")

    def withdraw(self, amount: float, note: str = "") -> dict:
        if amount > self.cash:
            raise ValueError(f"Cannot withdraw ₹{amount:,.0f} — only ₹{self.cash:,.0f} available")
        self.cash -= amount
        return self._append_ledger("WITHDRAWAL", -amount, note=note or "Manual withdrawal")

    # ------------------------------------------------------------------
    # Fill helpers — apply slippage + real charges
    # ------------------------------------------------------------------
    def _buy_fill(self, ref_price: float, shares: int) -> tuple[float, float, dict]:
        """Returns (fill_price, gross_cost, charges_dict). Slippage moves the
        fill price against the buyer (higher)."""
        fill_price = ref_price * (1 + self.slippage_bps / 1e4)
        gross_cost = fill_price * shares
        charges = buy_leg_cost(gross_cost)
        return fill_price, gross_cost, charges

    def _sell_fill(self, ref_price: float, shares: int) -> tuple[float, float, dict]:
        """Returns (fill_price, proceeds, charges_dict). Slippage moves the
        fill price against the seller (lower)."""
        fill_price = ref_price * (1 - self.slippage_bps / 1e4)
        proceeds = fill_price * shares
        charges = sell_leg_cost(proceeds)
        return fill_price, proceeds, charges

    @staticmethod
    def _estimate_breakeven(gross_cost: float, entry_charges: float, shares: int) -> float:
        """Exit price (pre-slippage) needed so net P&L ≈ 0, using the entry
        value as a proxy for the (unknown) future exit value when estimating
        sell-side charges."""
        est_exit_charges = sell_leg_cost(gross_cost)["total"]
        return (gross_cost + entry_charges + est_exit_charges) / max(shares, 1)

    # ------------------------------------------------------------------
    # Internal: open / close a position with real cost accounting
    # ------------------------------------------------------------------
    def _open_position(
        self,
        ticker: str,
        ref_price: float,
        shares: int,
        stop_loss: float,
        target_price: float,
        horizon_days: int,
        prob_up: float,
        entry_date: str,
        opened_via: str = "signal",
        notes: str = "",
    ) -> Trade:
        fill_price, gross_cost, charges = self._buy_fill(ref_price, shares)
        total_debit = gross_cost + charges["total"]
        if total_debit > self.cash:
            raise ValueError(
                f"Insufficient cash for {ticker}: need ₹{total_debit:,.0f}, have ₹{self.cash:,.0f}"
            )

        self.cash -= gross_cost
        self._append_ledger("BUY", -gross_cost, trade_id="", ticker=ticker,
                             qty=shares, price=fill_price, note=f"Bought {shares} {ticker}")
        self.cash -= charges["total"]
        self._append_ledger("CHARGE", -charges["total"], ticker=ticker,
                             qty=shares, price=fill_price,
                             note=f"Entry charges: {charges}")

        slippage_amt = (fill_price - ref_price) * shares
        breakeven = self._estimate_breakeven(gross_cost, charges["total"], shares)

        trade = Trade(
            trade_id        = str(uuid.uuid4())[:8],
            ticker          = ticker,
            entry_date      = entry_date,
            entry_price     = round(fill_price, 2),
            shares          = shares,
            signal          = "LONG",
            stop_loss       = round(stop_loss, 2),
            target_price    = round(target_price, 2),
            horizon_days    = horizon_days,
            prob_up         = round(prob_up, 4),
            entry_charges   = round(charges["total"], 2),
            entry_slippage  = round(slippage_amt, 2),
            breakeven_price = round(breakeven, 2),
            opened_via      = opened_via,
            notes           = notes,
        )
        # backfill trade_id on the ledger rows just written
        self.ledger[-1]["trade_id"] = trade.trade_id
        self.ledger[-2]["trade_id"] = trade.trade_id
        self.trades.append(trade)
        return trade

    def _close_trade(
        self, trade: Trade, ref_price: float, exit_date: str, reason: str,
    ) -> None:
        fill_price, proceeds, charges = self._sell_fill(ref_price, trade.shares)
        gross_cost = trade.shares * trade.entry_price

        self.cash += proceeds
        self._append_ledger("SELL", proceeds, trade_id=trade.trade_id, ticker=trade.ticker,
                             qty=trade.shares, price=fill_price,
                             note=f"Sold {trade.shares} {trade.ticker} ({reason})")
        self.cash -= charges["total"]
        self._append_ledger("CHARGE", -charges["total"], trade_id=trade.trade_id,
                             ticker=trade.ticker, qty=trade.shares, price=fill_price,
                             note=f"Exit charges: {charges}")

        gross_pnl = proceeds - gross_cost
        net_pnl   = gross_pnl - charges["total"] - trade.entry_charges

        trade.status        = "closed"
        trade.exit_date      = exit_date
        trade.exit_price     = round(fill_price, 2)
        trade.exit_reason    = reason
        trade.exit_charges   = round(charges["total"], 2)
        trade.exit_slippage  = round((ref_price - fill_price) * trade.shares, 2)
        trade.gross_pnl      = round(gross_pnl, 2)
        trade.gross_pnl_pct  = round(gross_pnl / max(gross_cost, 1e-9) * 100, 2)
        trade.pnl            = round(net_pnl, 2)
        trade.pnl_pct        = round(net_pnl / max(gross_cost, 1e-9) * 100, 2)

        pnl_sign = "+" if trade.pnl >= 0 else ""
        logger.info(
            "  CLOSE %-16s @ %8.2f  |  %-8s  |  net P&L: %s₹%.0f  (%s%.1f%%)  charges ₹%.0f+%.0f",
            trade.ticker, fill_price, reason,
            pnl_sign, trade.pnl, pnl_sign, trade.pnl_pct,
            trade.entry_charges, trade.exit_charges,
        )

    # ------------------------------------------------------------------
    # Update: close positions that hit stop / target / expiry
    # ------------------------------------------------------------------
    def update(self, price_df: pd.DataFrame) -> list[Trade]:
        """Close any open position whose stop, target, or horizon was hit."""
        if not self.open_trades:
            return []

        lp     = self._latest_prices(price_df)
        as_of  = self._as_of_str(price_df)
        closed = []

        for trade in self.open_trades:
            price = lp.get(trade.ticker)
            if price is None:
                continue

            days   = trade.days_held(as_of)
            reason = None

            if price >= trade.target_price:
                reason = "target"
            elif price <= trade.stop_loss:
                reason = "stop"
            elif days >= trade.horizon_days:
                reason = "expired"

            if reason:
                self._close_trade(trade, price, as_of, reason)
                closed.append(trade)

        if closed:
            logger.info(
                "Closed %d position(s): %s",
                len(closed),
                [f"{t.ticker}({t.exit_reason})" for t in closed],
            )
        return closed

    # ------------------------------------------------------------------
    # Add signals: open new positions
    # ------------------------------------------------------------------
    def add_signals(
        self,
        signals:   pd.DataFrame,
        price_df:  pd.DataFrame,
        cfg,
    ) -> list[Trade]:
        """Open LONG positions for signals not already held."""
        if signals.empty:
            return []

        long_signals = signals[signals["signal"] == "LONG"]
        if long_signals.empty:
            logger.info("No LONG signals — no new positions opened")
            return []

        open_tickers  = {t.ticker for t in self.open_trades}
        lp            = self._latest_prices(price_df)
        entry_date    = self._as_of_str(price_df)
        opened: list[Trade] = []

        for _, row in long_signals.iterrows():
            if len(self.open_trades) + len(opened) >= self.max_positions:
                logger.info("Max positions (%d) reached — skipping remaining signals", self.max_positions)
                break

            ticker = row["ticker"]
            if ticker in open_tickers:
                continue

            price = lp.get(ticker)
            if price is None or price <= 0:
                logger.warning("No price for %s — skipping", ticker)
                continue

            # Position sizing: fraction of current portfolio value
            portfolio_val = self.portfolio_value(price_df)
            alloc  = portfolio_val * self.position_size_pct
            shares = max(1, int(alloc // price))

            stop    = float(row.get("stop_loss",    price * 0.97))
            target  = float(row.get("target_price", price * 1.04))
            horizon = int(row.get("horizon_days",   getattr(cfg, "horizon", 5)))
            prob    = float(row.get("prob_up",       0.5))

            try:
                trade = self._open_position(
                    ticker, price, shares, stop, target, horizon, prob,
                    entry_date, opened_via="signal",
                )
            except ValueError as exc:
                logger.info("  Skipping %-16s — %s", ticker, exc)
                continue

            open_tickers.add(ticker)
            opened.append(trade)
            logger.info(
                "  OPEN  %-16s @ %8.2f  |  shares=%4d  |  stop=%8.2f  target=%8.2f  |  charges ₹%.0f",
                ticker, trade.entry_price, shares, stop, target, trade.entry_charges,
            )

        if opened:
            logger.info("Opened %d new position(s)", len(opened))
        return opened

    # ------------------------------------------------------------------
    # Manual open / close (UI-driven, ad-hoc trades)
    # ------------------------------------------------------------------
    def open_manual(
        self,
        ticker: str,
        ref_price: float,
        shares: int,
        stop_loss: float,
        target_price: float,
        horizon_days: int = 5,
        prob_up: float = 0.5,
        entry_date: Optional[str] = None,
        opened_via: str = "manual",
        notes: str = "",
    ) -> Trade:
        if shares <= 0 or ref_price <= 0:
            raise ValueError("shares and ref_price must be positive")
        entry_date = entry_date or datetime.now().strftime(_DATE_FMT)
        return self._open_position(
            ticker.upper(), ref_price, int(shares), stop_loss, target_price,
            int(horizon_days), prob_up, entry_date, opened_via=opened_via, notes=notes,
        )

    def close_manual(
        self,
        trade_id: str,
        ref_price: float,
        exit_date: Optional[str] = None,
        reason: str = "manual",
    ) -> Trade:
        for trade in self.open_trades:
            if trade.trade_id == trade_id:
                exit_date = exit_date or datetime.now().strftime(_DATE_FMT)
                self._close_trade(trade, ref_price, exit_date, reason)
                return trade
        raise ValueError(f"No open trade with trade_id={trade_id}")

    def edit_trade(self, trade_id: str, **fields) -> Trade:
        """Edit mutable fields (stop_loss, target_price, notes) on an open trade."""
        for trade in self.open_trades:
            if trade.trade_id == trade_id:
                for k, v in fields.items():
                    if hasattr(trade, k) and v is not None:
                        setattr(trade, k, v)
                return trade
        raise ValueError(f"No open trade with trade_id={trade_id}")

    def close_position(
        self,
        ticker:     str,
        price_df:   pd.DataFrame,
        reason:     str = "manual",
    ) -> bool:
        """Legacy ticker-based close (first open match), driven by a price frame."""
        lp = self._latest_prices(price_df)
        as_of = self._as_of_str(price_df)
        for trade in self.open_trades:
            if trade.ticker == ticker:
                price = lp.get(ticker, trade.entry_price)
                self._close_trade(trade, price, as_of, reason)
                return True
        logger.warning("No open position found for %s", ticker)
        return False

    # ------------------------------------------------------------------
    # Summary display
    # ------------------------------------------------------------------
    def print_summary(self, price_df: pd.DataFrame | None = None) -> None:
        pv        = self.portfolio_value(price_df) if price_df is not None else self.cash
        total_pnl = sum(t.pnl for t in self.closed_trades if t.pnl is not None)
        n_win     = sum(1 for t in self.closed_trades if t.pnl and t.pnl > 0)
        n_closed  = len(self.closed_trades)
        win_rate  = n_win / n_closed * 100 if n_closed else 0.0
        total_ret = (pv / self.initial_capital - 1) * 100

        w = 60
        print(f"\n{'═' * w}")
        print(f"  PAPER PORTFOLIO  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'═' * w}")
        print(f"  {'Initial capital':<28}  ₹{self.initial_capital:>12,.0f}")
        print(f"  {'Portfolio value':<28}  ₹{pv:>12,.0f}")
        print(f"  {'Cash':<28}  ₹{self.cash:>12,.0f}")
        sign = "+" if total_pnl >= 0 else ""
        print(f"  {'Realised P&L (net)':<28}  {sign}₹{total_pnl:>11,.0f}")
        print(f"  {'Total charges paid':<28}  ₹{self.total_charges_paid:>12,.0f}")
        sign = "+" if total_ret >= 0 else ""
        print(f"  {'Total return':<28}  {sign}{total_ret:>11.1f}%")
        print(f"{'─' * w}")
        print(f"  {'Open positions':<28}  {len(self.open_trades):>13d}")
        print(f"  {'Closed trades':<28}  {n_closed:>13d}")
        if n_closed:
            print(f"  {'Win rate':<28}  {win_rate:>12.1f}%")
        print(f"{'─' * w}")

        # Open positions table
        if self.open_trades and price_df is not None:
            lp    = self._latest_prices(price_df)
            as_of = self._as_of_str(price_df)
            print(f"\n  OPEN POSITIONS:")
            hdr = (
                f"  {'Ticker':<16}  {'Entry':>9}  {'Now':>9}  "
                f"{'Unrl P&L':>9}  {'Days':>5}  {'Stop':>9}  {'Target':>9}"
            )
            print(hdr)
            print(f"  {'─' * 74}")
            for t in self.open_trades:
                cp      = lp.get(t.ticker, t.entry_price)
                pnl_pct = t.unrealised_pnl_pct(cp) * 100
                days    = t.days_held(as_of)
                sign    = "+" if pnl_pct >= 0 else ""
                print(
                    f"  {t.ticker:<16}  {t.entry_price:>9.2f}  {cp:>9.2f}  "
                    f"{sign}{pnl_pct:>8.1f}%  {days:>5d}  {t.stop_loss:>9.2f}  {t.target_price:>9.2f}"
                )

        # Recent closed
        if self.closed_trades:
            recent = self.closed_trades[-5:]
            print(f"\n  RECENT CLOSED (last {len(recent)}, net of charges):")
            hdr = (
                f"  {'Ticker':<16}  {'Entry':>9}  {'Exit':>9}  "
                f"{'P&L':>10}  {'P&L%':>7}  {'Reason':<10}"
            )
            print(hdr)
            print(f"  {'─' * 66}")
            for t in recent:
                sign = "+" if (t.pnl or 0) >= 0 else ""
                print(
                    f"  {t.ticker:<16}  {t.entry_price:>9.2f}  {t.exit_price:>9.2f}  "
                    f"{sign}₹{t.pnl:>8,.0f}  {sign}{t.pnl_pct:>5.1f}%  {t.exit_reason:<10}"
                )

        print(f"{'═' * w}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = "outputs/portfolio.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "saved_at":        datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "cash":            self.cash,
            "position_size_pct": self.position_size_pct,
            "max_positions":   self.max_positions,
            "slippage_bps":    self.slippage_bps,
            "trades":          [asdict(t) for t in self.trades],
            "ledger":          self.ledger,
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))
        logger.info(
            "Portfolio saved → %s  (%d open, %d closed, %d ledger rows)",
            path, len(self.open_trades), len(self.closed_trades), len(self.ledger),
        )

    @classmethod
    def load(cls, path: str = "outputs/portfolio.json") -> "PaperPortfolio":
        p = Path(path)
        if not p.exists():
            logger.info("No portfolio at %s — starting fresh", path)
            return cls()
        data = json.loads(p.read_text())
        portfolio = cls.__new__(cls)
        portfolio.initial_capital   = data.get("initial_capital",   1_000_000)
        portfolio.position_size_pct = data.get("position_size_pct", 0.05)
        portfolio.max_positions     = data.get("max_positions",     10)
        portfolio.slippage_bps      = data.get("slippage_bps",      10.0)
        portfolio.cash               = data.get("cash", portfolio.initial_capital)
        portfolio.trades = []
        portfolio.ledger = data.get("ledger", [])
        known_fields = {f for f in Trade.__dataclass_fields__}
        for td in data.get("trades", []):
            portfolio.trades.append(Trade(**{k: v for k, v in td.items() if k in known_fields}))
        if not portfolio.ledger:
            portfolio._seed_opening_balance()
        logger.info(
            "Portfolio loaded ← %s  (%d open, %d closed, %d ledger rows)",
            path, len(portfolio.open_trades), len(portfolio.closed_trades), len(portfolio.ledger),
        )
        return portfolio
