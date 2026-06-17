"""Paper-trading portfolio tracker.

Tracks open and closed positions driven by model signals.
All prices come from the pipeline's price frame — no live data needed.

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
    entry_price:  float
    shares:       int
    signal:       str            # "LONG"
    stop_loss:    float
    target_price: float
    horizon_days: int
    prob_up:      float
    status:       str = "open"   # open | closed
    exit_date:    Optional[str]   = None
    exit_price:   Optional[float] = None
    exit_reason:  Optional[str]   = None   # target | stop | expired | manual
    pnl:          Optional[float] = None
    pnl_pct:      Optional[float] = None

    # ------------------------------------------------------------------
    def unrealised_pnl(self, current_price: float) -> float:
        return self.shares * (current_price - self.entry_price)

    def unrealised_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price

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
    """Paper-trading portfolio with automatic stop / target / expiry logic.

    Parameters
    ----------
    initial_capital   : starting cash in INR
    position_size_pct : fraction of *current* portfolio value allocated per position
    max_positions     : maximum concurrent open positions
    """

    def __init__(
        self,
        initial_capital:   float = 1_000_000,
        position_size_pct: float = 0.05,
        max_positions:     int   = 10,
    ) -> None:
        self.initial_capital   = initial_capital
        self.cash              = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions     = max_positions
        self.trades: list[Trade] = []

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def open_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.status == "open"]

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.status == "closed"]

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

    def _close_trade(
        self, trade: Trade, exit_price: float, exit_date: str, reason: str
    ) -> None:
        trade.status      = "closed"
        trade.exit_date   = exit_date
        trade.exit_price  = round(exit_price, 2)
        trade.exit_reason = reason
        trade.pnl         = round(trade.shares * (exit_price - trade.entry_price), 2)
        trade.pnl_pct     = round(
            (exit_price - trade.entry_price) / max(trade.entry_price, 1e-9) * 100, 2
        )
        self.cash += trade.shares * exit_price
        pnl_sign = "+" if trade.pnl >= 0 else ""
        logger.info(
            "  CLOSE %-16s @ %8.2f  |  %-8s  |  P&L: %s₹%.0f  (%s%.1f%%)",
            trade.ticker, exit_price, reason,
            pnl_sign, trade.pnl, pnl_sign, trade.pnl_pct,
        )

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
            cost   = shares * price

            if cost > self.cash:
                logger.info("  Skipping %-16s — need ₹%.0f, have ₹%.0f", ticker, cost, self.cash)
                continue

            self.cash -= cost

            stop    = float(row.get("stop_loss",    price * 0.97))
            target  = float(row.get("target_price", price * 1.04))
            horizon = int(row.get("horizon_days",   getattr(cfg, "horizon", 5)))
            prob    = float(row.get("prob_up",       0.5))

            trade = Trade(
                trade_id    = str(uuid.uuid4())[:8],
                ticker      = ticker,
                entry_date  = entry_date,
                entry_price = round(price, 2),
                shares      = shares,
                signal      = "LONG",
                stop_loss   = round(stop,   2),
                target_price= round(target, 2),
                horizon_days= horizon,
                prob_up     = round(prob, 4),
            )
            self.trades.append(trade)
            open_tickers.add(ticker)
            opened.append(trade)
            logger.info(
                "  OPEN  %-16s @ %8.2f  |  shares=%4d  |  stop=%8.2f  target=%8.2f",
                ticker, price, shares, stop, target,
            )

        if opened:
            logger.info("Opened %d new position(s)", len(opened))
        return opened

    # ------------------------------------------------------------------
    # Manual close (for paper-trade override)
    # ------------------------------------------------------------------
    def close_position(
        self,
        ticker:     str,
        price_df:   pd.DataFrame,
        reason:     str = "manual",
    ) -> bool:
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
        print(f"  {'Realised P&L':<28}  {sign}₹{total_pnl:>11,.0f}")
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
            print(f"\n  RECENT CLOSED (last {len(recent)}):")
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
            "trades":          [asdict(t) for t in self.trades],
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))
        logger.info(
            "Portfolio saved → %s  (%d open, %d closed)",
            path, len(self.open_trades), len(self.closed_trades),
        )

    @classmethod
    def load(cls, path: str = "outputs/portfolio.json") -> "PaperPortfolio":
        p = Path(path)
        if not p.exists():
            logger.info("No portfolio at %s — starting fresh", path)
            return cls()
        data = json.loads(p.read_text())
        portfolio = cls(
            initial_capital   = data.get("initial_capital",   1_000_000),
            position_size_pct = data.get("position_size_pct", 0.05),
            max_positions     = data.get("max_positions",     10),
        )
        portfolio.cash = data.get("cash", portfolio.initial_capital)
        for td in data.get("trades", []):
            portfolio.trades.append(Trade(**td))
        logger.info(
            "Portfolio loaded ← %s  (%d open, %d closed)",
            path, len(portfolio.open_trades), len(portfolio.closed_trades),
        )
        return portfolio
