"""NSE equity market-hours gate (Mon-Fri, 09:15-15:30 IST).

Does not account for exchange holidays — only weekday + time-of-day. Used to
block paper-trade entries outside trading hours so every fill happens at a
real, current CMP rather than a stale price from an off-hours signal run.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 30)


def now_ist() -> datetime:
    return datetime.now(IST)


def is_market_open(now: datetime | None = None) -> bool:
    now = now or now_ist()
    now = now.astimezone(IST)
    if now.weekday() >= 5:  # Sat=5, Sun=6
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def next_market_open(now: datetime | None = None) -> datetime:
    """Next IST timestamp the market opens, strictly after `now`."""
    now = now or now_ist()
    now = now.astimezone(IST)
    candidate = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now.time() < MARKET_OPEN:
        pass  # today's open hasn't happened yet
    else:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def market_status_message(now: datetime | None = None) -> str:
    now = now or now_ist()
    if is_market_open(now):
        return f"🟢 Market open — trades fill at CMP (as of {now.astimezone(IST):%H:%M} IST)"
    nxt = next_market_open(now)
    return f"🔴 Market closed — next open {nxt:%a %d %b, %H:%M} IST"
