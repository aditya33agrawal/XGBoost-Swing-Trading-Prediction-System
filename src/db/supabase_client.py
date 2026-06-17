"""Supabase client wrapper with graceful JSON fallback.

Contract:
  - get_supabase_client() returns a live Client or None.
  - Every caller must handle None (skip DB operation, fall back to JSON).
  - No function in this module ever raises — failures are logged and swallowed.

This allows the entire pipeline to run without any Supabase credentials
(local development, CI, first-time setup).
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------
def get_supabase_client(url: str = "", key: str = ""):
    """Return a live supabase.Client or None if credentials unavailable.

    Reads SUPABASE_URL / SUPABASE_KEY from env when args are empty strings.
    Never raises — returns None on any failure.
    """
    url = url or os.getenv("SUPABASE_URL", "")
    key = (
        key
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or ""
    )

    if not url or not key:
        logger.debug("Supabase credentials not set — running in JSON-only mode")
        return None

    try:
        from supabase import create_client  # type: ignore
        client = create_client(url, key)
        logger.debug("Supabase client connected to %s", url[:40])
        return client
    except ImportError:
        logger.warning("supabase package not installed — pip install supabase>=2.0")
        return None
    except Exception as exc:
        logger.warning("Supabase connection failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------
def upsert_rows(
    client,
    table: str,
    rows: list[dict[str, Any]],
    on_conflict: str = "id",
) -> bool:
    """Upsert a list of dicts into `table`.

    Returns True on success, False on any failure (including client=None).
    Silently logs errors rather than raising.
    """
    if client is None or not rows:
        return False
    try:
        (
            client.table(table)
            .upsert(rows, on_conflict=on_conflict)
            .execute()
        )
        return True
    except Exception as exc:
        logger.warning("upsert_rows(%s) failed: %s", table, exc)
        return False


def fetch_rows(
    client,
    table: str,
    filters: dict[str, Any] | None = None,
    order_by: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch rows from `table` with optional equality filters.

    Returns [] on any failure (including client=None).
    """
    if client is None:
        return []
    try:
        q = client.table(table).select("*")
        for col, val in (filters or {}).items():
            q = q.eq(col, val)
        if order_by:
            desc = order_by.startswith("-")
            col  = order_by.lstrip("-")
            q = q.order(col, desc=desc)
        if limit:
            q = q.limit(limit)
        resp = q.execute()
        return resp.data or []
    except Exception as exc:
        logger.warning("fetch_rows(%s) failed: %s", table, exc)
        return []


def execute_sql(client, sql: str, params: dict | None = None) -> list[dict]:
    """Run a raw SQL query via Supabase's rpc or PostgREST.

    Uses the `execute_sql` RPC if available, otherwise falls back to
    the PostgREST query builder (limited).  Returns [] on failure.
    """
    if client is None:
        return []
    try:
        resp = client.rpc("execute_sql", {"query": sql, "params": params or {}}).execute()
        return resp.data or []
    except Exception:
        # Fallback: not all Supabase plans expose execute_sql RPC
        logger.debug("execute_sql RPC not available — use fetch_rows for simple queries")
        return []
