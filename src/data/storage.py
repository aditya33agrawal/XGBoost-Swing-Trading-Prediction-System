"""DuckDB-backed storage layer.

Schema
------
prices(symbol TEXT, date DATE, open DOUBLE, high DOUBLE, low DOUBLE,
       close DOUBLE, volume DOUBLE, PRIMARY KEY (symbol, date))

All writes are idempotent upserts keyed on (symbol, date).
The raw Parquet landing zone lives alongside the DB for immutable archival.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _get_con(db_path: str):
    try:
        import duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required for persistent storage: pip install duckdb\n"
            "The pipeline will still run; storage is skipped when DuckDB is absent."
        )
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(db_path)


def init_db(db_path: str) -> None:
    """Create tables if they don't exist."""
    con = _get_con(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol  TEXT,
            date    DATE,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS index_prices (
            symbol  TEXT,
            date    DATE,
            close   DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)
    con.close()


def upsert_prices(df: pd.DataFrame, db_path: str) -> int:
    """Idempotent upsert of OHLCV rows.  Returns number of new rows inserted."""
    init_db(db_path)
    con = _get_con(db_path)
    df = df.rename(columns={"ticker": "symbol"})
    required = ["symbol", "date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in required if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # pandas 2.x may use the new 'str' StringDtype which older DuckDB can't scan;
    # force plain object dtype so the dataframe SELECT works everywhere.
    df["symbol"] = df["symbol"].astype(object)

    before = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    con.execute("""
        INSERT OR REPLACE INTO prices
        SELECT symbol, date, open, high, low, close, volume
        FROM df
    """)
    after = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    con.close()
    return after - before


def upsert_index(df: pd.DataFrame, db_path: str) -> None:
    init_db(db_path)
    con = _get_con(db_path)
    df = df.rename(columns={"ticker": "symbol"})[["symbol", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = df["symbol"].astype(object)
    con.execute("""
        INSERT OR REPLACE INTO index_prices SELECT symbol, date, close FROM df
    """)
    con.close()


def load_prices(
    db_path: str,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load price data from DuckDB; returns long-format DataFrame."""
    con = _get_con(db_path)
    where_clauses = []
    if symbols:
        sym_list = ", ".join(f"'{s}'" for s in symbols)
        where_clauses.append(f"symbol IN ({sym_list})")
    if start:
        where_clauses.append(f"date >= '{start}'")
    if end:
        where_clauses.append(f"date <= '{end}'")
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    df = con.execute(f"SELECT * FROM prices {where} ORDER BY symbol, date").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"symbol": "ticker"})
    return df


def load_index(db_path: str) -> pd.DataFrame:
    con = _get_con(db_path)
    df = con.execute("SELECT * FROM index_prices ORDER BY symbol, date").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_parquet(df: pd.DataFrame, raw_dir: str, name: str) -> None:
    """Immutable raw landing zone — append only, partitioned by date."""
    path = Path(raw_dir) / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df]).drop_duplicates(
            subset=["ticker", "date"], keep="last"
        )
    df.to_parquet(path, index=False)
