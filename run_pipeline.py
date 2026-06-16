#!/usr/bin/env python3
"""Entry point for the Nifty 50 swing prediction pipeline.

Usage:
    python run_pipeline.py                        # default config
    python run_pipeline.py --horizon 10           # 2-week horizon
    python run_pipeline.py --label-type fwd_ret   # regression mode
    python run_pipeline.py --mode long_short       # long/short (needs F&O)
    python run_pipeline.py --trials 0             # skip Optuna (fast dev run)
    python run_pipeline.py --start 2018-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.pipeline.runner import run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nifty 50 XGBoost Swing Pipeline")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--horizon", type=int, default=5, help="Swing horizon (trading days)")
    p.add_argument(
        "--label-type",
        choices=["triple_barrier", "fwd_ret"],
        default="triple_barrier",
    )
    p.add_argument(
        "--mode",
        choices=["long_only", "long_short"],
        default="long_only",
        help="long_short requires overnight short via F&O",
    )
    p.add_argument("--trials", type=int, default=50, help="Optuna trials (0 = skip)")
    p.add_argument("--db", default="data/market.duckdb", help="DuckDB path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        label_type=args.label_type,
        mode=args.mode,
        xgb_n_trials=args.trials,
        db_path=args.db,
        rebalance_every=args.horizon,  # rebalance at horizon frequency
        embargo=args.horizon,
    )
    run(cfg)


if __name__ == "__main__":
    main()
