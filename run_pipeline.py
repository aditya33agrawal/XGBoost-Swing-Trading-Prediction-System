#!/usr/bin/env python3
"""Entry point for the Nifty 50 swing prediction pipeline.

Common usage
------------
# Full run (Optuna tuning + walk-forward backtest + paper trading)
python run_pipeline.py

# Fast signals only — ~2 min on CPU (loads saved params, skips backtest)
python run_pipeline.py --fast-signals

# Full run without paper trading
python run_pipeline.py --no-paper-trade

# Different horizon / universe period
python run_pipeline.py --horizon 10
python run_pipeline.py --start 2018-01-01 --end 2024-12-31

# Regression mode (predict returns instead of barrier labels)
python run_pipeline.py --label-type fwd_ret

# Long/short mode (requires F&O capability)
python run_pipeline.py --mode long_short

# Tune with more Optuna trials for better hyperparameters
python run_pipeline.py --trials 100

# Set paper-trading capital
python run_pipeline.py --capital 500000 --max-positions 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.logging_setup import setup_logging
from src.pipeline.runner import run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nifty 50 XGBoost Swing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data / universe
    g_data = p.add_argument_group("data")
    g_data.add_argument("--start",  default="2015-01-01", help="History start date")
    g_data.add_argument("--end",    default=None, help="History end date (default: today)")
    g_data.add_argument("--db",     default="data/market.duckdb", help="DuckDB path")

    # Model
    g_model = p.add_argument_group("model")
    g_model.add_argument("--horizon",    type=int, default=5,
                         help="Swing horizon in trading days (default: 5)")
    g_model.add_argument("--label-type", choices=["triple_barrier", "fwd_ret"],
                         default="triple_barrier")
    g_model.add_argument("--mode",       choices=["long_only", "long_short"],
                         default="long_only")
    g_model.add_argument("--trials",     type=int, default=50,
                         help="Optuna trials (0 = skip, loads saved params if available)")
    g_model.add_argument("--device",     choices=["auto", "cuda", "cpu"], default="auto",
                         help="XGBoost device (auto detects GPU)")

    # Run modes
    g_run = p.add_argument_group("run modes")
    g_run.add_argument("--fast-signals", action="store_true",
                       help="Skip Optuna + walk-forward; only generate today's signals (~2 min)")
    g_run.add_argument("--skip-backtest", action="store_true",
                       help="Skip walk-forward backtest but still run Optuna")
    g_run.add_argument("--no-save", action="store_true",
                       help="Do not write outputs/ files")

    # Paper trading
    g_pt = p.add_argument_group("paper trading")
    g_pt.add_argument("--no-paper-trade", action="store_true",
                      help="Disable paper-trading portfolio update")
    g_pt.add_argument("--capital",       type=float, default=1_000_000,
                      help="Paper-trading initial capital in INR (default: 10,00,000)")
    g_pt.add_argument("--position-size", type=float, default=0.05,
                      help="Fraction of portfolio per position (default: 0.05 = 5%%)")
    g_pt.add_argument("--max-positions", type=int,   default=10,
                      help="Max concurrent open positions (default: 10)")
    g_pt.add_argument("--portfolio",     default="outputs/portfolio.json",
                      help="Portfolio state file path")

    # Supabase / tracking
    g_db = p.add_argument_group("tracking")
    g_db.add_argument("--model-version", default="",
                      help="Override model version tag (default: auto v{YYYYMMDD})")
    g_db.add_argument("--no-supabase", action="store_true",
                      help="Disable Supabase writes; fall back to JSON files only")

    p.add_argument("--log-level", default="INFO", help="DEBUG / INFO / WARNING")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    fast = args.fast_signals
    cfg_overrides = {}
    if args.end is not None:
        cfg_overrides["end"] = args.end
    cfg = Config(
        start          = args.start,
        horizon        = args.horizon,
        label_type     = args.label_type,
        mode           = args.mode,
        db_path        = args.db,
        device         = args.device,
        rebalance_every= args.horizon,
        embargo        = args.horizon,
        # Optuna: fast-signals forces 0 trials (loads saved params)
        xgb_n_trials   = 0 if fast else args.trials,
        # Backtest
        skip_backtest  = fast or args.skip_backtest,
        # Outputs
        save_outputs   = not args.no_save,
        # Paper trading
        paper_trade      = not args.no_paper_trade,
        initial_capital  = args.capital,
        position_size_pct= args.position_size,
        max_positions    = args.max_positions,
        portfolio_path   = args.portfolio,
        # Tracking / DB
        model_version    = args.model_version,
        save_to_supabase = not args.no_supabase,
        **cfg_overrides,
    )

    run(cfg)


if __name__ == "__main__":
    main()
