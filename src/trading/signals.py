"""Enrich model signals with ATR-based entry/exit levels and persist to disk.

Usage:
    from src.trading.signals import enrich_signals, save_signals, print_signal_table

    signals = predict_latest(...)
    signals = enrich_signals(signals, price_df, cfg)
    save_signals(signals, cfg.output_dir)
    print_signal_table(signals)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ATR computation (raw price units, not z-scored)
# ---------------------------------------------------------------------------
def _compute_raw_atr14(price_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[ticker, atr14, close] using latest bar per ticker."""
    rows = []
    required = {"high", "low", "close"}
    if not required.issubset(price_df.columns):
        logger.warning("price_df missing high/low/close — cannot compute ATR")
        return pd.DataFrame(columns=["ticker", "atr14", "close"])

    for ticker, grp in price_df.groupby("ticker"):
        grp = grp.sort_values("date")
        if len(grp) < 15:
            continue
        hi = grp["high"].values.astype(float)
        lo = grp["low"].values.astype(float)
        cl = grp["close"].values.astype(float)
        tr = np.maximum(
            hi[1:] - lo[1:],
            np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])),
        )
        tr = np.concatenate([[tr[0]], tr])
        alpha = 2.0 / 15          # span-14 EWM
        atr = tr.copy()
        for i in range(1, len(atr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        rows.append({"ticker": ticker, "atr14": float(atr[-1]), "close": float(cl[-1])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Signal enrichment: add entry / stop / target columns
# ---------------------------------------------------------------------------
def enrich_signals(
    signals: pd.DataFrame,
    price_df: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Add ATR-based stop/target levels and metadata to the signal frame.

    Parameters
    ----------
    signals  : output of predict_latest — must have [ticker, signal] + score column
    price_df : raw OHLCV frame with high / low / close in rupee terms
    cfg      : Config object
    """
    if signals.empty:
        return signals

    atr_df = _compute_raw_atr14(price_df)
    if atr_df.empty:
        return signals

    signals = signals.merge(atr_df, on="ticker", how="left")

    # Fallback: 1.5 % of price when ATR unavailable
    signals["close"] = signals["close"].fillna(0)
    signals["atr14"] = signals["atr14"].fillna(signals["close"] * 0.015)
    signals = signals.rename(columns={"close": "entry_price"})

    # Dynamic RR (docs/dynamic-horizon-rr-plan.md Phase 3) — when the signal
    # frame carries the predicted quantile surface at h* (q10_star/q90_star,
    # from predict_latest_surface) and the feature is enabled, derive stop/
    # target from the model's own up/down skew instead of a fixed multiple.
    has_surface = (
        getattr(cfg, "dynamic_horizon_enabled", False)
        and {"q10_star", "q90_star"}.issubset(signals.columns)
    )
    if has_surface:
        k = getattr(cfg, "rr_k", 1.0)
        stop_lo, stop_hi = getattr(cfg, "stop_atr_clamp", (0.8, 3.0))
        tgt_lo, tgt_hi = getattr(cfg, "target_atr_clamp", (1.0, 6.0))

        atr_safe = signals["atr14"].replace(0, np.nan)
        q10_abs = signals["q10_star"].abs()
        q90 = signals["q90_star"].clip(lower=0)

        # risk_reward (diagnostic, pre-clamp) — the asymmetry the model
        # actually predicted, before the executed multiples are clamped.
        signals["risk_reward"] = (q90 / q10_abs.replace(0, np.nan)).round(2)

        stop_mult_row = (k * q10_abs * signals["entry_price"] / atr_safe).clip(stop_lo, stop_hi).fillna(stop_lo)
        target_mult_row = (k * q90 * signals["entry_price"] / atr_safe).clip(tgt_lo, tgt_hi).fillna(tgt_hi)

        signals["stop_loss"] = (signals["entry_price"] - stop_mult_row * signals["atr14"]).round(2)
        signals["target_price"] = (signals["entry_price"] + target_mult_row * signals["atr14"]).round(2)
        signals["horizon_days"] = signals["horizon_days"].fillna(cfg.horizon).astype(int) \
            if "horizon_days" in signals.columns else cfg.horizon
    else:
        # Legacy fixed-multiple path — unchanged. Use the dedicated signal
        # exit multipliers, NOT the label barriers. The label barriers
        # (barrier_*_mult) define the classification target; the paper/live
        # stop & target are a separate risk decision. Default 1.5×ATR stop /
        # 3.0×ATR target gives a ~2:1 payoff instead of the old symmetric 1:1.
        stop_mult   = getattr(cfg, "signal_stop_atr_mult", 1.5)
        target_mult = getattr(cfg, "signal_target_atr_mult", 3.0)

        signals["stop_loss"]    = (signals["entry_price"] - stop_mult   * signals["atr14"]).round(2)
        signals["target_price"] = (signals["entry_price"] + target_mult * signals["atr14"]).round(2)

        risk   = (signals["entry_price"] - signals["stop_loss"]).replace(0, np.nan)
        reward = signals["target_price"] - signals["entry_price"]
        signals["risk_reward"]  = (reward / risk).round(2)
        signals["horizon_days"] = cfg.horizon

    signals["atr14"]        = signals["atr14"].round(2)
    signals["entry_price"]  = signals["entry_price"].round(2)

    latest_date = price_df["date"].max()
    signals["signal_date"] = str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date)

    return signals


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_signals(signals: pd.DataFrame, output_dir: str = "outputs") -> Path:
    """Persist signals as a dated JSON file and a rolling CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    date_str  = datetime.now().strftime("%Y%m%d")
    json_path = out / f"signals_{date_str}.json"
    csv_path  = out / "signals_latest.csv"

    long_count = int((signals.get("signal", pd.Series(dtype=str)) == "LONG").sum())
    payload = {
        "generated_at": datetime.now().isoformat(),
        "signal_count": len(signals),
        "long_count":   long_count,
        "signals": signals.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    signals.to_csv(csv_path, index=False)

    logger.info("Signals saved → %s  |  CSV → %s", json_path.name, csv_path.name)
    return json_path


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------
def print_signal_table(signals: pd.DataFrame, title: str = "SIGNALS",
                        max_rows: int = 25) -> None:
    """Print a clean, aligned signal table.

    Only the first ``max_rows`` (highest-ranked) rows are printed to keep the
    console/notebook readable when the full scored universe (~200 names) is
    surfaced; the footer still reports the true LONG/NEUTRAL totals.
    """
    w_total = 90
    print(f"\n{'─' * w_total}")
    print(f"  {title}")
    print(f"{'─' * w_total}")

    if signals.empty:
        print("  (no signals generated)")
        print(f"{'─' * w_total}")
        return

    col_order = [
        "ticker", "signal", "prob_up",
        "entry_price", "stop_loss", "target_price",
        "risk_reward", "atr14", "horizon_days",
    ]
    cols = [c for c in col_order if c in signals.columns]
    widths = {
        "ticker": 18, "signal": 9, "prob_up": 10,
        "entry_price": 13, "stop_loss": 12, "target_price": 13,
        "risk_reward": 12, "atr14": 9, "horizon_days": 12,
    }

    header = "".join(c.rjust(widths.get(c, 12)) for c in cols)
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    for _, row in signals.head(max_rows).iterrows():
        line = ""
        for c in cols:
            v = row[c]
            w = widths.get(c, 12)
            if isinstance(v, float):
                fmt = f"{v:.4f}" if c == "prob_up" else f"{v:,.2f}"
                line += fmt.rjust(w)
            else:
                line += str(v).rjust(w)
        print(line)

    if len(signals) > max_rows:
        print(f"  … {len(signals) - max_rows} more rows (showing top {max_rows})".rjust(w_total))

    print(sep)
    long_n = int((signals.get("signal", pd.Series(dtype=str)) == "LONG").sum())
    print(f"  {long_n} LONG  |  {len(signals) - long_n} NEUTRAL  |  horizon = {signals['horizon_days'].iloc[0] if 'horizon_days' in signals.columns else '?'} days")
    print(f"{'─' * w_total}")
