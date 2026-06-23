"""Tests for the dynamic stop/target/RR branch in
src.trading.signals.enrich_signals (docs/dynamic-horizon-rr-plan.md Phase 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.trading.signals import enrich_signals


def _price_df():
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for t in ("AAA", "BBB"):
        close = np.full(30, 100.0)
        rows.append(pd.DataFrame({
            "ticker": t, "date": dates,
            "high": close * 1.01, "low": close * 0.99, "close": close,
        }))
    return pd.concat(rows, ignore_index=True)


def test_dynamic_rr_derives_stop_target_from_surface():
    cfg = Config(dynamic_horizon_enabled=True, rr_k=1.0)
    signals = pd.DataFrame({
        "ticker": ["AAA"],
        "signal": ["LONG"],
        "q10_star": [-0.02],   # 2% downside
        "q90_star": [0.05],    # 5% upside
        "horizon_days": [21],
    })
    out = enrich_signals(signals, _price_df(), cfg)

    entry = out["entry_price"].iloc[0]
    atr = out["atr14"].iloc[0]
    stop = out["stop_loss"].iloc[0]
    target = out["target_price"].iloc[0]

    assert stop < entry < target
    # risk_reward should reflect the predicted skew (q90/|q10| = 0.05/0.02 = 2.5)
    assert np.isclose(out["risk_reward"].iloc[0], 2.5, atol=0.05)
    assert out["horizon_days"].iloc[0] == 21


def test_dynamic_rr_clamps_extreme_quantiles():
    """q10≈0 must not divide-by-near-zero into an absurd stop multiple — it
    should clamp to the configured floor instead."""
    cfg = Config(dynamic_horizon_enabled=True, rr_k=1.0, stop_atr_clamp=(0.8, 3.0), target_atr_clamp=(1.0, 6.0))
    signals = pd.DataFrame({
        "ticker": ["AAA"],
        "signal": ["LONG"],
        "q10_star": [-1e-9],   # ~0 downside
        "q90_star": [50.0],    # absurdly large upside
        "horizon_days": [63],
    })
    out = enrich_signals(signals, _price_df(), cfg)

    entry = out["entry_price"].iloc[0]
    atr = out["atr14"].iloc[0]
    stop_lo_mult = cfg.stop_atr_clamp[0]
    tgt_hi_mult = cfg.target_atr_clamp[1]

    expected_stop_floor = round(entry - stop_lo_mult * atr, 2)
    expected_target_ceiling = round(entry + tgt_hi_mult * atr, 2)

    assert out["stop_loss"].iloc[0] == expected_stop_floor
    assert out["target_price"].iloc[0] == expected_target_ceiling


def test_falls_back_to_fixed_multiples_without_surface_columns():
    """No q10_star/q90_star columns (or flag off) → legacy fixed-RR path,
    byte-identical to today's behaviour."""
    cfg = Config(dynamic_horizon_enabled=True)  # flag on, but no surface columns
    signals = pd.DataFrame({"ticker": ["AAA"], "signal": ["LONG"]})
    out = enrich_signals(signals, _price_df(), cfg)

    entry = out["entry_price"].iloc[0]
    atr = out["atr14"].iloc[0]
    assert np.isclose(out["stop_loss"].iloc[0], round(entry - cfg.signal_stop_atr_mult * atr, 2))
    assert np.isclose(out["target_price"].iloc[0], round(entry + cfg.signal_target_atr_mult * atr, 2))
    assert out["horizon_days"].iloc[0] == cfg.horizon


def test_legacy_path_unchanged_when_flag_off():
    cfg = Config(dynamic_horizon_enabled=False)
    signals = pd.DataFrame({
        "ticker": ["AAA"], "signal": ["LONG"],
        "q10_star": [-0.02], "q90_star": [0.05], "horizon_days": [21],
    })
    out = enrich_signals(signals, _price_df(), cfg)
    entry = out["entry_price"].iloc[0]
    atr = out["atr14"].iloc[0]
    # Flag off → must ignore the surface columns entirely, use fixed multiples.
    assert np.isclose(out["stop_loss"].iloc[0], round(entry - cfg.signal_stop_atr_mult * atr, 2))
    assert np.isclose(out["target_price"].iloc[0], round(entry + cfg.signal_target_atr_mult * atr, 2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
