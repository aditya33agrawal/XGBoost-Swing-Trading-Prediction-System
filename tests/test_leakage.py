"""Leakage test suite — CI gate (swing-bot-plan §12.1, build-step 3).

Fails the build if the classic look-ahead leaks reappear:
  (a) the last-`h`-rows label drop is missing,
  (b) triple-barrier labels peek beyond the horizon window,
  (c) walk-forward folds have train dates >= test dates (no purge/embargo),
  (d) forward returns are correctly future-shifted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels.targets import (
    triple_barrier_labels, forward_log_return, forward_log_return_grid, add_labels,
)
from src.validation.walk_forward import PurgedWalkForward


def _synthetic_panel(n_days: int = 120, tickers=("AAA", "BBB"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames = []
    for t in tickers:
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
        high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        frames.append(pd.DataFrame({
            "ticker": t, "date": dates,
            "open": close, "high": high, "low": low, "close": close,
            "volume": rng.integers(1e5, 1e6, n_days),
        }))
    return pd.concat(frames, ignore_index=True)


# (a) + (d) last-h-rows must be unlabeled --------------------------------------
def test_triple_barrier_drops_last_h_rows():
    h = 5
    df = _synthetic_panel()
    out = triple_barrier_labels(df, h)
    for ticker, grp in out.groupby("ticker"):
        grp = grp.sort_values("date")
        tail = grp["tb_label"].to_numpy()[-h:]
        assert np.all(np.isnan(tail)), f"{ticker}: last {h} labels must be NaN (no future bars)"


def test_forward_return_is_future_shift():
    h = 3
    df = _synthetic_panel(tickers=("AAA",))
    out = forward_log_return(df, h).sort_values("date").reset_index(drop=True)
    c = out["close"].to_numpy()
    expected = np.log(c[h] / c[0])
    assert np.isclose(out["fwd_ret"].iloc[0], expected), "fwd_ret must use close[t+h]/close[t]"
    assert np.all(np.isnan(out["fwd_ret"].to_numpy()[-h:])), "last h forward returns must be NaN"


# (b) labels must not depend on bars beyond the horizon ------------------------
def test_triple_barrier_ignores_bars_beyond_horizon():
    """Mutating prices strictly after the horizon window must not change a label."""
    h = 5
    df = _synthetic_panel(tickers=("AAA",), n_days=60).sort_values("date").reset_index(drop=True)
    base = triple_barrier_labels(df, h)
    i = 10  # label at row i depends only on rows (i, i+h]
    perturbed = df.copy()
    far = i + h + 3  # strictly beyond the window
    perturbed.loc[far:, ["high", "low", "close"]] *= 5.0
    after = triple_barrier_labels(perturbed, h)
    assert base["tb_label"].iloc[i] == after["tb_label"].iloc[i], \
        "label changed when only post-horizon bars moved → look-ahead leak"


# (c) walk-forward purge + embargo --------------------------------------------
def test_walk_forward_train_strictly_before_test_with_gap():
    h, embargo = 5, 5
    df = _synthetic_panel(n_days=200)
    wf = PurgedWalkForward(n_splits=4, embargo=embargo, label_h=h, min_train_size=20)
    splits = wf.split(df)
    assert splits, "expected at least one fold"
    for train_idx, test_idx in splits:
        train_dates = df.iloc[train_idx]["date"]
        test_dates = df.iloc[test_idx]["date"]
        max_train, min_test = train_dates.max(), test_dates.min()
        assert max_train < min_test, "train date >= test date — temporal leak"
        gap_days = (min_test - max_train).days
        assert gap_days >= (h + embargo), \
            f"purge gap {gap_days}d < required {h + embargo}d (calendar approx)"


# (dynamic-horizon-rr-plan.md Phase 1) multi-horizon labels -------------------
def test_forward_log_return_grid_matches_single_horizon():
    grid = [5, 21, 63]
    df = _synthetic_panel(n_days=150, tickers=("AAA", "BBB"))
    out = forward_log_return_grid(df, grid)
    for h in grid:
        expected = forward_log_return(df, h).sort_values(["ticker", "date"]).reset_index(drop=True)
        actual = out.sort_values(["ticker", "date"]).reset_index(drop=True)
        pd.testing.assert_series_equal(
            actual[f"fwd_ret_{h}"], expected["fwd_ret"], check_names=False,
            check_exact=False, atol=1e-12,
        )


def test_triple_barrier_first_passage_time_bounded():
    h = 10
    df = _synthetic_panel(n_days=80, tickers=("AAA",))
    out = triple_barrier_labels(df, h, record_first_passage=True)
    out = out.sort_values("date").reset_index(drop=True)

    labeled = out.dropna(subset=["tb_label"])
    fpt = labeled["tb_first_passage_time"].to_numpy()
    assert np.all(fpt >= 1) and np.all(fpt <= h), "first-passage time must be in [1, h]"

    tail = out["tb_first_passage_time"].to_numpy()[-h:]
    assert np.all(np.isnan(tail)), "last h rows must have no first-passage time (unlabeled)"

    # tb_barrier_hit must agree with tb_label's sign/value.
    sign_map = {1: "up", -1: "down", 0: "vertical"}
    for _, row in labeled.iterrows():
        assert row["tb_barrier_hit"] == sign_map[int(row["tb_label"])]


def test_purge_embargo_uses_max_grid():
    """Walk-forward purge for the dynamic-horizon path must use max(horizon_grid),
    not the legacy cfg.horizon — the A1 leakage-budget cost the plan calls out."""
    grid = [5, 21, 63]
    h_max, embargo = max(grid), 5
    df = _synthetic_panel(n_days=400)
    wf = PurgedWalkForward(n_splits=3, embargo=embargo, label_h=h_max, min_train_size=100)
    splits = wf.split(df)
    assert splits, "expected at least one fold at the widened (max-grid) purge gap"
    for train_idx, test_idx in splits:
        train_dates = df.iloc[train_idx]["date"]
        test_dates = df.iloc[test_idx]["date"]
        gap_days = (test_dates.min() - train_dates.max()).days
        assert gap_days >= (h_max + embargo), \
            f"purge gap {gap_days}d < required {h_max + embargo}d (max(horizon_grid)+embargo)"


def test_add_labels_supports_both_targets():
    df = _synthetic_panel()
    for lt in ("triple_barrier", "fwd_ret"):
        out = add_labels(df, h=5, label_type=lt)
        assert "target" in out.columns
        assert out["target"].notna().any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
