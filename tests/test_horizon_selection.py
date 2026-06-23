"""Tests for src.models.horizon_selection (docs/dynamic-horizon-rr-plan.md Phase 2)."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.horizon_selection import select_horizon, diagnose_horizon_distribution


def _surface_one_row(q10_by_h: dict, q50_by_h: dict, q90_by_h: dict, taus=(0.1, 0.5, 0.9)):
    """Build a 1-row surface dict {(h, tau): array([value])}."""
    surface = {}
    for h in q10_by_h:
        surface[(h, taus[0])] = np.array([q10_by_h[h]])
        surface[(h, taus[1])] = np.array([q50_by_h[h]])
        surface[(h, taus[2])] = np.array([q90_by_h[h]])
    return surface


def test_select_horizon_picks_the_peaking_edge():
    """Worked example from the plan's §1 table: edge peaks at h=21 then decays."""
    grid = [5, 21, 63]
    q10 = {5: -0.021, 21: -0.039, 63: -0.060}
    q50 = {5: 0.004, 21: 0.018, 63: 0.031}
    q90 = {5: 0.030, 21: 0.065, 63: 0.100}
    surface = _surface_one_row(q10, q50, q90)

    h_star, score_star, q_at_hstar = select_horizon(surface, grid, lambda_t=0.0005, h_max=63)

    assert h_star[0] == 21
    assert np.isclose(q_at_hstar["q50"][0], q50[21])
    assert np.isclose(q_at_hstar["q10"][0], q10[21])
    assert np.isclose(q_at_hstar["q90"][0], q90[21])


def test_select_horizon_respects_h_max_cap():
    grid = [5, 21, 63]
    # The best raw score is at h=63, but cap it out.
    q10 = {5: -0.01, 21: -0.01, 63: -0.01}
    q50 = {5: 0.001, 21: 0.001, 63: 0.05}
    q90 = {5: 0.01, 21: 0.01, 63: 0.01}
    surface = _surface_one_row(q10, q50, q90)

    h_star, _, _ = select_horizon(surface, grid, lambda_t=0.0, h_max=21)
    assert h_star[0] in (5, 21)
    assert h_star[0] != 63


def test_select_horizon_degenerate_all_equal_picks_cheapest():
    """All cells identical (no real signal) → ties go to the smallest/cheapest
    horizon since the time-decay penalty strictly favours it."""
    grid = [5, 21, 63]
    q10 = {h: -0.01 for h in grid}
    q50 = {h: 0.005 for h in grid}
    q90 = {h: 0.01 for h in grid}
    surface = _surface_one_row(q10, q50, q90)

    h_star, score_star, _ = select_horizon(surface, grid, lambda_t=0.0005, h_max=63)
    assert h_star[0] == 5
    assert np.isfinite(score_star[0])


def test_select_horizon_does_not_crash_on_inverted_spread():
    """q90 <= q10 (inverted/degenerate quantile prediction) must not raise or
    propagate inf/NaN uncontrolled — falls back to the cheapest horizon."""
    grid = [5, 21]
    surface = _surface_one_row(
        {5: 0.01, 21: -0.01},     # h=5 has q10 > q90 below (inverted)
        {5: 0.0, 21: 0.0},
        {5: 0.005, 21: 0.02},
    )
    h_star, score_star, q_at_hstar = select_horizon(surface, grid, lambda_t=0.0005, h_max=21)
    assert h_star[0] in grid
    assert not np.isnan(h_star[0])


def test_select_horizon_vectorised_over_multiple_rows():
    grid = [5, 21]
    n = 4
    surface = {
        (5, 0.1): np.array([-0.01] * n),
        (5, 0.5): np.array([0.001] * n),
        (5, 0.9): np.array([0.01] * n),
        (21, 0.1): np.array([-0.02] * n),
        (21, 0.5): np.array([0.02] * n),   # h=21 clearly better for every row
        (21, 0.9): np.array([0.05] * n),
    }
    h_star, _, _ = select_horizon(surface, grid, lambda_t=0.0, h_max=21)
    assert h_star.shape == (n,)
    assert np.all(h_star == 21)


def test_select_horizon_missing_h_max_raises():
    surface = _surface_one_row({5: -0.01}, {5: 0.0}, {5: 0.01})
    with pytest.raises(ValueError):
        select_horizon(surface, [5], lambda_t=0.0, h_max=1)


def test_diagnose_horizon_distribution_flags_collapse():
    grid = [5, 21, 63]
    h_star = np.array([21] * 99 + [5])  # 99% one horizon
    diag = diagnose_horizon_distribution(h_star, grid)
    assert diag["collapsed"] is True
    assert diag["looks_uniform"] is False
    assert diag["counts"][21] == 99


def test_diagnose_horizon_distribution_flags_uniform_noise():
    grid = [5, 21, 63]
    rng = np.random.default_rng(0)
    h_star = rng.choice(grid, size=3000)
    diag = diagnose_horizon_distribution(h_star, grid)
    assert diag["collapsed"] is False
    assert diag["looks_uniform"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
