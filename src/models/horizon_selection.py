"""Horizon selection from a predicted quantile surface (docs/dynamic-horizon-rr-plan.md §1, §2).

Given per-row quantile predictions at every (horizon, tau) cell, pick the
horizon h* that maximises predicted risk-adjusted edge net of a time-decay
penalty:

    h* = argmax_h  [ q50_h - lambda_t * h ] / (q90_h - q10_h)

This is "max predicted Sharpe per unit of predicted spread, penalised for
time" — the core of Design A (the recommended primary design in the plan).
"""
from __future__ import annotations

import numpy as np


def select_horizon(
    surface: dict[tuple[int, float], np.ndarray],
    grid: list[int],
    taus: tuple[float, float, float] = (0.1, 0.5, 0.9),
    lambda_t: float = 0.0005,
    h_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Pick h* per row from a predicted quantile surface.

    Parameters
    ----------
    surface : {(h, tau): array of shape (n_rows,)} — predicted quantiles,
              one array per (horizon, tau) cell. Every h in `grid` must have
              all three of `taus` present.
    grid    : horizons to consider, e.g. [5, 21, 63].
    taus    : (q_low, q_mid, q_high) — defaults to (0.1, 0.5, 0.9).
    lambda_t: per-day penalty subtracted from the median edge.
    h_max   : hard cap — horizons > h_max are excluded from the argmax.

    Returns
    -------
    h_star       : (n_rows,) int array — chosen horizon per row.
    score_star    : (n_rows,) float array — the score that won at h*.
    q_at_hstar    : {"q10": arr, "q50": arr, "q90": arr} — quantiles gathered
                    at the chosen horizon, ready for Phase 3 RR derivation.
    """
    q_lo, q_mid, q_hi = taus
    usable_grid = [h for h in grid if h_max is None or h <= h_max]
    if not usable_grid:
        raise ValueError(f"no horizons in grid {grid} are <= h_max={h_max}")

    n_rows = next(iter(surface.values())).shape[0]
    n_h = len(usable_grid)

    scores = np.full((n_h, n_rows), -np.inf)
    q10s = np.full((n_h, n_rows), np.nan)
    q50s = np.full((n_h, n_rows), np.nan)
    q90s = np.full((n_h, n_rows), np.nan)

    for j, h in enumerate(usable_grid):
        q10 = surface[(h, q_lo)]
        q50 = surface[(h, q_mid)]
        q90 = surface[(h, q_hi)]
        spread = q90 - q10
        # Guard against a degenerate/inverted spread (q90<=q10): treat as a
        # non-informative cell rather than dividing by zero/negative.
        safe_spread = np.where(spread > 1e-9, spread, np.nan)
        score = (q50 - lambda_t * h) / safe_spread
        scores[j] = np.where(np.isnan(score), -np.inf, score)
        q10s[j], q50s[j], q90s[j] = q10, q50, q90

    # If every cell is non-informative for a row, fall back to the cheapest
    # (smallest) horizon rather than propagating -inf/NaN downstream.
    all_bad = np.all(np.isneginf(scores), axis=0)
    best_j = np.argmax(scores, axis=0)
    best_j = np.where(all_bad, 0, best_j)

    h_star = np.array(usable_grid)[best_j]
    rows = np.arange(n_rows)
    score_star = scores[best_j, rows]
    score_star = np.where(all_bad, np.nan, score_star)

    q_at_hstar = {
        "q10": q10s[best_j, rows],
        "q50": q50s[best_j, rows],
        "q90": q90s[best_j, rows],
    }
    return h_star, score_star, q_at_hstar


def diagnose_horizon_distribution(h_star: np.ndarray, grid: list[int]) -> dict:
    """Histogram + degenerate-collapse check on a batch of chosen horizons.

    Flags:
      collapsed       — >=95% of rows picked the same horizon (grid/penalty
                         likely miscalibrated, plan §2 item 9).
      looks_uniform    — distribution is close to uniform over the grid
                         (within 10pp of 1/len(grid) each) — horizon signal
                         may not be real, just noise.
    """
    h_star = np.asarray(h_star)
    n = len(h_star)
    counts = {int(h): int(np.sum(h_star == h)) for h in grid}
    fractions = {h: c / n if n else 0.0 for h, c in counts.items()}

    collapsed = n > 0 and max(fractions.values()) >= 0.95
    uniform_frac = 1.0 / len(grid) if grid else 0.0
    looks_uniform = n > 0 and all(
        abs(f - uniform_frac) <= 0.10 for f in fractions.values()
    )

    return {
        "n": n,
        "counts": counts,
        "fractions": fractions,
        "collapsed": collapsed,
        "looks_uniform": looks_uniform,
    }
