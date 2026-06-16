"""Sample weighting for overlapping labels (plan §2, §8.4).

Two weighting schemes composed multiplicatively:
  1. Uniqueness weight  — down-weight samples whose label windows overlap many
     other label windows (López de Prado ch. 4).
  2. Time-decay weight  — exponential decay so recent regimes count more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Uniqueness weights (concurrency-based)
# ---------------------------------------------------------------------------
def label_uniqueness_weights(
    df: pd.DataFrame,
    h: int,
    label_col: str = "target",
) -> pd.Series:
    """Compute per-sample uniqueness weight in [0, 1].

    For each (ticker, date) we count how many other labels in the same ticker
    have a forward window that overlaps with this one.  More overlap → lower
    uniqueness → lower weight.

    Algorithm: O(h × n) vectorised, not O(n²).
    Two label windows at positions i and j overlap iff |i-j| < h.
    The overlap fraction is (h - |i-j|) / h.
    We accumulate concurrency by sliding over offsets d = 1 … h-1.

    Returns a Series aligned to df.index.
    """
    weights = pd.Series(1.0, index=df.index)

    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").dropna(subset=[label_col])
        n = len(grp)
        if n == 0:
            continue

        # concurrency[i] = 1 + sum_{d=1}^{h-1} (h-d)/h for each valid neighbour
        # at distance d in either direction.
        concurrency = np.ones(n)
        for d in range(1, h):
            frac = (h - d) / h
            concurrency[d:] += frac    # neighbours to the left  (i-d exists)
            concurrency[:n - d] += frac  # neighbours to the right (i+d exists)

        uniqueness = 1.0 / concurrency
        uniqueness = uniqueness / uniqueness.mean()
        weights.loc[grp.index] = uniqueness

    return weights.clip(lower=0.01)


# ---------------------------------------------------------------------------
# 2. Time-decay weights
# ---------------------------------------------------------------------------
def time_decay_weights(
    df: pd.DataFrame,
    halflife_days: int = 252,
) -> pd.Series:
    """Exponential time-decay: most recent row has weight 1.0.

    Returns a Series aligned to df.index.
    """
    dates = pd.to_datetime(df["date"])
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.astype(float)
    decay = np.exp(-np.log(2) * age_days / halflife_days)
    return pd.Series(decay.values, index=df.index)


# ---------------------------------------------------------------------------
# Combined weight
# ---------------------------------------------------------------------------
def sample_weights(
    df: pd.DataFrame,
    h: int,
    halflife_days: int = 252,
    label_col: str = "target",
) -> np.ndarray:
    """Composed sample_weight array for XGBoost fit().

    w = uniqueness × time_decay  (normalised so mean = 1)
    """
    uniq = label_uniqueness_weights(df, h, label_col)
    decay = time_decay_weights(df, halflife_days)
    combined = uniq * decay
    combined = combined / (combined.mean() + 1e-9)
    return combined.to_numpy(dtype=float)
