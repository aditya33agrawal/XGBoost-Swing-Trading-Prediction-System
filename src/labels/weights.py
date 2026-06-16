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

    Returns a Series aligned to df.index.
    """
    df = df.copy()
    weights = pd.Series(1.0, index=df.index)

    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").dropna(subset=[label_col])
        dates = pd.to_datetime(grp["date"]).values
        n = len(dates)
        if n == 0:
            continue

        # Concurrency: for each sample i, count samples j whose window
        # [j, j+h) overlaps [i, i+h)
        concurrency = np.ones(n)
        for i in range(n):
            start_i, end_i = i, i + h
            for j in range(n):
                start_j, end_j = j, j + h
                overlap = min(end_i, end_j) - max(start_i, start_j)
                if overlap > 0 and j != i:
                    concurrency[i] += overlap / h

        uniqueness = 1.0 / concurrency
        uniqueness = uniqueness / uniqueness.mean()  # normalise to mean=1
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
