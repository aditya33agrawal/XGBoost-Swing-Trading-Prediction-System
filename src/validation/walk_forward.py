"""Purged walk-forward cross-validation (plan §7).

PurgedWalkForward.split(dates) yields (train_idx, test_idx) pairs where:
  - All test dates are strictly after all train dates.
  - A purge gap of `embargo` bars is removed between train and test so that
    overlapping forward labels cannot straddle the boundary.
  - Splits are by unique date, not by row — all tickers on a given date go to
    the same fold (prevents cross-ticker leakage through shared dates).

Usage:
    splitter = PurgedWalkForward(n_splits=8, embargo=5, label_h=5)
    for train_idx, test_idx in splitter.split(df):
        X_train, y_train = ...
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedWalkForward:
    """Time-series walk-forward splitter with purge + embargo."""

    def __init__(
        self,
        n_splits: int = 8,
        embargo: int = 5,
        label_h: int = 5,
        min_train_size: int = 504,
    ):
        self.n_splits = n_splits
        self.embargo = embargo
        self.label_h = label_h
        self.min_train_size = min_train_size

    def split(self, df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) into `df`.

        df must have a 'date' column.  Splits are date-based so all tickers on
        a date are in the same fold.
        """
        dates = np.sort(df["date"].unique())
        n_dates = len(dates)

        # Total purge gap per boundary
        gap = self.label_h + self.embargo

        # Reserve the last portion for out-of-sample test
        # Split remaining dates into n_splits folds
        usable = n_dates - gap
        if usable <= self.min_train_size:
            raise ValueError(
                f"Not enough dates ({n_dates}) for {self.n_splits} walk-forward "
                f"splits with min_train_size={self.min_train_size}"
            )

        test_size = max(1, (usable - self.min_train_size) // self.n_splits)
        splits = []
        for k in range(self.n_splits):
            test_start_idx = self.min_train_size + k * test_size
            test_end_idx = test_start_idx + test_size
            if test_end_idx > n_dates:
                break

            test_dates = set(dates[test_start_idx:test_end_idx])
            train_cutoff = dates[max(0, test_start_idx - gap)]

            train_dates = set(dates[: max(0, test_start_idx - gap)])

            train_mask = df["date"].isin(train_dates)
            test_mask = df["date"].isin(test_dates)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) < self.min_train_size or len(test_idx) == 0:
                continue
            splits.append((train_idx, test_idx))

        return splits

    def final_train_test_split(
        self, df: pd.DataFrame, test_fraction: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hold-out the most-recent test_fraction of dates as the final OOS test.

        This test set is never touched during hyperparameter search.
        """
        dates = np.sort(df["date"].unique())
        split_point = int(len(dates) * (1 - test_fraction))
        gap = self.label_h + self.embargo

        train_cutoff = dates[max(0, split_point - gap)]
        test_start = dates[split_point]

        train_mask = df["date"] <= train_cutoff
        test_mask = df["date"] >= test_start

        return np.where(train_mask)[0], np.where(test_mask)[0]
