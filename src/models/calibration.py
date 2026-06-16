"""Probability calibration (plan §8.5).

Tree models produce poorly-calibrated probabilities.
We fit isotonic regression on a time-ordered holdout fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression


class TimeOrderedCalibrator:
    """Isotonic calibration fitted on a chronologically-ordered holdout."""

    def __init__(self):
        self._iso: IsotonicRegression | None = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Fit on time-ordered calibration set (after training cutoff)."""
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(probs, labels)

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if self._iso is None:
            return probs
        return self._iso.predict(probs)

    def calibration_error(
        self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> float:
        cal_probs = self.predict_proba(probs)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(cal_probs)
        for i in range(n_bins):
            mask = (cal_probs >= bins[i]) & (cal_probs < bins[i + 1])
            if mask.sum() == 0:
                continue
            ece += mask.sum() / n * abs(cal_probs[mask].mean() - labels[mask].mean())
        return float(ece)
