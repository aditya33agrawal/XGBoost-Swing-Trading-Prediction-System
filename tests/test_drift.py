"""Drift monitor sanity checks — PSI/KS, CUSUM, concept + calibration drift."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    population_stability_index, ks_statistic, cusum_drift,
    concept_drift_from_outcomes, calibration_drift,
    feature_drift_report, build_drift_report, write_drift_report,
)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert population_stability_index(x, x.copy()) < 0.01


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(3, 1, 2000)          # mean shifted by 3 sigma
    assert population_stability_index(ref, cur) > 0.25
    assert ks_statistic(ref, cur) > 0.5


def test_cusum_alarms_on_error_jump():
    calm = np.full(50, 0.3)
    assert not cusum_drift(calm)["alarm"]
    jumpy = np.concatenate([np.full(25, 0.2), np.full(25, 0.9)])
    assert cusum_drift(jumpy, threshold=3.0)["alarm"]


def test_feature_drift_report_flags():
    rng = np.random.default_rng(2)
    n = 1000
    ref = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    cur = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(4, 1, n)})
    rep = feature_drift_report(ref, cur, ["f1", "f2"])
    flags = dict(zip(rep["feature"], rep["flag"]))
    assert flags["f2"] == "RETRAIN"
    assert flags["f1"] == "ok"


def test_concept_and_calibration_drift_handle_empty():
    assert concept_drift_from_outcomes(pd.DataFrame())["n"] == 0
    assert np.isnan(calibration_drift(pd.DataFrame())["ece"])


def test_concept_drift_computes_live_ic():
    rng = np.random.default_rng(3)
    prob = rng.uniform(0, 1, 200)
    ret = (prob - 0.5) * 0.1 + rng.normal(0, 0.01, 200)   # genuinely predictive
    df = pd.DataFrame({"prob_up": prob, "actual_fwd_ret": ret,
                       "is_correct": (ret > 0) == (prob > 0.5)})
    out = concept_drift_from_outcomes(df, backtest_ic=0.3)
    assert out["live_ic"] > 0.5
    assert out["ic_gap"] is not None


def test_write_drift_report(tmp_path):
    rep = build_drift_report(
        feature_drift=pd.DataFrame([{"feature": "f2", "psi": 0.4, "ks": 0.6, "flag": "RETRAIN"}]),
        concept={"live_ic": 0.01, "cusum": {"alarm": False}},
        calibration={"ece": 0.05, "n": 100},
    )
    assert rep["retrain_recommended"] is True
    jp, hp = write_drift_report(rep, str(tmp_path), tag="testrun")
    assert (tmp_path / "drift_testrun.json").exists()
    assert (tmp_path / "drift_testrun.html").exists()
