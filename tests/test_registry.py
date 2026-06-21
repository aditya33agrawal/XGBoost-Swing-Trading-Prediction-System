"""Registry bundle round-trip + champion/challenger gate + rollback."""
from __future__ import annotations

import numpy as np
import pytest

from src.registry.bundle import (
    save_bundle, load_bundle, set_prod_pointer, load_prod_bundle,
    rollback_prod, list_bundles, prune_old_bundles, features_hash,
)
from src.registry.promotion import evaluate_promotion


def _toy_model():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)); y = (X[:, 0] + rng.normal(0, 0.1, 200) > 0).astype(int)
    m = xgb.XGBClassifier(n_estimators=10, max_depth=2)
    m.fit(X, y)
    return m


def test_bundle_roundtrip(tmp_path):
    feats = ["ret_1", "rsi_14", "atr_norm", "vol_z20"]
    model = _toy_model()
    bundle_dir = save_bundle(
        str(tmp_path), model=model, calibrator=None, features=feats,
        hyperparams={"max_depth": 2}, metrics={"oof_ic": 0.02, "sharpe_net": 0.8},
        train_window={"start": "2015-01-01", "end": "2025-12-31"},
        horizon_days=5, embargo_days=5, model_version="20260621",
    )
    loaded = load_bundle(bundle_dir)
    assert loaded["features"] == feats
    assert loaded["manifest"]["features_hash"] == features_hash(feats)
    assert loaded["manifest"]["model_version"] == "20260621"
    assert loaded["metrics"]["oof_ic"] == 0.02
    # native model loaded and usable
    pred = loaded["model"].predict_proba(np.zeros((1, 4)))
    assert pred.shape == (1, 2)


def test_prod_pointer_and_rollback(tmp_path):
    model = _toy_model()
    common = dict(calibrator=None, features=["a", "b", "c", "d"], hyperparams={},
                  metrics={}, train_window={"start": "x", "end": "y"},
                  horizon_days=5, embargo_days=5)
    b1 = save_bundle(str(tmp_path), model=model, model_version="20260601", **common)
    b2 = save_bundle(str(tmp_path), model=model, model_version="20260608", **common)

    set_prod_pointer(str(tmp_path), b1)
    set_prod_pointer(str(tmp_path), b2)
    assert load_prod_bundle(str(tmp_path))["bundle_dir"].endswith("model_20260608")

    rolled = rollback_prod(str(tmp_path))
    assert rolled.endswith("model_20260601")
    assert load_prod_bundle(str(tmp_path))["bundle_dir"].endswith("model_20260601")


def test_prune_keeps_prod_and_recent(tmp_path):
    model = _toy_model()
    common = dict(calibrator=None, features=["a", "b"], hyperparams={}, metrics={},
                  train_window={}, horizon_days=5, embargo_days=5)
    dirs = [save_bundle(str(tmp_path), model=model, model_version=f"2026060{i}", **common)
            for i in range(1, 6)]
    set_prod_pointer(str(tmp_path), dirs[0])     # oldest is live
    prune_old_bundles(str(tmp_path), keep=2)
    remaining = list_bundles(str(tmp_path))
    assert dirs[0] in remaining, "prod bundle must never be pruned"
    assert len(remaining) <= 3


# --- promotion gate ----------------------------------------------------------
def test_first_model_promotes_unless_drift():
    assert evaluate_promotion(challenger={"ic": 0.01}, champion=None)["promote"]
    assert not evaluate_promotion(challenger={"ic": 0.01}, champion=None, drift_alarm=True)["promote"]


def test_sharpe_margin_gate():
    win = evaluate_promotion(
        challenger={"sharpe_net": 1.2, "calib_err": 0.05},
        champion={"sharpe_net": 1.0})
    assert win["promote"]
    lose = evaluate_promotion(
        challenger={"sharpe_net": 1.05}, champion={"sharpe_net": 1.0})
    assert not lose["promote"]  # +0.05 < 0.10 margin


def test_calibration_and_drift_block():
    assert not evaluate_promotion(
        challenger={"sharpe_net": 2.0, "calib_err": 0.5},
        champion={"sharpe_net": 1.0})["promote"]
    assert not evaluate_promotion(
        challenger={"sharpe_net": 2.0}, champion={"sharpe_net": 1.0},
        drift_alarm=True)["promote"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
