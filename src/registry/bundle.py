"""Model registry — versioned bundles + atomic prod pointer.

Implements §7 of `docs/bot-implementaion-plan.md`. A model is never a single
pickle; it is a **bundle** directory so any prediction is fully reproducible:

    registry/bundles/model_<version>/
    ├── model.json          # xgboost booster — native save_model (version-safe)
    ├── calibrator.pkl      # isotonic calibrator (joblib)
    ├── features.json       # ordered feature list (+ definition hash)
    ├── manifest.json       # provenance: train window, hashes, hyperparams, git sha
    └── metrics.json        # OOS IC, hit-rate, calib error, net Sharpe, drawdown

The daily/VM loop reads ``registry/prod/manifest.json`` to learn which bundle to
load. ``set_prod_pointer`` writes that file atomically so a concurrent reader
never sees a half-written pointer.

Rules (see plan §7 / §14):
  - Use XGBoost native ``save_model('*.json')`` — NOT pickle (pickle breaks across
    library versions, a classic production failure). The xgboost version is pinned
    in the manifest.
  - Keep the last N bundles for instant rollback (``prune_old_bundles``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def features_hash(features: list[str]) -> str:
    """Stable fingerprint of the ordered feature list."""
    blob = "|".join(features).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _xgboost_version() -> str:
    try:
        import xgboost as xgb
        return xgb.__version__
    except Exception:
        return "unknown"


def _as_booster(model):
    """Return the native xgboost Booster from a sklearn wrapper or a Booster."""
    return model.get_booster() if hasattr(model, "get_booster") else model


class BundleModel:
    """Thin, version-robust predictor wrapping a native xgboost Booster.

    We persist the booster (not the sklearn wrapper) because XGBoost's sklearn
    ``save_model`` is brittle across versions (e.g. the ``_estimator_type`` bug in
    2.0.x) and a booster-only reload drops ``n_classes_`` needed by
    ``predict_proba``. ``inplace_predict`` sidesteps feature-name validation and
    returns P(class=1) directly for ``binary:logistic``.
    """

    def __init__(self, booster, features: list[str], task: str = "classification"):
        self.booster = booster
        self.features = features
        self.task = task

    def _matrix(self, X):
        import numpy as np
        if hasattr(X, "columns"):
            X = X[self.features]
        return np.ascontiguousarray(np.asarray(X, dtype="float32"))

    def predict(self, X):
        return self.booster.inplace_predict(self._matrix(X))

    def predict_proba(self, X):
        import numpy as np
        p = np.asarray(self.booster.inplace_predict(self._matrix(X))).reshape(-1)
        return np.column_stack([1.0 - p, p])


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON to a temp file then os.replace — readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def build_manifest(
    *,
    model_version: str,
    train_window: dict,
    embargo_days: int,
    horizon_days: int,
    features: list[str],
    hyperparams: dict,
    metrics: dict,
    label_type: str = "triple_barrier",
    universe_ref: str = "config/universe.json",
    promoted: bool = False,
    champion_replaced: str | None = None,
) -> dict:
    return {
        "model_version": model_version,
        "created_utc": _utc_now(),
        "train_window": train_window,            # {"start": ..., "end": ...}
        "embargo_days": embargo_days,
        "horizon_days": horizon_days,
        "label_type": label_type,
        "universe_asof": universe_ref,
        "features_hash": features_hash(features),
        "n_features": len(features),
        "hyperparams": hyperparams,
        "xgboost_version": _xgboost_version(),
        "code_git_sha": _git_sha(),
        "promoted": promoted,
        "champion_replaced": champion_replaced,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# save / load a bundle
# ---------------------------------------------------------------------------
def save_bundle(
    root: str,
    *,
    model,
    calibrator,
    features: list[str],
    hyperparams: dict,
    metrics: dict,
    train_window: dict,
    horizon_days: int,
    embargo_days: int,
    model_version: str | None = None,
    label_type: str = "triple_barrier",
    task: str = "classification",
    quantile_model=None,
) -> str:
    """Persist a model bundle under ``root/registry/bundles/model_<version>/``.

    ``model`` is an xgboost sklearn estimator (XGBClassifier/XGBRegressor) saved via
    native ``save_model`` to ``model.json`` — portable and version-robust.
    Returns the bundle directory path.
    """
    version = model_version or datetime.today().strftime("%Y%m%d")
    bundle_dir = Path(root) / "registry" / "bundles" / f"model_{version}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. booster — native JSON, never pickle (and never the brittle sklearn wrapper)
    _as_booster(model).save_model(str(bundle_dir / "model.json"))

    # 2. calibrator — small sklearn/isotonic object; joblib is fine here
    if calibrator is not None:
        try:
            import joblib
            joblib.dump(calibrator, bundle_dir / "calibrator.pkl")
        except Exception as exc:
            logger.warning("Could not save calibrator: %s", exc)

    # 3. optional sizing / quantile model (also native JSON)
    if quantile_model is not None:
        try:
            _as_booster(quantile_model).save_model(str(bundle_dir / "quantile_reg.json"))
        except Exception as exc:
            logger.warning("Could not save quantile model: %s", exc)

    # 4. features (ordered — order matters for inference)
    _atomic_write_json(bundle_dir / "features.json", {
        "features": features,
        "features_hash": features_hash(features),
    })

    # 5. metrics
    _atomic_write_json(bundle_dir / "metrics.json", metrics)

    # 6. manifest
    manifest = build_manifest(
        model_version=version,
        train_window=train_window,
        embargo_days=embargo_days,
        horizon_days=horizon_days,
        features=features,
        hyperparams=hyperparams,
        metrics=metrics,
        label_type=label_type,
    )
    manifest["task"] = task
    _atomic_write_json(bundle_dir / "manifest.json", manifest)

    logger.info("Saved model bundle → %s", bundle_dir)
    return str(bundle_dir)


def load_bundle(bundle_dir: str, task: str | None = None):
    """Load a bundle into ``{model, calibrator, features, manifest, metrics}``.

    ``model`` is a :class:`BundleModel` wrapping the native booster — it exposes
    ``predict`` / ``predict_proba`` regardless of xgboost version. ``task`` is read
    from the manifest unless overridden.
    """
    bd = Path(bundle_dir)
    if not (bd / "model.json").exists():
        raise FileNotFoundError(f"No model.json in {bundle_dir}")

    manifest = json.loads((bd / "manifest.json").read_text()) if (bd / "manifest.json").exists() else {}
    feats_blob = json.loads((bd / "features.json").read_text()) if (bd / "features.json").exists() else {}
    features = feats_blob.get("features", [])
    task = task or manifest.get("task", "classification")

    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(bd / "model.json"))
    model = BundleModel(booster, features, task=task)

    calibrator = None
    cal_path = bd / "calibrator.pkl"
    if cal_path.exists():
        try:
            import joblib
            calibrator = joblib.load(cal_path)
        except Exception as exc:
            logger.warning("Could not load calibrator: %s", exc)

    metrics = json.loads((bd / "metrics.json").read_text()) if (bd / "metrics.json").exists() else {}

    return {
        "model": model,
        "calibrator": calibrator,
        "features": features,
        "manifest": manifest,
        "metrics": metrics,
        "bundle_dir": str(bd),
    }


# ---------------------------------------------------------------------------
# prod pointer (atomic promote) + rollback
# ---------------------------------------------------------------------------
def set_prod_pointer(root: str, bundle_dir: str) -> None:
    """Atomically point ``registry/prod/manifest.json`` at ``bundle_dir``.

    Records the previous bundle so ``rollback_prod`` can flip back instantly.
    """
    prod_path = Path(root) / "registry" / "prod" / "manifest.json"
    previous = None
    if prod_path.exists():
        try:
            previous = json.loads(prod_path.read_text()).get("prod_bundle")
        except Exception:
            previous = None
    _atomic_write_json(prod_path, {
        "prod_bundle": str(bundle_dir),
        "previous_bundle": previous,
        "ts": _utc_now(),
    })
    logger.info("PROD pointer → %s (was %s)", bundle_dir, previous or "none")


def get_prod_bundle_dir(root: str) -> str | None:
    prod_path = Path(root) / "registry" / "prod" / "manifest.json"
    if not prod_path.exists():
        return None
    try:
        return json.loads(prod_path.read_text()).get("prod_bundle")
    except Exception:
        return None


def load_prod_bundle(root: str, task: str = "classification"):
    """Load whatever ``registry/prod`` currently points at (the VM daily loop entry)."""
    bd = get_prod_bundle_dir(root)
    if not bd:
        raise FileNotFoundError("No prod pointer set — train + promote a bundle first")
    return load_bundle(bd, task=task)


def rollback_prod(root: str) -> str | None:
    """Flip the prod pointer back to the previously-promoted bundle. Returns it."""
    prod_path = Path(root) / "registry" / "prod" / "manifest.json"
    if not prod_path.exists():
        logger.warning("No prod pointer to roll back")
        return None
    data = json.loads(prod_path.read_text())
    prev = data.get("previous_bundle")
    if not prev:
        logger.warning("No previous bundle recorded — cannot roll back")
        return None
    set_prod_pointer(root, prev)
    logger.warning("ROLLED BACK prod → %s", prev)
    return prev


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------
def list_bundles(root: str) -> list[str]:
    """Return bundle dirs sorted oldest→newest by version suffix."""
    base = Path(root) / "registry" / "bundles"
    if not base.exists():
        return []
    return [str(p) for p in sorted(base.glob("model_*")) if p.is_dir()]


def prune_old_bundles(root: str, keep: int = 8) -> list[str]:
    """Delete all but the most recent ``keep`` bundles. Never deletes the prod target.

    Returns the list of removed bundle dirs.
    """
    bundles = list_bundles(root)
    prod = get_prod_bundle_dir(root)
    removed: list[str] = []
    if len(bundles) <= keep:
        return removed
    for bd in bundles[:-keep]:
        if prod and Path(bd) == Path(prod):
            continue  # never remove the live model
        shutil.rmtree(bd, ignore_errors=True)
        removed.append(bd)
    if removed:
        logger.info("Pruned %d old bundle(s); kept last %d", len(removed), keep)
    return removed
