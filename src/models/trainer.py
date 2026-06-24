"""XGBoost model trainer with Optuna hyperparameter optimisation (plan §8).

Supports:
  - Classification (triple-barrier: binary for {+1} vs {-1,0} or multiclass)
  - Regression (forward log-return)

Workflow:
  1. PurgedWalkForward CV for hyperparameter search (Optuna)
  2. Retrain best model on all available training data
  3. Return fitted model + params + OOF predictions
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection (CPU vs GPU).  XGBoost 2.x uses tree_method="hist" + device.
# ---------------------------------------------------------------------------
_DEVICE = "cpu"


def resolve_device(device: str = "auto") -> str:
    """Resolve "auto" to "cuda" when a usable GPU is present, else "cpu"."""
    if device != "auto":
        return device
    try:
        import xgboost as xgb  # noqa: F401
        # Cheapest reliable probe: ask CUDA how many devices it sees.
        try:
            from xgboost import collective  # noqa: F401
        except Exception:
            pass
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and "GPU" in out.stdout:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def set_device(device: str = "auto") -> str:
    """Set the global device used for every XGBoost model built in this module."""
    global _DEVICE
    _DEVICE = resolve_device(device)
    logger.info("XGBoost device set to '%s'", _DEVICE)
    return _DEVICE


def _with_device(params: dict) -> dict:
    """Return a copy of params with tree_method/device set for the active device."""
    p = dict(params)
    p["tree_method"] = "hist"
    p["device"] = _DEVICE
    return p


# Public aliases so other modules (e.g. the runner) can tag params + query device.
apply_device = _with_device


def get_device() -> str:
    return _DEVICE


# ---------------------------------------------------------------------------
# XGBoost defaults (plan §8.2)
# ---------------------------------------------------------------------------
_BASE_PARAMS_CLF = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "n_estimators": 400,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "gamma": 0.5,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}

_BASE_PARAMS_REG = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 400,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "gamma": 0.5,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}


def _get_xgb():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        raise ImportError("xgboost is required: pip install xgboost")


# ---------------------------------------------------------------------------
# Train a single XGBoost model
# ---------------------------------------------------------------------------
def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    sample_weight: np.ndarray | None = None,
    early_stopping: int = 50,
    task: str = "classification",
) -> Any:
    xgb = _get_xgb()

    # For triple-barrier {-1, 0, +1}, binarize: 1 → 1, else → 0
    # This gives a "probability of an up-move" signal
    if task == "classification":
        y_train_bin = (y_train == 1).astype(int)
        y_val_bin = (y_val == 1).astype(int)
    else:
        y_train_bin = y_train
        y_val_bin = y_val

    params = _with_device(params)
    model = xgb.XGBClassifier(**params) if task == "classification" else xgb.XGBRegressor(**params)
    model.set_params(early_stopping_rounds=early_stopping)
    model.fit(
        X_train,
        y_train_bin,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val_bin)],
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Multi-seed bagging (plan §Phase 3.16) — averaging independent noisy fits
# reduces the run-to-run OOF IC instability of a single point-estimate model.
# Same data, same hyperparameters, only `random_state` differs per member.
# ---------------------------------------------------------------------------
def train_xgb_bag(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    sample_weight: np.ndarray | None = None,
    early_stopping: int = 50,
    task: str = "classification",
    n_seeds: int = 3,
    base_seed: int = 42,
) -> list:
    """Train `n_seeds` independent XGBoost models (only random_state differs).

    Returns the list of fitted models; average their predictions with
    `predict_bag` rather than picking any single one.
    """
    models = []
    for i in range(max(1, n_seeds)):
        seed_params = dict(params)
        seed_params["random_state"] = base_seed + i
        models.append(
            train_xgb(X_train, y_train, X_val, y_val, seed_params,
                      sample_weight=sample_weight, early_stopping=early_stopping, task=task)
        )
    return models


def predict_bag(models: list, X: pd.DataFrame, task: str = "classification") -> np.ndarray:
    """Average predictions across a bagged ensemble from `train_xgb_bag`."""
    if task == "classification":
        preds = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=0)
    else:
        preds = np.stack([m.predict(X) for m in models], axis=0)
    return preds.mean(axis=0)


# ---------------------------------------------------------------------------
# Quantile regression heads (docs/dynamic-horizon-rr-plan.md Phase 2) —
# one head per (horizon, tau) cell, trained with XGBoost's native pinball-loss
# objective. Mirrors train_xgb/train_xgb_bag's structure so the rest of the
# walk-forward plumbing (device handling, early stopping) is reused as-is.
# ---------------------------------------------------------------------------
_QUANTILE_BASE_PARAMS = {
    "objective": "reg:quantileerror",
    "n_estimators": 400,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "gamma": 0.5,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}


def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tau: float,
    params: dict | None = None,
    sample_weight: np.ndarray | None = None,
    early_stopping: int = 50,
) -> Any:
    """Train a single XGBoost quantile regressor at quantile `tau`."""
    xgb = _get_xgb()
    p = dict(params) if params else dict(_QUANTILE_BASE_PARAMS)
    # Strip params that don't apply to the quantile objective / aren't model knobs.
    for k in ("objective", "eval_metric", "early_stopping_rounds"):
        p.pop(k, None)
    p["objective"] = "reg:quantileerror"
    p["quantile_alpha"] = tau
    p = _with_device(p)

    model = xgb.XGBRegressor(**p)
    model.set_params(early_stopping_rounds=early_stopping)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_quantile_surface(
    X_train: pd.DataFrame,
    y_train_by_h: dict[int, pd.Series],
    X_val: pd.DataFrame,
    y_val_by_h: dict[int, pd.Series],
    taus: list[float],
    params: dict | None = None,
    sample_weight_by_h: dict[int, np.ndarray] | None = None,
    early_stopping: int = 50,
    n_seeds: int = 1,
    base_seed: int = 42,
) -> dict[tuple[int, float], list]:
    """Train one bagged quantile-head ensemble per (horizon, tau) cell.

    Returns {(h, tau): [model, ...]} — `n_seeds` models per cell, average
    with `predict_surface` rather than trusting a single fit.
    """
    surface: dict[tuple[int, float], list] = {}
    for h, y_tr in y_train_by_h.items():
        y_vl = y_val_by_h[h]
        sw = sample_weight_by_h.get(h) if sample_weight_by_h else None
        for tau in taus:
            models = []
            for i in range(max(1, n_seeds)):
                seed_params = dict(params) if params else dict(_QUANTILE_BASE_PARAMS)
                seed_params["random_state"] = base_seed + i
                models.append(
                    train_quantile_model(
                        X_train, y_tr, X_val, y_vl, tau,
                        params=seed_params, sample_weight=sw,
                        early_stopping=early_stopping,
                    )
                )
            surface[(h, tau)] = models
    return surface


def train_quantile_model_no_es(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tau: float,
    params: dict | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Final-model quantile fit with no early stopping (mirrors predict_latest's
    fixed-n_estimators pattern for the classifier/regressor — a small val window
    here would collapse the model the same way it does for the scalar score)."""
    xgb = _get_xgb()
    p = dict(params) if params else dict(_QUANTILE_BASE_PARAMS)
    for k in ("objective", "eval_metric", "early_stopping_rounds"):
        p.pop(k, None)
    p["objective"] = "reg:quantileerror"
    p["quantile_alpha"] = tau
    p = _with_device(p)
    model = xgb.XGBRegressor(**p)
    model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
    return model


def predict_surface(surface: dict[tuple[int, float], list], X: pd.DataFrame) -> dict[tuple[int, float], np.ndarray]:
    """Average each (h, tau) cell's bagged predictions on `X`."""
    out = {}
    for key, models in surface.items():
        preds = np.stack([m.predict(X) for m in models], axis=0)
        out[key] = preds.mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------
def _optuna_objective(
    trial,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    task: str,
    sample_weights_arr: np.ndarray | None,
    eval_target: pd.Series | None = None,
    threads_per_trial: int = 0,
):
    from scipy import stats as sp_stats
    xgb = _get_xgb()

    params = {
        "objective": "binary:logistic" if task == "classification" else "reg:squarederror",
        "eval_metric": "aucpr" if task == "classification" else "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }
    params = _with_device(params)
    # Cap each trial's CPU thread pool so N trials running concurrently
    # (see tune_hyperparameters n_jobs) don't oversubscribe the machine.
    if threads_per_trial > 0:
        params["n_jobs"] = threads_per_trial

    fold_scores = []
    for train_idx, val_idx in splits:
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_vl, y_vl = X.iloc[val_idx], y.iloc[val_idx]
        sw = sample_weights_arr[train_idx] if sample_weights_arr is not None else None

        if task == "classification":
            y_tr_b = (y_tr == 1).astype(int)
            y_vl_b = (y_vl == 1).astype(int)
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr_b, sample_weight=sw,
                      eval_set=[(X_vl, y_vl_b)], verbose=False)
            preds = model.predict_proba(X_vl)[:, 1]
        else:
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, sample_weight=sw,
                      eval_set=[(X_vl, y_vl)], verbose=False)
            preds = model.predict(X_vl)

        # Objective: Information Coefficient (Spearman) on realised forward return.
        # The strategy P&L and the reported OOF IC both rank against `fwd_ret`, so
        # we tune against `fwd_ret` (eval_target) rather than the discrete
        # triple-barrier label {-1,0,1}.  Falls back to y when no eval_target given.
        target_vl = eval_target.iloc[val_idx].values if eval_target is not None else y_vl.values
        if len(target_vl) > 1:
            ic, _ = sp_stats.spearmanr(preds, target_vl)
            fold_scores.append(ic if not np.isnan(ic) else 0.0)

    return float(np.mean(fold_scores)) if fold_scores else 0.0


def tune_hyperparameters(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    splits: list,
    task: str = "classification",
    n_trials: int = 50,
    sample_weights: np.ndarray | None = None,
    eval_target_col: str = "fwd_ret",
    max_workers: int = 8,
) -> dict:
    """Run Optuna search; return best params dict.

    The trial objective ranks predictions against ``eval_target_col`` (the
    realised forward return) when that column is present, so HPO optimises the
    same signal the backtest trades — not the discrete classification label.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna not installed — skipping tuning, using defaults")
        return _BASE_PARAMS_CLF if task == "classification" else _BASE_PARAMS_REG

    X = df_train[feature_cols]
    y = df_train[target_col]
    if eval_target_col in df_train.columns:
        eval_target = df_train[eval_target_col].reset_index(drop=True)
        logger.info("Optuna objective: ranking IC vs '%s'", eval_target_col)
    else:
        eval_target = None
        logger.info("Optuna objective: ranking IC vs target '%s' (no %s column)",
                    target_col, eval_target_col)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    # Run multiple Optuna trials concurrently — XGBoost's fit() releases the
    # GIL during tree building, so Optuna's thread-based n_jobs gives real
    # parallelism here. Each trial's XGBoost is capped to cpu_count // n_parallel
    # threads so the trials don't oversubscribe and fight each other for cores
    # on the CPU-side work (DMatrix construction, histogram binning) that runs
    # regardless of device. On GPU, concurrent trials overlap that per-call
    # host-side overhead instead of paying it serially for one device — the
    # actual tree-building kernels still queue on the GPU, but at this model
    # size (shallow trees, small per-fold data) the overhead dominates, not
    # the kernel time, so concurrency still helps.
    cpu_count = os.cpu_count() or 1
    if n_trials > 1:
        n_parallel = max(1, min(cpu_count, max(1, max_workers), n_trials))
        threads_per_trial = max(1, cpu_count // n_parallel)
    else:
        n_parallel = 1
        threads_per_trial = 0

    logger.info(
        "Optuna search: %d trials | %d CV folds | device=%s | parallel_trials=%d | threads/trial=%s",
        n_trials, len(splits), _DEVICE, n_parallel, threads_per_trial or "default",
    )

    def _log_trial(study_, trial):
        # Log every trial's IC and the running best so progress is visible in Colab.
        logger.info(
            "  trial %3d/%d | IC=%+.4f | best=%+.4f",
            trial.number + 1, n_trials,
            trial.value if trial.value is not None else float("nan"),
            study_.best_value,
        )

    study.optimize(
        lambda t: _optuna_objective(t, X, y, splits, task, sample_weights, eval_target, threads_per_trial),
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=n_parallel,
        callbacks=[_log_trial],
    )

    best = study.best_params
    best["objective"] = "binary:logistic" if task == "classification" else "reg:squarederror"
    best["eval_metric"] = "aucpr" if task == "classification" else "rmse"
    best["tree_method"] = "hist"
    best["device"] = _DEVICE
    best["random_state"] = 42
    best["verbosity"] = 0
    best["early_stopping_rounds"] = 50

    logger.info("Tuning done — best IC = %.4f", study.best_value)
    logger.info("Best params: %s", best)
    return best


# ---------------------------------------------------------------------------
# Final model: train on all data, score held-out OOS
# ---------------------------------------------------------------------------
def fit_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,  # small holdout for early stopping
    params: dict,
    task: str = "classification",
    sample_weights: np.ndarray | None = None,
):
    xgb = _get_xgb()

    X = df[feature_cols]
    y = df[target_col]

    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_vl, y_vl = X.iloc[val_idx], y.iloc[val_idx]
    sw = sample_weights[train_idx] if sample_weights is not None else None

    params = _with_device(params)
    if task == "classification":
        y_tr_b = (y_tr == 1).astype(int)
        y_vl_b = (y_vl == 1).astype(int)
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr_b, sample_weight=sw,
                  eval_set=[(X_vl, y_vl_b)], verbose=False)
    else:
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=sw,
                  eval_set=[(X_vl, y_vl)], verbose=False)

    return model
