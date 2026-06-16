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
# Optuna hyperparameter search
# ---------------------------------------------------------------------------
def _optuna_objective(
    trial,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    task: str,
    sample_weights_arr: np.ndarray | None,
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

        # Objective: Information Coefficient (Spearman) on realised forward return
        # For classification, we use the raw probability as the predicted signal
        if len(y_vl) > 1:
            ic, _ = sp_stats.spearmanr(preds, y_vl.values)
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
) -> dict:
    """Run Optuna search; return best params dict."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("optuna not installed — skipping tuning, using defaults")
        return _BASE_PARAMS_CLF if task == "classification" else _BASE_PARAMS_REG

    X = df_train[feature_cols]
    y = df_train[target_col]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    logger.info(
        "Optuna search: %d trials | %d CV folds | device=%s",
        n_trials, len(splits), _DEVICE,
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
        lambda t: _optuna_objective(t, X, y, splits, task, sample_weights),
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=1,
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
