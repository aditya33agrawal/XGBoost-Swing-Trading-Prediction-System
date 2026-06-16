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
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }

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
    study.optimize(
        lambda t: _optuna_objective(t, X, y, splits, task, sample_weights),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )

    best = study.best_params
    best["objective"] = "binary:logistic" if task == "classification" else "reg:squarederror"
    best["eval_metric"] = "aucpr" if task == "classification" else "rmse"
    best["tree_method"] = "hist"
    best["random_state"] = 42
    best["verbosity"] = 0
    best["early_stopping_rounds"] = 50

    print(f"[tuning] best IC = {study.best_value:.4f} | params = {best}")
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
