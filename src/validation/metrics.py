"""Evaluation metrics (plan §9).

ML metrics: IC (Spearman), AUC-PR, log-loss, calibration error, directional accuracy.
Financial metrics: Sharpe, Sortino, Calmar, max drawdown, hit rate, profit factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# ML metrics
# ---------------------------------------------------------------------------
def information_coefficient(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank correlation between predicted and realised returns."""
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 2:
        return np.nan
    ic, _ = stats.spearmanr(y_pred[mask], y_true[mask])
    return float(ic)


def directional_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / n * abs(avg_conf - avg_acc)
    return float(ece)


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------
def annualised_sharpe(
    period_returns: pd.Series | np.ndarray,
    periods_per_year: float = 252.0,
) -> float:
    r = np.asarray(period_returns)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def annualised_sortino(
    period_returns: pd.Series | np.ndarray,
    periods_per_year: float = 252.0,
    mar: float = 0.0,
) -> float:
    r = np.asarray(period_returns)
    downside = r[r < mar]
    if len(downside) == 0:
        return np.inf
    downside_std = downside.std()
    if downside_std == 0:
        return np.inf
    return float(r.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series | np.ndarray) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-9)
    return float(dd.min())


def calmar_ratio(
    period_returns: pd.Series | np.ndarray,
    periods_per_year: float = 252.0,
) -> float:
    r = np.asarray(period_returns)
    equity = (1 + r).cumprod()
    md = abs(max_drawdown(equity))
    if md == 0:
        return np.inf
    cagr = equity[-1] ** (periods_per_year / len(r)) - 1
    return float(cagr / md)


def profit_factor(period_returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(period_returns)
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    if gross_loss == 0:
        return np.inf
    return float(gross_profit / gross_loss)


def summarise(
    period_returns: pd.Series | np.ndarray,
    periods_per_year: float = 252.0,
    label: str = "",
) -> dict:
    r = np.asarray(period_returns)
    if len(r) == 0:
        return {"error": "no data"}
    equity = (1 + r).cumprod()
    cagr = equity[-1] ** (periods_per_year / len(r)) - 1
    return {
        "label": label,
        "n_periods": len(r),
        "CAGR": round(float(cagr), 4),
        "Sharpe": round(annualised_sharpe(r, periods_per_year), 3),
        "Sortino": round(annualised_sortino(r, periods_per_year), 3),
        "Calmar": round(calmar_ratio(r, periods_per_year), 3),
        "max_drawdown": round(float(max_drawdown(equity)), 4),
        "hit_rate": round(float((r > 0).mean()), 3),
        "profit_factor": round(profit_factor(r), 3),
        "avg_period_ret": round(float(r.mean()), 5),
        "final_equity": round(float(equity[-1]), 4),
    }
