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


# ---------------------------------------------------------------------------
# Daily (per-period) IC — plan §A1 / Phase 0.
#
# `information_coefficient` above pools every (date, ticker) OOF row into one
# Spearman correlation, which conflates cross-sectional stock-picking skill
# with time-series/market-level effects (a bull run inflates it even with
# zero genuine stock-picking skill). These two functions isolate the
# per-period cross-sectional signal and its statistical significance.
# ---------------------------------------------------------------------------
def daily_information_coefficient(
    preds: pd.DataFrame,
    pred_col: str = "pred",
    target_col: str = "fwd_ret",
    date_col: str = "date",
) -> pd.Series:
    """Per-day cross-sectional Spearman IC, indexed by date.

    Each value is the Spearman correlation between `pred_col` and
    `target_col` across all tickers traded on that single date — the
    standard definition of "IC" in cross-sectional equity research.
    """
    def _ic(g: pd.DataFrame) -> float:
        return information_coefficient(g[pred_col].to_numpy(), g[target_col].to_numpy())

    ic_by_date = preds.groupby(date_col, sort=True).apply(_ic, include_groups=False)
    return ic_by_date.dropna()


def ic_information_ratio(daily_ic: pd.Series | np.ndarray) -> dict:
    """Mean daily IC, its information ratio (mean/std), and significance t-stat.

    t-stat ≈ IC_IR × √n_days is the standard test for "is this cross-sectional
    signal distinguishable from zero" — a single pooled IC number cannot
    answer that question.
    """
    ic = np.asarray(daily_ic, dtype=float)
    ic = ic[~np.isnan(ic)]
    n = len(ic)
    if n < 2:
        return {"mean_ic": float("nan"), "ic_ir": float("nan"), "t_stat": float("nan"), "n_days": n}
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1))
    ic_ir = mean_ic / std_ic if std_ic > 0 else float("nan")
    t_stat = ic_ir * np.sqrt(n)
    return {"mean_ic": mean_ic, "ic_ir": float(ic_ir), "t_stat": float(t_stat), "n_days": n}


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


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_periods: int,
    periods_per_year: float = 252.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true Sharpe > 0) after correcting for picking the best of `n_trials`
    noisy hyperparameter-search draws (Bailey & Lopez de Prado, 2014).

    With no correction, the headline Sharpe from a run that searched
    `n_trials` Optuna configurations is optimistically biased — some of that
    Sharpe may just be the best of many noisy draws, not real, repeatable
    edge. This is an approximation (it uses the SR_hat-based variance
    estimator from the paper rather than the empirical variance across all
    `n_trials` actual trial Sharpes, which this codebase doesn't log).
    Values well below ~0.95 mean "don't trust this Sharpe yet."
    """
    if n_trials <= 1 or n_periods <= 2 or sharpe != sharpe:  # NaN check
        return float("nan")
    sr = sharpe / np.sqrt(periods_per_year)  # de-annualise to per-period SR
    euler_gamma = 0.5772156649015329
    var_sr = (1 - skew * sr + (kurtosis - 1) / 4.0 * sr ** 2) / (n_periods - 1)
    if var_sr <= 0:
        return float("nan")
    sr_std = np.sqrt(var_sr)
    sr0 = sr_std * (
        (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
        + euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    z = (sr - sr0) / sr_std
    return float(stats.norm.cdf(z))


def block_bootstrap_ci(
    period_returns: pd.Series | np.ndarray,
    periods_per_year: float = 252.0,
    n_boot: int = 1000,
    block_size: int = 20,
    seed: int = 42,
) -> dict:
    """Block-bootstrap confidence interval for Sharpe / CAGR / max_drawdown.

    Resamples the period-return series in overlapping blocks (preserves
    short-range autocorrelation, unlike an i.i.d. bootstrap) to produce a
    distribution instead of a single point estimate — answers "how lucky or
    unlucky was this one walk-forward path", which a single backtest run
    cannot.
    """
    r = np.asarray(period_returns, dtype=float)
    n = len(r)
    if n < block_size * 2:
        return {"error": "too few periods for block bootstrap"}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))

    sharpes, cagrs, mdds = [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        sharpes.append(annualised_sharpe(sample, periods_per_year))
        equity = (1 + sample).cumprod()
        cagrs.append(equity[-1] ** (periods_per_year / len(sample)) - 1)
        mdds.append(max_drawdown(equity))

    def _pct(arr: list[float], p: float) -> float:
        return float(np.percentile(arr, p))

    return {
        "sharpe_p05": _pct(sharpes, 5), "sharpe_p50": _pct(sharpes, 50), "sharpe_p95": _pct(sharpes, 95),
        "cagr_p05": _pct(cagrs, 5), "cagr_p50": _pct(cagrs, 50), "cagr_p95": _pct(cagrs, 95),
        "max_drawdown_p05": _pct(mdds, 5), "max_drawdown_p50": _pct(mdds, 50), "max_drawdown_p95": _pct(mdds, 95),
        "n_boot": n_boot, "block_size": block_size,
    }


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
