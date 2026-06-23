"""Quantile calibration + horizon-stability diagnostics
(docs/dynamic-horizon-rr-plan.md Phase 6, items 13/21/22).

A dynamic stop/target is only as trustworthy as the q10/q90 calibration —
verify the quantiles are calibrated (realised exceedance rate matches the
nominal tau) before trading on them, per the plan's own warning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_calibration_report(
    preds_with_quantiles: pd.DataFrame,
    taus: list[float] = (0.1, 0.5, 0.9),
    realised_col: str = "fwd_ret_hstar",
    quantile_col_fmt: str = "q{tau}",
) -> pd.DataFrame:
    """For each tau, the realised exceedance rate vs the nominal tau.

    `preds_with_quantiles` must have one column per tau (named via
    `quantile_col_fmt`, default "q0.1"/"q0.5"/"q0.9") holding the predicted
    quantile, and a `realised_col` holding the realised forward return. A
    well-calibrated tau=0.1 quantile should have ~10% of realised returns
    fall below it; large deviations mean the dynamic RR derived from these
    quantiles (Phase 3) is not trustworthy yet.

    Returns DataFrame[tau, nominal_rate, realised_rate, n, abs_gap].
    """
    rows = []
    realised = preds_with_quantiles.get(realised_col)
    for tau in taus:
        col = quantile_col_fmt.format(tau=tau)
        if col not in preds_with_quantiles.columns or realised is None:
            continue
        mask = preds_with_quantiles[col].notna() & realised.notna()
        n = int(mask.sum())
        if n == 0:
            continue
        below = (realised[mask] < preds_with_quantiles.loc[mask, col]).mean()
        rows.append({
            "tau": tau,
            "nominal_rate": tau,
            "realised_rate": round(float(below), 4),
            "n": n,
            "abs_gap": round(abs(float(below) - tau), 4),
        })
    return pd.DataFrame(rows)


def quantile_calibration_from_surface(
    oof_preds: pd.DataFrame,
    taus: tuple[float, float, float] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Convenience wrapper for the `_walk_forward_predict_surface` output
    shape: columns `q10_star`/`q50_star`/`q90_star` (quantiles at each row's
    chosen h*) and `fwd_ret` (realised return at that same h*)."""
    tau_to_col = {taus[0]: "q10_star", taus[1]: "q50_star", taus[2]: "q90_star"}
    rows = []
    for tau, col in tau_to_col.items():
        if col not in oof_preds.columns or "fwd_ret" not in oof_preds.columns:
            continue
        mask = oof_preds[col].notna() & oof_preds["fwd_ret"].notna()
        n = int(mask.sum())
        if n == 0:
            continue
        below = (oof_preds.loc[mask, "fwd_ret"] < oof_preds.loc[mask, col]).mean()
        rows.append({
            "tau": tau,
            "nominal_rate": tau,
            "realised_rate": round(float(below), 4),
            "n": n,
            "abs_gap": round(abs(float(below) - tau), 4),
        })
    return pd.DataFrame(rows)


def horizon_stability_report(h_star_by_seed: list[np.ndarray]) -> dict:
    """Cross-seed agreement on the chosen horizon h* (plan item 22) — an
    unstable horizon choice (same data, different random_state landing on
    very different h*) is a red flag, the same logic as the IC-instability
    concern in model-improvement-plan.md.

    Parameters
    ----------
    h_star_by_seed : list of (n_rows,) int arrays, one per seed, all aligned
                     to the same (date, ticker) ordering.

    Returns
    -------
    dict with `mode_agreement_rate` (fraction of rows where every seed picked
    the same horizon as the row-wise mode) and `mean_abs_seed_spread` (mean
    |max-min| horizon across seeds, per row) — both should be high/low
    respectively for a trustworthy horizon signal.
    """
    if not h_star_by_seed or len(h_star_by_seed) < 2:
        return {"mode_agreement_rate": float("nan"), "mean_abs_seed_spread": float("nan"), "n_seeds": len(h_star_by_seed)}

    stacked = np.stack(h_star_by_seed, axis=0)  # (n_seeds, n_rows)
    n_seeds, n_rows = stacked.shape

    modes = []
    for col in range(n_rows):
        vals, counts = np.unique(stacked[:, col], return_counts=True)
        modes.append(vals[np.argmax(counts)])
    modes = np.array(modes)

    agreement = (stacked == modes[None, :]).mean(axis=0)  # fraction of seeds agreeing w/ mode, per row
    mode_agreement_rate = float((agreement == 1.0).mean())

    spread = stacked.max(axis=0) - stacked.min(axis=0)
    mean_abs_seed_spread = float(spread.mean())

    return {
        "mode_agreement_rate": round(mode_agreement_rate, 4),
        "mean_abs_seed_spread": round(mean_abs_seed_spread, 2),
        "n_seeds": n_seeds,
        "n_rows": n_rows,
    }
