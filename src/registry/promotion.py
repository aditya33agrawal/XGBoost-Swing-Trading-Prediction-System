"""Champion / challenger promotion gate — §8 of the implementation plan.

A challenger is promoted to prod ONLY if it beats the champion on cost-adjusted
Sharpe by a margin AND keeps calibration honest AND there is no drift alarm.
This composes the existing IC comparison (`src.models.improvement.compare_models`)
with the Sharpe/calibration/drift checks the plan requires, and returns a fully
explained decision so the weekly report can show *why* a model shipped or didn't.
"""
from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

SHARPE_MARGIN = 0.10     # challenger must beat champion Sharpe by this much
IC_MARGIN = 0.005        # …or improve IC by this much (fallback when Sharpe is noisy)
CALIB_TOL = 0.10         # max acceptable calibration error (ECE)


def _is_num(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def evaluate_promotion(
    *,
    challenger: dict,
    champion: dict | None,
    drift_alarm: bool = False,
    sharpe_margin: float = SHARPE_MARGIN,
    ic_margin: float = IC_MARGIN,
    calib_tol: float = CALIB_TOL,
) -> dict:
    """Decide whether `challenger` should replace `champion`.

    `challenger` / `champion` are metric dicts with any of:
        sharpe_net (or Sharpe), ic (or oof_ic), calib_err (or ECE).

    Returns {promote: bool, reasons: [...], checks: {...}}.
    """
    def g(d, *keys):
        for k in keys:
            if d and d.get(k) is not None:
                return d.get(k)
        return None

    reasons: list[str] = []

    # No champion yet ⇒ first model always ships (still blocked by drift alarm).
    if champion is None:
        if drift_alarm:
            return {"promote": False, "reasons": ["drift alarm on first model"], "checks": {}}
        return {"promote": True, "reasons": ["no champion — first model promoted"], "checks": {}}

    ch_sharpe, cp_sharpe = g(challenger, "sharpe_net", "Sharpe"), g(champion, "sharpe_net", "Sharpe")
    ch_ic, cp_ic = g(challenger, "ic", "oof_ic"), g(champion, "ic", "oof_ic")
    ch_calib = g(challenger, "calib_err", "ECE", "ece")

    checks: dict = {}

    # 1. performance — prefer Sharpe; fall back to IC when Sharpe is unavailable
    perf_ok = False
    if _is_num(ch_sharpe) and _is_num(cp_sharpe):
        perf_ok = ch_sharpe >= cp_sharpe + sharpe_margin
        checks["sharpe"] = {"challenger": ch_sharpe, "champion": cp_sharpe,
                            "margin": sharpe_margin, "pass": perf_ok}
        if not perf_ok:
            reasons.append(f"Sharpe {ch_sharpe:.2f} < champion {cp_sharpe:.2f}+{sharpe_margin}")
    elif _is_num(ch_ic) and _is_num(cp_ic):
        perf_ok = ch_ic >= cp_ic + ic_margin
        checks["ic"] = {"challenger": ch_ic, "champion": cp_ic, "margin": ic_margin, "pass": perf_ok}
        if not perf_ok:
            reasons.append(f"IC {ch_ic:.4f} < champion {cp_ic:.4f}+{ic_margin}")
    else:
        reasons.append("no comparable performance metric")
        checks["perf"] = {"pass": False}

    # 2. calibration — only blocks if we actually measured it
    calib_ok = True
    if _is_num(ch_calib):
        calib_ok = ch_calib <= calib_tol
        checks["calibration"] = {"ece": ch_calib, "tol": calib_tol, "pass": calib_ok}
        if not calib_ok:
            reasons.append(f"calibration error {ch_calib:.3f} > tol {calib_tol}")

    # 3. drift gate
    drift_ok = not drift_alarm
    checks["drift"] = {"alarm": drift_alarm, "pass": drift_ok}
    if drift_alarm:
        reasons.append("drift alarm active — holding champion")

    promote = perf_ok and calib_ok and drift_ok
    if promote:
        reasons = ["beats champion on performance, calibration + drift clean"]
    logger.info("Promotion decision: %s — %s", "PROMOTE" if promote else "KEEP CHAMPION",
                "; ".join(reasons))
    return {"promote": promote, "reasons": reasons, "checks": checks}
