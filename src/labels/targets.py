"""Target/label engineering (plan §2).

Two label types:
  1. forward_log_return — regression target: ln(P_{t+h} / P_t)
  2. triple_barrier     — classification: {+1, -1, 0}

Critical leakage rule: the LAST h rows of each ticker have no valid label.
They are left as NaN and must be dropped before training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Forward log-return (regression)
# ---------------------------------------------------------------------------
def forward_log_return(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Add column `fwd_ret` = ln(close_{t+h} / close_t).

    Last `h` rows per ticker will be NaN — drop before training.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["fwd_ret"] = df.groupby("ticker")["close"].transform(
        lambda s: np.log(s.shift(-h) / s)
    )
    return df


# ---------------------------------------------------------------------------
# 2. Triple-barrier label (classification)
# ---------------------------------------------------------------------------
def triple_barrier_labels(
    df: pd.DataFrame,
    h: int,
    up_mult: float = 2.0,
    dn_mult: float = 2.0,
) -> pd.DataFrame:
    """Add column `tb_label` ∈ {+1, -1, 0} and `fwd_ret`.

    Barriers are ±(mult × ATR14) from entry close.
    Label = which barrier is hit first within the next h bars:
      +1  upper barrier (take-profit)
      -1  lower barrier (stop-loss)
       0  vertical barrier (time limit)

    Last `h` rows per ticker will be NaN.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df = forward_log_return(df, h)  # also compute fwd_ret

    # ATR14 (need high/low/close in df)
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("triple_barrier_labels requires 'high' and 'low' columns")

    all_labels = []
    for ticker, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").copy()
        c = grp["close"].to_numpy(dtype=float)
        hi = grp["high"].to_numpy(dtype=float)
        lo = grp["low"].to_numpy(dtype=float)
        n = len(c)

        # ATR14 (exponential)
        tr = np.maximum(
            hi[1:] - lo[1:],
            np.maximum(
                np.abs(hi[1:] - c[:-1]),
                np.abs(lo[1:] - c[:-1]),
            ),
        )
        tr = np.concatenate([[tr[0]], tr])
        atr = _ewm_mean(tr, span=14)

        labels = np.full(n, np.nan)
        for i in range(n - h):
            upper = c[i] + up_mult * atr[i]
            lower = c[i] - dn_mult * atr[i]
            window_hi = hi[i + 1: i + 1 + h]
            window_lo = lo[i + 1: i + 1 + h]

            up_times = np.where(window_hi >= upper)[0]
            dn_times = np.where(window_lo <= lower)[0]

            t_up = up_times[0] if len(up_times) else n
            t_dn = dn_times[0] if len(dn_times) else n

            if t_up == n and t_dn == n:
                labels[i] = 0
            elif t_up <= t_dn:
                labels[i] = 1
            else:
                labels[i] = -1

        grp = grp.copy()
        grp["tb_label"] = labels
        all_labels.append(grp)

    result = pd.concat(all_labels, ignore_index=True)
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def _ewm_mean(arr: np.ndarray, span: int) -> np.ndarray:
    """Vectorised exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


# ---------------------------------------------------------------------------
# Convenience: add whichever label type the config requests
# ---------------------------------------------------------------------------
def add_labels(df: pd.DataFrame, h: int, label_type: str = "triple_barrier") -> pd.DataFrame:
    if label_type == "triple_barrier":
        df = triple_barrier_labels(df, h)
        df["target"] = df["tb_label"]
    else:
        df = forward_log_return(df, h)
        df["target"] = df["fwd_ret"]
    return df
