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
# 1b. Multi-horizon forward log-returns (dynamic-horizon-rr-plan.md Phase 1)
# ---------------------------------------------------------------------------
def forward_log_return_grid(df: pd.DataFrame, grid: list[int]) -> pd.DataFrame:
    """Add one `fwd_ret_{h}` column per horizon in `grid`.

    Each column is computed with the same logic as `forward_log_return(df, h)`
    — last `h` rows per ticker are NaN for that column. Used to train one
    quantile head per horizon without re-deriving the label math per-h.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    close_by_ticker = df.groupby("ticker")["close"]
    for h in grid:
        df[f"fwd_ret_{h}"] = close_by_ticker.transform(lambda s, h=h: np.log(s.shift(-h) / s))
    return df


# ---------------------------------------------------------------------------
# 1c. Cross-sectional relevance grades (Learning-to-Rank target)
# ---------------------------------------------------------------------------
def cross_sectional_relevance(
    df: pd.DataFrame,
    bins: int = 8,
    ret_col: str = "fwd_ret",
    out_col: str = "rank_rel",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add an integer relevance grade in [0, bins-1] per (date) from ``ret_col``.

    The LambdaMART ranker (objective="rank:ndcg") needs a graded relevance
    label per query group; here the query group is a date and the relevance is
    "how good was this stock's forward return relative to its peers *that day*"
    — exactly the cross-sectional ordering the strategy trades.

    Implementation:
      - Within each date, sort by ``ret_col`` and assign integer buckets
        0..bins-1 (higher grade = better forward return).
      - ``pd.qcut`` is used when there are enough distinct values; it falls back
        to a dense rank scaled into [0, bins-1] for short/degenerate days (e.g.
        the most-recent dates where few tickers are labeled, or ties).
      - Rows where ``ret_col`` is NaN (the last ``h`` bars per ticker) keep a
        NaN grade and are dropped before training — same contract as the other
        label columns. The grade is computed strictly within a date, so no
        information crosses the train/test boundary.

    Returns a copy of ``df`` with ``out_col`` added.
    """
    df = df.copy()

    def _grade(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        out = pd.Series(np.nan, index=s.index)
        n = len(valid)
        if n == 0:
            return out
        if n < 2:
            out.loc[valid.index] = 0.0
            return out
        # Dense rank in [0, 1] → scale to [0, bins-1] integer grades. This is
        # robust to ties and to days with fewer than `bins` tickers, where
        # qcut would raise; it yields the same monotonic ordering qcut targets.
        r = valid.rank(method="first")  # 1..n, breaks ties by order
        grade = np.floor((r - 1) / n * bins).astype(int)
        grade = np.clip(grade, 0, bins - 1)
        out.loc[valid.index] = grade.astype(float)
        return out

    df[out_col] = df.groupby(date_col, sort=False)[ret_col].transform(_grade)
    return df


# ---------------------------------------------------------------------------
# 2. Triple-barrier label (classification)
# ---------------------------------------------------------------------------
def triple_barrier_labels(
    df: pd.DataFrame,
    h: int,
    up_mult: float = 2.0,
    dn_mult: float = 2.0,
    record_first_passage: bool = False,
) -> pd.DataFrame:
    """Add column `tb_label` ∈ {+1, -1, 0} and `fwd_ret`.

    Barriers are ±(mult × ATR14) from entry close.
    Label = which barrier is hit first within the next h bars:
      +1  upper barrier (take-profit)
      -1  lower barrier (stop-loss)
       0  vertical barrier (time limit)

    Last `h` rows per ticker will be NaN.

    When `record_first_passage=True`, also emits:
      tb_first_passage_time — bars until whichever barrier hit (1..h), or h
                               if the vertical barrier fired (no early hit).
      tb_barrier_hit        — "up" | "down" | "vertical".
    Used for Design B/C cross-checks and horizon diagnostics
    (docs/dynamic-horizon-rr-plan.md §1 Design B/C); does not change the
    existing tb_label/fwd_ret contract when left False.
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
        fpt = np.full(n, np.nan)
        hit = np.full(n, None, dtype=object)
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
                fpt[i] = h
                hit[i] = "vertical"
            elif t_up <= t_dn:
                labels[i] = 1
                fpt[i] = t_up + 1
                hit[i] = "up"
            else:
                labels[i] = -1
                fpt[i] = t_dn + 1
                hit[i] = "down"

        grp = grp.copy()
        grp["tb_label"] = labels
        if record_first_passage:
            grp["tb_first_passage_time"] = fpt
            grp["tb_barrier_hit"] = hit
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
