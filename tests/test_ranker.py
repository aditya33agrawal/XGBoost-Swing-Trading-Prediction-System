"""Learning-to-Rank (LambdaMART) path tests.

Covers the SOTA upgrade: cross-sectional relevance grading, the XGBRanker
training wrappers, and the bagged ranking prediction — all on tiny synthetic
data (no network, no full pipeline run, per project convention).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels.targets import cross_sectional_relevance, forward_log_return
from src.models.trainer import (
    train_xgb_ranker, train_xgb_bag_ranker, train_xgb_ranker_no_es,
    predict_bag, _qid_codes, _sort_by_qid, _BASE_PARAMS_RANKER,
)

pytest.importorskip("xgboost")


# ---------------------------------------------------------------------------
# Synthetic cross-sectional panel: many tickers per date, with a genuine
# signal so the ranker has something to learn (feature x0 ~ forward return).
# ---------------------------------------------------------------------------
def _xs_panel(n_days: int = 40, n_tickers: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    rows = []
    for d in dates:
        for k in range(n_tickers):
            x0 = rng.normal()
            x1 = rng.normal()
            # forward return loads on x0 plus noise → x0 should rank stocks
            fwd = 0.03 * x0 + rng.normal(0, 0.01)
            rows.append({"date": d, "ticker": f"T{k:02d}", "x0": x0, "x1": x1,
                         "fwd_ret": fwd})
    df = pd.DataFrame(rows)
    # Blank the last `h` dates' fwd_ret to mimic the unlabeled tail.
    last_dates = dates[-3:]
    df.loc[df["date"].isin(last_dates), "fwd_ret"] = np.nan
    return df, ["x0", "x1"]


# ---------------------------------------------------------------------------
# 1. Relevance grading
# ---------------------------------------------------------------------------
def test_relevance_is_per_date_monotonic_and_bounded():
    bins = 5
    df, _ = _xs_panel()
    out = cross_sectional_relevance(df, bins=bins)

    graded = out.dropna(subset=["rank_rel"])
    # Bounded integer grades.
    vals = graded["rank_rel"].to_numpy()
    assert vals.min() >= 0 and vals.max() <= bins - 1
    assert np.allclose(vals, np.round(vals)), "grades must be integers"

    # Monotone with fwd_ret *within* a date (Spearman ~ 1).
    for _, grp in graded.groupby("date"):
        if len(grp) < bins:
            continue
        corr = grp["rank_rel"].corr(grp["fwd_ret"], method="spearman")
        assert corr > 0.9, f"grade must track fwd_ret rank within a date (got {corr:.2f})"


def test_relevance_nan_where_fwd_ret_nan():
    df, _ = _xs_panel()
    out = cross_sectional_relevance(df, bins=8)
    assert out["rank_rel"].isna().equals(out["fwd_ret"].isna()), \
        "rank_rel must be NaN exactly where fwd_ret is NaN (unlabeled tail)"


def test_relevance_handles_short_day_without_qcut_error():
    # A date with fewer tickers than bins must not raise (rank fallback).
    df = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-01"] * 3),
        "ticker": ["A", "B", "C"],
        "fwd_ret": [0.01, -0.02, 0.03],
    })
    out = cross_sectional_relevance(df, bins=8)
    assert out["rank_rel"].notna().all()
    # Best return → highest grade.
    assert out.loc[out["ticker"] == "C", "rank_rel"].iloc[0] == out["rank_rel"].max()


# ---------------------------------------------------------------------------
# 2. qid helpers
# ---------------------------------------------------------------------------
def test_sort_by_qid_groups_are_contiguous():
    X = pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0]})
    y = np.array([0, 1, 2, 3])
    qid = np.array([2, 1, 2, 1])
    Xs, ys, qids, sws = _sort_by_qid(X, y, qid, sw=np.array([1.0, 2.0, 3.0, 4.0]))
    # Equal qids must be adjacent after the stable sort.
    assert list(qids) == [1, 1, 2, 2]
    # Stable: within a group the original order is preserved.
    assert list(ys) == [1, 3, 0, 2]
    assert list(sws) == [2.0, 4.0, 1.0, 3.0]


def test_qid_codes_dense_and_sorted():
    codes = _qid_codes(pd.to_datetime(["2021-01-03", "2021-01-01", "2021-01-03"]))
    assert list(codes) == [1, 0, 1]


# ---------------------------------------------------------------------------
# 3. Ranker fit / predict
# ---------------------------------------------------------------------------
def _split_by_date(df):
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * 0.7)]
    tr = df[df["date"] <= cut]
    vl = df[df["date"] > cut]
    return tr, vl


def test_ranker_fits_and_predicts_dispersed_scores():
    df, feats = _xs_panel(seed=1)
    df = cross_sectional_relevance(df, bins=8).dropna(subset=["rank_rel"])
    tr, vl = _split_by_date(df)

    params = dict(_BASE_PARAMS_RANKER, n_estimators=40, learning_rate=0.1)
    model = train_xgb_ranker(
        tr[feats], tr["rank_rel"], tr["date"].to_numpy(),
        vl[feats], vl["rank_rel"], vl["date"].to_numpy(),
        params, sample_weight=np.ones(len(tr)), early_stopping=10,
    )
    scores = model.predict(vl[feats])
    assert np.std(scores) > 0, "ranker must produce dispersed (non-constant) scores"

    # The score should rank stocks: positive IC vs realised fwd_ret on val.
    from scipy import stats as sp
    ic, _ = sp.spearmanr(scores, vl["fwd_ret"].to_numpy())
    assert ic > 0.1, f"ranker IC vs fwd_ret should be clearly positive (got {ic:.3f})"


def test_ranker_accepts_per_instance_sample_weight():
    # Validates the plan's assumption that XGBRanker takes per-row weights.
    df, feats = _xs_panel(seed=2)
    df = cross_sectional_relevance(df, bins=6).dropna(subset=["rank_rel"])
    tr, vl = _split_by_date(df)
    rng = np.random.default_rng(0)
    sw = rng.uniform(0.5, 1.5, len(tr))
    model = train_xgb_ranker(
        tr[feats], tr["rank_rel"], tr["date"].to_numpy(),
        vl[feats], vl["rank_rel"], vl["date"].to_numpy(),
        dict(_BASE_PARAMS_RANKER, n_estimators=20), sample_weight=sw, early_stopping=10,
    )
    assert model.predict(vl[feats]).shape[0] == len(vl)


def test_predict_bag_ranking_averages_and_preserves_order():
    df, feats = _xs_panel(seed=3)
    df = cross_sectional_relevance(df, bins=8).dropna(subset=["rank_rel"])
    tr, vl = _split_by_date(df)
    models = train_xgb_bag_ranker(
        tr[feats], tr["rank_rel"], tr["date"].to_numpy(),
        vl[feats], vl["rank_rel"], vl["date"].to_numpy(),
        dict(_BASE_PARAMS_RANKER, n_estimators=30, learning_rate=0.1),
        sample_weight=np.ones(len(tr)), early_stopping=10, n_seeds=3,
    )
    assert len(models) == 3
    bag = predict_bag(models, vl[feats], task="ranking")
    assert bag.shape[0] == len(vl)
    # Bagged score is the mean of members.
    members = np.stack([m.predict(vl[feats]) for m in models], axis=0)
    assert np.allclose(bag, members.mean(axis=0))


def test_ranker_no_es_trains_on_full_data():
    df, feats = _xs_panel(seed=4)
    df = cross_sectional_relevance(df, bins=8).dropna(subset=["rank_rel"])
    model = train_xgb_ranker_no_es(
        df[feats], df["rank_rel"], df["date"].to_numpy(),
        params=dict(_BASE_PARAMS_RANKER, n_estimators=25),
    )
    scores = model.predict(df[feats])
    assert np.std(scores) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
