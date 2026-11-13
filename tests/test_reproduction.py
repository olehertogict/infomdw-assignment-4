"""Correctness checks: the exact-gradient LFR against AIF360, the measures, the data, the split and
the model selection. Run: uv run pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.preprocessing.lfr_helpers import helpers as aif360_helpers
from aif360.datasets import BinaryLabelDataset
from scipy.optimize import approx_fprime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datasets as D  # noqa: E402
import lfr as L  # noqa: E402
import run_experiments as E  # noqa: E402


def toy_problem(seed=0, n=60, D=4, k=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, D))
    s = rng.integers(0, 2, size=n)
    y = (x[:, 0] + 0.5 * s + rng.normal(scale=0.5, size=n) > 0).astype(float)
    params = np.concatenate([rng.uniform(size=k), rng.normal(size=k * D)])
    return x, s, y, params, k


def toy_dataset(x, s, y):
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    df["s"], df["y"] = s.astype(float), y
    return BinaryLabelDataset(df=df, label_names=["y"], protected_attribute_names=["s"],
                              favorable_label=1.0, unfavorable_label=0.0)


@pytest.mark.parametrize("weights", [(0.01, 1.0, 50.0), (1.0, 0.1, 0.0), (0.5, 5.0, 10.0)])
def test_objective_equals_aif360(weights):
    x, s, y, params, k = toy_problem()
    Ax, Ay, Az = weights
    ours, _ = L.objective_and_gradient(params, x[s == 0], x[s == 1], y[s == 0], y[s == 1], k, Ax, Ay, Az)
    aif360_helpers.LFR_optim_objective.steps = 0
    theirs = aif360_helpers.LFR_optim_objective(params, x[s == 0], x[s == 1], y[s == 0], y[s == 1],
                                                k, Ax, Ay, Az, 250, 0)
    assert ours == pytest.approx(theirs, rel=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_matches_finite_differences(seed):
    x, s, y, params, k = toy_problem(seed)
    args = (x[s == 0], x[s == 1], y[s == 0], y[s == 1], k, 0.5, 1.0, 2.0)
    _, grad = L.objective_and_gradient(params, *args)
    numeric = approx_fprime(params, lambda p: L.objective_and_gradient(p, *args)[0], 1e-7)
    assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-6)


def test_exact_lfr_reaches_lower_objective_than_aif360_and_reuses_transform():
    x, s, y, _, _ = toy_problem(n=200, D=6)
    ds = toy_dataset(x, s, y)
    args = dict(unprivileged_groups=[{"s": 0}], privileged_groups=[{"s": 1}], k=4, Ax=0.01, Ay=1.0, Az=1.0, seed=0)
    exact = L.ExactGradientLFR(**args).fit(ds)
    released = L.fit_aif360(LFR(**args), ds, maxfun=200)
    assert exact.optimizer_info["objective"] <= released.optimizer_info["objective"] + 1e-9
    assert released.optimizer_info["stop_reason"].startswith("STOP: TOTAL NO. OF F,G EVALUATIONS")
    # transform() is AIF360's own: its labels are LFR's thresholded predictions
    out = exact.transform(ds)
    assert np.array_equal(out.labels.ravel(), (L.predict_proba(exact, ds.features) > 0.5).astype(float))

def toy():
    # 4 unprivileged (s=0) and 6 privileged (s=1) rows on a line
    x = np.arange(10, dtype=float).reshape(-1, 1)
    s = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=float)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1], dtype=float)
    df = pd.DataFrame({"x": x.ravel(), "s": s, "y": y})
    ds = BinaryLabelDataset(df=df, label_names=["y"], protected_attribute_names=["s"],
                            favorable_label=1.0, unfavorable_label=0.0)
    return ds, x, s, y


def test_dp_diff_and_dir_follow_the_definitions():
    ds, x, s, y = toy()
    out = E.fairness(ds, "s", y, ds.features)
    p_u, p_p = 1 / 4, 5 / 6
    assert out["DP_diff"] == pytest.approx(p_u - p_p)
    assert out["DIR"] == pytest.approx(p_u / p_p)
    assert (out["pos_rate_unpriv"], out["pos_rate_priv"]) == pytest.approx((p_u, p_p))


def test_consistency_equals_yNN_with_the_row_itself_among_its_5_neighbours():
    ds, x, s, y = toy()
    features = np.column_stack([x.ravel(), s])
    expected = []
    for i in range(len(y)):
        nearest = np.argsort(np.linalg.norm(features - features[i], axis=1), kind="stable")[:5]
        expected.append(abs(y[i] - y[nearest].mean()))
    out = E.fairness(ds, "s", y, features)
    assert out["consistency"] == pytest.approx(1 - np.mean(expected))


def test_dir_is_nan_when_privileged_group_gets_no_positive_prediction():
    ds, x, s, y = toy()
    out = E.fairness(ds, "s", (s == 0).astype(float), ds.features)
    assert np.isnan(out["DIR"]) and out["DP_diff"] == pytest.approx(1.0)


def test_paper_selection_scores():
    _, _, s, y = toy()
    pred = np.ones_like(y)
    out = E.paper_selection_scores(y, pred, s)
    assert out == pytest.approx({"yAcc": 0.6, "yDiscrim": 0.0, "Delta": 0.6})   # 6 of 10 labels are 1


@pytest.mark.parametrize("name,rows", [("adult", 45222), ("german", 1000), ("compas", 6172)])
def test_datasets_have_expected_size_and_binary_coding(name, rows):
    ds, pa = D.load(name)
    assert ds.features.shape[0] == rows
    assert pa == D.DESCRIPTION[name]["protected"]
    assert set(np.unique(ds.labels)) == {0.0, 1.0} and set(np.unique(ds.protected_attributes)) == {0.0, 1.0}


def test_compas_favourable_label_is_no_recidivism():
    ds, _ = D.load("compas")
    raw = pd.read_csv(D.raw_path("compas", "compas-scores-two-years.csv"), index_col="id")
    recid = raw.loc[ds.instance_names_as_int if hasattr(ds, "instance_names_as_int") else
                    [int(float(i)) for i in ds.instance_names], "two_year_recid"].to_numpy()
    assert np.array_equal(ds.labels.ravel(), (recid == 0).astype(float))


def test_split_is_stratified_and_disjoint():
    ds, _ = D.load("german")
    idx = E.split(ds, seed=0)
    all_idx = np.concatenate(list(idx.values()))
    assert len(all_idx) == len(set(all_idx)) == 1000
    strata = 2 * ds.labels.ravel() + ds.protected_attributes.ravel()
    overall = np.bincount(strata.astype(int)) / len(strata)
    for part in idx.values():
        assert np.allclose(np.bincount(strata[part].astype(int), minlength=4) / len(part), overall, atol=0.01)


def test_changed_cached_data_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "DATA_DIR", tmp_path)
    path = D.raw_path("german", "german.data")
    path.parent.mkdir(parents=True)
    path.write_text("not the German credit data\n")
    with pytest.raises(ValueError, match="SHA256"):
        D.download("german")


def test_selection_uses_validation_rows_only():
    rows = pd.DataFrame([
        {"method": E.LFR_CLF, "part": "val", "optimizer": "exact", "Ay": 1, "Az": 0, "Delta": 0.6, "yAcc": 0.8, "yDiscrim": 0.2},
        {"method": E.LFR_CLF, "part": "val", "optimizer": "exact", "Ay": 1, "Az": 1, "Delta": 0.7, "yAcc": 0.75, "yDiscrim": 0.05},
        {"method": E.LFR_CLF, "part": "val", "optimizer": "exact", "Ay": 1, "Az": 10, "Delta": 0.7, "yAcc": 0.70, "yDiscrim": 0.0},
        {"method": E.LFR_CLF, "part": "test", "optimizer": "exact", "Ay": 5, "Az": 5, "Delta": 0.99, "yAcc": 0.99, "yDiscrim": 0.0},
    ])
    assert E.select(rows, E.LFR_CLF, "delta") == {"Ay": 1, "Az": 1}      # tie on Delta -> higher accuracy
    assert E.select(rows, E.LFR_CLF, "min_disc") == {"Ay": 1, "Az": 10}


def test_one_quick_run_end_to_end():
    df = E.run_seed("german", 0, quick=True)
    test = df[df["part"] == "test"]
    assert set(test["method"]) == {"labels (no model)", E.BASELINE, E.LFR_CLF, E.LFR_LR}
    exact = test[(test["optimizer"] == "exact") & (test["method"] == E.LFR_CLF)]
    assert len(exact) == len(E.QUICK["ay"]) * len(E.QUICK["az"]) and exact["converged"].all()
    released = test[(test["optimizer"] == "aif360") & (test["method"] == E.LFR_CLF)]
    assert set(released["setting"]) == {"selected (max Delta)", "AIF360 default"}
    assert not released["converged"].any()                      # stops on the evaluation limit
    labels = test[test["method"] == "labels (no model)"].iloc[0]
    assert labels["accuracy"] == 1.0 and labels["n_test"] == 300
