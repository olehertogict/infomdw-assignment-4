"""The exact-gradient LFR must optimise exactly AIF360's objective. Run: uv run pytest"""
import sys
from pathlib import Path

import numpy as np
import pytest
from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.preprocessing.lfr_helpers import helpers as aif360_helpers
from aif360.datasets import BinaryLabelDataset
import pandas as pd
from scipy.optimize import approx_fprime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lfr as L  # noqa: E402


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
