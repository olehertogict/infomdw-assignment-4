"""Learning Fair Representations (Zemel et al., ICML 2013) as implemented in AIF360 0.6.1.

Two ways to fit the same model are used:

* ``fit_aif360``: AIF360's own ``LFR.fit``, unchanged. It calls ``scipy.optimize.fmin_l_bfgs_b``
  with ``approx_grad=True``: every gradient is a forward finite difference, i.e. one objective
  evaluation per parameter (k + k*D parameters). The default limit ``maxfun=5000`` counts these
  evaluations too, so on the datasets used here L-BFGS stops after only a handful of iterations.
  The call is wrapped only to record SciPy's convergence report, which AIF360 discards.

* ``ExactGradientLFR``: a subclass of AIF360's ``LFR`` whose ``fit`` minimises the identical
  objective (``aif360...lfr_helpers.LFR_optim_objective``) with the identical initialisation,
  bounds and limits, but passes the analytic gradient below to L-BFGS-B. ``transform`` is
  AIF360's, unchanged. tests/test_lfr.py checks that the objective equals AIF360's and that the
  gradient matches finite differences.

AIF360 objective (with u = unprivileged, p = privileged group; M = softmax(-||x - v_k||_2);
x_hat = M V; y_hat = clip(M w, eps, 1 - eps)):

  L = Ax * [mean((x_hat_u - x_u)^2) + mean((x_hat_p - x_p)^2)]
    + Ay * mean(cross-entropy(y, y_hat))           (over all N rows)
    + Az * mean_k |mean_n M_u[n, k] - mean_n M_p[n, k]|
"""
from contextlib import contextmanager

import numpy as np
import scipy.optimize as optim
from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.preprocessing import lfr as aif360_lfr_module
from scipy.spatial.distance import cdist
from scipy.special import softmax

EPS = np.finfo(float).eps


def _group_terms(V, w, x):
    dist = cdist(x, V)                       # (n, k) Euclidean distances, as in AIF360
    M = softmax(-dist, axis=1)               # (n, k)
    x_hat = M @ V                            # (n, D)
    lin = M @ w                              # (n,)
    y_hat = np.clip(lin, EPS, 1.0 - EPS)
    return dist, M, x_hat, lin, y_hat


def objective_and_gradient(params, x_u, x_p, y_u, y_p, k, Ax, Ay, Az):
    """Value and gradient of AIF360's LFR_optim_objective at ``params`` = [w (k), V (k*D)]."""
    D = x_u.shape[1]
    w = params[:k]
    V = params[k:].reshape(k, D)
    N = len(y_u) + len(y_p)

    groups = []
    for x, y in ((x_u, y_u), (x_p, y_p)):
        groups.append((x, np.asarray(y, dtype=float).ravel(), *_group_terms(V, w, x)))

    L_x = sum(np.mean((x_hat - x) ** 2) for x, _, _, _, x_hat, _, _ in groups)
    cross_entropy = sum(-np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
                        for _, y, _, _, _, _, y_hat in groups)
    L_y = cross_entropy / N
    diff = groups[0][3].mean(axis=0) - groups[1][3].mean(axis=0)   # (k,)
    L_z = np.mean(np.abs(diff))
    loss = Ax * L_x + Ay * L_y + Az * L_z

    grad_w = np.zeros(k)
    grad_V = np.zeros((k, D))
    sign = np.sign(diff)
    for g, (x, y, dist, M, x_hat, lin, y_hat) in enumerate(groups):
        n = len(x)
        # reconstruction term
        R = (2 * Ax / (n * D)) * (x_hat - x)                        # dL/dx_hat
        G_M = R @ V.T                                                # dL/dM
        grad_V += M.T @ R
        # prediction term (np.clip has zero derivative outside [eps, 1 - eps])
        inside = (lin > EPS) & (lin < 1 - EPS)
        dy = (Ay / N) * (-(y / y_hat) + (1 - y) / (1 - y_hat)) * inside
        G_M += np.outer(dy, w)
        grad_w += M.T @ dy
        # parity term
        G_M += ((1 if g == 0 else -1) * Az / (k * n)) * sign[None, :]
        # softmax(-dist) backward, then distance backward
        G_A = M * (G_M - np.sum(G_M * M, axis=1, keepdims=True))   # dL/d(-dist)
        W = -G_A / np.maximum(dist, 1e-12)                          # dL/ddist / dist
        grad_V += W.sum(axis=0)[:, None] * V - W.T @ x
    return loss, np.concatenate([grad_w, grad_V.ravel()])


def _split_by_group(model, dataset):
    s = dataset.protected_attributes[:, dataset.protected_attribute_names.index(model.protected_attribute_name)]
    u = s == model.unprivileged_group_protected_attribute_value
    p = s == model.privileged_group_protected_attribute_value
    return dataset.features[u], dataset.features[p], dataset.labels[u, 0], dataset.labels[p, 0]


class ExactGradientLFR(LFR):
    """AIF360's LFR with the analytic gradient passed to L-BFGS-B; everything else unchanged."""

    def fit(self, dataset, maxiter=5000, maxfun=5000):
        if self.seed is not None:
            np.random.seed(self.seed)
        x_u, x_p, y_u, y_p = _split_by_group(self, dataset)
        self.features_dim = dataset.features.shape[1]
        x0 = np.random.uniform(size=self.k + self.features_dim * self.k)          # as AIF360
        bounds = [(0, 1)] * self.k + [(None, None)] * self.features_dim * self.k  # as AIF360
        x, f, info = optim.fmin_l_bfgs_b(objective_and_gradient, x0=x0, args=(x_u, x_p, y_u, y_p, self.k,
                                         self.Ax, self.Ay, self.Az), bounds=bounds, maxfun=maxfun,
                                         maxiter=maxiter)
        self.learned_model = x
        self.w = x[:self.k]
        self.prototypes = x[self.k:].reshape((self.k, self.features_dim))
        self.optimizer_info = _info(f, info)
        return self


def _info(f, info):
    return {"objective": float(f), "iterations": int(info["nit"]), "evaluations": int(info["funcalls"]),
            "converged": int(info["warnflag"]) == 0, "stop_reason": str(info["task"])}


@contextmanager
def _record_lbfgs():
    """Temporarily wrap the fmin_l_bfgs_b used inside AIF360's lfr module to keep SciPy's report."""
    original = aif360_lfr_module.optim.fmin_l_bfgs_b
    record = {}

    def wrapper(*args, **kwargs):
        x, f, info = original(*args, **kwargs)
        record.update(_info(f, info))
        return x, f, info

    aif360_lfr_module.optim.fmin_l_bfgs_b = wrapper
    try:
        yield record
    finally:
        aif360_lfr_module.optim.fmin_l_bfgs_b = original


def fit_aif360(model, dataset, maxiter=5000, maxfun=5000):
    """AIF360's unchanged LFR.fit; returns the model with SciPy's convergence report attached."""
    with _record_lbfgs() as record:
        model.fit(dataset, maxiter=maxiter, maxfun=maxfun)
    model.optimizer_info = dict(record)
    return model


def prototype_probabilities(model, features):
    """M = P(Z = k | x) for each row, as used inside AIF360's transform."""
    return softmax(-cdist(features, model.prototypes), axis=1)


def predict_proba(model, features):
    """LFR's own prediction y_hat = sum_k M_nk w_k (clipped as in AIF360)."""
    return np.clip(prototype_probabilities(model, features) @ model.w, EPS, 1 - EPS)
