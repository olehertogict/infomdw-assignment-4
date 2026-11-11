"""Fairness and performance measures.

All fairness measures are computed with AIF360's BinaryLabelDatasetMetric on a copy of the
dataset whose labels are replaced by the values being assessed (original labels or final
predictions):

* DP_diff  = P(y=1 | unprivileged) - P(y=1 | privileged)   (statistical_parity_difference; 0 is parity)
* DIR      = P(y=1 | unprivileged) / P(y=1 | privileged)   (disparate_impact; 1 is parity)
* consistency = 1 - mean_n |y_n - mean_{j in kNN(x_n)} y_j|, k = 5 (Zemel et al.'s yNN)

Consistency is computed on z-scored features (AIF360 uses the features as given; on raw
features distances are dominated by large-range columns such as capital-gain). Note that
AIF360 fits the neighbour search on the same rows it queries, so each row is its own first
neighbour: k = 5 means the row itself plus its 4 nearest other rows. The raw-feature variant is
also returned as ``consistency_raw``.
"""
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.metrics import accuracy_score, balanced_accuracy_score

K_NEIGHBOURS = 5


def _with_labels(dataset, labels, features=None):
    d = dataset.copy(deepcopy=True)
    d.labels = np.asarray(labels, dtype=float).reshape(-1, 1)
    if features is not None:
        d.features = np.asarray(features, dtype=float)
    return d


def _metric(dataset, pa):
    return BinaryLabelDatasetMetric(dataset, unprivileged_groups=[{pa: 0.0}], privileged_groups=[{pa: 1.0}])


def fairness(dataset, pa, labels, scaled_features):
    """DP_diff, DIR and consistency of ``labels`` (0/1, aligned with the rows of ``dataset``)."""
    m = _metric(_with_labels(dataset, labels, scaled_features), pa)
    m_raw = _metric(_with_labels(dataset, labels), pa)
    s = dataset.protected_attributes[:, dataset.protected_attribute_names.index(pa)]
    labels = np.asarray(labels, dtype=float).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        dir_ = m.disparate_impact()
    return {
        "DP_diff": float(m.statistical_parity_difference()),
        "DIR": float(dir_) if np.isfinite(dir_) else float("nan"),
        "consistency": float(np.ravel(m.consistency(n_neighbors=K_NEIGHBOURS))[0]),
        "consistency_raw": float(np.ravel(m_raw.consistency(n_neighbors=K_NEIGHBOURS))[0]),
        "pos_rate": float(labels.mean()),
        "pos_rate_unpriv": float(labels[s == 0].mean()),
        "pos_rate_priv": float(labels[s == 1].mean()),
    }


def performance(y_true, y_pred):
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}


def paper_selection_scores(y_true, y_pred, s):
    """Zemel et al. section 4.2: yAcc, yDiscrim = |P(y=1|s=1) - P(y=1|s=0)| and Delta = yAcc - yDiscrim."""
    y_true, y_pred, s = np.ravel(y_true), np.ravel(y_pred), np.ravel(s)
    acc = float(np.mean(y_true == y_pred))
    disc = float(abs(y_pred[s == 1].mean() - y_pred[s == 0].mean()))
    return {"yAcc": acc, "yDiscrim": disc, "Delta": acc - disc}
