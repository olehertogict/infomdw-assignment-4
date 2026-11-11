"""The complete mitigation pipeline and the experiment grid.

Per dataset and seed:
  1. stratified split (strata = label x protected group): 70% train / 30% test; the train part
     is split again, stratified, into 80% fit / 20% validation (Zemel et al. select
     hyper-parameters on a validation set);
  2. z-scoring fitted on the fit part (LFR input, logistic regression input, consistency kNN);
  3. baseline: logistic regression on the original fit data;
  4. LFR (pre-processing) for every grid setting, fitted on the fit part only, giving two
     pipelines on the held-out data:
       "LFR classifier"        the model's own prediction y_hat = sum_k M_nk w_k > 0.5
                               (what Zemel et al. evaluate; AIF360's transformed labels);
       "LR on LFR repr."       logistic regression trained on the LFR-transformed fit data
                               (AIF360's transformed features x_hat) with the original labels,
                               applied to the LFR-transformed test data;
  5. hyper-parameters selected on the validation set with the paper's two criteria
     (maximum Delta = yAcc - yDiscrim, and minimum yDiscrim);
  6. AIF360's unchanged LFR.fit, refitted with the selected setting and with AIF360's
     default setting, to show the effect of its finite-difference optimisation.
All measures are reported on validation and test predictions; the tables use the test part.
"""
import json
import logging
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# AIF360 logs a warning for every optional algorithm it cannot import, in every worker process.
logging.getLogger().setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "results" / "raw"

SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.30
VALIDATION_SIZE = 0.20          # share of the train part
K_PROTOTYPES = 10
AX = 0.01                       # Zemel et al. section 4.1: Ax = 0.01
AY_GRID = [0.1, 0.5, 1, 5, 10]  # Zemel et al.: Ay, Az from {0.1, 0.5, 1, 5, 10}, plus Az = 0
AZ_GRID = [0, 0.1, 0.5, 1, 5, 10]
AIF360_DEFAULT = {"k": 5, "Ax": 0.01, "Ay": 1.0, "Az": 50.0}
THRESHOLD = 0.5                 # AIF360 LFR.transform default
MAXITER = MAXFUN = 5000         # AIF360 LFR.fit defaults
LR_PARAMS = {"max_iter": 1000}  # scikit-learn defaults otherwise (L2, C = 1, lbfgs)

BASELINE = "LR baseline"
LFR_CLF = "LFR classifier"
LFR_LR = "LR on LFR repr."
QUICK = {"seeds": [0], "ay": [1], "az": [0, 1, 10]}   # smoke-test grid (REPRODUCE_QUICK=1)


def split(dataset, seed):
    from sklearn.model_selection import train_test_split
    y = dataset.labels.ravel().astype(int)
    s = dataset.protected_attributes[:, 0].astype(int)
    strata = 2 * y + s
    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=TEST_SIZE, stratify=strata, random_state=seed)
    fit, val = train_test_split(train, test_size=VALIDATION_SIZE, stratify=strata[train], random_state=seed)
    return {"fit": fit, "val": val, "test": test}


def _scaled_copy(dataset, scaler):
    d = dataset.copy(deepcopy=True)
    d.features = scaler.transform(dataset.features)
    return d


def _evaluate(part_ds, pa, scaled_features, y_pred):
    from metrics import fairness, paper_selection_scores, performance
    s = part_ds.protected_attributes[:, part_ds.protected_attribute_names.index(pa)]
    return {**fairness(part_ds, pa, y_pred, scaled_features), **performance(part_ds.labels, y_pred),
            **paper_selection_scores(part_ds.labels, y_pred, s)}


def _lr(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**LR_PARAMS).fit(X, y)


def _lfr_rows(model, parts, scaled, pa, base):
    """Evaluate both LFR pipelines of a fitted model on validation and test."""
    import lfr as L
    rows = []
    z_fit = model.transform(scaled["fit"], threshold=THRESHOLD).features
    downstream = _lr(z_fit, parts["fit"].labels.ravel())
    for part in ("val", "test"):
        y_lfr = (L.predict_proba(model, scaled[part].features) > THRESHOLD).astype(float)
        rows.append({**base, "method": LFR_CLF, "part": part,
                     **_evaluate(parts[part], pa, scaled[part].features, y_lfr)})
        z = model.transform(scaled[part], threshold=THRESHOLD).features
        rows.append({**base, "method": LFR_LR, "part": part,
                     **_evaluate(parts[part], pa, scaled[part].features, downstream.predict(z))})
    return rows


def select(rows, method, criterion):
    """Pick one grid setting from validation rows of ``method`` with the paper's criteria.

    'delta': maximise Delta = yAcc - yDiscrim; 'min_disc': minimise yDiscrim. The paper gives no
    tie-break; here ties go to higher validation accuracy, then lower Ay, then lower Az.
    """
    v = rows[(rows["method"] == method) & (rows["part"] == "val") & (rows["optimizer"] == "exact")].copy()
    if criterion == "delta":
        v = v.sort_values(["Delta", "yAcc", "Ay", "Az"], ascending=[False, False, True, True])
    elif criterion == "min_disc":
        v = v.sort_values(["yDiscrim", "yAcc", "Ay", "Az"], ascending=[True, False, True, True])
    else:
        raise ValueError(criterion)
    return v.iloc[0][["Ay", "Az"]].to_dict()


def run_seed(name, seed, quick=False):
    """All runs for one dataset and seed; returns a DataFrame (one row per method/setting/part)."""
    import lfr as L
    from aif360.algorithms.preprocessing import LFR
    from sklearn.preprocessing import StandardScaler
    import datasets

    ds, pa = datasets.load(name)
    idx = split(ds, seed)
    parts = {p: ds.subset(i) for p, i in idx.items()}
    scaler = StandardScaler().fit(parts["fit"].features)
    scaled = {p: _scaled_copy(d, scaler) for p, d in parts.items()}
    groups = {"unprivileged_groups": [{pa: 0.0}], "privileged_groups": [{pa: 1.0}]}
    common = {"dataset": name, "seed": seed, "n_fit": len(idx["fit"]), "n_val": len(idx["val"]),
              "n_test": len(idx["test"])}
    rows = []

    for part in ("val", "test"):
        rows.append({**common, "method": "labels (no model)", "part": part,
                     **_evaluate(parts[part], pa, scaled[part].features, parts[part].labels)})

    t0 = time.perf_counter()
    lr = _lr(scaled["fit"].features, parts["fit"].labels.ravel())
    for part in ("val", "test"):
        rows.append({**common, "method": BASELINE, "part": part, "fit_seconds": time.perf_counter() - t0,
                     **_evaluate(parts[part], pa, scaled[part].features, lr.predict(scaled[part].features))})

    ay_grid, az_grid = (QUICK["ay"], QUICK["az"]) if quick else (AY_GRID, AZ_GRID)
    for ay in ay_grid:
        for az in az_grid:
            t0 = time.perf_counter()
            model = L.ExactGradientLFR(**groups, k=K_PROTOTYPES, Ax=AX, Ay=ay, Az=az, seed=seed)
            model.fit(scaled["fit"], maxiter=MAXITER, maxfun=MAXFUN)
            base = {**common, "optimizer": "exact", "setting": "grid", "k": K_PROTOTYPES, "Ax": AX, "Ay": ay,
                    "Az": az, "fit_seconds": time.perf_counter() - t0, **model.optimizer_info}
            rows += _lfr_rows(model, parts, scaled, pa, base)

    df = pd.DataFrame(rows)
    chosen = select(df, LFR_CLF, "delta")
    settings = [("selected (max Delta)", {"k": K_PROTOTYPES, "Ax": AX, **chosen}), ("AIF360 default", AIF360_DEFAULT)]
    for label, hp in settings:
        t0 = time.perf_counter()
        model = L.fit_aif360(LFR(**groups, **hp, seed=seed), scaled["fit"], maxiter=MAXITER, maxfun=MAXFUN)
        base = {**common, "optimizer": "aif360", "setting": label, **hp,
                "fit_seconds": time.perf_counter() - t0, **model.optimizer_info}
        rows += _lfr_rows(model, parts, scaled, pa, base)
    return pd.DataFrame(rows)


def _task(args):
    name, seed, quick, out = args
    t0 = time.perf_counter()
    df = run_seed(name, seed, quick)
    df.to_csv(out, index=False)
    return name, seed, time.perf_counter() - t0


def label_metrics():
    """Measures of the original labels on the complete datasets, before any split or model."""
    from sklearn.preprocessing import StandardScaler
    from metrics import fairness
    import datasets
    rows = []
    for name in datasets.DATASETS:
        ds, pa = datasets.load(name)
        z = StandardScaler().fit_transform(ds.features)
        rows.append({"dataset": name, "rows": ds.features.shape[0], "features": ds.features.shape[1],
                     "protected": pa, "privileged_share": float(ds.protected_attributes.mean()),
                     **fairness(ds, pa, ds.labels, z)})
    return pd.DataFrame(rows)


def environment():
    import aif360, scipy, sklearn
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(), "aif360": aif360.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__}


def run_all(datasets_=None, seeds=None, quick=False, workers=None, out_dir=RAW_DIR, reuse=False, log=print):
    """Run every (dataset, seed) task in parallel processes and write results/raw/.

    reuse=True keeps per-task files that already exist (for resuming an interrupted run).
    """
    import datasets
    datasets_ = datasets_ or datasets.DATASETS
    seeds = seeds or (QUICK["seeds"] if quick else SEEDS)
    out_dir = Path(out_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    for name in datasets_:
        datasets.download(name)
    label_metrics().to_csv(out_dir / "label_metrics.csv", index=False)
    tasks = []
    for name in datasets_:
        for seed in seeds:
            out = out_dir / "runs" / f"{name}_seed{seed}.csv"
            if not (reuse and out.exists()):
                tasks.append((name, seed, quick, out))
    # One BLAS thread per process; parallelism comes from running tasks side by side.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    workers = workers or int(os.environ.get("REPRODUCE_WORKERS", max(1, min(len(tasks), (os.cpu_count() or 2) - 2))))
    t0 = time.perf_counter()
    if tasks:
        # Longest tasks (Adult) first
        tasks.sort(key=lambda t: datasets.DATASETS.index(t[0]))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, seed, seconds in pool.map(_task, tasks):
                log(f"{name} seed {seed}: {seconds:.0f} s")
    runs = pd.concat([pd.read_csv(out_dir / "runs" / f"{n}_seed{s}.csv") for n in datasets_ for s in seeds],
                     ignore_index=True)
    runs.to_csv(out_dir / "runs.csv", index=False)
    env_path = out_dir / "environment.json"
    if tasks or not env_path.exists():   # a fully reused run keeps the record of the run that made the files
        env = environment() | {"workers": workers, "wall_seconds": round(time.perf_counter() - t0, 1),
                               "tasks_run": len(tasks), "quick": quick, "datasets": list(datasets_),
                               "seeds": list(seeds)}
        env_path.write_text(json.dumps(env, indent=2) + "\n")
    return runs


if __name__ == "__main__":
    quick = os.environ.get("REPRODUCE_QUICK") == "1"
    run_all(quick=quick)
