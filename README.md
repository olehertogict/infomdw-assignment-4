# INFOMDW – Assignment 4: evaluating Learning Fair Representations

Evaluation of the bias-mitigation algorithm of

> R. Zemel, Y. Wu, K. Swersky, T. Pitassi and C. Dwork. *Learning Fair Representations.* ICML 2013,
> PMLR 28(3):325–333. [paper](https://proceedings.mlr.press/v28/zemel13.html) ·
> [PDF](https://proceedings.mlr.press/v28/zemel13.pdf) ·
> [AIF360 implementation](https://github.com/Trusted-AI/AIF360) · [our fork](https://github.com/olehertogict/AIF360)

Assignment: [Assignment 4](https://hqahtan.github.io/Teaching/infomdw/assignments/Assignment_4_26_27.html).

LFR is AIF360 0.6.1's pre-processing algorithm `aif360.algorithms.preprocessing.LFR`. AIF360 is installed from
our fork [olehertogict/AIF360](https://github.com/olehertogict/AIF360), included as the git submodule
`external/AIF360`. Its branch `infomdw-a4` is release `v0.6.1` with
[instructions](https://github.com/olehertogict/AIF360/blob/infomdw-a4/INFOMDW_A4.md) added; the code is unchanged
and byte-identical to the PyPI wheel 0.6.1. Exact package versions are in `uv.lock`. The notebook [`fairness_lfr.ipynb`](fairness_lfr.ipynb)
runs the pipeline on Adult, German and COMPAS:
1. measure the original labels;
2. make a stratified train/test split;
3. learn LFR on the training data;
4. train a classifier on the transformed data;
5. measure DP_diff, DIR, consistency and (balanced) accuracy on the test data.

AIF360's `LFR.fit` approximates every gradient by finite differences, and its evaluation limit
(`maxfun=5000`) counts those evaluations. L-BFGS therefore stops after 4–30 iterations, and the model mostly
predicts a single class. `lfr.ExactGradientLFR` minimises the same objective with its analytic gradient.
Both versions are reported. The saved results in `results/` were produced at
[commit `cb171c2`](https://github.com/olehertogict/infomdw-assignment-4/tree/cb171c2f97bbcac0f452d866dbe089f1990a90cc).
Later commits restructure files and documentation and install the same AIF360 code from the fork; they do not
change the computations or the saved results.

## Setup and running

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), `git` and internet access for the first
run. macOS on Apple Silicon is tested. Linux is expected to work but has not been tested.

```bash
git clone --recurse-submodules https://github.com/olehertogict/infomdw-assignment-4
cd infomdw-assignment-4
bash reproduce.sh
```

`reproduce.sh` does the following:
- fetches the AIF360 fork into `external/AIF360` if the clone was made without `--recurse-submodules`;
- downloads Python 3.12.15 and the locked dependencies if needed, and builds AIF360 from the fork;
- checks the three datasets against recorded SHA256 hashes;
- runs the tests;
- executes the notebook, which runs all 480 LFR fits and writes `results/`.

The saved experiments took about 50 minutes on an M1 Pro with 6 parallel processes. A complete
`reproduce.sh` from a fresh copy took 1 h 24 min and reproduced every saved value except running times.
A rerun overwrites `results/`.

For a quick check, run `REPRODUCE_QUICK=1 bash reproduce.sh` (one seed, three settings, a few minutes)
or `uv run --locked pytest -q`.

## What is and is not evaluated

| Part of the paper | Here |
|---|---|
| LFR model and objective (§2, Eq. 1–11) | Yes, as implemented in AIF360 0.6.1. AIF360 has no learned feature weights α (Eq. 12), uses a plain Euclidean distance and takes means instead of sums. |
| Optimisation: L-BFGS (§4.1) | Yes: AIF360's `fit` as released (finite-difference gradient) and `ExactGradientLFR` (same objective, analytic gradient). |
| Hyper-parameter grid and model selection (§4.1–4.2) | Yes: the paper's grid, k = 10 (the paper does not report k), both selection criteria on a validation set. |
| Measures yAcc, yDiscrim, yNN (Eq. 13–15) | Yes, plus the assignment's DP_diff, DIR and balanced accuracy. Consistency is AIF360's yNN with the intended averaging. |
| Datasets German and Adult | Yes, with AIF360's preprocessing. |
| Dataset Heritage Health | No: the competition data is not publicly available. COMPAS is used as the third dataset. |
| Baselines FNB and RLR (Figure 1) | Not run. The unmitigated logistic regression is the baseline. |
| Figure 3 (predicting S) and transfer learning (§4.3) | Not run; outside the assignment's pipeline. |

## Datasets

| Dataset | Source | Rows | Protected (privileged) | Favourable label |
|---|---|---|---|---|
| Adult | [UCI 2](https://archive.ics.uci.edu/dataset/2), rows with `?` removed | 45,222 | sex (male) | income > 50K |
| German | [UCI 144](https://archive.ics.uci.edu/dataset/144) | 1,000 | age (> 25) | good credit |
| COMPAS | [ProPublica](https://github.com/propublica/compas-analysis) at `bafff5d`, ProPublica's filters | 6,172 | race (Caucasian) | no recidivism within two years |

All three are built with AIF360's `StandardDataset`, using the options of AIF360's own dataset classes and
one protected attribute each. The deviations are commented in `datasets.py`: COMPAS without `c_charge_desc`,
and German keeping `personal_status`.

## Metric and parameters

All measures come from AIF360. They are applied to the original labels before any split, and to the final
test predictions.
- **DP_diff** = P(ŷ=1 | unprivileged) − P(ŷ=1 | privileged).
- **DIR** = P(ŷ=1 | unprivileged) / P(ŷ=1 | privileged).
- **Consistency** = Zemel et al.'s yNN with 5 neighbours, on z-scored features.
- **Accuracy** and **balanced accuracy**.

| Parameter | Value |
|---|---|
| Split | Stratified on label × protected group, 70/30 train/test. The train part is split again 80/20 into fit/validation. Seeds 0–4 |
| LFR | k = 10; A_x = 0.01; A_y ∈ {0.1, 0.5, 1, 5, 10}; A_z ∈ {0, 0.1, 0.5, 1, 5, 10} (the paper's grid); L-BFGS-B with maxiter = maxfun = 5000; threshold 0.5 |
| Selection | Per seed, the highest validation Delta = yAcc − yDiscrim (the paper's criterion). Lowest yDiscrim is also reported |
| Final classifiers | LFR's own prediction, and logistic regression on the LFR-transformed data |
| Baseline | Logistic regression on the original z-scored features (`max_iter=1000`, C = 1) |
| AIF360 as released | `LFR.fit` unchanged, with the selected setting and with AIF360's default (k = 5, A_z = 50) |

AIF360's LFR differs from the paper:
- it learns no feature weights α;
- it uses a plain Euclidean distance;
- it takes means instead of sums;
- its default A_z = 50 lies outside the paper's grid.

The paper does not report k. These experiments keep AIF360's objective and only compute its gradient exactly.
AIF360 itself is not modified: `lfr.ExactGradientLFR` subclasses its `LFR` and replaces only `fit`.

AIF360 0.6.1 needs pandas < 3 and SciPy < 1.18. The saved runs used an Apple M1 Pro with 16 GiB RAM,
macOS 27.0.1, Python 3.12.15, pandas 2.3.3, SciPy 1.17.1 and scikit-learn 1.9.1. The 19 correctness tests pass.

## Results

[`results/results.md`](results/results.md) is generated by `make_tables.py` from `results/raw/`:
- `runs.csv`: every run, on validation and test data;
- `label_metrics.csv`: the original labels;
- `environment.json`: the environment of the saved run.

`results/summary.csv` has the means per pipeline, and `results/tradeoff.png` shows the trade-off figure.

With exact gradients, LFR brings |DP_diff| below 0.03 and DIR above 0.95 on all three datasets. It costs
0.003–0.075 in balanced accuracy. AIF360 as released predicts a single class on Adult and German.

## Files

```
datasets.py          datasets: download, SHA256 check, AIF360 construction
lfr.py               AIF360's LFR: unchanged fit (with SciPy's report) and ExactGradientLFR
external/AIF360      AIF360 v0.6.1, our fork with instructions (git submodule; code unchanged)
run_experiments.py   pipeline, measures, grid, model selection, all runs -> results/raw/
make_tables.py       results/raw/ -> results/results.md, summary.csv, tradeoff.png
fairness_lfr.ipynb   the notebook: the pipeline step by step and all runs, with outputs
reproduce.sh         complete setup, checks and notebook execution in one command
tests/               correctness tests (uv run pytest)
```
