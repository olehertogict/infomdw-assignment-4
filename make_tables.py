"""results/raw/ -> results/results.md, results/summary.csv and results/tradeoff.png.

Run: uv run python make_tables.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_experiments as E

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results"
NAMES = {"adult": "Adult (sex)", "german": "German (age)", "compas": "COMPAS (race)"}
MEASURES = ["DP_diff", "DIR", "consistency", "accuracy", "balanced_accuracy", "pos_rate"]


def load():
    return pd.read_csv(RAW / "runs.csv"), pd.read_csv(RAW / "label_metrics.csv")


def selected_rows(runs, part="test"):
    """Per dataset and seed, the rows of every reported pipeline on ``part``."""
    out = []
    for (name, seed), g in runs.groupby(["dataset", "seed"], sort=False):
        def pick(mask, label, **extra):
            r = g[mask & (g["part"] == part)]
            assert len(r) == 1, (name, seed, label, len(r))
            out.append(r.iloc[0].to_dict() | {"pipeline": label} | extra)
        pick(g["method"] == "labels (no model)", "Test labels (no model)")
        pick(g["method"] == E.BASELINE, "LR baseline (no mitigation)")
        for method in (E.LFR_CLF, E.LFR_LR):
            for criterion, crit_label in (("delta", "max Delta"), ("min_disc", "min yDiscrim")):
                hp = E.select(g, method, criterion)
                pick((g["method"] == method) & (g["optimizer"] == "exact") & (g["Ay"] == hp["Ay"]) & (g["Az"] == hp["Az"]),
                     f"{method}, exact gradient, {crit_label}", criterion=criterion)
            for setting in ("selected (max Delta)", "AIF360 default"):
                pick((g["method"] == method) & (g["optimizer"] == "aif360") & (g["setting"] == setting),
                     f"{method}, AIF360 as released, {setting}")
    return pd.DataFrame(out)


def mean_sd(df, cols, by):
    agg = df.groupby(by, sort=False)[cols].agg(["mean", "std", "count"])
    return agg


def fmt(mean, sd, digits=3):
    if pd.isna(mean):
        return "–"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if pd.notna(sd) else f"{mean:.{digits}f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


ORDER = ["Test labels (no model)", "LR baseline (no mitigation)",
         f"{E.LFR_CLF}, exact gradient, max Delta", f"{E.LFR_LR}, exact gradient, max Delta",
         f"{E.LFR_CLF}, exact gradient, min yDiscrim", f"{E.LFR_LR}, exact gradient, min yDiscrim",
         f"{E.LFR_CLF}, AIF360 as released, selected (max Delta)", f"{E.LFR_LR}, AIF360 as released, selected (max Delta)",
         f"{E.LFR_CLF}, AIF360 as released, AIF360 default", f"{E.LFR_LR}, AIF360 as released, AIF360 default"]


def summary(sel):
    rows = []
    for name in NAMES:
        for pipe in ORDER:
            g = sel[(sel["dataset"] == name) & (sel["pipeline"] == pipe)]
            if g.empty:
                continue
            r = {"dataset": name, "pipeline": pipe, "seeds": len(g)}
            for m in MEASURES:
                r[m + "_mean"], r[m + "_sd"] = g[m].mean(), g[m].std()
            r["DIR_undefined"] = int(g["DIR"].isna().sum())
            rows.append(r)
    return pd.DataFrame(rows)


def tradeoff_figure(runs, sel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted, grid, surface = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    test = runs[(runs["part"] == "test")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), facecolor=surface)
    for ax, name in zip(axes, NAMES):
        ax.set_facecolor(surface)
        t = test[test["dataset"] == name]
        grid_rows = t[t["optimizer"] == "exact"].groupby(["method", "Ay", "Az"])[["DP_diff", "balanced_accuracy"]].mean()
        for method, color, label in ((E.LFR_CLF, blue, "LFR classifier (grid)"), (E.LFR_LR, orange, "LR on LFR repr. (grid)")):
            g = grid_rows.loc[method]
            ax.scatter(g["DP_diff"].abs(), g["balanced_accuracy"], s=26, color=color, edgecolor=surface,
                       linewidth=1.2, label=label, zorder=3)
        s = sel[sel["dataset"] == name].groupby("pipeline")[["DP_diff", "balanced_accuracy"]].mean()
        y_min = min(grid_rows["balanced_accuracy"].min(), s["balanced_accuracy"].min())
        refs = [("LR baseline (no mitigation)", "D", ink, "LR baseline", (-8, -13), "right"),
                (f"{E.LFR_CLF}, exact gradient, max Delta", "o", blue, "LFR, selected", (8, 5), "left"),
                (f"{E.LFR_CLF}, AIF360 as released, selected (max Delta)", "X", muted, "AIF360 as released",
                 (8, -13), "left")]
        for pipe, marker, color, text, offset, ha in refs:
            x, y = abs(s.loc[pipe, "DP_diff"]), s.loc[pipe, "balanced_accuracy"]
            if offset[1] < 0 and y - y_min < 0.01:   # no room below the lowest point: label above instead
                offset = (offset[0], 5)
            ax.scatter([x], [y], s=70, marker=marker, facecolor="none" if marker == "o" else color,
                       edgecolor=color, linewidth=1.6, zorder=4)
            ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", fontsize=7.5, color=muted, ha=ha, zorder=5,
                        bbox={"boxstyle": "round,pad=0.15", "fc": surface, "ec": "none", "alpha": 0.85})
        ax.set_title(NAMES[name], fontsize=10, color=ink, loc="left")
        ax.set_xlabel("|DP_diff| on test set (lower is fairer)", fontsize=8, color=muted)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=7.5)
    axes[0].set_ylabel("Balanced accuracy on test set", fontsize=8, color=muted)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fairness–accuracy trade-off of the 30 LFR settings and the selected setting (exact gradient; test set, mean over 5 seeds)", fontsize=10.5,
                 color=ink, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


def main():
    runs, labels = load()
    env = json.loads((RAW / "environment.json").read_text())
    sel = selected_rows(runs)
    summ = summary(sel)
    summ.to_csv(OUT / "summary.csv", index=False)
    tradeoff_figure(runs, sel, OUT / "tradeoff.png")

    md = ["# Results", "",
          "Generated by `make_tables.py` from `results/raw/`. DP_diff = P(ŷ=1 | unprivileged) − P(ŷ=1 | privileged) "
          "(0 = parity); DIR = P(ŷ=1 | unprivileged) / P(ŷ=1 | privileged) (1 = parity); consistency = yNN with "
          "k = 5 on z-scored features (1 = similar rows get the same label). Values on held-out test sets are "
          f"mean ± standard deviation over {len(env['seeds'])} stratified splits (seeds {env['seeds']}).", ""]

    md += ["## 1. Original labels, before any split, mitigation or classifier", ""]
    rows = []
    for _, r in labels.iterrows():
        rows.append([NAMES[r["dataset"]], f"{r['rows']:,}", r["features"], f"{r['privileged_share']:.3f}",
                     f"{r['pos_rate_unpriv']:.3f}", f"{r['pos_rate_priv']:.3f}", f"{r['DP_diff']:.3f}", f"{r['DIR']:.3f}",
                     f"{r['consistency']:.3f}", f"{r['consistency_raw']:.3f}"])
    md += [md_table(["dataset", "rows", "features (one-hot)", "privileged share", "P(y=1) unpriv.", "P(y=1) priv.",
                     "DP_diff", "DIR", "consistency (z-scored)", "consistency (raw features, AIF360 as is)"], rows), ""]

    md += ["## 2. Held-out test set: before and after mitigation", "",
           "'exact gradient' = AIF360's LFR objective minimised with its analytic gradient (`lfr.ExactGradientLFR`); "
           "'AIF360 as released' = `aif360.algorithms.preprocessing.LFR.fit` unchanged. The setting is chosen per seed "
           "on the validation set with the paper's criteria (max Delta = yAcc − yDiscrim, or min yDiscrim). "
           "'LFR classifier' = the model's own prediction; 'LR on LFR repr.' = logistic regression trained on the "
           "LFR-transformed training data and applied to the LFR-transformed test data.", ""]
    for name in NAMES:
        md += [f"### {NAMES[name]}", ""]
        rows = []
        for _, r in summ[summ["dataset"] == name].iterrows():
            dir_txt = fmt(r["DIR_mean"], r["DIR_sd"]) + (f" ({r['DIR_undefined']} undefined)" if r["DIR_undefined"] else "")
            rows.append([r["pipeline"], fmt(r["DP_diff_mean"], r["DP_diff_sd"]), dir_txt,
                         fmt(r["consistency_mean"], r["consistency_sd"]), fmt(r["accuracy_mean"], r["accuracy_sd"]),
                         fmt(r["balanced_accuracy_mean"], r["balanced_accuracy_sd"]), fmt(r["pos_rate_mean"], r["pos_rate_sd"])])
        md += [md_table(["pipeline", "DP_diff", "DIR", "consistency", "accuracy", "balanced accuracy",
                         "share predicted favourable"], rows), ""]

    md += ["## 3. Selected settings per seed (Ax = 0.01, k = 10)", ""]
    rows = []
    for name in NAMES:
        for pipe in ORDER[2:6]:
            g = sel[(sel["dataset"] == name) & (sel["pipeline"] == pipe)].sort_values("seed")
            rows.append([NAMES[name], pipe, "; ".join(f"Ay={a:g}, Az={z:g}" for a, z in zip(g["Ay"], g["Az"]))])
    md += [md_table(["dataset", "pipeline", "seed 0; 1; 2; 3; 4"], rows), ""]

    md += ["## 4. Optimisation: AIF360 as released vs exact gradient", "",
           "Same objective, initialisation, bounds and limits (maxiter = maxfun = 5000). 'Iterations' = L-BFGS-B "
           "iterations; 'evaluations' = objective evaluations counted by SciPy (with AIF360's finite differences, "
           "one per parameter for every gradient). Objective = final value of AIF360's objective on the fit data.", ""]
    rows = []
    lfr = runs[(runs["method"] == E.LFR_CLF) & (runs["part"] == "test")]
    for name in NAMES:
        exact_grid = lfr[(lfr["dataset"] == name) & (lfr["optimizer"] == "exact")]
        for label, g in (("exact gradient, all 30 grid settings", exact_grid),
                         ("exact gradient, selected setting",
                          sel[(sel["dataset"] == name) & (sel["pipeline"] == ORDER[2])]),
                         ("AIF360 as released, same selected setting",
                          sel[(sel["dataset"] == name) & (sel["pipeline"] == ORDER[6])]),
                         ("AIF360 as released, AIF360 default (k=5, Az=50)",
                          sel[(sel["dataset"] == name) & (sel["pipeline"] == ORDER[8])])):
            rows.append([NAMES[name], label, len(g), f"{g['iterations'].median():.0f} ({g['iterations'].min():.0f}–{g['iterations'].max():.0f})",
                         f"{g['evaluations'].median():.0f}", f"{g['converged'].astype(bool).mean():.0%}",
                         f"{g['objective'].mean():.4f}", f"{g['fit_seconds'].median():.1f}"])
    md += [md_table(["dataset", "fits", "n", "iterations, median (range)", "evaluations, median", "converged",
                     "objective, mean", "seconds per fit, median"], rows), ""]
    stops = runs[(runs["method"] == E.LFR_CLF) & (runs["part"] == "test")].groupby(["optimizer", "stop_reason"]).size()
    md += ["Stop reasons reported by SciPy (all LFR fits):", ""]
    md += [f"- {opt}: `{reason}` × {n}" for (opt, reason), n in stops.items()] + [""]

    md += ["## 5. Degenerate solutions in the grid (exact gradient, test set)", "",
           "A fit is counted as degenerate when the pipeline predicts a single class for every test row "
           "(DP_diff = 0 and balanced accuracy = 0.5 by construction).", ""]
    rows = []
    grid = runs[(runs["optimizer"] == "exact") & (runs["part"] == "test")]
    for name in NAMES:
        for method in (E.LFR_CLF, E.LFR_LR):
            g = grid[(grid["dataset"] == name) & (grid["method"] == method)]
            const = g["pos_rate"].isin([0.0, 1.0])
            by_az = g.assign(const=const).groupby("Az")["const"].mean()
            rows.append([NAMES[name], method, f"{const.mean():.0%} of {len(g)}",
                         ", ".join(f"Az={az:g}: {v:.0%}" for az, v in by_az.items())])
    md += [md_table(["dataset", "pipeline", "degenerate fits", "by Az (over Ay and seeds)"], rows), ""]

    md += ["## Environment", "", "```", json.dumps(env, indent=2), "```", ""]
    (OUT / "results.md").write_text("\n".join(md))
    print("wrote results/results.md, results/summary.csv and results/tradeoff.png")


if __name__ == "__main__":
    main()
