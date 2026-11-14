#!/usr/bin/env bash
# Complete setup, checks, experiments and tables in one command.
#   bash reproduce.sh                      full run (5 seeds x 30 LFR settings x 3 datasets)
#   REPRODUCE_QUICK=1 bash reproduce.sh    smoke test: 1 seed, 3 settings (writes results/ too)
#   REPRODUCE_WORKERS=4 bash reproduce.sh  number of parallel processes (default: CPU count - 2)
# Windows: reproduce.cmd (PowerShell or Command Prompt) does the same.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for tool in uv git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Missing required tool: %s. See README.md for installation requirements.\n' "$tool" >&2
        exit 1
    fi
done

# AIF360 is built from the fork in external/AIF360 (git submodule; release v0.6.1, code unchanged).
if [ ! -f external/AIF360/setup.py ]; then
    git submodule update --init external/AIF360
fi
uv sync --locked
# Downloads the three datasets on first use and checks their SHA256 hashes.
uv run --locked python -W ignore datasets.py
uv run --locked pytest -q
# A full run replaces the saved per-task results.
rm -rf -- results/raw/runs
# Executes the notebook in place: every step of the pipeline, all experiments, results/results.md.
uv run --locked jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 fairness_lfr.ipynb
printf 'Done: see results/results.md and the executed fairness_lfr.ipynb\n'
