@echo off
rem Windows version of reproduce.sh: complete setup, checks, experiments and tables in one command.
rem Run it from PowerShell or Command Prompt. See README.md for the quick run and the settings.
setlocal
set PYTHONUTF8=1
cd /d "%~dp0" || exit /b 1

for %%T in (uv git) do (
    where %%T >nul 2>nul
    if errorlevel 1 (
        echo Missing required tool: %%T. See README.md for installation requirements.
        exit /b 1
    )
)

rem AIF360 is built from the fork in external\AIF360 (git submodule; release v0.6.1, code unchanged).
if not exist "external\AIF360\setup.py" (
    git submodule update --init external/AIF360 || exit /b 1
)
uv sync --locked || exit /b 1
rem Downloads the three datasets on first use and checks their SHA256 hashes.
uv run --locked python -W ignore datasets.py || exit /b 1
uv run --locked pytest -q || exit /b 1
rem A full run replaces the saved per-task results.
if exist "results\raw\runs" rmdir /s /q "results\raw\runs"
if exist "results\raw\runs" exit /b 1
rem Executes the notebook in place: every step of the pipeline, all experiments, results\results.md.
uv run --locked jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 fairness_lfr.ipynb || exit /b 1
echo Done: see results\results.md and the executed fairness_lfr.ipynb
