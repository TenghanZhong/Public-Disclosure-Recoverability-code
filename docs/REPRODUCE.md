# Reproduction Guide

## Environment

```bash
python -m pip install -r requirements.txt
```

The code was written for Python 3.9 or newer and uses NumPy, pandas, SciPy,
statsmodels, matplotlib, PyYAML, tqdm, pyarrow, and pytest.

## Tests

```bash
pytest tests
```

These tests use synthetic data to check core package behavior.

## Figures From Committed Tables

```bash
python scripts/analysis/make_reported_figures.py
```

This command reads CSV files from `tables/` and writes PDF/PNG files to
`figures/`.

## Full Pipeline

After placing disclosure inputs under `data/raw/`, run:

```bash
python scripts/pipeline/run_full_experiment.py --config configs/main_32gb.yaml
```

The full pipeline writes intermediate and row-level outputs to `outputs/`.
Analysis scripts under `scripts/analysis/` then build reported diagnostics
and summaries.

Examples:

```bash
python scripts/analysis/boundary_inference_checks.py
python scripts/analysis/recoverability_tau_menu_checks.py
python scripts/analysis/out_of_proxy_mechanical_robustness.py
python scripts/analysis/build_persistence_benchmarks.py
```

Some scripts require row-level outputs from the full pipeline. When those files
are absent, the committed tables remain the inspection layer.
