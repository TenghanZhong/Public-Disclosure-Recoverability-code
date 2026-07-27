# Reproduction Guide

## Environment

```bash
python -m pip install -r requirements.txt
```

The code was written for Python 3.9 or newer and uses NumPy, pandas, SciPy,
statsmodels, matplotlib, PyYAML, tqdm, pyarrow, scikit-learn, and pytest.

## Tests

```bash
pytest tests
```

These tests use synthetic data to check core package behavior, including the
20% supervised recommendation-set selection rule.

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

Run the supervised ranker at the paper's 20% recommendation frontier with:

```bash
python scripts/pipeline/run_supervised_ranker_f20.py \
  --config configs/main_32gb.yaml \
  --panel_root outputs/full_pipeline \
  --out_dir outputs/supervised_ranker \
  --recommendation_fraction 0.20
```

The global rule selects `ceil(0.20 * number of non-observed candidates)`.
The feasibility-filtered rule first restricts to non-observed candidates
inside the exogenous mandate proxy and then selects
`ceil(0.20 * number of feasible candidates)`. Coverage and distance are
computed from the closest candidate in each selected set.

To regenerate the compact supervised table after the full-data run:

```bash
python scripts/analysis/build_supervised_f20_reported_table.py \
  --row_level outputs/supervised_ranker/supervised_ranker_f20_row_level.parquet \
  --out tables/supervised_ranker_f20_reported.csv
```

Other examples:

```bash
python scripts/analysis/boundary_inference_checks.py
python scripts/analysis/recoverability_tau_menu_checks.py
python scripts/analysis/out_of_proxy_mechanical_robustness.py
python scripts/analysis/build_persistence_benchmarks.py
```

Some scripts require row-level outputs from the full pipeline. When those files
are absent, the committed tables remain the inspection layer.
