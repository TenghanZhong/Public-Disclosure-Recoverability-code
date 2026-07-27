# Reproduction Guide

## Environment

```bash
python -m pip install -r requirements.txt
```

The code requires Python 3.9 or newer and uses NumPy, pandas, SciPy,
statsmodels, matplotlib, PyYAML, tqdm, pyarrow, scikit-learn, and pytest.

## Tests

```bash
pytest -q
```

The tests use synthetic data to check core package behavior, including the
20% supervised recommendation-set selection rule.

## Figures

```bash
python scripts/analysis/make_reported_figures.py
```

This command reads CSV files from `tables/` and writes PDF and PNG files to
`figures/`.

## Full Pipeline

After placing disclosure inputs under `data/raw/`, run:

```bash
python scripts/pipeline/run_full_experiment.py --config configs/main_32gb.yaml
```

The full pipeline writes intermediate and row-level outputs to `outputs/`.
Analysis scripts under `scripts/analysis/` create compact summaries and
diagnostics.

Run the supervised 20% recommendation-set baseline with:

```bash
python -m scripts.pipeline.run_supervised_ranker_f20 \
  --config configs/main_32gb.yaml \
  --panel_root outputs/full_pipeline \
  --out_dir outputs/supervised_ranker \
  --recommendation_fraction 0.20
```

The global rule selects `ceil(0.20 * number of non-observed candidates)`.
The feasibility-filtered rule first restricts eligibility to non-observed
candidates inside the exogenous mandate proxy and then selects
`ceil(0.20 * number of feasible candidates)`. Coverage and distance use the
closest candidate in each selected set.

Create the compact supervised summary with:

```bash
python -m scripts.analysis.build_supervised_f20_reported_table \
  --row_level outputs/supervised_ranker/supervised_ranker_f20_row_level.parquet \
  --out tables/supervised_ranker_f20_reported.csv
```

Additional entry points:

```bash
python scripts/analysis/boundary_inference_checks.py
python scripts/analysis/recoverability_tau_menu_checks.py
python scripts/analysis/out_of_proxy_mechanical_robustness.py
python scripts/analysis/build_persistence_benchmarks.py
```

Some scripts require row-level outputs from the full pipeline. The committed
CSV files provide compact reference summaries when those files are absent.
