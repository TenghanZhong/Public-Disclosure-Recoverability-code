# Public-Disclosure Recoverability Code Artifact

This repository contains code, configuration files, tests, and precomputed
outputs for *Public-Disclosure Recoverability in Fund Allocation
Recommendations*.

The repository covers three components:

1. mandate-proxy bounds and feasible candidate menus;
2. revealed-preference and benchmark recommendation scores;
3. summary tables, diagnostics, and figures.

The supervised baseline uses the same 20% recommendation-set frontier as the
RP comparisons. The global rule retains the highest-scoring 20% of the
non-observed menu. The feasibility-filtered rule first removes candidates
outside the empirical mandate proxy and then retains the highest-scoring 20%
of the remaining feasible menu.

Raw regulatory filings are not redistributed. The code expects the disclosure
inputs described in `docs/DATA.md`; compact reference outputs are included
under `tables/` and `figures/`.

## Repository Layout

```text
mandate_setirl/       Bounds, candidates, features, and ranking package
configs/              Pipeline configuration files
scripts/pipeline/     Experiment entry points
scripts/analysis/     Summary and figure builders
tables/               Precomputed CSV summaries
figures/              Precomputed PDF and PNG figures
docs/                 Data, execution, output, and scope documentation
tests/                Synthetic-data tests
```

## Quick Checks

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Regenerate figures from the committed CSV files:

```bash
python scripts/analysis/make_reported_figures.py
```

## Output Regeneration

Full runs require the disclosure-derived panel files documented in
`docs/DATA.md`. The configured entry points write row-level results to
`outputs/` and compact summaries to `tables/`.

Run the supervised 20% recommendation-set baseline with:

```bash
python -m scripts.pipeline.run_supervised_ranker_f20 \
  --recommendation_fraction 0.20
```

See:

- `docs/REPRODUCE.md` for execution commands;
- `docs/OUTPUT_MAP.md` for output locations;
- `docs/ARTIFACT_SCOPE.md` for included and excluded material.
