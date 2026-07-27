# Public-Disclosure Recoverability Code Artifact

This repository contains code, configuration files, tests, and precomputed
outputs for *Public-Disclosure Recoverability in Fund Allocation
Recommendations*.

The artifact builds three layers:

1. mandate-proxy bounds and feasible candidate menus;
2. revealed-preference and benchmark recommendation scores;
3. reported tables, diagnostics, and figures.

The supervised baseline uses the same 20% recommendation-set frontier as the
RP comparisons. The global rule retains the highest-scoring 20% of the
non-observed menu. The feasibility-filtered rule first removes candidates
outside the empirical mandate proxy and then retains the highest-scoring 20%
of the remaining feasible menu.

Raw regulatory filings are not redistributed. The code expects the public
disclosure inputs described in `docs/DATA.md`; compact reported outputs are
included under `tables/` and `figures/` for inspection.

## Repository Layout

```text
mandate_setirl/       Core package for bounds, candidates, features, and ranking
configs/             Pipeline configuration files
scripts/pipeline/    End-to-end experiment entry points
scripts/analysis/    Table, diagnostic, and figure builders
tables/              Precomputed CSV outputs reported in the study
figures/             Precomputed PDF/PNG figures
docs/                Data, reproduction, output, and artifact-scope notes
tests/               Synthetic-data checks for core logic
```

## Quick Checks

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run synthetic-data tests:

```bash
pytest tests
```

Regenerate figures from the committed CSV outputs:

```bash
python scripts/analysis/make_reported_figures.py
```

## Reproducing Reported Outputs

The committed tables and figures provide the reported numerical
outputs. Full reruns require the disclosure-derived panel files documented in
`docs/DATA.md`. With those files in place, the configured entry points write
results to `outputs/` and reported summaries to `tables/`.

The supervised 20% recommendation-set baseline is generated with:

```bash
python scripts/pipeline/run_supervised_ranker_f20.py \
  --recommendation_fraction 0.20
```

See:

- `docs/REPRODUCE.md` for execution commands;
- `docs/OUTPUT_MAP.md` for the link between reported outputs and files;
- `docs/ARTIFACT_SCOPE.md` for included and excluded material.
