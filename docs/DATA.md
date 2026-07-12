# Data Inputs

Raw regulatory filings are not redistributed in this repository. The pipeline
expects a disclosure-derived, user-provided data directory with these inputs:

```text
data/raw/
  nport_choice_with_market_adviser.csv
  nport_asset_class_exposure_long.csv
  nport_fund_choice_summary.csv
  adv_contract_text_constraints.csv
```

The study uses quarter-end N-PORT month-three reports, N-CEN fund-series
metadata, adviser disclosure fields, and market-state controls. The scripts
convert those inputs into:

```text
outputs/full_pipeline/
  panel_model_ready.parquet
  evaluation_row_level.parquet
  prediction_compatibility.parquet
  action_columns.json
  main_metrics.csv
  mandate_feasible_set_metrics_exogenous.csv
  scope_preference_summary_exogenous_polytope.csv
```

Large raw files and row-level intermediate outputs are excluded to keep the
repository compact and to avoid redistributing regulatory-data extracts. The
included `tables/` and `figures/` directories provide the compact reported
outputs.
