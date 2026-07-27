# Output Map

## Figures

| Output | File |
| --- | --- |
| Decomposition diagram | `figures/fig_decomposition.pdf` |
| Risk-signal forest plot | `figures/fig_risk_signal_forest.pdf` |
| Recoverability heatmap | `figures/fig_recoverability_heatmap.pdf` |
| Persistence ladder | `figures/fig_persistence_ladder.pdf` |
| Direction and source attribution | `figures/fig_direction_source_attribution.pdf` |

## Tables and Diagnostics

| Output | File |
| --- | --- |
| Risk-signal summary | `tables/reported_risk_signal_table.csv` |
| Persistence, RP, and inverse-optimization benchmarks | `tables/persistence_ladder_with_ci.csv` |
| Supervised global and feasibility-filtered 20% sets | `tables/supervised_ranker_f20_reported.csv` |
| Move-bin proxy diagnostic | `tables/reported_proxy_by_move_bin.csv` |
| Recoverability menu support | `tables/repeated_menu_support_summary.csv` |
| Transition-candidate selection diagnostic | `tables/repeated_transition_selection_summary.csv` |
| Candidate-source attribution | `tables/transition_source_by_move_bin.csv` |
| Preference-set characterization | `tables/set_point_characterization_by_category.csv` |
| Filing availability | `tables/filing_date_availability.csv` |

`scripts/pipeline/run_supervised_ranker_f20.py` evaluates global and
feasibility-filtered 20% recommendation sets. A full run writes
`outputs/supervised_ranker/supervised_ranker_f20_row_level.parquet`.
`scripts/analysis/build_supervised_f20_reported_table.py` creates the compact
summary. `scripts/pipeline/run_blackbox_ranker_baseline.py` provides a separate
top-1 diagnostic.

Additional full-data diagnostics write the holdings-count exclusion check to
`tables/mechanical_robustness/reported_mechanical_robustness.csv` and the
buy-and-hold adjusted drift check to
`tables/candidate_drift_diagnostics/buyhold_adjusted_asset_drift_regressions.csv`.

`figures/figure_manifest.json` records the source tables used by the figure
builder.
