# Output Map

This file maps reported outputs to repository files.

## Figures

| Reported output | File |
| --- | --- |
| Decomposition diagram | `figures/fig_decomposition.pdf` |
| Risk-signal forest plot | `figures/fig_risk_signal_forest.pdf` |
| Recoverability heatmap | `figures/fig_recoverability_heatmap.pdf` |
| Persistence ladder figure | `figures/fig_persistence_ladder.pdf` |
| Direction and source attribution | `figures/fig_direction_source_attribution.pdf` |

## Tables and Diagnostics

| Reported output | File |
| --- | --- |
| Risk-signal table | `tables/reported_risk_signal_table.csv` |
| Persistence and ranking benchmarks | `tables/persistence_ladder_with_ci.csv` |
| Move-bin proxy diagnostic | `tables/reported_proxy_by_move_bin.csv` |
| Recoverability menu support | `tables/repeated_menu_support_summary.csv` |
| Transition-candidate selection diagnostic | `tables/repeated_transition_selection_summary.csv` |
| Candidate-source attribution | `tables/transition_source_by_move_bin.csv` |
| Preference-set characterization | `tables/set_point_characterization_by_category.csv` |
| Filing availability | `tables/filing_date_availability.csv` |

Full-data diagnostics write the holdings-count exclusion check to
`tables/mechanical_robustness/reported_mechanical_robustness.csv` and the
buy-and-hold adjusted drift check to
`tables/candidate_drift_diagnostics/buyhold_adjusted_asset_drift_regressions.csv`.

The figure builder records its source table list in
`figures/figure_manifest.json` when it is run.
