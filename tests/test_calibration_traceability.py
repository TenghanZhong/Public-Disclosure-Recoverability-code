from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.pipeline.build_calibration_traceability import (
    binding_constraint_summary,
    cluster_bootstrap_ci,
    preference_intervals_long,
)


def test_cluster_bootstrap_ci_reports_model_and_paired_difference():
    df = pd.DataFrame({
        "series_id": ["a", "a", "b", "b", "a", "a", "b", "b"],
        "model": ["main", "main", "main", "main", "baseline", "baseline", "baseline", "baseline"],
        "coverage": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "distance": [0.1, 0.2, 0.4, 0.5, 0.3, 0.4, 0.7, 0.8],
    })

    summary, diffs = cluster_bootstrap_ci(
        df,
        cluster_col="series_id",
        model_col="model",
        metric_cols=["coverage", "distance"],
        primary_model="main",
        baseline_models=["baseline"],
        n_boot=25,
        seed=7,
    )

    main_cov = summary[(summary["model"] == "main") & (summary["metric"] == "coverage")].iloc[0]
    assert np.isclose(main_cov["point_estimate"], 0.5)
    diff_cov = diffs[(diffs["comparison"] == "main_minus_baseline") & (diffs["metric"] == "coverage")].iloc[0]
    assert np.isclose(diff_cov["point_estimate"], 0.25)
    assert {"ci_low", "ci_high", "bootstrap_mean", "bootstrap_se"}.issubset(summary.columns)


def test_preference_intervals_long_parses_theta_columns():
    pref = pd.DataFrame({
        "model": ["exogenous_polytope"],
        "scope_id": ["0"],
        "theta_point_ret_proxy": [0.2],
        "theta_min_ret_proxy": [0.1],
        "theta_max_ret_proxy": [0.4],
        "theta_width_ret_proxy": [0.3],
    })

    out = preference_intervals_long(pref)

    assert len(out) == 1
    assert out.loc[0, "feature"] == "ret_proxy"
    assert np.isclose(out.loc[0, "theta_width"], 0.3)


def test_binding_constraint_summary_finds_dimension_violations():
    actions = pd.DataFrame({"a": [0.0, 0.5, 1.2], "b": [0.2, -0.1, 0.4]})
    lower = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    upper = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.3]])

    out = binding_constraint_summary(actions, lower, upper, ["a", "b"])

    row_a = out[out["action_col"] == "a"].iloc[0]
    row_b = out[out["action_col"] == "b"].iloc[0]
    assert row_a["upper_violation_n"] == 1
    assert row_b["lower_violation_n"] == 1
    assert row_b["upper_violation_n"] == 1
