from __future__ import annotations

import numpy as np

from scripts.out_of_proxy_mechanical_robustness import (
    boundary_metrics_subset,
    nonviolating_next_drift,
    training_coordinate_scale,
)


def test_boundary_metrics_subset_can_exclude_holdings_coordinate():
    actions = np.array([[0.5, 1.4], [1.2, 0.6]])
    lower = np.zeros_like(actions)
    upper = np.ones_like(actions)

    all_metrics = boundary_metrics_subset(actions, lower, upper, include_idx=[0, 1], tol=1e-7)
    no_holdings = boundary_metrics_subset(actions, lower, upper, include_idx=[0], tol=1e-7)

    assert all_metrics["out_of_proxy"].to_list() == [1, 1]
    assert np.allclose(all_metrics["severity_l1"], [0.4, 0.2])
    assert no_holdings["out_of_proxy"].to_list() == [0, 1]
    assert np.allclose(no_holdings["severity_l1"], [0.0, 0.2])


def test_boundary_metrics_subset_uses_training_standardization():
    actions = np.array([[1.2, -0.5]])
    lower = np.array([[0.0, 0.0]])
    upper = np.array([[1.0, 1.0]])
    scale = np.array([0.1, 0.5])

    got = boundary_metrics_subset(
        actions,
        lower,
        upper,
        include_idx=[0, 1],
        scale=scale,
        tol=1e-7,
    )

    # Raw violation is 0.2 + 0.5. Standardized violation is 0.2/0.1 + 0.5/0.5.
    assert np.allclose(got["severity_l1"], [0.7])
    assert np.allclose(got["severity_standardized"], [3.0])


def test_nonviolating_next_drift_excludes_current_violation_coordinates():
    actions = np.array([[1.2, 0.4], [0.8, 0.9], [0.2, 0.3]])
    lower = np.zeros_like(actions)
    upper = np.ones_like(actions)
    lead_actions = np.array([[0.7, 0.9], [0.1, 1.5], [np.nan, np.nan]])

    got = nonviolating_next_drift(actions, lead_actions, lower, upper, tol=1e-7)

    # Row 0 violates coordinate 0, so only coordinate 1 contributes: |0.9 - 0.4|.
    # Row 1 has no current violation, so both coordinates contribute.
    assert np.allclose(got[:2], [0.5, 1.3])
    assert np.isnan(got[2])


def test_training_coordinate_scale_uses_train_rows_only():
    actions = np.array([[0.0, 0.0], [2.0, 4.0], [100.0, 100.0]])
    train_mask = np.array([True, True, False])

    scale = training_coordinate_scale(actions, train_mask)

    assert np.allclose(scale, [1.0, 2.0])
