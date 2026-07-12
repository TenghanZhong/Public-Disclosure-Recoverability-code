from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.out_of_proxy_extremeness_incremental import (
    cluster_extremeness_metrics,
    compute_boundary_metrics,
    training_cluster_reference,
)


def test_compute_boundary_metrics_matches_expected_violations():
    actions = np.array(
        [
            [0.5, 0.5],
            [1.2, 0.2],
            [-0.1, 0.7],
        ]
    )
    lower = np.zeros_like(actions)
    upper = np.ones_like(actions)

    got = compute_boundary_metrics(actions, lower, upper, tol=1e-7)

    assert got["out_of_proxy"].to_list() == [0, 1, 1]
    assert np.allclose(got["exception_severity_l1"], [0.0, 0.2, 0.1])
    assert np.allclose(got["max_normalized_violation"], [0.0, 0.2, 0.1])
    assert np.allclose(got["min_signed_margin_to_bound"], [0.5, -0.2, -0.1])


def test_training_cluster_reference_uses_train_split_only():
    panel = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "mandate_cluster": [1, 1, 1],
            "asset_EC": [0.0, 2.0, 100.0],
            "asset_DBT": [2.0, 4.0, 100.0],
        }
    )

    means, stds, _, _ = training_cluster_reference(
        panel,
        ["asset_EC", "asset_DBT"],
        "mandate_cluster",
    )

    assert np.allclose(means.loc[1].to_numpy(), [1.0, 3.0])
    assert np.allclose(stds.loc[1].to_numpy(), [1.0, 1.0])


def test_cluster_extremeness_falls_back_to_training_center():
    panel = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "mandate_cluster": [1, 1, 2],
            "asset_EC": [0.0, 2.0, 1.0],
            "asset_DBT": [2.0, 4.0, 3.0],
        }
    )

    got = cluster_extremeness_metrics(panel, ["asset_EC", "asset_DBT"], "mandate_cluster")

    # Cluster 2 is unseen in training, so the test row uses the global training
    # center [1, 3] and has zero L1 distance.
    assert np.isclose(got.loc[2, "cluster_center_l1"], 0.0)
