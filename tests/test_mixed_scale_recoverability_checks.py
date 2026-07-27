from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mixed_scale_recoverability_checks import (  # noqa: E402
    calibrated_scaled_tau,
    metric_threshold_table,
    scaled_l1,
    summarize_across_seeds,
    summarize_seed,
    train_iqr_scales,
)


def test_scaled_l1_uses_coordinate_scales():
    a = np.array([[2.0, 4.0]])
    b = np.array([[1.0, 2.0]])
    scale = np.array([1.0, 2.0])
    assert np.allclose(scaled_l1(a, b, scale), [2.0])


def test_train_iqr_scales_use_floor_for_constant_coordinate():
    panel = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0], "y": [0.0, 1.0, 2.0, 3.0]})
    mask = np.array([True, True, True, True])
    scales = train_iqr_scales(panel, ["x", "y"], mask, floor=1e-3)
    assert scales[0] == 1.0
    assert scales[1] > 0


def test_calibrated_scaled_tau_matches_training_stable_share_quantile():
    panel = pd.DataFrame(
        {
            "split": ["train"] * 4,
            "a": [0.00, 0.01, 0.10, 0.20],
            "lag_a": [0.00, 0.00, 0.00, 0.00],
        }
    )
    mask = panel["split"].eq("train").to_numpy()
    tau, share = calibrated_scaled_tau(panel, ["a"], mask, np.array([1.0]), full_tau=0.05)
    assert share == 0.5
    assert np.isclose(tau, np.quantile([0.0, 0.01, 0.10, 0.20], 0.5))


def test_summarize_seed_reports_metric_specific_support():
    rows = pd.DataFrame(
        {
            "move_bin": ["stable", "large"],
            "observed_in_proxy": [True, True],
            "min_full_l1": [0.01, 0.20],
            "min_asset_l1": [0.01, 0.04],
            "min_scaled_l1": [0.50, 2.00],
            "full_tau": [0.05, 0.05],
            "asset_tau": [0.05, 0.05],
            "scaled_tau": [1.00, 1.00],
        }
    )
    out = summarize_seed(rows)
    overall = out[out["group"].eq("overall")]
    full = overall[overall["metric"].eq("full_19_coordinate_l1")].iloc[0]
    asset = overall[overall["metric"].eq("asset_block_l1")].iloc[0]
    scaled = overall[overall["metric"].eq("training_iqr_scaled_full_l1")].iloc[0]
    assert full["menu_support"] == 0.5
    assert asset["menu_support"] == 1.0
    assert scaled["menu_support"] == 0.5


def test_summary_across_seeds_averages_rates():
    seed_df = pd.DataFrame(
        {
            "group": ["overall", "overall"],
            "metric": ["full_19_coordinate_l1", "full_19_coordinate_l1"],
            "move_bin": [np.nan, np.nan],
            "observed_in_proxy": [np.nan, np.nan],
            "seed": [1, 2],
            "n": [10, 20],
            "menu_support": [0.2, 0.4],
            "mean_min_distance": [0.1, 0.2],
            "median_min_distance": [0.1, 0.2],
            "tolerance": [0.05, 0.05],
        }
    )
    out = summarize_across_seeds(seed_df).iloc[0]
    assert out["n_seeds"] == 2
    assert np.isclose(out["menu_support_mean"], 0.3)
    assert np.isclose(out["n_mean"], 15.0)


def test_metric_threshold_table_records_scaled_rule():
    tab = metric_threshold_table(0.05, 0.05, 1.25, 0.4)
    assert set(tab["metric"]) == {
        "full_19_coordinate_l1",
        "asset_block_l1",
        "training_iqr_scaled_full_l1",
    }
    assert tab.loc[tab["metric"].eq("training_iqr_scaled_full_l1"), "tolerance"].iloc[0] == 1.25
