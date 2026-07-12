from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.boundary_timing_diagnostics import (
    action_violations,
    boundary_gap,
    build_regression_panel,
    fast_exogenous_bounds,
    training_iqr_scale,
    zscore_on_sample,
)


def _cfg() -> dict:
    return {
        "bounds": {
            "global_lower_q": 0.0,
            "global_upper_q": 1.0,
            "peer_lower_q": 0.0,
            "peer_upper_q": 1.0,
            "min_width": 0.01,
            "use_adv_soft_caps": False,
        }
    }


def test_fast_exogenous_bounds_make_asset_caps_simplex_feasible():
    panel = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "mandate_cluster": [0, 0, 0],
            "asset_a": [0.20, 0.20, 0.50],
            "asset_b": [0.20, 0.20, 0.50],
            "overlay": [0.0, 0.1, 0.2],
        }
    )

    bounds = fast_exogenous_bounds(
        panel,
        action_cols=["asset_a", "asset_b", "overlay"],
        asset_cols=["asset_a", "asset_b"],
        overlay_cols=["overlay"],
        cfg=_cfg(),
    )

    # Even if raw train asset caps sum to less than one, cleaned caps admit a
    # simplex allocation.
    assert np.all(bounds.upper[:, :2].sum(axis=1) >= 1.0 - 1e-8)
    assert np.allclose(bounds.lower[:, :2], 0.0)


def test_training_iqr_scale_uses_training_rows_only():
    actions = np.array(
        [
            [0.0, 0.0],
            [2.0, 4.0],
            [100.0, 100.0],
        ]
    )
    scale = training_iqr_scale(actions, np.array([True, True, False]), floor=1e-8)

    assert np.allclose(scale, [1.0, 2.0])


def test_regression_panel_nonviolating_drift_excludes_violated_coordinates():
    panel = pd.DataFrame(
        {
            "series_id": ["a", "a"],
            "split": ["test", "test"],
            "mandate_cluster": [0, 0],
            "report_date": pd.to_datetime(["2024-01-31", "2024-04-30"]),
            "filing_date": pd.to_datetime(["2024-03-15", "2024-06-15"]),
        }
    )
    actions = np.array([[1.2, 0.4], [0.8, 0.9]])
    lag = np.array([[0.8, 0.3], [1.2, 0.4]])
    lead = np.array([[0.8, 0.9], [np.nan, np.nan]])
    lower = np.zeros_like(actions)
    upper = np.ones_like(actions)
    gap = boundary_gap(actions, lower, upper)

    out = build_regression_panel(
        panel,
        actions,
        lag,
        lead,
        gap,
        scale_iqr=np.ones(2),
        tol=1e-7,
        entity_col="series_id",
        date_col="report_date",
    )

    # First row violates coordinate 0, so next nonviolating drift only includes
    # coordinate 1: |0.9 - 0.4|.
    assert np.isclose(out.loc[0, "next_nonviolating_drift_l1"], 0.5)
    assert action_violations(actions, lower, upper, tol=1e-7).tolist() == [True, False]


def test_zscore_on_sample_is_centered_on_requested_sample():
    df = pd.DataFrame({"split": ["train", "test", "test"], "x": [100.0, 2.0, 4.0]})
    zcol = zscore_on_sample(df, "x", df["split"].eq("test"))

    assert np.isclose(df.loc[df["split"].eq("test"), zcol].mean(), 0.0)
    assert np.isclose(df.loc[df["split"].eq("test"), zcol].std(ddof=0), 1.0)
