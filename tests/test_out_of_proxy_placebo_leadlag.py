from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.out_of_proxy_placebo_leadlag import (
    fwl_beta,
    permute_within_groups,
)


def test_permute_within_groups_preserves_group_multisets():
    df = pd.DataFrame(
        {
            "date": ["a", "a", "a", "b", "b", "b"],
            "cluster": [1, 1, 1, 2, 2, 2],
            "flag": [0, 1, 1, 0, 0, 1],
        }
    )

    out = permute_within_groups(df, "flag", ["date", "cluster"], np.random.default_rng(7))

    for _, g in df.groupby(["date", "cluster"], sort=False):
        got = out.loc[g.index].sort_values().to_list()
        want = g["flag"].sort_values().to_list()
        assert got == want
    assert set(out.index) == set(df.index)


def test_fwl_beta_matches_manual_residualized_coefficient():
    y = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
    x = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    controls = np.column_stack([np.ones_like(y), np.array([0.0, 0.0, 1.0, 1.0, 2.0])])

    beta = fwl_beta(y, x, controls)

    mx = np.eye(len(y)) - controls @ np.linalg.pinv(controls.T @ controls) @ controls.T
    y_res = mx @ y
    x_res = mx @ x
    expected = float(x_res @ y_res / (x_res @ x_res))
    assert np.isclose(beta, expected)
