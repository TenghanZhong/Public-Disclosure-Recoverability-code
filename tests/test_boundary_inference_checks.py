import numpy as np

from scripts.boundary_inference_checks import (
    training_iqr_scale,
    wild_cluster_bootstrap_p,
)


def test_training_iqr_scale_uses_train_rows_only():
    actions = np.array(
        [
            [0.0, 10.0],
            [2.0, 14.0],
            [100.0, 100.0],
        ]
    )
    train_mask = np.array([True, True, False])
    scale = training_iqr_scale(actions, train_mask, floor=1e-8)
    np.testing.assert_allclose(scale, np.array([1.0, 2.0]))


def test_wild_cluster_bootstrap_returns_valid_p_value():
    rng = np.random.default_rng(123)
    n = 120
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n)
    X = np.column_stack([x, rng.normal(size=n)])
    clusters = np.repeat(np.arange(12), 10)
    res = wild_cluster_bootstrap_p(y, X, clusters, n_boot=49, seed=456)
    assert 0.0 <= res["wild_cluster_p"] <= 1.0
    assert res["wild_boot_draws"] == 49
    assert np.isfinite(res["date_cluster_t"])
