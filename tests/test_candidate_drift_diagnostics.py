from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from candidate_drift_diagnostics import (  # noqa: E402
    buyhold_next_weights,
    lead_values,
    normalize_simplex,
    order_persistence,
    order_transition_source,
    selected_min_distance,
    summarize_rates,
    unique_representatives,
)


def test_unique_representatives_keeps_first_slot_and_counts_duplicate_actions():
    actions = np.array(
        [
            [0.20, 0.80],
            [0.2000000001, 0.7999999999],
            [0.50, 0.50],
        ]
    )
    sources = np.array(["lag", "transition_small", "transition_moderate"], dtype=object)
    unique_actions, unique_sources, keep_idx = unique_representatives(actions, sources, decimals=6)
    assert unique_actions.shape == (2, 2)
    assert keep_idx.tolist() == [0, 2]
    assert unique_sources.tolist() == ["lag", "transition_moderate"]


def test_selected_min_distance_uses_frontier_count_after_ranking():
    actions = np.array([[0.0, 1.0], [0.4, 0.6], [0.5, 0.5], [0.9, 0.1]])
    observed = np.array([0.5, 0.5])
    lag = np.array([0.0, 1.0])
    order = order_persistence(actions, lag)
    dist, k = selected_min_distance(actions, order, observed, frontier=0.50)
    assert k == 2
    assert np.isclose(dist, 0.2)


def test_transition_order_prioritizes_transition_sources_then_lag_distance():
    actions = np.array([[0.0, 1.0], [0.3, 0.7], [0.1, 0.9]])
    sources = np.array(["lag_exact", "transition_moderate", "transition_small"], dtype=object)
    lag = np.array([0.0, 1.0])
    order = order_transition_source(actions, sources, lag)
    assert order.tolist() == [2, 1, 0]


def test_summarize_rates_reports_conditional_coverage_given_support():
    rows = pd.DataFrame(
        {
            "move_bin": ["stable", "stable", "large", "large"],
            "observed_in_proxy": [True, True, True, True],
            "entry_oracle_min_l1": [0.01, 0.02, 0.20, 0.30],
            "unique_oracle_min_l1": [0.01, 0.02, 0.20, 0.30],
            "entry_persistence_min_l1": [0.01, 0.08, 0.50, 0.50],
            "unique_persistence_min_l1": [0.01, 0.08, 0.50, 0.50],
            "entry_transition_min_l1": [0.06, 0.02, 0.50, 0.50],
            "unique_transition_min_l1": [0.06, 0.02, 0.50, 0.50],
        }
    )
    out = summarize_rates(rows, tau=0.05)
    overall = out[out["group"].eq("overall")].iloc[0]
    assert overall["entry_support"] == 0.5
    assert overall["entry_persistence_cov"] == 0.25
    assert overall["entry_persistence_conditional_on_support"] == 0.5


def test_normalize_simplex_and_buyhold_zero_return_identity():
    weights = np.array([[2.0, 1.0], [0.0, 0.0]])
    normalized = normalize_simplex(weights)
    assert np.allclose(normalized[0], [2 / 3, 1 / 3])
    assert np.isnan(normalized[1]).all()
    returns = np.zeros((1, 2))
    buyhold = buyhold_next_weights(normalized[:1], returns)
    assert np.allclose(buyhold[0], normalized[0])


def test_buyhold_shifts_weights_toward_higher_return_asset():
    weights = np.array([[0.5, 0.5]])
    returns = np.array([[0.10, 0.00]])
    buyhold = buyhold_next_weights(weights, returns)
    assert buyhold[0, 0] > 0.5
    assert np.isclose(buyhold.sum(), 1.0)


def test_lead_values_allows_repeated_source_columns():
    panel = pd.DataFrame(
        {
            "series_id": ["a", "a", "b", "b"],
            "report_date": pd.to_datetime(["2020-01-01", "2020-04-01", "2020-01-01", "2020-04-01"]),
            "ret": [0.1, 0.2, 0.3, 0.4],
        }
    )
    lead = lead_values(panel, ["ret", "ret"], "series_id", "report_date").to_numpy(dtype=float)
    assert lead.shape == (4, 2)
    assert np.allclose(lead[0], [0.2, 0.2])
    assert np.isnan(lead[1]).all()
