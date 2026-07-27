import numpy as np

from scripts.pipeline.run_supervised_ranker_f20 import (
    recommendation_metrics_for_set,
    select_ranker_sets,
)


def test_global_f20_excludes_observed_and_uses_stable_score_order():
    scores = np.array([[100.0, 0.1, 0.9, 0.8, 0.2, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]])
    violations = np.zeros_like(scores, dtype=bool)
    selected = select_ranker_sets(scores, violations, observed_index=0, recommendation_fraction=0.20)
    # Ten non-observed candidates imply ceil(0.20 * 10) = 2 selections.
    assert selected[0].tolist() == [2, 3]


def test_filtered_f20_uses_feasible_menu_denominator_and_refills():
    scores = np.array([[100.0, 0.99, 0.98, 0.80, 0.70, 0.60, 0.50]])
    violations = np.array([[False, True, True, False, False, False, False]])
    selected = select_ranker_sets(
        scores,
        violations,
        observed_index=0,
        filter_compliant=True,
        recommendation_fraction=0.20,
    )
    # Four feasible non-observed candidates imply one selected candidate.
    assert selected[0].tolist() == [3]


def test_set_metrics_use_closest_candidate_and_candidate_level_violation_rate():
    actions = np.array(
        [
            [0.50, 0.50],  # observed
            [0.80, 0.20],
            [0.51, 0.49],
            [0.40, 0.60],
        ]
    )
    violations = np.array([False, True, False, False])
    metrics = recommendation_metrics_for_set(
        actions=actions,
        selected_indices=np.array([1, 2]),
        violations=violations,
        asset_idx=np.array([0, 1]),
        overlay_idx=np.array([], dtype=int),
        risk_overlay_idx=np.array([], dtype=int),
        l1_tol=0.05,
        observed_index=0,
    )
    assert metrics["recommendation_choice_coverage"]
    assert np.isclose(metrics["recommendation_distance_to_observed_l1"], 0.02)
    assert np.isclose(metrics["recommendation_violation_rate"], 0.50)
    assert metrics["selected_candidate_count"] == 2
