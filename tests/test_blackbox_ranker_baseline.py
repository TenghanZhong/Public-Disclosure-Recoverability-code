from __future__ import annotations

import numpy as np

from scripts.run_blackbox_ranker_baseline import (
    DEFAULT_STATE_FEATURES,
    make_ranker_feature_matrix,
    recommendation_metrics_for_choice,
    select_ranker_indices,
)


ADV_TEXT_KEYWORD_FEATURES = {
    "allows_or_discusses_derivatives",
    "shorting",
    "leverage",
    "liquidity_or_redemption_constraint",
    "performance_fee",
    "wrap_fee",
    "crypto",
    "esg",
    "custody",
}


def test_ranker_feature_matrix_excludes_candidate_source_leakage():
    actions = np.array([[[1.0, 0.0], [0.25, 0.75]]])
    utility = np.array([[[0.1, -0.2], [0.3, 0.4]]])
    state = np.array([[2.0, -1.0]])

    X, names = make_ranker_feature_matrix(
        actions,
        utility,
        action_cols=["asset_EC", "asset_STIV"],
        utility_feature_names=["ret_proxy", "neg_risk_proxy"],
        state_features=state,
        state_feature_names=["mandate_cluster", "log_net_assets"],
    )

    assert X.shape == (2, 6)
    assert names == [
        "action:asset_EC",
        "action:asset_STIV",
        "utility:ret_proxy",
        "utility:neg_risk_proxy",
        "state:mandate_cluster",
        "state:log_net_assets",
    ]
    assert not any("source" in name.lower() or "observed" in name.lower() for name in names)


def test_blackbox_default_state_features_keep_drp_not_adv_text_keywords():
    features = set(DEFAULT_STATE_FEATURES)

    assert ADV_TEXT_KEYWORD_FEATURES.isdisjoint(features)
    assert "adv_total_drp_records" in features
    assert "legal_disciplinary" in features


def test_select_ranker_indices_excludes_observed_candidate():
    scores = np.array([[10.0, 0.5, 0.1]])
    violations = np.array([[False, False, False]])

    chosen = select_ranker_indices(scores, violations, observed_index=0, filter_compliant=False)

    assert chosen.tolist() == [1]


def test_posthoc_filter_skips_noncompliant_high_score_candidate():
    scores = np.array([[0.0, 0.9, 0.7]])
    violations = np.array([[False, True, False]])

    raw = select_ranker_indices(scores, violations, observed_index=0, filter_compliant=False)
    filt = select_ranker_indices(scores, violations, observed_index=0, filter_compliant=True)

    assert raw.tolist() == [1]
    assert filt.tolist() == [2]


def test_recommendation_metrics_for_choice_reports_grouped_distances():
    actions = np.array([
        [1.0, 0.0, 0.10, 0.20],
        [0.6, 0.4, 0.30, 0.15],
    ])
    violations = np.array([False, True])

    row = recommendation_metrics_for_choice(
        actions=actions,
        selected_idx=1,
        violations=violations,
        asset_idx=np.array([0, 1]),
        overlay_idx=np.array([2, 3]),
        risk_overlay_idx=np.array([2]),
        l1_tol=0.05,
    )

    assert row["recommendation_choice_coverage"] is False
    assert np.isclose(row["recommendation_distance_to_observed_l1"], 1.05)
    assert np.isclose(row["recommendation_distance_to_observed_asset_only_l1"], 0.8)
    assert np.isclose(row["recommendation_distance_to_observed_overlay_only_l1"], 0.25)
    assert np.isclose(row["recommendation_distance_to_observed_risk_overlay_l1"], 0.2)
    assert row["point_pred_mandate_violation_ex_observed"] is True


def test_recommendation_metrics_respects_nonzero_observed_index():
    actions = np.array([
        [0.0, 1.0],
        [0.8, 0.2],
        [0.7, 0.3],
    ])
    violations = np.array([False, False, True])

    row = recommendation_metrics_for_choice(
        actions=actions,
        selected_idx=2,
        violations=violations,
        asset_idx=np.array([0, 1]),
        overlay_idx=np.array([], dtype=int),
        risk_overlay_idx=np.array([], dtype=int),
        l1_tol=0.05,
        observed_index=1,
    )

    assert np.isclose(row["recommendation_distance_to_observed_l1"], 0.2)
    assert row["point_pred_mandate_violation_ex_observed"] is True
