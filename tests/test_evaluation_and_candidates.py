from __future__ import annotations

import numpy as np
import pandas as pd

from mandate_setirl.bounds import BoundModel, fit_mandate_bounds
from mandate_setirl.candidates import CandidateBatch, CandidateGenerator
from mandate_setirl.irl import IRLModel, evaluate_irl_model, summarize_frontier_metrics
from mandate_setirl.irl import summarize_metrics
from mandate_setirl.panel import _prepare_summary_action_columns
from mandate_setirl.utils import read_yaml
from mandate_setirl.polytope import LinearPreferencePolytope
import mandate_setirl.irl as irl_module


ADV_TEXT_KEYWORD_FEATURES = {
    "allows_or_discusses_derivatives",
    "shorting",
    "allows_or_discusses_shorting",
    "leverage",
    "allows_or_discusses_leverage",
    "liquidity_or_redemption_constraint",
    "discretionary_management",
    "wrap_fee",
    "performance_fee",
    "mentions_performance_fee",
    "esg",
    "crypto",
    "custody",
}


def _pref() -> LinearPreferencePolytope:
    theta = np.array([1.0])
    return LinearPreferencePolytope(
        scope_id="0",
        G=np.zeros((0, 1)),
        eps_star=0.0,
        eps_bound=0.0,
        theta_point=theta,
        theta_min=theta,
        theta_max=theta,
        support_thetas=theta[None, :],
        n_constraints=0,
        n_train_obs=1,
        feasible=True,
    )


class _FakeCandGen:
    entity_col = "series_id"

    def __init__(self):
        self.bounds = BoundModel(
            action_cols=["asset_EC", "asset_STIV"],
            lower=np.zeros((1, 2)),
            upper=np.ones((1, 2)),
            row_index=pd.RangeIndex(1),
            asset_action_cols=["asset_EC", "asset_STIV"],
            overlay_action_cols=[],
            mode="exogenous",
        )

    def generate_for_positions(self, positions, n_candidates=None):
        # Candidate 0 is observed and cannot enter recommendations.
        actions = np.array([[[1.0, 0.0], [0.0, 1.0], [0.4, 0.6]]], dtype=float)
        sources = np.array([["observed", "peer_train", "frontier_conservative"]], dtype=object)
        return CandidateBatch(row_positions=np.asarray(positions), actions=actions, sources=sources, observed_index=0)


def test_evaluation_prediction_and_frontier_exclude_observed(monkeypatch):
    panel = pd.DataFrame({"split": ["test"], "mandate_cluster": ["0"], "series_id": ["fundA"]})

    class _FeatureModel:
        feature_names = ["score"]

    def fake_features(panel, batch, feature_model):
        # Observed score is 10; alternative scores are lower.
        return np.array([[[10.0], [1.0], [0.0]]], dtype=float)

    monkeypatch.setattr(irl_module, "compute_scaled_utility_features", fake_features)
    model = IRLModel(preferences={"0": _pref(), "__fallback__": _pref()}, feature_names=["score"], scope_col="mandate_cluster", fallback_scope_id="__fallback__")
    cfg = {
        "keys": {"entity_col": "series_id"},
        "candidates": {"n_candidates": 3, "chunk_size": 8, "prediction_exclude_observed": True},
        "evaluation": {"selected_fractions": [0.05], "frontier_exclude_observed": True, "coverage_l1_tolerance": 1e-12},
        "irl": {"set_near_optimal_eta": 0.0, "membership_mode": "support"},
    }
    ev, _ = evaluate_irl_model(panel, _FakeCandGen(), _FeatureModel(), model, cfg, split="test")
    row = ev.iloc[0]
    assert row["point_pred_source"] != "observed"
    assert row["point_pred_source_ex_observed"] != "observed"
    assert bool(row["point_top1_observed_in_coverage_menu"]) is True
    assert bool(row["choice_coverage"]) is False
    assert row["frontier_f05_contains_observed_candidate"] is False or bool(row["frontier_f05_contains_observed_candidate"]) is False
    fm = summarize_frontier_metrics(ev)
    assert float(fm.loc[0, "choice_coverage"]) == 0.0
    assert float(fm.loc[0, "contains_observed_candidate"]) == 0.0


def test_recommendation_benchmark_excludes_observed_candidate(monkeypatch):
    panel = pd.DataFrame({"split": ["test"], "mandate_cluster": ["0"], "series_id": ["fundA"]})

    class _FeatureModel:
        feature_names = ["score"]

    class _CloseAlternativeCandGen(_FakeCandGen):
        def generate_for_positions(self, positions, n_candidates=None):
            actions = np.array([[[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]]], dtype=float)
            sources = np.array([["observed", "peer_train", "frontier_conservative"]], dtype=object)
            return CandidateBatch(row_positions=np.asarray(positions), actions=actions, sources=sources, observed_index=0)

    def fake_features(panel, batch, feature_model):
        # Excluding the observed candidate admits the best alternative at eta=0.
        return np.array([[[10.0], [1.0], [0.0]]], dtype=float)

    monkeypatch.setattr(irl_module, "compute_scaled_utility_features", fake_features)
    model = IRLModel(preferences={"0": _pref(), "__fallback__": _pref()}, feature_names=["score"], scope_col="mandate_cluster", fallback_scope_id="__fallback__")
    cfg = {
        "keys": {"entity_col": "series_id"},
        "candidates": {"n_candidates": 3, "chunk_size": 8, "prediction_exclude_observed": True},
        "evaluation": {"selected_fractions": [0.05], "frontier_exclude_observed": True, "coverage_l1_tolerance": 0.01},
        "irl": {"set_near_optimal_eta": 0.0, "membership_mode": "support"},
    }
    ev, _ = evaluate_irl_model(panel, _CloseAlternativeCandGen(), _FeatureModel(), model, cfg, split="test")
    row = ev.iloc[0]
    assert bool(row["recommendation_benchmark_excludes_observed"]) is True
    assert bool(row["choice_coverage"]) is True
    assert bool(row["rationalization_choice_coverage_including_observed"]) is True
    assert int(row["recommendation_action_set_size_ex_observed"]) == 1
    assert row["frontier_f05_contains_observed_candidate"] is False or bool(row["frontier_f05_contains_observed_candidate"]) is False


def test_same_date_peer_off_does_not_use_test_date_peers():
    panel = pd.DataFrame({
        "split": ["train", "train", "test", "test"],
        "mandate_cluster": [0, 0, 0, 0],
        "report_date": pd.to_datetime(["2020-01-31", "2020-02-29", "2021-01-31", "2021-01-31"]),
        "series_id": ["a", "b", "c", "d"],
        "asset_EC": [0.1, 0.2, 0.9, 0.8],
        "asset_STIV": [0.9, 0.8, 0.1, 0.2],
    })
    bounds = BoundModel(
        action_cols=["asset_EC", "asset_STIV"],
        lower=np.zeros((len(panel), 2)),
        upper=np.ones((len(panel), 2)),
        row_index=panel.index,
        asset_action_cols=["asset_EC", "asset_STIV"],
        overlay_action_cols=[],
        mode="exogenous",
    )
    cfg = {"keys": {"date_col": "report_date"}, "candidates": {
        "include_observed_candidate": True,
        "same_date_peer_mode": "off",
        "peer_same_date_frac": 1.0,
        "peer_train_frac": 0.0,
        "self_frac": 0.0,
        "dirichlet_frac": 0.0,
        "n_candidates": 4,
    }}
    gen = CandidateGenerator(panel, ["asset_EC", "asset_STIV"], ["asset_EC", "asset_STIV"], [], "series_id", cfg, np.random.default_rng(1), bounds)
    batch = gen.generate_for_positions([2], n_candidates=4)
    assert "peer_same_date" not in set(batch.sources[0])
    assert not any(src.startswith("peer_same_date") for src in batch.sources[0])
    # The other test-date fund has action [0.8, 0.2]; it must not be pulled into
    # the default out-of-sample menu when same-date peers are disabled.
    alts = batch.actions[0, 1:, :]
    assert not np.any(np.all(np.isclose(alts, np.array([0.8, 0.2])), axis=1))


def test_summary_action_signed_split_and_drop_rows():
    cfg = {
        "actions": {
            "summary_action_cols": ["derivative_pct", "short_pct"],
            "percent_cols_are_0_100": True,
            "summary_action_cleaning": {
                "enabled": True,
                "nonnegative_cols": ["derivative_pct", "short_pct"],
                "upper_bounds": {"derivative_pct": 2.5, "short_pct": 2.5},
                "signed_split_cols": ["derivative_pct"],
                "drop_rows": {
                    "enabled": True,
                    "columns": ["short_pct"],
                    "lower_bounds": {"short_pct": 0.0},
                },
            },
        }
    }
    df = pd.DataFrame({
        "derivative_pct": [-20.0, 30.0, -10.0],
        "short_pct": [5.0, 10.0, -5.0],
    })
    out, overlay_cols, reports = _prepare_summary_action_columns(df, cfg)

    assert len(out) == 2
    assert "derivative_pct_pos" in overlay_cols
    assert "derivative_pct_neg_abs" in overlay_cols
    assert "derivative_pct" not in overlay_cols
    assert "short_pct" in overlay_cols
    assert np.allclose(out["derivative_pct_pos"].to_numpy(), [0.0, 0.30])
    assert np.allclose(out["derivative_pct_neg_abs"].to_numpy(), [0.20, 0.0])
    assert np.allclose(out["short_pct"].to_numpy(), [0.05, 0.10])
    assert any(r["column"] == "__row_filter__" and r["n_rows_dropped"] == 1 for r in reports)


def test_grouped_l1_metrics_are_reported(monkeypatch):
    panel = pd.DataFrame({"split": ["test"], "mandate_cluster": ["0"], "series_id": ["fundA"]})

    class _FeatureModel:
        feature_names = ["score"]

    class _GroupedCandGen:
        entity_col = "series_id"
        action_cols = ["asset_EC", "asset_STIV", "short_pct", "derivative_pct"]
        asset_cols = ["asset_EC", "asset_STIV"]
        overlay_cols = ["short_pct", "derivative_pct"]

        def __init__(self):
            self.bounds = BoundModel(
                action_cols=self.action_cols,
                lower=np.zeros((1, 4)),
                upper=np.ones((1, 4)),
                row_index=pd.RangeIndex(1),
                asset_action_cols=self.asset_cols,
                overlay_action_cols=self.overlay_cols,
                mode="exogenous",
            )

        def generate_for_positions(self, positions, n_candidates=None):
            actions = np.array([[
                [1.0, 0.0, 0.20, 0.30],
                [0.0, 1.0, 0.20, 0.30],
                [1.0, 0.0, 0.50, 0.70],
            ]], dtype=float)
            sources = np.array([["observed", "peer_train", "frontier_style"]], dtype=object)
            return CandidateBatch(row_positions=np.asarray(positions), actions=actions, sources=sources, observed_index=0)

    def fake_features(panel, batch, feature_model):
        return np.zeros((1, 3, 1), dtype=float)

    monkeypatch.setattr(irl_module, "compute_scaled_utility_features", fake_features)
    model = IRLModel(preferences={"0": _pref(), "__fallback__": _pref()}, feature_names=["score"], scope_col="mandate_cluster", fallback_scope_id="__fallback__")
    cfg = {
        "keys": {"entity_col": "series_id"},
        "candidates": {"n_candidates": 3, "chunk_size": 8, "prediction_exclude_observed": True},
        "evaluation": {"selected_fractions": [0.05], "frontier_exclude_observed": True, "coverage_l1_tolerance": 1e-12},
        "irl": {"set_near_optimal_eta": 0.0, "membership_mode": "support"},
    }
    ev, _ = evaluate_irl_model(panel, _GroupedCandGen(), _FeatureModel(), model, cfg, split="test")
    row = ev.iloc[0]

    assert np.isclose(row["distance_to_set_l1"], 0.70)
    assert np.isclose(row["distance_to_set_asset_only_l1"], 0.0)
    assert np.isclose(row["distance_to_set_overlay_only_l1"], 0.0)
    assert np.isclose(row["distance_to_set_risk_overlay_l1"], 0.0)
    metrics = summarize_metrics(ev)
    assert "distance_to_set_asset_only_l1_mean" in metrics.columns
    assert np.isclose(metrics.loc[0, "distance_to_set_asset_only_l1_mean"], 0.0)


def test_adv_keyword_soft_caps_are_opt_in_not_default():
    panel = pd.DataFrame({
        "split": ["train", "train", "test"],
        "series_id": ["a", "b", "c"],
        "mandate_cluster": [0, 0, 0],
        "asset_EC": [0.5, 0.5, 0.5],
        "asset_STIV": [0.5, 0.5, 0.5],
        "derivative_pct": [0.00, 0.12, 0.12],
        "short_pct": [0.00, 0.08, 0.08],
        "restricted_pct": [0.00, 0.20, 0.20],
        "allows_or_discusses_derivatives": [0, 0, 0],
        "shorting": [0, 0, 0],
        "liquidity_or_redemption_constraint": [1, 1, 1],
    })
    action_cols = ["asset_EC", "asset_STIV", "derivative_pct", "short_pct", "restricted_pct"]
    cfg = {"bounds": {"main_mode": "exogenous", "global_lower_q": 0.0, "global_upper_q": 1.0, "peer_lower_q": 0.0, "peer_upper_q": 1.0}}

    default_bounds = fit_mandate_bounds(
        panel, action_cols, ["asset_EC", "asset_STIV"], ["derivative_pct", "short_pct", "restricted_pct"], "series_id", cfg, mode="exogenous"
    )
    default_upper = pd.DataFrame(default_bounds.upper, columns=action_cols)
    assert np.isclose(default_upper["derivative_pct"].iloc[0], 0.12)
    assert np.isclose(default_upper["short_pct"].iloc[0], 0.08)
    assert np.isclose(default_upper["restricted_pct"].iloc[0], 0.20)

    with_adv_cfg = {"bounds": {**cfg["bounds"], "use_adv_soft_caps": True}}
    with_adv_bounds = fit_mandate_bounds(
        panel, action_cols, ["asset_EC", "asset_STIV"], ["derivative_pct", "short_pct", "restricted_pct"], "series_id", with_adv_cfg, mode="exogenous"
    )
    with_adv_upper = pd.DataFrame(with_adv_bounds.upper, columns=action_cols)
    assert np.isclose(with_adv_upper["derivative_pct"].iloc[0], 0.05)
    assert np.isclose(with_adv_upper["short_pct"].iloc[0], 0.03)
    assert np.isclose(with_adv_upper["restricted_pct"].iloc[0], 0.10)


def test_main_config_removes_adv_text_keywords_from_mandate_clusters_but_keeps_drp_controls():
    cfg = read_yaml("configs/main_32gb.yaml")
    features = set(cfg["clusters"]["feature_candidates"])

    assert ADV_TEXT_KEYWORD_FEATURES.isdisjoint(features)
    assert "adv_total_drp_records" in features
    assert "total_drp_records" in features
    assert "legal_disciplinary" in features
    assert cfg["bounds"]["use_adv_soft_caps"] is False
