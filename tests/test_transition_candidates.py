from __future__ import annotations

import numpy as np
import pandas as pd

from mandate_setirl.bounds import fit_mandate_bounds
from mandate_setirl.candidates import CandidateGenerator
from mandate_setirl.transitions import build_transition_library
from scripts.run_transition_menu_diagnostics import direction_cosine, sign_agreement


def _panel() -> pd.DataFrame:
    rows = []
    specs = [
        ("A", "train", 0, [0.70, 0.20, 0.10], [0.65, 0.25, 0.10], 10.0),
        ("A", "train", 1, [0.65, 0.25, 0.10], [0.55, 0.35, 0.10], 11.0),
        ("A", "train", 2, [0.55, 0.35, 0.10], [0.50, 0.40, 0.10], 12.0),
        ("B", "train", 0, [0.30, 0.60, 0.10], [0.35, 0.55, 0.10], 10.0),
        ("B", "train", 1, [0.35, 0.55, 0.10], [0.45, 0.45, 0.10], 11.0),
        ("B", "train", 2, [0.45, 0.45, 0.10], [0.55, 0.35, 0.10], 12.0),
        ("D", "train", 0, [0.90, 0.05, 0.05], [0.30, 0.60, 0.10], 13.0),
        ("C", "test", 0, [0.60, 0.30, 0.10], [0.58, 0.32, 0.10], 11.5),
    ]
    for sid, split, step, lag, cur, vix in specs:
        rows.append(
            {
                "series_id": sid,
                "report_date": pd.Timestamp("2020-01-31") + pd.DateOffset(months=3 * step),
                "split": split,
                "mandate_cluster": 0,
                "asset_EC": cur[0],
                "asset_DBT": cur[1],
                "asset_STIV": cur[2],
                "lag_asset_EC": lag[0],
                "lag_asset_DBT": lag[1],
                "lag_asset_STIV": lag[2],
                "long_pct": 1.0,
                "lag_long_pct": 1.0,
                "short_pct": 0.0,
                "lag_short_pct": 0.0,
                "derivative_pct": 0.0,
                "lag_derivative_pct": 0.0,
                "restricted_pct": 0.0,
                "lag_restricted_pct": 0.0,
                "log_net_assets": 10.0 + step,
                "net_flow_3m_scaled": 0.01 * step,
                "vix": vix,
            }
        )
    return pd.DataFrame(rows)


def _cfg() -> dict:
    return {
        "keys": {"entity_col": "series_id", "date_col": "report_date"},
        "bounds": {
            "global_lower_q": 0.0,
            "global_upper_q": 1.0,
            "peer_lower_q": 0.0,
            "peer_upper_q": 1.0,
            "min_width": 0.01,
            "use_adv_soft_caps": False,
        },
        "candidates": {
            "include_observed_candidate": True,
            "generator_mode": "transition_augmented",
            "same_date_peer_mode": "off",
            "n_candidates": 16,
            "peer_train_count": 2,
            "self_lag_exact_count": 1,
            "self_lag_projected_count": 1,
            "transition_knn_count": 4,
            "transition_regime_count": 2,
            "transition_shrunk_count": 2,
            "clean_frontier_count": 2,
        },
        "transition": {
            "enabled": True,
            "donor_split": "train",
            "min_move_l1": 0.001,
            "max_move_l1_quantile": 1.0,
            "scales": [0.5, 1.0],
            "retrieval_k": 4,
            "retrieval_state_weight": 0.7,
            "retrieval_lag_weight": 0.3,
            "state_columns": ["vix", "log_net_assets", "net_flow_3m_scaled"],
            "regime_min_count": 1,
        },
    }


def test_transition_library_uses_train_donors_only():
    panel = _panel()
    action_cols = ["asset_EC", "asset_DBT", "asset_STIV", "long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    lib = build_transition_library(panel, action_cols, "series_id", "report_date", _cfg())
    assert len(lib.donor_pos) > 0
    assert set(panel.loc[lib.donor_pos, "split"]) == {"train"}
    assert np.all(np.abs(lib.donor_delta).sum(axis=1) > 0.001)


def test_transition_library_excludes_query_row_as_donor_for_train_rows():
    panel = _panel()
    action_cols = ["asset_EC", "asset_DBT", "asset_STIV", "long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    lib = build_transition_library(panel, action_cols, "series_id", "report_date", _cfg())
    train_row = int(lib.donor_pos[0])
    picked, dist = lib.query_with_distance(train_row, k=4)
    assert train_row not in set(picked.tolist())
    assert len(picked) == len(dist)


def test_transition_augmented_candidates_are_no_leakage_and_feasible():
    panel = _panel()
    action_cols = ["asset_EC", "asset_DBT", "asset_STIV", "long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    asset_cols = ["asset_EC", "asset_DBT", "asset_STIV"]
    overlay_cols = ["long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    cfg = _cfg()
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, "series_id", cfg, mode="exogenous")
    gen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, "series_id", cfg, np.random.default_rng(1), bounds)
    pos = np.flatnonzero(panel["split"].eq("test").to_numpy())
    batch = gen.generate_for_positions(pos, n_candidates=16)
    sources = set(batch.sources[0].tolist())
    assert "observed" in sources
    assert "lag_exact" in sources or "lag_projected" in sources
    assert any(str(s).startswith("transition_knn") for s in sources)
    assert "frontier_style" not in sources
    assert hasattr(batch, "donor_pos")
    assert hasattr(batch, "donor_move_l1")
    assert hasattr(batch, "same_series_donor")
    assert hasattr(batch, "donor_retrieval_distance")
    transition_idx = [i for i, s in enumerate(batch.sources[0].tolist()) if str(s).startswith("transition_knn")]
    assert transition_idx
    assert np.all(batch.donor_pos[0, transition_idx] >= 0)
    assert np.all(np.isfinite(batch.donor_move_l1[0, transition_idx]))
    assert np.all(np.isfinite(batch.donor_retrieval_distance[0, transition_idx]))
    assert np.all(np.isfinite(batch.post_projection_l1_from_lag[0, 1:]))
    assert np.all(np.isfinite(batch.post_projection_l1_from_raw_transition_candidate[0, transition_idx]))
    assert np.allclose(batch.actions[0, 1:, :3].sum(axis=1), 1.0)
    lower, upper = bounds.bounds_for_positions(pos)
    assert np.all(batch.actions[0, 1:] >= lower[0] - 1e-8)
    assert np.all(batch.actions[0, 1:] <= upper[0] + 1e-8)


def test_transition_selection_deduplicates_before_truncation():
    panel = _panel()
    action_cols = ["asset_EC", "asset_DBT", "asset_STIV", "long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    asset_cols = ["asset_EC", "asset_DBT", "asset_STIV"]
    overlay_cols = ["long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    cfg = _cfg()
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, "series_id", cfg, mode="exogenous")
    gen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, "series_id", cfg, np.random.default_rng(1), bounds)
    pos = int(np.flatnonzero(panel["split"].eq("test").to_numpy())[0])
    lower, upper = bounds.bounds_for_positions([pos])
    a = gen.project_action(gen.lag_actions_all[pos].copy(), lower[0], upper[0])
    b = a.copy()
    b[0] -= 0.02
    b[1] += 0.02
    records = [
        gen._candidate_record("lag_exact", a.copy(), raw_action=a.copy()),
        gen._candidate_record("lag_projected", a.copy(), raw_action=a.copy()),
        gen._candidate_record("peer_static_train", b.copy(), raw_action=b.copy()),
    ]
    selected = gen._dedupe_and_select_records(pos, records, target_count=3, lower=lower[0], upper=upper[0])
    rounded = {tuple(np.round(r["action"], 6)) for r in selected}
    assert len(selected) == 2
    assert len(rounded) == 2


def test_large_transition_candidates_use_directional_target_magnitudes():
    panel = _panel()
    action_cols = ["asset_EC", "asset_DBT", "asset_STIV", "long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    asset_cols = ["asset_EC", "asset_DBT", "asset_STIV"]
    overlay_cols = ["long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    cfg = _cfg()
    cfg["candidates"].update(
        {
            "n_candidates": 20,
            "transition_knn_count": 0,
            "transition_shrunk_count": 0,
            "transition_regime_count": 0,
            "transition_small_count": 0,
            "transition_moderate_count": 0,
            "transition_large_count": 4,
            "transition_stress_count": 0,
            "clean_frontier_count": 0,
            "peer_train_count": 0,
        }
    )
    cfg["transition"]["direction_magnitudes"] = [0.08, 0.15]
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, "series_id", cfg, mode="exogenous")
    gen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, "series_id", cfg, np.random.default_rng(1), bounds)
    pos = np.flatnonzero(panel["split"].eq("test").to_numpy())
    batch = gen.generate_for_positions(pos, n_candidates=20)
    idx = [i for i, s in enumerate(batch.sources[0]) if str(s).startswith("transition_large_dir_m")]
    assert idx
    move_from_lag = batch.post_projection_l1_from_lag[0, idx]
    assert np.nanmax(move_from_lag) <= 0.151 + 1e-8
    assert np.nanmin(move_from_lag) >= 0.079 - 1e-8


def test_predicted_mover_routing_changes_source_quotas_without_test_move():
    panel = _panel()
    action_cols = ["asset_EC", "asset_DBT", "asset_STIV", "long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    asset_cols = ["asset_EC", "asset_DBT", "asset_STIV"]
    overlay_cols = ["long_pct", "short_pct", "derivative_pct", "restricted_pct"]
    cfg = _cfg()
    cfg["candidates"]["mover_routing_enabled"] = True
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, "series_id", cfg, mode="exogenous")
    gen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, "series_id", cfg, np.random.default_rng(1), bounds)
    stayer_q = gen._source_quotas(39, predicted_move_bin="stayer")
    large_q = gen._source_quotas(39, predicted_move_bin="large")
    assert large_q["peer_large_stress"] > stayer_q["peer_large_stress"]
    assert stayer_q["peer_small"] > large_q["peer_small"]
    p_move, pred_mag = gen.transition_library.predict_move_probability_and_magnitude(0, k=4)
    assert 0.0 <= p_move <= 1.0
    assert pred_mag >= 0.0


def test_direction_helpers_are_stable_for_zero_and_signed_moves():
    assert np.isnan(direction_cosine(np.array([0.0, 0.0]), np.array([1.0, 0.0])))
    assert np.isclose(direction_cosine(np.array([1.0, 0.0]), np.array([2.0, 0.0])), 1.0)
    assert np.isclose(direction_cosine(np.array([-1.0, 0.0]), np.array([2.0, 0.0])), -1.0)
    assert np.isclose(
        sign_agreement(np.array([1.0, -1.0, 0.0]), np.array([2.0, -3.0, 0.0])),
        1.0,
    )
    assert np.isclose(
        sign_agreement(np.array([1.0, -1.0, 0.0]), np.array([-2.0, -3.0, 1.0])),
        1.0 / 3.0,
    )
