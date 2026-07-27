from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .candidates import CandidateGenerator
from .features import UtilityFeatureModel, compute_scaled_utility_features
from .polytope import LinearPreferencePolytope, build_preference_polytope, candidate_is_near_optimal_exact
from .utils import chunk_indices, smart_write_table
from .bounds import action_violations, BoundModel


@dataclass
class IRLModel:
    preferences: dict[str, LinearPreferencePolytope]
    feature_names: list[str]
    scope_col: str
    fallback_scope_id: str
    estimator: str = "polyhedral_revealed_preference"

    def get_scope(self, scope_id: str) -> LinearPreferencePolytope:
        return self.preferences.get(str(scope_id), self.preferences[self.fallback_scope_id])


def make_scope_column(panel: pd.DataFrame, cfg: dict[str, Any], entity_col: str) -> tuple[pd.Series, str]:
    mode = cfg.get("irl", {}).get("preference_scope", "cluster")
    if mode == "cluster" or "mandate_cluster" not in panel.columns:
        return panel.get("mandate_cluster", pd.Series(0, index=panel.index)).astype(str), "mandate_cluster"
    if mode == "fund_cluster_fallback":
        train_counts = panel[panel["split"] == "train"].groupby(entity_col).size()
        min_obs = int(cfg.get("irl", {}).get("min_train_obs_per_scope", 60))
        scopes = []
        for _, row in panel.iterrows():
            ent = str(row[entity_col])
            if train_counts.get(row[entity_col], 0) >= min_obs:
                scopes.append(f"fund:{ent}")
            else:
                scopes.append(f"cluster:{row.get('mandate_cluster', 0)}")
        return pd.Series(scopes, index=panel.index), "preference_scope"
    return panel.get("mandate_cluster", pd.Series(0, index=panel.index)).astype(str), "mandate_cluster"


def collect_pairwise_constraints(
    positions: np.ndarray,
    panel: pd.DataFrame,
    candgen: CandidateGenerator,
    feature_model: UtilityFeatureModel,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Collect G rows where G theta <= eps encodes alt no better than observed.

    For each observation and each non-observed candidate m,
        G_{t,m} = phi(s_t, a_m) - phi(s_t, a_obs).
    """
    positions = np.asarray(positions, dtype=int)
    if len(positions) == 0:
        return np.zeros((0, len(feature_model.feature_names)), dtype=float)
    if cfg.get("irl", {}).get("train_require_observed_feasible", True):
        lower, upper = candgen.bounds.bounds_for_positions(positions)
        obs = candgen.actions_all[positions]
        feasible = ~action_violations(obs, lower, upper, tol=float(cfg.get("irl", {}).get("feasibility_tol", 1e-7)))
        positions = positions[feasible]
        if len(positions) == 0:
            return np.zeros((0, len(feature_model.feature_names)), dtype=float)
    chunk_size = int(cfg.get("candidates", {}).get("chunk_size", 512))
    M = int(cfg.get("candidates", {}).get("n_candidates", 64))
    rows = []
    for start, end in chunk_indices(len(positions), chunk_size):
        pos = positions[start:end]
        batch = candgen.generate_for_positions(pos, n_candidates=M)
        F = compute_scaled_utility_features(panel, batch, feature_model)  # B x M x D
        if F.shape[1] <= 1:
            continue
        delta = F[:, 1:, :] - F[:, [0], :]
        rows.append(delta.reshape(-1, F.shape[-1]))
    if not rows:
        return np.zeros((0, len(feature_model.feature_names)), dtype=float)
    G = np.vstack(rows)
    # Remove numerically empty constraints and clip extreme standardized values.
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(G, axis=1)
    G = G[norm > 1e-12]
    clip = float(cfg.get("irl", {}).get("constraint_clip", 12.0))
    if clip > 0:
        G = np.clip(G, -clip, clip)
    return G


def fit_set_valued_irl(
    panel: pd.DataFrame,
    candgen: CandidateGenerator,
    feature_model: UtilityFeatureModel,
    cfg: dict[str, Any],
    entity_col: str,
    rng: np.random.Generator,
    model_name: str = "mandate_polytope",
    out_dir: str | None = None,
) -> IRLModel:
    """Fit polyhedral revealed-preference sets by linear programming.

    The routine computes minimax slack and coordinatewise preference intervals.
    """
    irl_cfg = cfg.get("irl", {})
    scope_series, _ = make_scope_column(panel, cfg, entity_col)
    panel = panel.copy()
    panel["_scope_id"] = scope_series.astype(str)
    train_mask = panel["split"].to_numpy() == "train"
    train_pos_all = np.flatnonzero(train_mask)
    scopes = sorted(panel.loc[train_pos_all, "_scope_id"].dropna().astype(str).unique().tolist())
    if not scopes:
        scopes = ["0"]
        panel["_scope_id"] = "0"
        train_pos_all = np.arange(len(panel))

    max_train = int(irl_cfg.get("max_train_obs_per_scope", 6000))
    min_train = int(irl_cfg.get("min_train_obs_per_scope", 60))
    max_constraints = int(irl_cfg.get("max_constraints_per_scope", 50000))
    eps_add = float(irl_cfg.get("eps_additive", 0.02))
    eps_mult = float(irl_cfg.get("eps_multiplier", 1.05))

    prefs: dict[str, LinearPreferencePolytope] = {}
    fallback_positions = []
    d = len(feature_model.feature_names)

    for scope in tqdm(scopes, desc=f"fit {model_name} LP polytopes"):
        pos = np.flatnonzero(train_mask & (panel["_scope_id"].astype(str).to_numpy() == str(scope)))
        if len(pos) < min_train:
            continue
        if len(pos) > max_train:
            pos = rng.choice(pos, size=max_train, replace=False)
        fallback_positions.append(pos)
        G = collect_pairwise_constraints(pos, panel, candgen, feature_model, cfg)
        if G.shape[0] == 0:
            G = np.zeros((0, d), dtype=float)
        pref = build_preference_polytope(
            str(scope), G, len(pos), eps_additive=eps_add, eps_multiplier=eps_mult,
            max_constraints=max_constraints, rng=rng,
        )
        prefs[str(scope)] = pref

    # Fallback preference from pooled train observations.
    if fallback_positions:
        pooled = np.concatenate(fallback_positions)
    else:
        pooled = train_pos_all if len(train_pos_all) else np.arange(len(panel))
    if len(pooled) > max_train:
        pooled = rng.choice(pooled, size=max_train, replace=False)
    G_global = collect_pairwise_constraints(pooled, panel, candgen, feature_model, cfg)
    if G_global.shape[0] == 0:
        G_global = np.zeros((0, d), dtype=float)
    fallback_id = "__fallback__"
    prefs[fallback_id] = build_preference_polytope(
        fallback_id, G_global, len(pooled), eps_additive=eps_add, eps_multiplier=eps_mult,
        max_constraints=max_constraints, rng=rng,
    )

    model = IRLModel(preferences=prefs, feature_names=feature_model.feature_names, scope_col="_scope_id", fallback_scope_id=fallback_id)
    if out_dir:
        save_preference_summary(model, out_dir, model_name)
    return model


def save_preference_summary(model: IRLModel, out_dir: str, model_name: str):
    rows = []
    for scope, pref in model.preferences.items():
        if scope == model.fallback_scope_id:
            scope_type = "fallback"
        elif scope.startswith("fund:"):
            scope_type = "fund"
        elif scope.startswith("cluster:"):
            scope_type = "cluster"
        else:
            scope_type = "scope"
        row = {
            "model": model_name,
            "scope_id": scope,
            "scope_type": scope_type,
            "n_train_obs": pref.n_train_obs,
            "n_constraints": pref.n_constraints,
            "eps_star_minimax_regret": pref.eps_star,
            "eps_bound": pref.eps_bound,
            "feasible": pref.feasible,
            "status": pref.status,
            "support_thetas": int(pref.support_thetas.shape[0]),
            "theta_interval_l1_upper_diameter": pref.interval_l1_upper_diameter,
        }
        for j, name in enumerate(model.feature_names):
            row[f"theta_point_{name}"] = float(pref.theta_point[j])
            row[f"theta_min_{name}"] = float(pref.theta_min[j])
            row[f"theta_max_{name}"] = float(pref.theta_max[j])
            row[f"theta_width_{name}"] = float(pref.theta_max[j] - pref.theta_min[j])
        rows.append(row)
    smart_write_table(pd.DataFrame(rows), f"{out_dir}/scope_preference_summary_{model_name}.csv")


def _support_membership(
    pref: LinearPreferencePolytope,
    F: np.ndarray,
    eta: float,
    use_point: bool = False,
    benchmark_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta = pref.theta_point[None, :] if use_point else pref.support_thetas
    if theta.size == 0:
        theta = pref.theta_point[None, :]
    scores = F @ theta.T  # M x T
    if benchmark_mask is None:
        benchmark_scores = scores
    else:
        mask = np.asarray(benchmark_mask, dtype=bool)
        if mask.shape[0] != scores.shape[0]:
            raise ValueError("benchmark_mask must have one entry per candidate")
        benchmark_scores = scores[mask]
        if benchmark_scores.shape[0] == 0:
            benchmark_scores = scores
    max_by_theta = benchmark_scores.max(axis=0, keepdims=True)
    gaps = max_by_theta - scores
    near = gaps <= eta
    in_set = near.any(axis=1)
    regret_by_theta = max_by_theta.ravel() - scores[0, :]
    # Smallest support gap required to include each candidate.
    best_gap = gaps.min(axis=1)
    return in_set, regret_by_theta, scores, best_gap


def _exact_membership(
    pref: LinearPreferencePolytope,
    F: np.ndarray,
    eta: float,
    max_candidates: int | None = None,
    benchmark_mask: np.ndarray | None = None,
) -> np.ndarray:
    M = F.shape[0]
    in_set = np.zeros(M, dtype=bool)
    # Always test observed and then candidates with strong support scores first.
    order = np.arange(M)
    if max_candidates is not None and max_candidates > 0 and M > max_candidates:
        supp_scores = F @ pref.theta_point
        rest = np.argsort(-supp_scores)
        order = np.unique(np.concatenate([[0], rest[:max_candidates]])).astype(int)
    for m in order:
        in_set[m] = candidate_is_near_optimal_exact(pref, F, int(m), eta, benchmark_mask=benchmark_mask)
    return in_set


def evaluate_irl_model(
    panel: pd.DataFrame,
    candgen: CandidateGenerator,
    feature_model: UtilityFeatureModel,
    model: IRLModel,
    cfg: dict[str, Any],
    split: str = "test",
    use_point: bool = False,
    model_name: str = "mandate_polytope",
    reference_bounds: BoundModel | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate rationalization and recommendation metrics.

    Candidate 0 is the observed allocation. Rationalization metrics include it;
    recommendation, coverage, frontier, and distance metrics exclude it.
    """
    irl_cfg = cfg.get("irl", {})
    eval_cfg = cfg.get("evaluation", {})
    eta = float(irl_cfg.get("set_near_optimal_eta", 0.05))
    recommendation_l1_tol = float(eval_cfg.get(
        "recommendation_distance_tol_l1",
        eval_cfg.get("coverage_l1_tolerance", eval_cfg.get("coverage_distance_tol", 0.05)),
    ))
    frontier_distance_tol = float(eval_cfg.get(
        "frontier_distance_tol_l1",
        eval_cfg.get("recommendation_distance_tol_l1", eval_cfg.get("coverage_l1_tolerance", eval_cfg.get("coverage_distance_tol", 0.05))),
    ))
    exclude_obs_frontier = bool(eval_cfg.get("frontier_exclude_observed", True))
    report_rp_frontier = bool(eval_cfg.get("report_including_observed_frontier", True))
    prediction_exclude_observed = bool(
        cfg.get("candidates", {}).get(
            "prediction_exclude_observed",
            eval_cfg.get("exclude_observed_candidate_for_recommendation", True),
        )
    )

    positions = np.flatnonzero(panel["split"].to_numpy() == split)
    if len(positions) == 0:
        return pd.DataFrame(), pd.DataFrame()
    max_eval = irl_cfg.get("max_eval_obs")
    if max_eval is not None and len(positions) > int(max_eval):
        idx = np.linspace(0, len(positions) - 1, int(max_eval)).round().astype(int)
        positions = positions[idx]

    chunk_size = int(cfg.get("candidates", {}).get("chunk_size", 512))
    M = int(cfg.get("candidates", {}).get("n_candidates", 64))
    membership_mode = str(irl_cfg.get("membership_mode", "support")).lower()
    exact_max_candidates = irl_cfg.get("exact_max_candidates_per_obs")
    exact_max_candidates = None if exact_max_candidates is None else int(exact_max_candidates)
    selected_fractions = [float(x) for x in eval_cfg.get("selected_fractions", [0.05, 0.10, 0.20, 0.30, 0.50])]

    ref_bounds = reference_bounds or candgen.bounds
    action_cols = list(getattr(candgen, "action_cols", getattr(ref_bounds, "action_cols", [])))
    asset_cols = list(getattr(candgen, "asset_cols", getattr(ref_bounds, "asset_action_cols", [])))
    overlay_cols = list(getattr(candgen, "overlay_cols", getattr(ref_bounds, "overlay_action_cols", [])))
    risk_overlay_cols = list(eval_cfg.get("risk_overlay_cols", [
        "short_pct", "short_pct_pos", "short_pct_neg_abs",
        "derivative_pct", "derivative_pct_pos", "derivative_pct_neg_abs",
        "restricted_pct",
    ]))
    idx_by_col = {c: j for j, c in enumerate(action_cols)}
    asset_idx = np.asarray([idx_by_col[c] for c in asset_cols if c in idx_by_col], dtype=int)
    overlay_idx = np.asarray([idx_by_col[c] for c in overlay_cols if c in idx_by_col], dtype=int)
    risk_overlay_idx = np.asarray([idx_by_col[c] for c in risk_overlay_cols if c in idx_by_col], dtype=int)
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    scope_series, _ = make_scope_column(panel, cfg, candgen.entity_col)
    scope_arr = scope_series.astype(str).to_numpy()

    def _dist_to_obs(actions: np.ndarray, obs: np.ndarray, idx: np.ndarray | None = None) -> float:
        if actions is None or len(actions) == 0:
            return float(np.nan)
        if idx is not None:
            if len(idx) == 0:
                return float(np.nan)
            actions = actions[:, idx]
            obs = obs[idx]
        return float(np.min(np.abs(actions - obs).sum(axis=1)))

    def _single_l1(action: np.ndarray, obs: np.ndarray, idx: np.ndarray | None = None) -> float:
        if idx is not None:
            if len(idx) == 0:
                return float(np.nan)
            action = action[idx]
            obs = obs[idx]
        return float(np.abs(action - obs).sum())

    for start, end in tqdm(list(chunk_indices(len(positions), chunk_size)), desc=f"eval {model_name} {split}"):
        pos = positions[start:end]
        batch = candgen.generate_for_positions(pos, n_candidates=M)
        F_all = compute_scaled_utility_features(panel, batch, feature_model)
        A = batch.actions
        lower_b, upper_b = ref_bounds.bounds_for_positions(pos)
        violations = action_violations(A, lower_b, upper_b)

        for bi, row_pos in enumerate(pos):
            pref = model.get_scope(scope_arr[row_pos])
            F = F_all[bi]
            obs_action = A[bi, 0]
            in_set_support_rat, regret_by_theta, scores, best_gap_rat = _support_membership(pref, F, eta, use_point=use_point)
            rec_benchmark_mask = np.ones(F.shape[0], dtype=bool)
            if F.shape[0] > 0:
                rec_benchmark_mask[0] = False
            in_set_support_rec, _, _, best_gap_rec = _support_membership(
                pref, F, eta, use_point=use_point, benchmark_mask=rec_benchmark_mask
            )
            if membership_mode == "lp_exact" and not use_point:
                in_set_rat = _exact_membership(pref, F, eta, exact_max_candidates)
                in_set_rec = _exact_membership(pref, F, eta, exact_max_candidates, benchmark_mask=rec_benchmark_mask)
            else:
                in_set_rat = in_set_support_rat
                in_set_rec = in_set_support_rec

            lower_i, upper_i = ref_bounds.bounds_for_positions([row_pos])
            observed_feasible = bool(~action_violations(
                obs_action[None, :], lower_i, upper_i,
                tol=float(irl_cfg.get("feasibility_tol", 1e-7)),
            )[0])
            obs_in_set = bool(in_set_rat[0])

            # Rationalization set includes observed candidate.
            rp_idx = np.flatnonzero(in_set_rat)
            if len(rp_idx) == 0:
                rp_idx = np.array([int(np.argmax(scores.max(axis=1)))])
            rp_actions = A[bi, rp_idx]
            rp_viol = violations[bi, rp_idx]
            rp_compliant = rp_actions[~rp_viol]
            rp_dist = _dist_to_obs(rp_actions, obs_action)
            rp_dist_comp = _dist_to_obs(rp_compliant, obs_action)
            rp_dist_asset = _dist_to_obs(rp_actions, obs_action, asset_idx)
            rp_dist_overlay = _dist_to_obs(rp_actions, obs_action, overlay_idx)
            rp_dist_risk = _dist_to_obs(rp_actions, obs_action, risk_overlay_idx)
            rp_dist_comp_asset = _dist_to_obs(rp_compliant, obs_action, asset_idx)
            rp_dist_comp_overlay = _dist_to_obs(rp_compliant, obs_action, overlay_idx)
            rp_dist_comp_risk = _dist_to_obs(rp_compliant, obs_action, risk_overlay_idx)
            if len(rp_idx) > 1:
                rp_diameter = float(np.max(np.abs(rp_actions[:, None, :] - rp_actions[None, :, :]).sum(axis=2)))
            else:
                rp_diameter = 0.0

            # Recommendation set excludes observed candidate.
            rec_idx = np.flatnonzero(in_set_rec)
            rec_idx = rec_idx[rec_idx != 0]
            rec_actions = A[bi, rec_idx] if len(rec_idx) else np.empty((0, A.shape[-1]), dtype=float)
            rec_viol = violations[bi, rec_idx] if len(rec_idx) else np.asarray([], dtype=bool)
            rec_compliant = rec_actions[~rec_viol] if len(rec_actions) else np.empty((0, A.shape[-1]), dtype=float)
            rec_dist = _dist_to_obs(rec_actions, obs_action)
            rec_dist_comp = _dist_to_obs(rec_compliant, obs_action)
            rec_dist_asset = _dist_to_obs(rec_actions, obs_action, asset_idx)
            rec_dist_overlay = _dist_to_obs(rec_actions, obs_action, overlay_idx)
            rec_dist_risk = _dist_to_obs(rec_actions, obs_action, risk_overlay_idx)
            rec_dist_comp_asset = _dist_to_obs(rec_compliant, obs_action, asset_idx)
            rec_dist_comp_overlay = _dist_to_obs(rec_compliant, obs_action, overlay_idx)
            rec_dist_comp_risk = _dist_to_obs(rec_compliant, obs_action, risk_overlay_idx)
            rec_choice_coverage = bool(np.isfinite(rec_dist) and rec_dist <= recommendation_l1_tol)
            rec_comp_coverage = bool(np.isfinite(rec_dist_comp) and rec_dist_comp <= recommendation_l1_tol)
            rec_exact_match = bool(np.isfinite(rec_dist) and rec_dist <= 1e-10)
            if len(rec_idx) > 1:
                rec_diameter = float(np.max(np.abs(rec_actions[:, None, :] - rec_actions[None, :, :]).sum(axis=2)))
            elif len(rec_idx) == 1:
                rec_diameter = 0.0
            else:
                rec_diameter = float(np.nan)
            rec_denom = max(A.shape[1] - 1, 1)

            point_scores = F @ pref.theta_point
            pred_idx_rationalization = int(np.argmax(point_scores))
            if len(point_scores) > 1:
                pred_idx_ex_obs = int(1 + np.argmax(point_scores[1:]))
            else:
                pred_idx_ex_obs = -1
            # Candidate 0 enters rationalization constraints but not recommendations.
            pred_idx = pred_idx_ex_obs if prediction_exclude_observed and pred_idx_ex_obs >= 0 else pred_idx_rationalization
            if pred_idx >= 0:
                point_pred_violation = bool(violations[bi, pred_idx])
                point_pred_source = str(batch.sources[bi, pred_idx])
                point_l1 = _single_l1(A[bi, pred_idx], obs_action)
                point_l1_asset = _single_l1(A[bi, pred_idx], obs_action, asset_idx)
                point_l1_overlay = _single_l1(A[bi, pred_idx], obs_action, overlay_idx)
                point_l1_risk = _single_l1(A[bi, pred_idx], obs_action, risk_overlay_idx)
            else:
                point_pred_violation = np.nan
                point_pred_source = "none_ex_observed"
                point_l1 = float(np.nan)
                point_l1_asset = float(np.nan)
                point_l1_overlay = float(np.nan)
                point_l1_risk = float(np.nan)
            if pred_idx_ex_obs >= 0:
                point_pred_violation_ex = bool(violations[bi, pred_idx_ex_obs])
                point_pred_source_ex = str(batch.sources[bi, pred_idx_ex_obs])
                point_l1_ex = _single_l1(A[bi, pred_idx_ex_obs], obs_action)
                point_l1_ex_asset = _single_l1(A[bi, pred_idx_ex_obs], obs_action, asset_idx)
                point_l1_ex_overlay = _single_l1(A[bi, pred_idx_ex_obs], obs_action, overlay_idx)
                point_l1_ex_risk = _single_l1(A[bi, pred_idx_ex_obs], obs_action, risk_overlay_idx)
            else:
                point_pred_violation_ex = np.nan
                point_pred_source_ex = "none_ex_observed"
                point_l1_ex = float(np.nan)
                point_l1_ex_asset = float(np.nan)
                point_l1_ex_overlay = float(np.nan)
                point_l1_ex_risk = float(np.nan)
            point_pred_violation_rp = bool(violations[bi, pred_idx_rationalization])
            point_pred_source_rp = str(batch.sources[bi, pred_idx_rationalization])
            point_l1_rp = _single_l1(A[bi, pred_idx_rationalization], obs_action)
            point_l1_rp_asset = _single_l1(A[bi, pred_idx_rationalization], obs_action, asset_idx)
            point_l1_rp_overlay = _single_l1(A[bi, pred_idx_rationalization], obs_action, overlay_idx)
            point_l1_rp_risk = _single_l1(A[bi, pred_idx_rationalization], obs_action, risk_overlay_idx)

            row_out = {
                "model": model_name,
                "split": split,
                "row_pos": int(row_pos),
                "scope_id": str(scope_arr[row_pos]),
                "recommendation_l1_tolerance": recommendation_l1_tol,
                "frontier_l1_tolerance": frontier_distance_tol,
                "observed_feasible_under_bounds": observed_feasible,
                "observed_mandate_violation": bool(violations[bi, 0]),

                # Recommendation metrics exclude candidate 0.
                "choice_coverage": rec_choice_coverage,
                "choice_coverage_given_feasible": bool(rec_choice_coverage and observed_feasible),
                "distance_to_set_l1": rec_dist,
                "distance_to_compliant_set_l1": rec_dist_comp,
                "distance_to_set_asset_only_l1": rec_dist_asset,
                "distance_to_set_overlay_only_l1": rec_dist_overlay,
                "distance_to_set_risk_overlay_l1": rec_dist_risk,
                "distance_to_compliant_set_asset_only_l1": rec_dist_comp_asset,
                "distance_to_compliant_set_overlay_only_l1": rec_dist_comp_overlay,
                "distance_to_compliant_set_risk_overlay_l1": rec_dist_comp_risk,
                "action_set_size": int(len(rec_idx)),
                "action_set_fraction": float(len(rec_idx) / rec_denom),
                "action_set_diameter_l1": rec_diameter,
                "action_set_violation_rate": float(rec_viol.mean()) if len(rec_viol) else np.nan,
                "action_set_all_compliant": bool((~rec_viol).all()) if len(rec_viol) else False,
                "empty_predictive_action_set": bool(len(rec_idx) == 0),

                # Recommendation aliases.
                "recommendation_choice_coverage": rec_choice_coverage,
                "recommendation_choice_coverage_ex_observed": rec_choice_coverage,
                "recommendation_compliant_choice_coverage": rec_comp_coverage,
                "recommendation_compliant_choice_coverage_ex_observed": rec_comp_coverage,
                "recommendation_exact_match_coverage": rec_exact_match,
                "recommendation_exact_match_coverage_ex_observed": rec_exact_match,
                "recommendation_distance_to_observed_l1": rec_dist,
                "recommendation_distance_to_set_l1_ex_observed": rec_dist,
                "recommendation_distance_to_compliant_set_l1": rec_dist_comp,
                "recommendation_distance_to_compliant_set_l1_ex_observed": rec_dist_comp,
                "recommendation_distance_to_observed_asset_only_l1": rec_dist_asset,
                "recommendation_distance_to_observed_overlay_only_l1": rec_dist_overlay,
                "recommendation_distance_to_observed_risk_overlay_l1": rec_dist_risk,
                "recommendation_distance_to_compliant_set_asset_only_l1": rec_dist_comp_asset,
                "recommendation_distance_to_compliant_set_overlay_only_l1": rec_dist_comp_overlay,
                "recommendation_distance_to_compliant_set_risk_overlay_l1": rec_dist_comp_risk,
                "recommendation_action_set_size": int(len(rec_idx)),
                "recommendation_action_set_size_ex_observed": int(len(rec_idx)),
                "recommendation_action_set_fraction": float(len(rec_idx) / rec_denom),
                "recommendation_action_set_fraction_ex_observed": float(len(rec_idx) / rec_denom),
                "recommendation_action_set_diameter_l1": rec_diameter,
                "recommendation_action_set_diameter_l1_ex_observed": rec_diameter,
                "recommendation_action_set_violation_rate": float(rec_viol.mean()) if len(rec_viol) else np.nan,
                "recommendation_action_set_violation_rate_ex_observed": float(rec_viol.mean()) if len(rec_viol) else np.nan,

                # Rationalization diagnostics include candidate 0.
                "rationalization_choice_coverage": obs_in_set,
                "rationalization_choice_coverage_including_observed": obs_in_set,
                "rationalization_choice_coverage_given_feasible": bool(obs_in_set and observed_feasible),
                "rationalization_distance_to_set_l1": rp_dist,
                "rationalization_distance_to_set_l1_including_observed": rp_dist,
                "rationalization_distance_to_compliant_set_l1": rp_dist_comp,
                "rationalization_distance_to_compliant_set_l1_including_observed": rp_dist_comp,
                "rationalization_distance_to_set_asset_only_l1": rp_dist_asset,
                "rationalization_distance_to_set_overlay_only_l1": rp_dist_overlay,
                "rationalization_distance_to_set_risk_overlay_l1": rp_dist_risk,
                "rationalization_distance_to_compliant_set_asset_only_l1": rp_dist_comp_asset,
                "rationalization_distance_to_compliant_set_overlay_only_l1": rp_dist_comp_overlay,
                "rationalization_distance_to_compliant_set_risk_overlay_l1": rp_dist_comp_risk,
                "rationalization_action_set_size": int(len(rp_idx)),
                "rationalization_action_set_size_including_observed": int(len(rp_idx)),
                "rationalization_action_set_fraction": float(len(rp_idx) / max(A.shape[1], 1)),
                "rationalization_action_set_fraction_including_observed": float(len(rp_idx) / max(A.shape[1], 1)),
                "rationalization_action_set_diameter_l1": rp_diameter,
                "rationalization_action_set_diameter_l1_including_observed": rp_diameter,
                "rationalization_action_set_violation_rate": float(rp_viol.mean()) if len(rp_viol) else np.nan,
                "rationalization_action_set_violation_rate_including_observed": float(rp_viol.mean()) if len(rp_viol) else np.nan,

                "point_prediction_excluded_observed": bool(prediction_exclude_observed),
                "point_pred_mandate_violation": point_pred_violation,
                "point_pred_mandate_violation_ex_observed": point_pred_violation_ex,
                "point_pred_mandate_violation_rationalization_including_observed": point_pred_violation_rp,
                "point_pred_source": point_pred_source,
                "point_pred_source_ex_observed": point_pred_source_ex,
                "point_pred_source_rationalization_including_observed": point_pred_source_rp,
                "point_top1_observed_in_coverage_menu": bool(pred_idx_rationalization == 0),
                "point_l1_to_observed": point_l1,
                "point_l1_to_observed_ex_observed": point_l1_ex,
                "point_l1_to_observed_rationalization_including_observed": point_l1_rp,
                "point_l1_to_observed_asset_only": point_l1_asset,
                "point_l1_to_observed_overlay_only": point_l1_overlay,
                "point_l1_to_observed_risk_overlay": point_l1_risk,
                "point_l1_to_observed_ex_observed_asset_only": point_l1_ex_asset,
                "point_l1_to_observed_ex_observed_overlay_only": point_l1_ex_overlay,
                "point_l1_to_observed_ex_observed_risk_overlay": point_l1_ex_risk,
                "point_l1_to_observed_rationalization_asset_only": point_l1_rp_asset,
                "point_l1_to_observed_rationalization_overlay_only": point_l1_rp_overlay,
                "point_l1_to_observed_rationalization_risk_overlay": point_l1_rp_risk,

                "regret_min_support": float(np.min(regret_by_theta)),
                "regret_median_support": float(np.median(regret_by_theta)),
                "regret_max_support": float(np.max(regret_by_theta)),
                "observed_best_gap": float(best_gap_rat[0]),
                "recommendation_benchmark_excludes_observed": True,
                "theta_interval_l1_upper_diameter": pref.interval_l1_upper_diameter,
                "eps_bound": pref.eps_bound,
                "membership_mode": membership_mode if not use_point else "point",
                "candidate_bounds_mode": getattr(candgen.bounds, "mode", "unknown"),
                "violation_reference_bounds_mode": getattr(ref_bounds, "mode", "unknown"),
                "n_support_theta": int(pref.support_thetas.shape[0]),
            }

            order_all = np.argsort(best_gap_rat, kind="mergesort")
            order_rec_all = np.argsort(best_gap_rec, kind="mergesort")
            order_rec = order_rec_all[order_rec_all != 0] if exclude_obs_frontier else order_rec_all
            denom_rec_frontier = max(len(order_rec), 1)
            for frac in selected_fractions:
                if not np.isfinite(frac) or frac <= 0:
                    continue
                tag = str(int(round(frac * 100))).zfill(2)

                # Recommendation frontier excludes candidate 0.
                if len(order_rec):
                    k = int(np.ceil(min(max(frac, 0.0), 1.0) * len(order_rec)))
                    k = max(1, min(k, len(order_rec)))
                    idx_f = order_rec[:k]
                    if exclude_obs_frontier and np.any(idx_f == 0):
                        raise RuntimeError("Invariant error: observed candidate leaked into recommendation frontier")
                    actions_f = A[bi, idx_f]
                    viol_f = violations[bi, idx_f]
                    compliant_f = actions_f[~viol_f]
                    dist_f = _dist_to_obs(actions_f, obs_action)
                    dist_comp_f = _dist_to_obs(compliant_f, obs_action)
                    rec_cov_f = bool(np.isfinite(dist_f) and dist_f <= frontier_distance_tol)
                    viol_rate_f = float(viol_f.mean()) if len(viol_f) else np.nan
                else:
                    k = 0
                    dist_f = np.nan
                    dist_comp_f = np.nan
                    rec_cov_f = False
                    viol_rate_f = np.nan
                row_out[f"frontier_f{tag}_choice_coverage"] = rec_cov_f
                row_out[f"frontier_f{tag}_recommendation_choice_coverage"] = rec_cov_f
                row_out[f"frontier_f{tag}_recommendation_coverage_l1tol"] = rec_cov_f
                row_out[f"frontier_f{tag}_recommendation_exact_match_coverage"] = bool(np.isfinite(dist_f) and dist_f <= 1e-10)
                row_out[f"frontier_f{tag}_distance_to_set_l1"] = dist_f
                row_out[f"frontier_f{tag}_recommendation_distance_to_set_l1"] = dist_f
                row_out[f"frontier_f{tag}_action_set_size"] = int(k)
                row_out[f"frontier_f{tag}_action_set_fraction"] = float(k / denom_rec_frontier) if denom_rec_frontier else np.nan
                row_out[f"frontier_f{tag}_violation_rate"] = viol_rate_f
                row_out[f"frontier_f{tag}_distance_to_compliant_set_l1"] = dist_comp_f
                row_out[f"frontier_f{tag}_recommendation_distance_to_compliant_set_l1"] = dist_comp_f
                row_out[f"frontier_f{tag}_recommendation_compliant_choice_coverage"] = bool(np.isfinite(dist_comp_f) and dist_comp_f <= frontier_distance_tol)
                row_out[f"frontier_f{tag}_recommendation_action_set_size"] = int(k)
                row_out[f"frontier_f{tag}_recommendation_action_set_fraction"] = float(k / denom_rec_frontier) if denom_rec_frontier else np.nan
                row_out[f"frontier_f{tag}_recommendation_violation_rate"] = viol_rate_f
                row_out[f"frontier_f{tag}_contains_observed_candidate"] = False
                row_out[f"frontier_f{tag}_excluded_observed_candidate"] = bool(exclude_obs_frontier)
                row_out[f"frontier_f{tag}_candidate_menu_denominator"] = int(denom_rec_frontier)

                # Rationalization frontier includes candidate 0.
                if report_rp_frontier:
                    k_all = int(np.ceil(min(max(frac, 0.0), 1.0) * len(order_all)))
                    k_all = max(1, min(k_all, len(order_all)))
                    idx_all = order_all[:k_all]
                    actions_all = A[bi, idx_all]
                    viol_all = violations[bi, idx_all]
                    row_out[f"frontier_f{tag}_rationalization_choice_coverage"] = bool(np.any(idx_all == 0))
                    row_out[f"frontier_f{tag}_rp_choice_coverage_including_observed"] = bool(np.any(idx_all == 0))
                    row_out[f"frontier_f{tag}_rationalization_distance_to_set_l1"] = _dist_to_obs(actions_all, obs_action)
                    row_out[f"frontier_f{tag}_rp_distance_to_set_l1_including_observed"] = _dist_to_obs(actions_all, obs_action)
                    row_out[f"frontier_f{tag}_rationalization_action_set_size"] = int(k_all)
                    row_out[f"frontier_f{tag}_rp_action_set_size_including_observed"] = int(k_all)
                    row_out[f"frontier_f{tag}_rationalization_action_set_fraction"] = float(k_all / max(len(order_all), 1))
                    row_out[f"frontier_f{tag}_rp_action_set_fraction_including_observed"] = float(k_all / max(len(order_all), 1))
                    row_out[f"frontier_f{tag}_rationalization_violation_rate"] = float(viol_all.mean()) if len(viol_all) else np.nan
                    row_out[f"frontier_f{tag}_rp_violation_rate_including_observed"] = float(viol_all.mean()) if len(viol_all) else np.nan
            rows.append(row_out)

            compat = -float(np.min(regret_by_theta))
            pred_rows.append({
                "row_pos": int(row_pos),
                "scope_id": str(scope_arr[row_pos]),
                "compatibility_score": compat,
                "compatibility_regret_min_support": float(np.min(regret_by_theta)),
                "compatibility_regret_median_support": float(np.median(regret_by_theta)),
                "choice_coverage": rec_choice_coverage,
                "recommendation_choice_coverage_ex_observed": rec_choice_coverage,
                "rationalization_choice_coverage_including_observed": obs_in_set,
                "observed_feasible_under_bounds": observed_feasible,
                "action_set_size": int(len(rec_idx)),
                "recommendation_action_set_size_ex_observed": int(len(rec_idx)),
                "rationalization_action_set_size_including_observed": int(len(rp_idx)),
                "recommendation_distance_to_set_l1_ex_observed": rec_dist,
                "recommendation_distance_to_set_asset_only_l1": rec_dist_asset,
                "recommendation_distance_to_set_overlay_only_l1": rec_dist_overlay,
                "recommendation_distance_to_set_risk_overlay_l1": rec_dist_risk,
                "observed_mandate_violation": bool(violations[bi, 0]),
                "action_set_violation_rate": float(rec_viol.mean()) if len(rec_viol) else np.nan,
                "recommendation_action_set_violation_rate_ex_observed": float(rec_viol.mean()) if len(rec_viol) else np.nan,
                "rationalization_action_set_violation_rate_including_observed": float(rp_viol.mean()) if len(rp_viol) else np.nan,
                "model": model_name,
            })
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def summarize_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    rows = []
    for (model, split), g in eval_df.groupby(["model", "split"], dropna=False):
        def mean_col(col: str) -> float:
            return float(pd.to_numeric(g[col], errors="coerce").mean()) if col in g.columns else np.nan
        def q_col(col: str, q: float) -> float:
            return float(pd.to_numeric(g[col], errors="coerce").quantile(q)) if col in g.columns else np.nan
        row = {
            "model": model,
            "split": split,
            "n": len(g),
            # Recommendation metrics exclude candidate 0.
            "choice_coverage": mean_col("choice_coverage"),
            "predictive_choice_coverage": mean_col("predictive_choice_coverage"),
            "recommendation_choice_coverage": mean_col("recommendation_choice_coverage"),
            "recommendation_compliant_choice_coverage": mean_col("recommendation_compliant_choice_coverage"),
            "choice_distance_tol_l1_median": q_col("choice_distance_tol_l1", 0.5),
            "choice_coverage_given_feasible": float(g.loc[g["observed_feasible_under_bounds"].astype(bool), "choice_coverage"].mean()) if "observed_feasible_under_bounds" in g and g["observed_feasible_under_bounds"].astype(bool).any() else np.nan,
            "distance_to_set_l1_mean": mean_col("distance_to_set_l1"),
            "distance_to_compliant_set_l1_mean": mean_col("distance_to_compliant_set_l1"),
            "distance_to_set_asset_only_l1_mean": mean_col("distance_to_set_asset_only_l1"),
            "distance_to_set_overlay_only_l1_mean": mean_col("distance_to_set_overlay_only_l1"),
            "distance_to_set_risk_overlay_l1_mean": mean_col("distance_to_set_risk_overlay_l1"),
            "distance_to_compliant_set_asset_only_l1_mean": mean_col("distance_to_compliant_set_asset_only_l1"),
            "distance_to_compliant_set_overlay_only_l1_mean": mean_col("distance_to_compliant_set_overlay_only_l1"),
            "distance_to_compliant_set_risk_overlay_l1_mean": mean_col("distance_to_compliant_set_risk_overlay_l1"),
            "action_set_size_mean": mean_col("action_set_size"),
            "action_set_fraction_mean": mean_col("action_set_fraction"),
            "action_set_diameter_l1_mean": mean_col("action_set_diameter_l1"),
            "empty_predictive_action_set_rate": mean_col("empty_predictive_action_set"),
            "action_set_violation_rate_mean": mean_col("action_set_violation_rate"),
            "action_set_all_compliant_rate": mean_col("action_set_all_compliant"),
            "recommendation_distance_to_observed_l1_mean": mean_col("recommendation_distance_to_observed_l1"),
            "recommendation_distance_to_compliant_set_l1_mean": mean_col("recommendation_distance_to_compliant_set_l1"),
            "recommendation_action_set_size_mean": mean_col("recommendation_action_set_size"),
            "recommendation_action_set_fraction_mean": mean_col("recommendation_action_set_fraction"),
            "recommendation_action_set_diameter_l1_mean": mean_col("recommendation_action_set_diameter_l1"),
            "recommendation_action_set_violation_rate_mean": mean_col("recommendation_action_set_violation_rate"),
            "recommendation_exact_match_coverage": mean_col("recommendation_exact_match_coverage"),
            "recommendation_distance_to_observed_asset_only_l1_mean": mean_col("recommendation_distance_to_observed_asset_only_l1"),
            "recommendation_distance_to_observed_overlay_only_l1_mean": mean_col("recommendation_distance_to_observed_overlay_only_l1"),
            "recommendation_distance_to_observed_risk_overlay_l1_mean": mean_col("recommendation_distance_to_observed_risk_overlay_l1"),
            "recommendation_distance_to_compliant_set_asset_only_l1_mean": mean_col("recommendation_distance_to_compliant_set_asset_only_l1"),
            "recommendation_distance_to_compliant_set_overlay_only_l1_mean": mean_col("recommendation_distance_to_compliant_set_overlay_only_l1"),
            "recommendation_distance_to_compliant_set_risk_overlay_l1_mean": mean_col("recommendation_distance_to_compliant_set_risk_overlay_l1"),
            # Rationalization diagnostics include candidate 0.
            "rationalization_choice_coverage": mean_col("rationalization_choice_coverage"),
            "rationalization_choice_coverage_given_feasible": float(g.loc[g["observed_feasible_under_bounds"].astype(bool), "rationalization_choice_coverage"].mean()) if "observed_feasible_under_bounds" in g and "rationalization_choice_coverage" in g and g["observed_feasible_under_bounds"].astype(bool).any() else np.nan,
            "rationalization_distance_to_set_l1_mean": mean_col("rationalization_distance_to_set_l1"),
            "rationalization_distance_to_compliant_set_l1_mean": mean_col("rationalization_distance_to_compliant_set_l1"),
            "rationalization_distance_to_set_asset_only_l1_mean": mean_col("rationalization_distance_to_set_asset_only_l1"),
            "rationalization_distance_to_set_overlay_only_l1_mean": mean_col("rationalization_distance_to_set_overlay_only_l1"),
            "rationalization_distance_to_set_risk_overlay_l1_mean": mean_col("rationalization_distance_to_set_risk_overlay_l1"),
            "rationalization_distance_to_compliant_set_asset_only_l1_mean": mean_col("rationalization_distance_to_compliant_set_asset_only_l1"),
            "rationalization_distance_to_compliant_set_overlay_only_l1_mean": mean_col("rationalization_distance_to_compliant_set_overlay_only_l1"),
            "rationalization_distance_to_compliant_set_risk_overlay_l1_mean": mean_col("rationalization_distance_to_compliant_set_risk_overlay_l1"),
            "rationalization_action_set_size_mean": mean_col("rationalization_action_set_size"),
            "rationalization_action_set_fraction_mean": mean_col("rationalization_action_set_fraction"),
            "rationalization_action_set_diameter_l1_mean": mean_col("rationalization_action_set_diameter_l1"),
            "rationalization_action_set_violation_rate_mean": mean_col("rationalization_action_set_violation_rate"),
            "observed_feasible_under_bounds": mean_col("observed_feasible_under_bounds"),
            "observed_mandate_violation_rate": mean_col("observed_mandate_violation"),
            "point_pred_mandate_violation_rate": mean_col("point_pred_mandate_violation"),
            "point_pred_mandate_violation_rate_ex_observed": mean_col("point_pred_mandate_violation_ex_observed"),
            "point_top1_observed_in_coverage_menu": mean_col("point_top1_observed_in_coverage_menu"),
            "point_prediction_excluded_observed": mean_col("point_prediction_excluded_observed"),
            "point_l1_to_observed_mean": mean_col("point_l1_to_observed"),
            "point_l1_to_observed_ex_observed_mean": mean_col("point_l1_to_observed_ex_observed"),
            "point_l1_to_observed_rationalization_including_observed_mean": mean_col("point_l1_to_observed_rationalization_including_observed"),
            "point_l1_to_observed_asset_only_mean": mean_col("point_l1_to_observed_asset_only"),
            "point_l1_to_observed_overlay_only_mean": mean_col("point_l1_to_observed_overlay_only"),
            "point_l1_to_observed_risk_overlay_mean": mean_col("point_l1_to_observed_risk_overlay"),
            "point_l1_to_observed_ex_observed_asset_only_mean": mean_col("point_l1_to_observed_ex_observed_asset_only"),
            "point_l1_to_observed_ex_observed_overlay_only_mean": mean_col("point_l1_to_observed_ex_observed_overlay_only"),
            "point_l1_to_observed_ex_observed_risk_overlay_mean": mean_col("point_l1_to_observed_ex_observed_risk_overlay"),
            "theta_interval_l1_upper_diameter_mean": mean_col("theta_interval_l1_upper_diameter"),
            "regret_median_support_mean": mean_col("regret_median_support"),
            "regret_max_support_p90": q_col("regret_max_support", 0.90),
            "eps_bound_mean": mean_col("eps_bound"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_frontier_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize fixed-fraction recommendation frontiers.

    Main frontier metrics exclude candidate 0.  The rationalization frontier is
    stored in ``rationalization_*`` columns and includes candidate 0.
    """
    if eval_df.empty:
        return pd.DataFrame()
    prefixes = []
    for c in eval_df.columns:
        if not (c.startswith("frontier_f") and c.endswith("_choice_coverage")):
            continue
        base = c[: -len("_choice_coverage")]
        tag = base.replace("frontier_f", "", 1)
        # Keep base frontier columns only.
        if tag.isdigit():
            prefixes.append(base)
    prefixes = sorted(set(prefixes))
    rows = []
    for (model, split), g in eval_df.groupby(["model", "split"], dropna=False):
        for pref in prefixes:
            tag = pref.replace("frontier_f", "")
            row = {"model": model, "split": split, "selected_fraction_tag": tag, "n": len(g)}
            mapping = {
                # Recommendation frontier excludes candidate 0.
                "choice_coverage": f"{pref}_choice_coverage",
                "recommendation_choice_coverage": f"{pref}_recommendation_choice_coverage",
                "recommendation_coverage_l1tol": f"{pref}_recommendation_coverage_l1tol",
                "recommendation_exact_match_coverage": f"{pref}_recommendation_exact_match_coverage",
                "recommendation_compliant_choice_coverage": f"{pref}_recommendation_compliant_choice_coverage",
                "distance_to_set_l1": f"{pref}_distance_to_set_l1",
                "recommendation_distance_to_set_l1": f"{pref}_recommendation_distance_to_set_l1",
                "distance_to_compliant_set_l1": f"{pref}_distance_to_compliant_set_l1",
                "recommendation_distance_to_compliant_set_l1": f"{pref}_recommendation_distance_to_compliant_set_l1",
                "action_set_size": f"{pref}_action_set_size",
                "recommendation_action_set_size": f"{pref}_recommendation_action_set_size",
                "action_set_fraction": f"{pref}_action_set_fraction",
                "recommendation_action_set_fraction": f"{pref}_recommendation_action_set_fraction",
                "violation_rate": f"{pref}_violation_rate",
                "recommendation_violation_rate": f"{pref}_recommendation_violation_rate",
                "contains_observed_candidate": f"{pref}_contains_observed_candidate",
                # Rationalization frontier includes candidate 0.
                "rationalization_choice_coverage": f"{pref}_rationalization_choice_coverage",
                "rationalization_distance_to_set_l1": f"{pref}_rationalization_distance_to_set_l1",
                "rationalization_action_set_size": f"{pref}_rationalization_action_set_size",
                "rationalization_action_set_fraction": f"{pref}_rationalization_action_set_fraction",
                "rationalization_violation_rate": f"{pref}_rationalization_violation_rate",
            }
            for out_name, col in mapping.items():
                if col in g.columns:
                    row[out_name] = float(pd.to_numeric(g[col], errors="coerce").mean())
            rows.append(row)
    return pd.DataFrame(rows)
