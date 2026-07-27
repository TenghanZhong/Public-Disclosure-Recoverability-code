#!/usr/bin/env python
"""Run the supervised ranker at the paper's 20% recommendation frontier.

This is the reported Table 2 implementation. The older
``run_blackbox_ranker_baseline.py`` remains a top-1 diagnostic.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from mandate_setirl.bounds import action_violations, feasible_set_metrics, fit_mandate_bounds
from mandate_setirl.candidates import CandidateGenerator
from mandate_setirl.features import fit_utility_feature_model
from mandate_setirl.utils import chunk_indices, ensure_dir, read_yaml, set_seed, smart_write_table
from scripts.build_calibration_traceability import cluster_bootstrap_ci
from scripts.pipeline.run_blackbox_ranker_baseline import (
    DEFAULT_CONFIG,
    DEFAULT_OUT_DIR,
    DEFAULT_PANEL_ROOT,
    _available_state_columns,
    _build_classifier,
    _fit_state_scaler,
    _predict_scores,
    _state_matrix,
    build_training_data,
    make_ranker_feature_matrix,
    sample_positions,
)

DEFAULT_RECOMMENDATION_FRACTION = 0.20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the supervised 20% recommendation-set baseline.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--panel_root", default=str(DEFAULT_PANEL_ROOT))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--menu_bounds_mode", default="global")
    p.add_argument("--reference_bounds_mode", default="exogenous")
    p.add_argument("--n_candidates", type=int, default=40)
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--max_train_obs", type=int, default=30000)
    p.add_argument("--max_eval_obs", type=int, default=None)
    p.add_argument("--model", choices=["hist_gradient_boosting", "random_forest"], default="hist_gradient_boosting")
    p.add_argument("--recommendation_fraction", type=float, default=DEFAULT_RECOMMENDATION_FRACTION)
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--skip_bootstrap", action="store_true")
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def select_ranker_sets(
    scores: np.ndarray,
    violations: np.ndarray,
    *,
    observed_index: int = 0,
    filter_compliant: bool = False,
    recommendation_fraction: float = DEFAULT_RECOMMENDATION_FRACTION,
) -> list[np.ndarray]:
    """Return deterministic top-fraction sets, excluding the observed allocation.

    The global denominator is the full non-observed menu. The filtered
    denominator is the feasible non-observed menu, matching mandate-aware RP.
    """
    scores = np.asarray(scores, dtype=float)
    violations = np.asarray(violations, dtype=bool)
    if scores.ndim != 2 or violations.shape != scores.shape:
        raise ValueError("scores and violations must be matching B x M arrays")
    if not 0.0 < recommendation_fraction <= 1.0:
        raise ValueError("recommendation_fraction must lie in (0, 1]")

    eligible = np.isfinite(scores)
    eligible[:, observed_index] = False
    if filter_compliant:
        eligible &= ~violations

    out: list[np.ndarray] = []
    for row_scores, row_eligible in zip(scores, eligible):
        pool = np.flatnonzero(row_eligible)
        if pool.size == 0:
            out.append(np.empty(0, dtype=int))
            continue
        k = min(pool.size, max(1, int(np.ceil(recommendation_fraction * pool.size))))
        order = np.argsort(-row_scores[pool], kind="mergesort")
        out.append(pool[order[:k]].astype(int, copy=False))
    return out


def recommendation_metrics_for_set(
    actions: np.ndarray,
    selected_indices: np.ndarray,
    violations: np.ndarray,
    *,
    asset_idx: np.ndarray,
    overlay_idx: np.ndarray,
    risk_overlay_idx: np.ndarray,
    l1_tol: float,
    observed_index: int = 0,
) -> dict[str, Any]:
    selected = np.asarray(selected_indices, dtype=int)
    if selected.size == 0:
        return {
            "recommendation_choice_coverage": False,
            "recommendation_distance_to_observed_l1": np.nan,
            "recommendation_distance_to_observed_asset_only_l1": np.nan,
            "recommendation_distance_to_observed_overlay_only_l1": np.nan,
            "recommendation_distance_to_observed_risk_overlay_l1": np.nan,
            "recommendation_violation_rate": np.nan,
            "selected_candidate_count": 0,
            "selected_candidate_indices": "[]",
        }

    obs = np.asarray(actions[observed_index], dtype=float)
    pred = np.asarray(actions[selected], dtype=float)

    def closest_l1(idx: np.ndarray | None = None) -> float:
        if idx is None:
            distance = np.abs(pred - obs).sum(axis=1)
        elif len(idx) == 0:
            return float("nan")
        else:
            distance = np.abs(pred[:, idx] - obs[idx]).sum(axis=1)
        return float(np.min(distance))

    total = closest_l1()
    return {
        "recommendation_choice_coverage": bool(total <= l1_tol),
        "recommendation_distance_to_observed_l1": total,
        "recommendation_distance_to_observed_asset_only_l1": closest_l1(asset_idx),
        "recommendation_distance_to_observed_overlay_only_l1": closest_l1(overlay_idx),
        "recommendation_distance_to_observed_risk_overlay_l1": closest_l1(risk_overlay_idx),
        "recommendation_violation_rate": float(violations[selected].mean()),
        "selected_candidate_count": int(selected.size),
        "selected_candidate_indices": json.dumps(selected.tolist()),
    }


def evaluate_ranker_f20(
    panel: pd.DataFrame,
    candgen: CandidateGenerator,
    feature_model,
    clf,
    positions: np.ndarray,
    cfg: dict[str, Any],
    state_cols: list[str],
    state_scaler,
    reference_bounds,
    split: str,
    recommendation_fraction: float,
) -> pd.DataFrame:
    M = int(cfg["candidates"]["n_candidates"])
    chunk_size = int(cfg["candidates"]["chunk_size"])
    l1_tol = float(cfg.get("evaluation", {}).get("coverage_l1_tolerance", 0.05))
    idx_by_col = {c: j for j, c in enumerate(candgen.action_cols)}
    asset_idx = np.asarray([idx_by_col[c] for c in candgen.asset_cols if c in idx_by_col], dtype=int)
    overlay_idx = np.asarray([idx_by_col[c] for c in candgen.overlay_cols if c in idx_by_col], dtype=int)
    risk_cols = cfg.get("evaluation", {}).get(
        "risk_overlay_cols",
        ["short_pct", "short_pct_pos", "short_pct_neg_abs", "derivative_pct", "derivative_pct_pos", "derivative_pct_neg_abs", "restricted_pct"],
    )
    risk_idx = np.asarray([idx_by_col[c] for c in risk_cols if c in idx_by_col], dtype=int)
    panel_reset = panel.reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for start, end in tqdm(list(chunk_indices(len(positions), chunk_size)), desc=f"eval supervised f20 {split}"):
        pos = positions[start:end]
        batch = candgen.generate_for_positions(pos, n_candidates=M)
        from mandate_setirl.features import compute_scaled_utility_features

        utility = compute_scaled_utility_features(panel, batch, feature_model)
        state = _state_matrix(panel, pos, state_cols, state_scaler)
        X, _ = make_ranker_feature_matrix(batch.actions, utility, candgen.action_cols, feature_model.feature_names, state, state_cols)
        scores = _predict_scores(clf, X).reshape(len(pos), M)
        lower, upper = reference_bounds.bounds_for_positions(pos)
        violations = action_violations(batch.actions, lower, upper)
        global_sets = select_ranker_sets(scores, violations, observed_index=batch.observed_index, recommendation_fraction=recommendation_fraction)
        feasible_sets = select_ranker_sets(
            scores,
            violations,
            observed_index=batch.observed_index,
            filter_compliant=True,
            recommendation_fraction=recommendation_fraction,
        )

        for bi, row_pos in enumerate(pos):
            ex_observed = np.ones(M, dtype=bool)
            ex_observed[batch.observed_index] = False
            base = {
                "split": split,
                "row_pos": int(row_pos),
                "series_id": str(panel_reset.loc[row_pos, candgen.entity_col]),
                "report_date": panel_reset.loc[row_pos, "report_date"] if "report_date" in panel_reset.columns else "",
                "mandate_cluster": panel_reset.loc[row_pos, "mandate_cluster"] if "mandate_cluster" in panel_reset.columns else "",
                "observed_feasible_under_bounds": bool(not violations[bi, batch.observed_index]),
                "n_candidates_ex_observed": int(M - 1),
                "n_compliant_candidates_ex_observed": int((~violations[bi, ex_observed]).sum()),
                "recommendation_fraction": float(recommendation_fraction),
            }
            for model, selected in [
                ("supervised_global_f20", global_sets[bi]),
                ("supervised_feasibility_filter_f20", feasible_sets[bi]),
            ]:
                metrics = recommendation_metrics_for_set(
                    batch.actions[bi],
                    selected,
                    violations[bi],
                    asset_idx=asset_idx,
                    overlay_idx=overlay_idx,
                    risk_overlay_idx=risk_idx,
                    l1_tol=l1_tol,
                    observed_index=batch.observed_index,
                )
                selected_scores = scores[bi, selected] if selected.size else np.array([], dtype=float)
                rows.append(
                    {
                        **base,
                        "model": model,
                        "choice_coverage": metrics["recommendation_choice_coverage"],
                        "distance_to_set_l1": metrics["recommendation_distance_to_observed_l1"],
                        "distance_to_set_asset_only_l1": metrics["recommendation_distance_to_observed_asset_only_l1"],
                        "distance_to_set_overlay_only_l1": metrics["recommendation_distance_to_observed_overlay_only_l1"],
                        "distance_to_set_risk_overlay_l1": metrics["recommendation_distance_to_observed_risk_overlay_l1"],
                        "ranker_score_mean": float(selected_scores.mean()) if selected_scores.size else np.nan,
                        "ranker_score_cutoff": float(selected_scores.min()) if selected_scores.size else np.nan,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def summarize(row_level: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "choice_coverage",
        "distance_to_set_l1",
        "distance_to_set_asset_only_l1",
        "distance_to_set_overlay_only_l1",
        "distance_to_set_risk_overlay_l1",
        "recommendation_violation_rate",
        "selected_candidate_count",
    ]
    rows = []
    for (model, split), g in row_level.groupby(["model", "split"]):
        row = {"model": model, "split": split, "n": len(g)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(pd.to_numeric(g[metric], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "model"])


def write_bootstrap(row_level: pd.DataFrame, out_dir: Path, n_boot: int, seed: int) -> None:
    test = row_level[row_level["split"].astype(str).eq("test")].copy()
    metrics = ["choice_coverage", "distance_to_set_l1", "recommendation_violation_rate"]
    summary, differences = cluster_bootstrap_ci(
        test,
        cluster_col="series_id",
        model_col="model",
        metric_cols=metrics,
        primary_model="supervised_feasibility_filter_f20",
        baseline_models=["supervised_global_f20"],
        n_boot=n_boot,
        seed=seed,
    )
    summary.to_csv(out_dir / "supervised_ranker_f20_bootstrap_ci.csv", index=False)
    differences.to_csv(out_dir / "supervised_ranker_f20_bootstrap_differences.csv", index=False)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.recommendation_fraction <= 1.0:
        raise ValueError("--recommendation_fraction must lie in (0, 1]")
    if args.fast:
        args.max_train_obs = min(args.max_train_obs, 1000)
        args.max_eval_obs = min(args.max_eval_obs or 1000, 1000)
        args.n_candidates = min(args.n_candidates, 12)
        args.chunk_size = min(args.chunk_size, 64)
        args.n_boot = min(args.n_boot, 50)

    out_dir = ensure_dir(args.out_dir)
    cfg = read_yaml(args.config)
    cfg.setdefault("candidates", {})
    cfg["candidates"].update(
        {
            "n_candidates": int(args.n_candidates),
            "chunk_size": int(args.chunk_size),
            "include_observed_candidate": True,
            "prediction_exclude_observed": True,
            "same_date_peer_mode": "off",
            "allow_transductive_same_date": False,
        }
    )
    seed = int(cfg.get("sample", {}).get("random_seed", 42))
    rng = set_seed(seed)

    panel_root = Path(args.panel_root)
    panel = pd.read_parquet(panel_root / "panel_model_ready.parquet").reset_index(drop=True)
    with (panel_root / "action_columns.json").open("r", encoding="utf-8") as fh:
        action_meta = json.load(fh)
    action_cols = list(action_meta["action_cols"])
    asset_cols = list(action_meta.get("asset_action_cols", []))
    overlay_cols = list(action_meta.get("overlay_action_cols", []))
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")

    menu_bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode=args.menu_bounds_mode)
    reference_bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode=args.reference_bounds_mode)
    smart_write_table(
        pd.concat(
            [
                feasible_set_metrics(panel, menu_bounds, name=f"menu_{args.menu_bounds_mode}"),
                feasible_set_metrics(panel, reference_bounds, name=f"reference_{args.reference_bounds_mode}"),
            ],
            ignore_index=True,
        ),
        out_dir / "supervised_f20_feasible_set_metrics.csv",
    )

    candgen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, rng, menu_bounds)
    feature_model = fit_utility_feature_model(panel, candgen, cfg, rng)
    state_cols = _available_state_columns(panel, cfg)
    state_scaler = _fit_state_scaler(panel, state_cols)
    train_pos = sample_positions(panel, "train", args.max_train_obs, rng)
    X, y, sample_weight, feature_names = build_training_data(panel, candgen, feature_model, train_pos, cfg, state_cols, state_scaler)
    clf = _build_classifier(args.model, seed)
    clf.fit(X, y, sample_weight=sample_weight)
    del X, y, sample_weight
    gc.collect()

    evaluated = []
    for split in ["calib", "test"]:
        positions = sample_positions(panel, split, args.max_eval_obs, rng)
        evaluated.append(
            evaluate_ranker_f20(
                panel,
                candgen,
                feature_model,
                clf,
                positions,
                cfg,
                state_cols,
                state_scaler,
                reference_bounds,
                split,
                float(args.recommendation_fraction),
            )
        )
    row_level = pd.concat(evaluated, ignore_index=True)
    summary = summarize(row_level)
    smart_write_table(row_level, out_dir / "supervised_ranker_f20_row_level.parquet")
    smart_write_table(summary, out_dir / "supervised_ranker_f20_metrics.csv")
    if not args.skip_bootstrap:
        write_bootstrap(row_level, out_dir, args.n_boot, seed)

    manifest = {
        "reported_estimand": "20% recommendation set",
        "recommendation_fraction": float(args.recommendation_fraction),
        "global_rule": "top ceil(q * non-observed menu size)",
        "feasibility_filtered_rule": "filter to feasible candidates, then top ceil(q * feasible menu size)",
        "coverage_rule": "minimum selected-candidate L1 distance <= configured tolerance",
        "model": args.model,
        "random_seed": seed,
        "n_candidates_including_observed": int(args.n_candidates),
        "feature_names": feature_names,
    }
    (out_dir / "supervised_ranker_f20_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
