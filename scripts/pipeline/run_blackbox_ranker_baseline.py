#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mandate_setirl.bounds import action_violations, feasible_set_metrics, fit_mandate_bounds
from mandate_setirl.candidates import CandidateGenerator
from mandate_setirl.features import compute_scaled_utility_features, fit_utility_feature_model
from mandate_setirl.utils import SimpleStandardScaler, chunk_indices, ensure_dir, read_yaml, set_seed, smart_write_table
from scripts.pipeline.build_calibration_traceability import cluster_bootstrap_ci


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "main_32gb.yaml"
DEFAULT_PANEL_ROOT = PROJECT_ROOT / "outputs" / "full_pipeline"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "supervised_ranker"

DEFAULT_STATE_FEATURES = [
    "mandate_cluster",
    "log_net_assets",
    "net_flow_3m_scaled",
    "lag_net_flow_3m_scaled",
    "sales_flow_3m_scaled",
    "redemption_flow_3m_scaled",
    "MANAGEMENT_FEE",
    "NET_OPERATING_EXPENSES",
    "vix",
    "hy_oas",
    "rate_10y",
    "IS_ETF",
    "IS_INDEX",
    "IS_FUND_OF_FUND",
    "IS_NON_DIVERSIFIED",
    "adv_total_drp_records",
    "legal_disciplinary",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run black-box observed-choice ranker baseline.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--panel_root", default=str(DEFAULT_PANEL_ROOT), help="Directory containing panel_model_ready.parquet and action_columns.json.")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--menu_bounds_mode", default="global")
    p.add_argument("--reference_bounds_mode", default="exogenous")
    p.add_argument("--n_candidates", type=int, default=40)
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--max_train_obs", type=int, default=30000)
    p.add_argument("--max_eval_obs", type=int, default=None)
    p.add_argument("--model", choices=["hist_gradient_boosting", "random_forest"], default="hist_gradient_boosting")
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--skip_bootstrap", action="store_true")
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def make_ranker_feature_matrix(
    actions: np.ndarray,
    utility_features: np.ndarray,
    action_cols: list[str],
    utility_feature_names: list[str],
    state_features: np.ndarray | None = None,
    state_feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    actions = np.asarray(actions, dtype=np.float32)
    utility_features = np.asarray(utility_features, dtype=np.float32)
    if actions.ndim != 3 or utility_features.ndim != 3:
        raise ValueError("actions and utility_features must be B x M x K/D arrays")
    if actions.shape[:2] != utility_features.shape[:2]:
        raise ValueError("actions and utility_features must have matching B x M dimensions")
    B, M, _ = actions.shape
    parts = [actions.reshape(B * M, actions.shape[-1]), utility_features.reshape(B * M, utility_features.shape[-1])]
    names = [f"action:{c}" for c in action_cols] + [f"utility:{c}" for c in utility_feature_names]
    if state_features is not None:
        state_features = np.asarray(state_features, dtype=np.float32)
        if state_features.ndim != 2 or state_features.shape[0] != B:
            raise ValueError("state_features must be B x S")
        repeated = np.repeat(state_features[:, None, :], M, axis=1).reshape(B * M, state_features.shape[-1])
        parts.append(repeated)
        names.extend([f"state:{c}" for c in (state_feature_names or [])])
    X = np.hstack(parts).astype(np.float32, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def select_ranker_indices(
    scores: np.ndarray,
    violations: np.ndarray,
    observed_index: int = 0,
    filter_compliant: bool = False,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float).copy()
    violations = np.asarray(violations, dtype=bool)
    if scores.ndim != 2:
        raise ValueError("scores must be B x M")
    if violations.shape != scores.shape:
        raise ValueError("violations must match scores")
    scores[:, observed_index] = -np.inf
    if filter_compliant:
        scores[violations] = -np.inf
    chosen = np.argmax(scores, axis=1).astype(int)
    no_candidate = ~np.isfinite(scores[np.arange(scores.shape[0]), chosen])
    chosen[no_candidate] = -1
    return chosen


def recommendation_metrics_for_choice(
    actions: np.ndarray,
    selected_idx: int,
    violations: np.ndarray,
    asset_idx: np.ndarray,
    overlay_idx: np.ndarray,
    risk_overlay_idx: np.ndarray,
    l1_tol: float,
    observed_index: int = 0,
) -> dict[str, Any]:
    obs = np.asarray(actions[observed_index], dtype=float)
    if selected_idx < 0:
        return {
            "recommendation_choice_coverage": False,
            "recommendation_distance_to_observed_l1": np.nan,
            "recommendation_distance_to_observed_asset_only_l1": np.nan,
            "recommendation_distance_to_observed_overlay_only_l1": np.nan,
            "recommendation_distance_to_observed_risk_overlay_l1": np.nan,
            "point_pred_mandate_violation_ex_observed": np.nan,
            "selected_candidate_idx": -1,
        }
    pred = np.asarray(actions[selected_idx], dtype=float)

    def l1(idx: np.ndarray | None = None) -> float:
        if idx is None:
            return float(np.abs(pred - obs).sum())
        if len(idx) == 0:
            return float(np.nan)
        return float(np.abs(pred[idx] - obs[idx]).sum())

    total = l1()
    return {
        "recommendation_choice_coverage": bool(total <= l1_tol),
        "recommendation_distance_to_observed_l1": total,
        "recommendation_distance_to_observed_asset_only_l1": l1(asset_idx),
        "recommendation_distance_to_observed_overlay_only_l1": l1(overlay_idx),
        "recommendation_distance_to_observed_risk_overlay_l1": l1(risk_overlay_idx),
        "point_pred_mandate_violation_ex_observed": bool(violations[selected_idx]),
        "selected_candidate_idx": int(selected_idx),
    }


def _available_state_columns(panel: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    requested = cfg.get("blackbox_ranker", {}).get("state_feature_cols", DEFAULT_STATE_FEATURES)
    out = []
    for col in requested:
        if col in panel.columns and col not in out:
            out.append(col)
    return out


def _fit_state_scaler(panel: pd.DataFrame, state_cols: list[str]) -> SimpleStandardScaler:
    scaler = SimpleStandardScaler()
    if not state_cols:
        scaler.fit(np.zeros((1, 0), dtype=np.float32))
        return scaler
    train = panel[panel["split"].astype(str).eq("train")]
    if train.empty:
        train = panel
    X = train[state_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    scaler.fit(X)
    return scaler


def _state_matrix(panel: pd.DataFrame, positions: np.ndarray, state_cols: list[str], scaler: SimpleStandardScaler) -> np.ndarray | None:
    if not state_cols:
        return None
    X = panel.iloc[positions][state_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return scaler.transform(X).astype(np.float32, copy=False)


def _build_classifier(kind: str, seed: int):
    if kind == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )


def _predict_scores(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return clf.predict(X)


def sample_positions(panel: pd.DataFrame, split: str, max_obs: int | None, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(panel["split"].astype(str).to_numpy() == split)
    if max_obs is not None and len(pos) > int(max_obs):
        pos = rng.choice(pos, size=int(max_obs), replace=False)
        pos = np.sort(pos)
    return pos.astype(int)


def build_training_data(
    panel: pd.DataFrame,
    candgen: CandidateGenerator,
    feature_model,
    positions: np.ndarray,
    cfg: dict[str, Any],
    state_cols: list[str],
    state_scaler: SimpleStandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    M = int(cfg["candidates"]["n_candidates"])
    chunk_size = int(cfg["candidates"]["chunk_size"])
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    for start, end in tqdm(list(chunk_indices(len(positions), chunk_size)), desc="build blackbox train matrix"):
        pos = positions[start:end]
        batch = candgen.generate_for_positions(pos, n_candidates=M)
        utility = compute_scaled_utility_features(panel, batch, feature_model)
        state = _state_matrix(panel, pos, state_cols, state_scaler)
        X, feature_names = make_ranker_feature_matrix(
            batch.actions,
            utility,
            candgen.action_cols,
            feature_model.feature_names,
            state,
            state_cols,
        )
        B = len(pos)
        y = np.zeros(B * M, dtype=np.int8)
        y[np.arange(B) * M + batch.observed_index] = 1
        w = np.ones(B * M, dtype=np.float32)
        w[y == 1] = float(M - 1)
        X_parts.append(X)
        y_parts.append(y)
        w_parts.append(w)
    if not X_parts:
        raise ValueError("No training rows generated")
    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(w_parts), feature_names


def evaluate_ranker(
    panel: pd.DataFrame,
    candgen: CandidateGenerator,
    feature_model,
    clf,
    positions: np.ndarray,
    cfg: dict[str, Any],
    state_cols: list[str],
    state_scaler: SimpleStandardScaler,
    reference_bounds,
    split: str,
) -> pd.DataFrame:
    M = int(cfg["candidates"]["n_candidates"])
    chunk_size = int(cfg["candidates"]["chunk_size"])
    l1_tol = float(cfg.get("evaluation", {}).get("coverage_l1_tolerance", 0.05))
    action_cols = candgen.action_cols
    idx_by_col = {c: j for j, c in enumerate(action_cols)}
    asset_idx = np.asarray([idx_by_col[c] for c in candgen.asset_cols if c in idx_by_col], dtype=int)
    overlay_idx = np.asarray([idx_by_col[c] for c in candgen.overlay_cols if c in idx_by_col], dtype=int)
    risk_cols = cfg.get("evaluation", {}).get("risk_overlay_cols", [
        "short_pct", "short_pct_pos", "short_pct_neg_abs",
        "derivative_pct", "derivative_pct_pos", "derivative_pct_neg_abs",
        "restricted_pct",
    ])
    risk_idx = np.asarray([idx_by_col[c] for c in risk_cols if c in idx_by_col], dtype=int)
    rows: list[dict[str, Any]] = []
    panel_reset = panel.reset_index(drop=True)
    for start, end in tqdm(list(chunk_indices(len(positions), chunk_size)), desc=f"eval blackbox {split}"):
        pos = positions[start:end]
        batch = candgen.generate_for_positions(pos, n_candidates=M)
        utility = compute_scaled_utility_features(panel, batch, feature_model)
        state = _state_matrix(panel, pos, state_cols, state_scaler)
        X, _ = make_ranker_feature_matrix(
            batch.actions,
            utility,
            candgen.action_cols,
            feature_model.feature_names,
            state,
            state_cols,
        )
        scores = _predict_scores(clf, X).reshape(len(pos), M)
        lower, upper = reference_bounds.bounds_for_positions(pos)
        violations = action_violations(batch.actions, lower, upper)
        raw_idx = select_ranker_indices(scores, violations, observed_index=batch.observed_index, filter_compliant=False)
        filt_idx = select_ranker_indices(scores, violations, observed_index=batch.observed_index, filter_compliant=True)
        for bi, row_pos in enumerate(pos):
            obs_feasible = bool(not violations[bi, batch.observed_index])
            ex_observed = np.ones(M, dtype=bool)
            ex_observed[batch.observed_index] = False
            base = {
                "split": split,
                "row_pos": int(row_pos),
                "series_id": str(panel_reset.loc[row_pos, candgen.entity_col]) if candgen.entity_col in panel_reset.columns else "",
                "report_date": panel_reset.loc[row_pos, "report_date"] if "report_date" in panel_reset.columns else "",
                "mandate_cluster": panel_reset.loc[row_pos, "mandate_cluster"] if "mandate_cluster" in panel_reset.columns else "",
                "observed_feasible_under_bounds": obs_feasible,
                "observed_mandate_violation": not obs_feasible,
                "n_candidates": int(M),
                "n_candidates_ex_observed": int(M - 1),
                "n_compliant_candidates_ex_observed": int((~violations[bi, ex_observed]).sum()),
                "compliant_candidate_fraction_ex_observed": float((~violations[bi, ex_observed]).mean()),
            }
            for model_name, selected in [
                ("blackbox_ranker_global_menu", int(raw_idx[bi])),
                ("blackbox_ranker_posthoc_exogenous_filter", int(filt_idx[bi])),
            ]:
                metrics = recommendation_metrics_for_choice(
                    actions=batch.actions[bi],
                    selected_idx=selected,
                    violations=violations[bi],
                    asset_idx=asset_idx,
                    overlay_idx=overlay_idx,
                    risk_overlay_idx=risk_idx,
                    l1_tol=l1_tol,
                    observed_index=batch.observed_index,
                )
                score = float(scores[bi, selected]) if selected >= 0 else np.nan
                rows.append({
                    **base,
                    "model": model_name,
                    "ranker_score": score,
                    "no_compliant_candidate": bool(selected < 0 and model_name.endswith("filter")),
                    "choice_coverage": metrics["recommendation_choice_coverage"],
                    "recommendation_choice_coverage": metrics["recommendation_choice_coverage"],
                    "distance_to_set_l1": metrics["recommendation_distance_to_observed_l1"],
                    "distance_to_set_asset_only_l1": metrics["recommendation_distance_to_observed_asset_only_l1"],
                    "distance_to_set_overlay_only_l1": metrics["recommendation_distance_to_observed_overlay_only_l1"],
                    "distance_to_set_risk_overlay_l1": metrics["recommendation_distance_to_observed_risk_overlay_l1"],
                    **metrics,
                })
    return pd.DataFrame(rows)


def summarize_blackbox_metrics(row_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "choice_coverage",
        "recommendation_choice_coverage",
        "observed_feasible_under_bounds",
        "point_pred_mandate_violation_ex_observed",
        "recommendation_distance_to_observed_l1",
        "recommendation_distance_to_observed_asset_only_l1",
        "recommendation_distance_to_observed_overlay_only_l1",
        "recommendation_distance_to_observed_risk_overlay_l1",
        "distance_to_set_l1",
        "distance_to_set_asset_only_l1",
        "distance_to_set_overlay_only_l1",
        "distance_to_set_risk_overlay_l1",
        "compliant_candidate_fraction_ex_observed",
        "no_compliant_candidate",
        "ranker_score",
    ]
    for (model, split), g in row_level.groupby(["model", "split"], dropna=False):
        row = {"model": model, "split": split, "n": int(len(g))}
        for col in metric_cols:
            if col in g.columns:
                row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
        row["point_pred_mandate_violation_rate_ex_observed"] = row.get("point_pred_mandate_violation_ex_observed_mean", np.nan)
        row["distance_to_set_l1_mean"] = row.get("distance_to_set_l1_mean", np.nan)
        row["distance_to_set_asset_only_l1_mean"] = row.get("distance_to_set_asset_only_l1_mean", np.nan)
        row["distance_to_set_overlay_only_l1_mean"] = row.get("distance_to_set_overlay_only_l1_mean", np.nan)
        row["distance_to_set_risk_overlay_l1_mean"] = row.get("distance_to_set_risk_overlay_l1_mean", np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "model"])


def write_bootstrap(row_level: pd.DataFrame, out_dir: Path, n_boot: int, seed: int) -> None:
    metric_cols = [
        "choice_coverage",
        "point_pred_mandate_violation_ex_observed",
        "distance_to_set_l1",
        "distance_to_set_asset_only_l1",
        "distance_to_set_overlay_only_l1",
        "distance_to_set_risk_overlay_l1",
    ]
    test = row_level[row_level["split"].astype(str).eq("test")].copy()
    summary, diffs = cluster_bootstrap_ci(
        test,
        cluster_col="series_id",
        model_col="model",
        metric_cols=[c for c in metric_cols if c in test.columns],
        primary_model="blackbox_ranker_posthoc_exogenous_filter",
        baseline_models=["blackbox_ranker_global_menu"],
        n_boot=n_boot,
        seed=seed,
    )
    summary.to_csv(out_dir / "blackbox_ranker_bootstrap_ci_model_metrics.csv", index=False)
    diffs.to_csv(out_dir / "blackbox_ranker_bootstrap_ci_model_differences.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.fast:
        args.max_train_obs = min(args.max_train_obs, 1000)
        args.max_eval_obs = min(args.max_eval_obs or 1000, 1000)
        args.n_candidates = min(args.n_candidates, 12)
        args.chunk_size = min(args.chunk_size, 64)
        args.n_boot = min(args.n_boot, 50)

    out_dir = ensure_dir(args.out_dir)
    cfg = read_yaml(args.config)
    cfg.setdefault("candidates", {})
    cfg["candidates"]["n_candidates"] = int(args.n_candidates)
    cfg["candidates"]["chunk_size"] = int(args.chunk_size)
    cfg["candidates"]["include_observed_candidate"] = True
    cfg["candidates"]["prediction_exclude_observed"] = True
    cfg["candidates"]["same_date_peer_mode"] = "off"
    cfg["candidates"]["allow_transductive_same_date"] = False
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
    feasible = pd.concat([
        feasible_set_metrics(panel, menu_bounds, name=f"menu_{args.menu_bounds_mode}"),
        feasible_set_metrics(panel, reference_bounds, name=f"reference_{args.reference_bounds_mode}"),
    ], ignore_index=True)
    smart_write_table(feasible, out_dir / "blackbox_feasible_set_metrics.csv")

    candgen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, rng, menu_bounds)
    feature_model = fit_utility_feature_model(panel, candgen, cfg, rng)
    state_cols = _available_state_columns(panel, cfg)
    state_scaler = _fit_state_scaler(panel, state_cols)

    train_pos = sample_positions(panel, "train", args.max_train_obs, rng)
    X, y, sample_weight, feature_names = build_training_data(panel, candgen, feature_model, train_pos, cfg, state_cols, state_scaler)
    clf = _build_classifier(args.model, seed)
    clf.fit(X, y, sample_weight=sample_weight)
    train_pos_count = int(len(train_pos))
    train_candidate_rows = int(len(y))
    del X, y, sample_weight
    gc.collect()

    eval_frames = []
    for split in ["calib", "test"]:
        pos = sample_positions(panel, split, args.max_eval_obs, rng)
        ev = evaluate_ranker(panel, candgen, feature_model, clf, pos, cfg, state_cols, state_scaler, reference_bounds, split)
        eval_frames.append(ev)
    row_level = pd.concat(eval_frames, ignore_index=True)
    metrics = summarize_blackbox_metrics(row_level)

    smart_write_table(row_level, out_dir / "blackbox_ranker_row_level.parquet")
    smart_write_table(metrics, out_dir / "blackbox_ranker_metrics.csv")
    if not args.skip_bootstrap:
        write_bootstrap(row_level, out_dir, int(args.n_boot), seed)

    info = {
        "config": str(args.config),
        "config_path": str(args.config),
        "panel_root": str(args.panel_root),
        "data_panel": str(panel_root / "panel_model_ready.parquet"),
        "action_columns": str(panel_root / "action_columns.json"),
        "menu_bounds_mode": args.menu_bounds_mode,
        "reference_bounds_mode": args.reference_bounds_mode,
        "model": args.model,
        "ranker_kind": args.model,
        "random_seed": seed,
        "n_candidates": int(args.n_candidates),
        "chunk_size": int(args.chunk_size),
        "include_observed_candidate": bool(cfg["candidates"]["include_observed_candidate"]),
        "exclude_observed_candidate_for_recommendation": bool(cfg["candidates"]["prediction_exclude_observed"]),
        "same_date_peer_mode": cfg["candidates"]["same_date_peer_mode"],
        "allow_transductive_same_date": bool(cfg["candidates"]["allow_transductive_same_date"]),
        "max_train_obs": int(args.max_train_obs),
        "max_eval_obs": args.max_eval_obs,
        "train_obs_used": train_pos_count,
        "train_candidate_rows": train_candidate_rows,
        "positive_rows": train_pos_count,
        "negative_rows": train_candidate_rows - train_pos_count,
        "feature_names": feature_names,
        "state_feature_cols": state_cols,
        "no_source_feature_leakage": not any("source" in n.lower() or "observed" in n.lower() for n in feature_names),
    }
    with (out_dir / "blackbox_ranker_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False, default=str)

    if hasattr(clf, "feature_importances_"):
        imp = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_}).sort_values("importance", ascending=False)
        imp.to_csv(out_dir / "blackbox_ranker_feature_importance.csv", index=False)

    readme = [
        "# Black-box ranker baseline",
        "",
        "The ranker is trained to classify the observed candidate against global-menu alternatives.",
        "Candidate source strings and observed/source indicators are excluded from the feature matrix.",
        "Evaluation reports the raw global-menu top-1 candidate and a post-hoc exogenous-mandate-filtered top-1 candidate.",
        "The outputs compare observed-choice similarity and mandate compliance.",
    ]
    (out_dir / "README_blackbox_ranker.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(metrics.to_string(index=False))
    print(f"Black-box ranker baseline saved to: {out_dir}")


if __name__ == "__main__":
    main()
