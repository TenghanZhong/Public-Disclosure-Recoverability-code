from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mandate_setirl.bounds import action_violations, fit_mandate_bounds
from mandate_setirl.candidates import CandidateGenerator
from mandate_setirl.utils import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--selected_root", default=str(repo_root / "outputs" / "full_pipeline"))
    parser.add_argument("--config", default="configs/main_32gb.yaml")
    parser.add_argument("--out_dir", default=str(repo_root / "outputs" / "transition_menu_diagnostics"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_candidates", type=int, default=40)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--max_rows", type=int, default=None, help="Optional diagnostic subsample size from the selected split.")
    parser.add_argument("--write_row_level", action="store_true", help="Write row-level diagnostic CSVs. Summaries are always written.")
    parser.add_argument("--transition_magnitude_count", type=int, default=12)
    parser.add_argument("--transition_stress_count", type=int, default=6)
    return parser.parse_args()


def transition_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg.setdefault("candidates", {})
    cfg["candidates"].update(
        {
            "generator_mode": "transition_augmented",
            "include_observed_candidate": True,
            "prediction_exclude_observed": True,
            "same_date_peer_mode": "off",
            "allow_transductive_same_date": False,
            "n_candidates": args.n_candidates,
            "chunk_size": args.chunk_size,
            "peer_train_count": 2,
            "self_lag_exact_count": 1,
            "self_lag_projected_count": 1,
            "transition_knn_count": 0,
            "transition_small_count": 4,
            "transition_moderate_count": 12,
            "transition_large_count": 4,
            "transition_regime_count": 6,
            "transition_shrunk_count": 3,
            "transition_magnitude_count": args.transition_magnitude_count,
            "transition_stress_count": 4,
            "clean_frontier_count": 2,
            "raw_candidate_oversample_multiplier": 3,
            "dedupe_round_decimals": 6,
            "mover_routing_enabled": True,
            "source_quotas": {
                "lag": 2,
                "static_peer": 2,
                "peer_small": 4,
                "peer_moderate": 12,
                "peer_large_stress": 8,
                "regime": 6,
                "shrunk": 3,
                "clean": 2,
            },
            "source_quotas_by_predicted_move": {
                "stayer": {
                    "lag": 2,
                    "static_peer": 2,
                    "peer_small": 12,
                    "peer_moderate": 8,
                    "peer_large_stress": 2,
                    "regime": 6,
                    "shrunk": 5,
                    "clean": 2,
                },
                "moderate": {
                    "lag": 2,
                    "static_peer": 2,
                    "peer_small": 4,
                    "peer_moderate": 12,
                    "peer_large_stress": 8,
                    "regime": 6,
                    "shrunk": 3,
                    "clean": 2,
                },
                "large": {
                    "lag": 1,
                    "static_peer": 1,
                    "peer_small": 2,
                    "peer_moderate": 10,
                    "peer_large_stress": 14,
                    "regime": 6,
                    "shrunk": 3,
                    "clean": 2,
                },
            },
        }
    )
    cfg.setdefault("transition", {})
    cfg["transition"].update(
        {
            "enabled": True,
            "donor_split": "train",
            "min_move_l1": 0.005,
            "max_move_l1_quantile": 0.99,
            "scales": [0.5, 1.0],
            "same_cluster_first": True,
            "fallback_to_pooled": True,
            "retrieval_k": 16,
            "retrieval_state_weight": 0.7,
            "retrieval_lag_weight": 0.3,
            "regime_min_count": 50,
            "stress_move_quantile": 0.75,
            "direction_magnitudes": [0.08, 0.15, 0.30, 0.50],
            "mover_threshold": 0.05,
            "mover_prediction_k": 16,
        }
    )
    return cfg


def load_action_columns(selected_root: Path) -> dict:
    with open(selected_root / "action_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=-1)


def direction_cosine(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    pred = np.asarray(pred_delta, dtype=float)
    true = np.asarray(true_delta, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return np.nan
    pred = pred[mask]
    true = true[mask]
    denom = float(np.linalg.norm(pred) * np.linalg.norm(true))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(pred, true) / denom)


def sign_agreement(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    pred = np.asarray(pred_delta, dtype=float)
    true = np.asarray(true_delta, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.sign(pred[mask]) == np.sign(true[mask])))


def direction_l1_error(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    pred = np.asarray(pred_delta, dtype=float)
    true = np.asarray(true_delta, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return np.nan
    return float(np.abs(pred[mask] - true[mask]).sum())


def add_direction_metrics(row: dict, prefix: str, action: np.ndarray, lag: np.ndarray, obs: np.ndarray, asset_idx: np.ndarray) -> None:
    pred_delta = np.asarray(action, dtype=float) - np.asarray(lag, dtype=float)
    true_delta = np.asarray(obs, dtype=float) - np.asarray(lag, dtype=float)
    row[f"{prefix}_move_l1"] = float(np.abs(pred_delta).sum())
    row[f"{prefix}_cosine_to_true_move"] = direction_cosine(pred_delta, true_delta)
    row[f"{prefix}_sign_agreement_to_true_move"] = sign_agreement(pred_delta, true_delta)
    row[f"{prefix}_move_l1_error"] = direction_l1_error(pred_delta, true_delta)
    if len(asset_idx):
        row[f"{prefix}_asset_cosine_to_true_move"] = direction_cosine(pred_delta[asset_idx], true_delta[asset_idx])
        row[f"{prefix}_asset_sign_agreement_to_true_move"] = sign_agreement(pred_delta[asset_idx], true_delta[asset_idx])
        row[f"{prefix}_asset_move_l1_error"] = direction_l1_error(pred_delta[asset_idx], true_delta[asset_idx])


def source_group(source: str) -> str:
    s = str(source)
    if s == "observed":
        return "observed"
    if s.startswith("lag_"):
        return "lag"
    if s.startswith("transition_knn"):
        return "transition_knn"
    if s.startswith("transition_regime"):
        return "transition_regime"
    if s.startswith("transition_shrunk"):
        return "transition_shrunk"
    if s.startswith("transition_small"):
        return "transition_small"
    if s.startswith("transition_moderate"):
        return "transition_moderate"
    if s.startswith("transition_large") or s.startswith("transition_stress"):
        return "transition_large_stress"
    if s.startswith("peer_static") or s.startswith("peer_train"):
        return "peer_static_train"
    if s.startswith("clean_frontier"):
        return "clean_frontier"
    return s


def move_size_bucket(move_l1: float) -> str:
    if not np.isfinite(move_l1):
        return "unknown"
    if move_l1 < 0.05:
        return "small"
    if move_l1 <= 0.25:
        return "moderate"
    return "large"


def hit_source_bucket(source: str, donor_move_l1: float, same_series: bool) -> str:
    s = str(source)
    if s in {"lag_exact", "lag_projected"}:
        return "lag exact/projected"
    if s.startswith("transition_regime"):
        return "regime transition"
    if s.startswith("transition"):
        size = move_size_bucket(donor_move_l1)
        prefix = "same-series" if bool(same_series) else "peer"
        return f"{prefix} {size} transition"
    if s.startswith("peer_static") or s.startswith("peer_train"):
        return "static peer"
    if s.startswith("clean_frontier"):
        return "clean frontier"
    return s


def move_bins(lag_l1: np.ndarray) -> np.ndarray:
    return np.asarray(pd.cut(
        lag_l1,
        [-np.inf, 0.05, 0.25, np.inf],
        labels=["stayer_l1_le_005", "moderate_l1_005_025", "mover_l1_gt_025"],
    ).astype(str))


def summarize_group(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)} if group_cols else {}
        row["n"] = int(len(g))
        row["n_series"] = int(g["series_id"].nunique()) if "series_id" in g.columns else np.nan
        for m in metrics:
            row[m] = float(pd.to_numeric(g[m], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def frontier_metrics(dist: np.ndarray, viol: np.ndarray, sources: np.ndarray, order: np.ndarray, fractions: list[float], tol: float) -> list[dict]:
    rows = []
    n = len(dist)
    for q in fractions:
        k = max(1, int(np.ceil(q * n)))
        idx = order[:k]
        d = dist[idx]
        nearest_local = int(np.argmin(d))
        nearest_idx = int(idx[nearest_local])
        rows.append(
            {
                "frontier": q,
                "coverage": float(np.min(d) <= tol),
                "distance": float(np.min(d)),
                "violation": float(np.any(viol[idx])),
                "nearest_selected_source": str(sources[nearest_idx]),
                "nearest_selected_source_group": source_group(str(sources[nearest_idx])),
            }
        )
    return rows


def main():
    args = parse_args()
    selected_root = Path(args.selected_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = transition_cfg(read_yaml(args.config), args)
    ac = load_action_columns(selected_root)
    action_cols = ac["action_cols"]
    asset_cols = ac["asset_action_cols"]
    overlay_cols = ac["overlay_action_cols"]
    asset_idx = np.asarray([action_cols.index(c) for c in asset_cols if c in action_cols], dtype=int)
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
    rng = np.random.default_rng(args.seed)
    gen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, rng, bounds)
    positions = np.flatnonzero(panel["split"].astype(str).eq(args.split).to_numpy())
    if args.max_rows is not None and args.max_rows > 0 and args.max_rows < len(positions):
        positions = np.sort(rng.choice(positions, size=args.max_rows, replace=False))
    A_obs_all = panel[action_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    A_lag_all = np.zeros_like(A_obs_all)
    for j, c in enumerate(action_cols):
        lag_col = f"lag_{c}"
        A_lag_all[:, j] = pd.to_numeric(panel[lag_col] if lag_col in panel.columns else panel[c], errors="coerce").fillna(0.0)
    lag_l1_all = l1(A_obs_all, A_lag_all)
    bin_all = move_bins(lag_l1_all)

    row_rows = []
    selected_rows = []
    source_rows = []
    fractions = [0.05, 0.10, 0.20, 0.30, 0.50]
    for start in range(0, len(positions), args.chunk_size):
        pos = positions[start : start + args.chunk_size]
        batch = gen.generate_for_positions(pos, n_candidates=args.n_candidates)
        lower, upper = bounds.bounds_for_positions(pos)
        cand_viol = action_violations(batch.actions, lower, upper, tol=1e-7)
        for bi, row_pos in enumerate(pos):
            obs = A_obs_all[row_pos]
            lag = A_lag_all[row_pos]
            cand = batch.actions[bi]
            src = batch.sources[bi].astype(str)
            rec_idx = np.arange(cand.shape[0])
            rec_idx = rec_idx[rec_idx != batch.observed_index]
            dist = l1(cand[rec_idx], obs)
            lag_dist = l1(cand[rec_idx], lag)
            viol = cand_viol[bi, rec_idx]
            src_rec = src[rec_idx]
            donor_move_rec = batch.donor_move_l1[bi, rec_idx]
            same_series_rec = batch.same_series_donor[bi, rec_idx]
            donor_pos_rec = batch.donor_pos[bi, rec_idx]
            donor_dist_rec = batch.donor_retrieval_distance[bi, rec_idx]
            post_proj_lag_rec = batch.post_projection_l1_from_lag[bi, rec_idx]
            post_proj_raw_rec = batch.post_projection_l1_from_raw_transition_candidate[bi, rec_idx]
            source_bucket_rec = np.asarray(
                [hit_source_bucket(s, mv, ss) for s, mv, ss in zip(src_rec, donor_move_rec, same_series_rec)],
                dtype=object,
            )
            nearest = int(np.argmin(dist))
            nearest_idx = int(rec_idx[nearest])
            row = {
                "row_pos": int(row_pos),
                "series_id": str(panel.loc[row_pos, entity_col]),
                "report_date": panel.loc[row_pos, "report_date"],
                "move_bin": bin_all[row_pos],
                "lag_l1_to_obs": float(lag_l1_all[row_pos]),
                "predicted_move_probability": float(batch.predicted_move_probability[bi]) if batch.predicted_move_probability is not None else np.nan,
                "predicted_move_magnitude": float(batch.predicted_move_magnitude[bi]) if batch.predicted_move_magnitude is not None else np.nan,
                "predicted_move_bin": str(batch.predicted_move_bin[bi]) if batch.predicted_move_bin is not None else "",
                "oracle_cover": float(dist[nearest] <= args.tol),
                "oracle_min_l1": float(dist[nearest]),
                "oracle_nearest_source": str(src[nearest_idx]),
                "oracle_nearest_source_group": source_group(str(src[nearest_idx])),
                "oracle_nearest_source_bucket": hit_source_bucket(
                    str(src[nearest_idx]),
                    float(batch.donor_move_l1[bi, nearest_idx]),
                    bool(batch.same_series_donor[bi, nearest_idx]),
                ),
                "oracle_nearest_donor_pos": int(batch.donor_pos[bi, nearest_idx]),
                "oracle_nearest_donor_move_l1": float(batch.donor_move_l1[bi, nearest_idx]),
                "oracle_nearest_same_series_donor": bool(batch.same_series_donor[bi, nearest_idx]),
                "oracle_nearest_donor_retrieval_distance": float(batch.donor_retrieval_distance[bi, nearest_idx]),
                "oracle_nearest_post_projection_l1_from_lag": float(batch.post_projection_l1_from_lag[bi, nearest_idx]),
                "oracle_nearest_post_projection_l1_from_raw_transition_candidate": float(batch.post_projection_l1_from_raw_transition_candidate[bi, nearest_idx]),
                "candidate_count": int(len(rec_idx)),
                "duplicate_rate_rounded_1e6": float(1.0 - len({tuple(np.round(x, 6)) for x in cand[rec_idx]}) / max(len(rec_idx), 1)),
                "transition_candidate_frac": float(np.mean([source_group(s).startswith("transition") for s in src_rec])),
                "lag_candidate_frac": float(np.mean([source_group(s) == "lag" for s in src_rec])),
                "same_series_transition_candidate_frac": float(np.mean([str(g).startswith("same-series") for g in source_bucket_rec])),
                "peer_moderate_large_transition_candidate_frac": float(np.mean([str(g) in {"peer moderate transition", "peer large transition"} for g in source_bucket_rec])),
            }
            row_rows.append(row)
            for source_group_name in np.unique(source_bucket_rec):
                idx = np.flatnonzero(source_bucket_rec == source_group_name)
                d = dist[idx]
                source_rows.append(
                    {
                        "row_pos": int(row_pos),
                        "source_bucket": source_group_name,
                        "n_candidates": int(len(idx)),
                        "source_oracle_cover": float(np.min(d) <= args.tol),
                        "source_min_l1": float(np.min(d)),
                        "mean_donor_move_l1": float(np.nanmean(donor_move_rec[idx])) if np.isfinite(donor_move_rec[idx]).any() else np.nan,
                        "same_series_share": float(np.nanmean(same_series_rec[idx])) if len(idx) else np.nan,
                        "mean_donor_retrieval_distance": float(np.nanmean(donor_dist_rec[idx])) if np.isfinite(donor_dist_rec[idx]).any() else np.nan,
                        "mean_post_projection_l1_from_lag": float(np.nanmean(post_proj_lag_rec[idx])) if np.isfinite(post_proj_lag_rec[idx]).any() else np.nan,
                        "mean_post_projection_l1_from_raw_transition_candidate": float(np.nanmean(post_proj_raw_rec[idx])) if np.isfinite(post_proj_raw_rec[idx]).any() else np.nan,
                    }
                )
            persistence_order = np.argsort(lag_dist, kind="mergesort")
            transition_first_score = np.array([0 if source_group(s).startswith("transition") else 1 for s in src_rec], dtype=float)
            transition_order = np.lexsort((np.arange(len(rec_idx)), lag_dist, transition_first_score))
            donor_distance_order = np.lexsort((np.arange(len(rec_idx)), np.nan_to_num(donor_dist_rec, nan=1e18)))
            oracle_order = np.argsort(dist, kind="mergesort")
            for model, order in [
                ("size_matched_persistence", persistence_order),
                ("transition_source_then_persistence", transition_order),
                ("transition_knn_retrieval", donor_distance_order),
                ("oracle_upper_bound", oracle_order),
            ]:
                for m in frontier_metrics(dist, viol, src_rec, order, fractions, args.tol):
                    near_source = m["nearest_selected_source"]
                    local_near = int(np.flatnonzero(src_rec == near_source)[0]) if np.any(src_rec == near_source) else int(order[0])
                    # Recompute by selected nearest distance to avoid ambiguity when source labels repeat.
                    idx_sel = order[: max(1, int(np.ceil(m["frontier"] * len(order))))]
                    d_sel = dist[idx_sel]
                    near_local = int(idx_sel[int(np.argmin(d_sel))])
                    selected_actions = cand[rec_idx[idx_sel]]
                    top_action = cand[rec_idx[int(idx_sel[0])]]
                    nearest_action = cand[rec_idx[near_local]]
                    centroid_action = np.nanmean(selected_actions, axis=0)
                    m.update({
                        "row_pos": int(row_pos),
                        "model": model,
                        "move_bin": bin_all[row_pos],
                        "predicted_move_bin": str(batch.predicted_move_bin[bi]) if batch.predicted_move_bin is not None else "",
                        "series_id": str(panel.loc[row_pos, entity_col]),
                    })
                    m["nearest_selected_source_bucket"] = source_bucket_rec[near_local]
                    m["nearest_selected_donor_pos"] = int(donor_pos_rec[near_local])
                    m["nearest_selected_donor_move_l1"] = float(donor_move_rec[near_local])
                    m["nearest_selected_same_series_donor"] = bool(same_series_rec[near_local])
                    m["true_move_l1"] = float(lag_l1_all[row_pos])
                    add_direction_metrics(m, "top_selected", top_action, lag, obs, asset_idx)
                    add_direction_metrics(m, "nearest_selected", nearest_action, lag, obs, asset_idx)
                    add_direction_metrics(m, "centroid_selected", centroid_action, lag, obs, asset_idx)
                    selected_rows.append(m)

    row_df = pd.DataFrame(row_rows)
    selected_df = pd.DataFrame(selected_rows)
    source_df = pd.DataFrame(source_rows)
    if args.write_row_level:
        row_df.to_csv(out_dir / "transition_menu_row_level.csv", index=False)
        selected_df.to_csv(out_dir / "transition_menu_frontier_row_level.csv", index=False)
        source_df.to_csv(out_dir / "transition_menu_source_row_level.csv", index=False)
    summarize_group(row_df, [], ["oracle_cover", "oracle_min_l1", "duplicate_rate_rounded_1e6", "transition_candidate_frac", "lag_candidate_frac"]).to_csv(out_dir / "transition_menu_oracle_summary.csv", index=False)
    summarize_group(row_df, ["move_bin"], ["oracle_cover", "oracle_min_l1", "duplicate_rate_rounded_1e6"]).to_csv(out_dir / "transition_menu_oracle_by_move_bin.csv", index=False)
    summarize_group(row_df, ["predicted_move_bin"], ["oracle_cover", "oracle_min_l1", "duplicate_rate_rounded_1e6", "lag_l1_to_obs", "predicted_move_probability", "predicted_move_magnitude"]).to_csv(out_dir / "transition_menu_oracle_by_predicted_move_bin.csv", index=False)
    summarize_group(row_df, ["oracle_nearest_source_group"], ["oracle_cover", "oracle_min_l1"]).to_csv(out_dir / "transition_menu_oracle_nearest_source.csv", index=False)
    summarize_group(row_df, ["oracle_nearest_source_bucket"], ["oracle_cover", "oracle_min_l1", "oracle_nearest_donor_move_l1", "oracle_nearest_post_projection_l1_from_lag"]).to_csv(out_dir / "transition_menu_oracle_nearest_source_bucket.csv", index=False)
    summarize_group(source_df, ["source_bucket"], ["source_oracle_cover", "source_min_l1", "n_candidates", "mean_donor_move_l1", "same_series_share", "mean_donor_retrieval_distance", "mean_post_projection_l1_from_lag", "mean_post_projection_l1_from_raw_transition_candidate"]).to_csv(out_dir / "transition_menu_source_oracle_summary.csv", index=False)
    summarize_group(selected_df, ["model", "frontier"], ["coverage", "distance", "violation"]).to_csv(out_dir / "transition_menu_frontier_summary.csv", index=False)
    summarize_group(selected_df, ["model", "frontier", "move_bin"], ["coverage", "distance", "violation"]).to_csv(out_dir / "transition_menu_frontier_by_move_bin.csv", index=False)
    summarize_group(selected_df, ["model", "frontier", "predicted_move_bin"], ["coverage", "distance", "violation"]).to_csv(out_dir / "transition_menu_frontier_by_predicted_move_bin.csv", index=False)
    summarize_group(selected_df[selected_df["coverage"] > 0], ["model", "frontier", "nearest_selected_source_group"], ["coverage", "distance"]).to_csv(out_dir / "transition_menu_selected_hit_source.csv", index=False)
    summarize_group(selected_df[selected_df["coverage"] > 0], ["model", "frontier", "nearest_selected_source_bucket"], ["coverage", "distance", "nearest_selected_donor_move_l1"]).to_csv(out_dir / "transition_menu_selected_hit_source_bucket.csv", index=False)
    manifest = {
        "selected_root": str(selected_root),
        "config": str(Path(args.config).resolve()),
        "out_dir": str(out_dir),
        "split": args.split,
        "n_rows": int(len(row_df)),
        "max_rows": args.max_rows,
        "n_candidates": args.n_candidates,
        "tolerance": args.tol,
        "transition_donors": int(len(gen.transition_library.donor_pos)) if gen.transition_library is not None else 0,
        "notes": [
            "Candidate-only diagnostic. No RP fitting is performed.",
            "Observed candidate is excluded from oracle and frontier metrics.",
            "Transition donors come from train split only.",
        ],
    }
    with open(out_dir / "transition_menu_diagnostic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(pd.read_csv(out_dir / "transition_menu_oracle_summary.csv").to_string(index=False))
    print(pd.read_csv(out_dir / "transition_menu_frontier_summary.csv").to_string(index=False))
    print(pd.read_csv(out_dir / "transition_menu_frontier_by_move_bin.csv").to_string(index=False))


if __name__ == "__main__":
    main()
