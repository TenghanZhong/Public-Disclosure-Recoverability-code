"""Persistence and candidate-menu diagnostics.

This script measures whether reported coverage is driven by lagged
self-history candidates. It reads a selected-specification result directory,
rebuilds the exogenous mandate-proxy bounds, and evaluates persistence-based
baselines on the same candidate-menu construction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_ROOT = DEFAULT_PACKAGE_ROOT / "outputs" / "full_pipeline"
DEFAULT_OUT_DIR = DEFAULT_PACKAGE_ROOT / "outputs" / "persistence_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected_root",
        default=str(DEFAULT_SELECTED_ROOT),
        help="Existing selected-spec result directory.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for diagnostic CSV files.",
    )
    parser.add_argument(
        "--package_root",
        default=str(DEFAULT_PACKAGE_ROOT),
        help="Repository root.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML. Defaults to <package_root>/configs/main_32gb.yaml.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_package(package_root: Path):
    sys.path.insert(0, str(package_root))
    from mandate_setirl.bounds import action_violations, fit_mandate_bounds
    from mandate_setirl.candidates import CandidateGenerator
    from mandate_setirl.utils import read_yaml

    return action_violations, fit_mandate_bounds, CandidateGenerator, read_yaml


def l1_distance(a: np.ndarray, b: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    if idx is None:
        return np.abs(a - b).sum(axis=-1)
    return np.abs(a[..., idx] - b[..., idx]).sum(axis=-1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(a.shape[0], np.nan, dtype=float)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def sign_agreement(a: np.ndarray, b: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    if idx is not None:
        a = a[:, idx]
        b = b[:, idx]
    mask = (np.abs(a) > 1e-8) | (np.abs(b) > 1e-8)
    same = np.sign(a) == np.sign(b)
    denom = mask.sum(axis=1)
    out = np.full(a.shape[0], np.nan, dtype=float)
    ok = denom > 0
    out[ok] = (same & mask).sum(axis=1)[ok] / denom[ok]
    return out


def summarize_binary(rows: list[dict], group_cols: Iterable[str], metric_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for keys, g in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n"] = int(len(g))
        for c in metric_cols:
            if c not in g.columns:
                continue
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def load_action_columns(selected_root: Path) -> dict:
    with open(selected_root / "action_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_future_outcomes(panel: pd.DataFrame, action_cols: list[str], entity_col: str) -> pd.DataFrame:
    date_col = "report_date" if "report_date" in panel.columns else None
    sort_cols = [entity_col] + ([date_col] if date_col else [])
    work = panel.sort_values(sort_cols).copy()
    out = pd.DataFrame(index=work.index)
    out["row_pos"] = work.index.astype(int)
    action = work[action_cols].apply(pd.to_numeric, errors="coerce")
    next_action = action.groupby(work[entity_col]).shift(-1)
    out["next_style_drift_l1"] = (next_action - action).abs().sum(axis=1)
    out["has_next_report"] = next_action.notna().all(axis=1)
    for c in ["RETURN_AFTR_FEES_AND_EXPENSES", "RETURN_B4_FEES_AND_EXPENSES", "STDV_AFTR_FEES_AND_EXPENSES", "STDV_B4_FEES_AND_EXPENSES"]:
        if c in work.columns:
            out[f"next_{c.lower()}"] = pd.to_numeric(work[c], errors="coerce").groupby(work[entity_col]).shift(-1)
    return out.sort_index()


def residualized_beta(df: pd.DataFrame, y_col: str, x_col: str, fe_cols: list[str]) -> tuple[float, int]:
    data = df[[y_col, x_col] + fe_cols].copy()
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data = data.dropna()
    if len(data) < 50 or data[x_col].nunique() < 2:
        return np.nan, int(len(data))
    y = data[y_col].to_numpy(dtype=float)
    x = data[x_col].to_numpy(dtype=float)
    y = y - y.mean()
    x = x - x.mean()
    # Alternating demeaning for two low-cardinality fixed-effect sets.
    for _ in range(8):
        for fe in fe_cols:
            codes = pd.factorize(data[fe], sort=False)[0]
            for arr in [y, x]:
                sums = np.bincount(codes, weights=arr)
                counts = np.bincount(codes)
                arr -= sums[codes] / np.maximum(counts[codes], 1)
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return np.nan, int(len(data))
    return float(np.dot(x, y) / denom), int(len(data))


def main():
    args = parse_args()
    selected_root = Path(args.selected_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    package_root = Path(args.package_root)
    config_path = Path(args.config) if args.config else package_root / "configs" / "main_32gb.yaml"
    action_violations, fit_mandate_bounds, CandidateGenerator, read_yaml = add_package(package_root)

    cfg = read_yaml(config_path)
    cfg.setdefault("candidates", {})
    cfg["candidates"]["chunk_size"] = int(args.chunk_size)
    ac = load_action_columns(selected_root)
    action_cols = ac["action_cols"]
    asset_cols = ac["asset_action_cols"]
    overlay_cols = ac["overlay_action_cols"]
    asset_idx = np.array([action_cols.index(c) for c in asset_cols], dtype=int)
    overlay_idx = np.array([action_cols.index(c) for c in overlay_cols], dtype=int)
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")

    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    eval_cols = [
        "model",
        "split",
        "row_pos",
        "frontier_f20_choice_coverage",
        "frontier_f20_distance_to_set_l1",
        "frontier_f20_violation_rate",
    ]
    eval_df = pd.read_parquet(selected_root / "evaluation_row_level.parquet", columns=eval_cols)

    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
    rng = np.random.default_rng(int(args.seed))
    candgen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, rng, bounds)

    split_mask = panel["split"].astype(str).eq(args.split).to_numpy()
    positions = np.flatnonzero(split_mask)
    A_obs = panel.loc[split_mask, action_cols].to_numpy(dtype=float)
    A_lag = panel.loc[split_mask, [f"lag_{c}" for c in action_cols]].to_numpy(dtype=float)
    lag_l1 = l1_distance(A_obs, A_lag)
    move_bin = pd.cut(
        lag_l1,
        [-np.inf, args.tol, 0.25, np.inf],
        labels=["stayer_l1_le_005", "moderate_l1_005_025", "mover_l1_gt_025"],
    ).astype(str)
    lower, upper = bounds.bounds_for_positions(positions)
    lag_viol = action_violations(A_lag, lower, upper, tol=1e-7)
    obs_viol = action_violations(A_obs, lower, upper, tol=1e-7)

    projected = np.zeros_like(A_lag)
    for i in range(len(positions)):
        projected[i] = candgen.project_action(A_lag[i], lower[i], upper[i])
    proj_l1 = l1_distance(A_obs, projected)
    proj_viol = action_violations(projected, lower, upper, tol=1e-7)

    baseline_rows = []
    for i, row_pos in enumerate(positions):
        baseline_rows.append(
            {
                "row_pos": int(row_pos),
                "split": args.split,
                "move_bin": move_bin[i],
                "lag_l1": float(lag_l1[i]),
                "bare_persistence_coverage": bool(lag_l1[i] <= args.tol),
                "bare_persistence_distance_l1": float(lag_l1[i]),
                "bare_persistence_asset_l1": float(l1_distance(A_obs[i], A_lag[i], asset_idx)),
                "bare_persistence_overlay_l1": float(l1_distance(A_obs[i], A_lag[i], overlay_idx)),
                "bare_persistence_violation": bool(lag_viol[i]),
                "mandate_filtered_persistence_coverage": bool((lag_l1[i] <= args.tol) and (not lag_viol[i])),
                "mandate_filtered_persistence_distance_l1": float(lag_l1[i]) if not lag_viol[i] else np.nan,
                "mandate_filtered_persistence_violation": False if not lag_viol[i] else np.nan,
                "projected_persistence_coverage": bool(proj_l1[i] <= args.tol),
                "projected_persistence_distance_l1": float(proj_l1[i]),
                "projected_persistence_asset_l1": float(l1_distance(A_obs[i], projected[i], asset_idx)),
                "projected_persistence_overlay_l1": float(l1_distance(A_obs[i], projected[i], overlay_idx)),
                "projected_persistence_violation": bool(proj_viol[i]),
                "observed_out_of_proxy": bool(obs_viol[i]),
            }
        )

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(out_dir / "persistence_single_point_baselines_row_level.csv", index=False)

    frontier_fracs = [0.05, 0.10, 0.20, 0.30, 0.50]
    size_rows = []
    oracle_rows = []
    source_rows = []
    direction_rows = []

    for start in range(0, len(positions), args.chunk_size):
        pos = positions[start : start + args.chunk_size]
        batch = candgen.generate_for_positions(pos)
        B, M, _ = batch.actions.shape
        lo, hi = bounds.bounds_for_positions(pos)
        violations = action_violations(batch.actions.reshape(-1, len(action_cols)), np.repeat(lo, M, axis=0), np.repeat(hi, M, axis=0), tol=1e-7).reshape(B, M)
        for bi, row_pos in enumerate(pos):
            global_i = start + bi
            obs = batch.actions[bi, 0]
            lag = A_lag[global_i]
            feasible = (~violations[bi]).copy()
            feasible[0] = False
            idx = np.flatnonzero(feasible)
            if len(idx) == 0:
                oracle_rows.append(
                    {
                        "row_pos": int(row_pos),
                        "oracle_hit": False,
                        "oracle_distance_l1": np.nan,
                        "oracle_source": "empty",
                        "oracle_candidate_count": 0,
                    }
                )
                continue
            menu = batch.actions[bi, idx]
            src = batch.sources[bi, idx].astype(str)
            dist_obs = l1_distance(menu, obs)
            dist_lag = l1_distance(menu, lag)
            nearest_j = int(np.argmin(dist_obs))
            oracle_hit = bool(dist_obs[nearest_j] <= args.tol)
            oracle_rows.append(
                {
                    "row_pos": int(row_pos),
                    "oracle_hit": oracle_hit,
                    "oracle_distance_l1": float(dist_obs[nearest_j]),
                    "oracle_source": str(src[nearest_j]),
                    "oracle_candidate_count": int(len(idx)),
                }
            )
            source_counts = pd.Series(src).value_counts()
            for source, count in source_counts.items():
                source_rows.append({"row_pos": int(row_pos), "source": str(source), "candidate_count": int(count)})

            order = np.argsort(dist_lag, kind="mergesort")
            for frac in frontier_fracs:
                tag = f"f{int(round(frac * 100)):02d}"
                k = max(1, int(math.ceil(frac * len(order))))
                chosen_local = order[:k]
                chosen = menu[chosen_local]
                chosen_src = src[chosen_local]
                chosen_dist = l1_distance(chosen, obs)
                nearest_sel = int(np.argmin(chosen_dist))
                coverage = bool(chosen_dist[nearest_sel] <= args.tol)
                centroid = chosen.mean(axis=0)
                top_action = chosen[0]
                true_move = obs - lag
                centroid_move = centroid - lag
                top_move = top_action - lag
                size_rows.append(
                    {
                        "row_pos": int(row_pos),
                        "split": args.split,
                        "frontier": tag,
                        "move_bin": move_bin[global_i],
                        "coverage": coverage,
                        "distance_l1": float(chosen_dist[nearest_sel]),
                        "asset_l1": float(l1_distance(chosen[nearest_sel], obs, asset_idx)),
                        "overlay_l1": float(l1_distance(chosen[nearest_sel], obs, overlay_idx)),
                        "violation_rate": 0.0,
                        "selected_count": int(k),
                        "candidate_count": int(len(order)),
                        "nearest_selected_source": str(chosen_src[nearest_sel]),
                    }
                )
                if tag == "f20":
                    direction_rows.append(
                        {
                            "row_pos": int(row_pos),
                            "move_bin": move_bin[global_i],
                            "true_move_l1": float(l1_distance(obs, lag)),
                            "centroid_cosine": float(cosine_similarity(true_move[None, :], centroid_move[None, :])[0]),
                            "top_cosine": float(cosine_similarity(true_move[None, :], top_move[None, :])[0]),
                            "centroid_sign_agreement": float(sign_agreement(true_move[None, :], centroid_move[None, :])[0]),
                            "top_sign_agreement": float(sign_agreement(true_move[None, :], top_move[None, :])[0]),
                            "centroid_asset_cosine": float(cosine_similarity(true_move[None, asset_idx], centroid_move[None, asset_idx])[0]),
                            "top_asset_cosine": float(cosine_similarity(true_move[None, asset_idx], top_move[None, asset_idx])[0]),
                            "centroid_asset_sign_agreement": float(sign_agreement(true_move[None, :], centroid_move[None, :], asset_idx)[0]),
                            "top_asset_sign_agreement": float(sign_agreement(true_move[None, :], top_move[None, :], asset_idx)[0]),
                        }
                    )

    oracle_df = pd.DataFrame(oracle_rows)
    size_df = pd.DataFrame(size_rows)
    source_df = pd.DataFrame(source_rows)
    direction_df = pd.DataFrame(direction_rows)
    oracle_df.to_csv(out_dir / "oracle_menu_source_row_level.csv", index=False)
    size_df.to_csv(out_dir / "size_matched_persistence_frontier_row_level.csv", index=False)
    source_df.to_csv(out_dir / "candidate_source_counts_row_level.csv", index=False)
    direction_df.to_csv(out_dir / "size_matched_persistence_direction_row_level.csv", index=False)

    # Summary tables.
    single_summary_rows = []
    for name, cov_col, dist_col, viol_col in [
        ("bare_lagged_self", "bare_persistence_coverage", "bare_persistence_distance_l1", "bare_persistence_violation"),
        ("lagged_self_if_mandate_feasible", "mandate_filtered_persistence_coverage", "mandate_filtered_persistence_distance_l1", "mandate_filtered_persistence_violation"),
        ("projected_lagged_self", "projected_persistence_coverage", "projected_persistence_distance_l1", "projected_persistence_violation"),
    ]:
        vals = baseline_df
        single_summary_rows.append(
            {
                "model": name,
                "n": int(len(vals)),
                "coverage_pct": float(vals[cov_col].mean() * 100),
                "distance_l1": float(pd.to_numeric(vals[dist_col], errors="coerce").mean()),
                "asset_l1": float(pd.to_numeric(vals.get(dist_col.replace("distance_l1", "asset_l1"), pd.Series(np.nan, index=vals.index)), errors="coerce").mean()),
                "overlay_l1": float(pd.to_numeric(vals.get(dist_col.replace("distance_l1", "overlay_l1"), pd.Series(np.nan, index=vals.index)), errors="coerce").mean()),
                "violation_pct": float(pd.to_numeric(vals[viol_col], errors="coerce").mean() * 100),
            }
        )
    pd.DataFrame(single_summary_rows).to_csv(out_dir / "persistence_single_point_baselines_summary.csv", index=False)

    size_summary = (
        size_df.groupby("frontier", as_index=False)
        .agg(
            n=("row_pos", "size"),
            coverage_pct=("coverage", lambda x: float(np.mean(x) * 100)),
            distance_l1=("distance_l1", "mean"),
            asset_l1=("asset_l1", "mean"),
            overlay_l1=("overlay_l1", "mean"),
            violation_pct=("violation_rate", lambda x: float(np.mean(x) * 100)),
            selected_count_mean=("selected_count", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
        )
    )
    size_summary.to_csv(out_dir / "size_matched_persistence_frontier_summary.csv", index=False)

    oracle_summary = pd.DataFrame(
        [
            {
                "n": int(len(oracle_df)),
                "oracle_coverage_pct": float(oracle_df["oracle_hit"].mean() * 100),
                "oracle_distance_l1": float(oracle_df["oracle_distance_l1"].mean()),
                "candidate_count_mean": float(oracle_df["oracle_candidate_count"].mean()),
            }
        ]
    )
    oracle_summary.to_csv(out_dir / "oracle_menu_ceiling_summary.csv", index=False)

    hit_source = (
        oracle_df[oracle_df["oracle_hit"]]
        .groupby("oracle_source", as_index=False)
        .agg(n_hits=("row_pos", "size"), distance_l1=("oracle_distance_l1", "mean"))
    )
    hit_source["pct_of_oracle_hits"] = hit_source["n_hits"] / max(hit_source["n_hits"].sum(), 1) * 100
    hit_source["pct_of_test_reports"] = hit_source["n_hits"] / max(len(oracle_df), 1) * 100
    hit_source.to_csv(out_dir / "oracle_hit_source_decomposition.csv", index=False)

    selected_hit_source = (
        size_df[(size_df["frontier"] == "f20") & (size_df["coverage"])]
        .groupby("nearest_selected_source", as_index=False)
        .agg(n_hits=("row_pos", "size"), distance_l1=("distance_l1", "mean"))
    )
    selected_hit_source["pct_of_size_matched_hits"] = selected_hit_source["n_hits"] / max(selected_hit_source["n_hits"].sum(), 1) * 100
    selected_hit_source["pct_of_test_reports"] = selected_hit_source["n_hits"] / max(len(oracle_df), 1) * 100
    selected_hit_source.to_csv(out_dir / "size_matched_persistence_hit_source_f20.csv", index=False)

    # Stayer / mover comparison table.
    rp = eval_df[(eval_df["split"] == args.split) & (eval_df["model"] == "exogenous_polytope")].copy()
    point = eval_df[(eval_df["split"] == args.split) & (eval_df["model"] == "point_minimax_preference")].copy()
    move_lookup = baseline_df[["row_pos", "move_bin"]].copy()
    stayer_rows = []
    for model_name, frame, cov_col, dist_col, viol_col in [
        ("bare_lagged_self", baseline_df, "bare_persistence_coverage", "bare_persistence_distance_l1", "bare_persistence_violation"),
        ("lagged_self_if_mandate_feasible", baseline_df, "mandate_filtered_persistence_coverage", "mandate_filtered_persistence_distance_l1", "mandate_filtered_persistence_violation"),
        ("projected_lagged_self", baseline_df, "projected_persistence_coverage", "projected_persistence_distance_l1", "projected_persistence_violation"),
        ("size_matched_persistence_f20", size_df[size_df["frontier"] == "f20"], "coverage", "distance_l1", "violation_rate"),
        ("mandate_aware_rp_f20", rp.merge(move_lookup, on="row_pos", how="inner"), "frontier_f20_choice_coverage", "frontier_f20_distance_to_set_l1", "frontier_f20_violation_rate"),
        ("point_minimax_f20", point.merge(move_lookup, on="row_pos", how="inner"), "frontier_f20_choice_coverage", "frontier_f20_distance_to_set_l1", "frontier_f20_violation_rate"),
    ]:
        for mb, g in frame.groupby("move_bin", dropna=False):
            stayer_rows.append(
                {
                    "model": model_name,
                    "move_bin": str(mb),
                    "n": int(len(g)),
                    "coverage_pct": float(pd.to_numeric(g[cov_col], errors="coerce").mean() * 100),
                    "distance_l1": float(pd.to_numeric(g[dist_col], errors="coerce").mean()),
                    "violation_pct": float(pd.to_numeric(g[viol_col], errors="coerce").mean() * 100),
                }
            )
    pd.DataFrame(stayer_rows).to_csv(out_dir / "stayer_mover_model_comparison.csv", index=False)

    direction_summary = (
        direction_df.groupby("move_bin", as_index=False)
        .agg(
            n=("row_pos", "size"),
            true_move_l1=("true_move_l1", "mean"),
            centroid_cosine=("centroid_cosine", "mean"),
            top_cosine=("top_cosine", "mean"),
            centroid_sign_agreement=("centroid_sign_agreement", "mean"),
            top_sign_agreement=("top_sign_agreement", "mean"),
            centroid_asset_cosine=("centroid_asset_cosine", "mean"),
            top_asset_cosine=("top_asset_cosine", "mean"),
            centroid_asset_sign_agreement=("centroid_asset_sign_agreement", "mean"),
            top_asset_sign_agreement=("top_asset_sign_agreement", "mean"),
        )
    )
    direction_summary.to_csv(out_dir / "size_matched_persistence_direction_summary.csv", index=False)

    # Future outcome association diagnostics.
    future = prepare_future_outcomes(panel, action_cols, entity_col)
    all_lower, all_upper = bounds.bounds_for_positions(np.arange(len(panel)))
    obs_all = panel[action_cols].to_numpy(dtype=float)
    out_of_proxy = action_violations(obs_all, all_lower, all_upper, tol=1e-7)
    assoc = pd.DataFrame(
        {
            "row_pos": np.arange(len(panel)),
            "split": panel["split"].astype(str).to_numpy(),
            "report_date": panel["report_date"].astype(str).to_numpy() if "report_date" in panel.columns else "unknown",
            "mandate_cluster": panel["mandate_cluster"].astype(str).to_numpy() if "mandate_cluster" in panel.columns else "0",
            "observed_out_of_proxy": out_of_proxy.astype(float),
        }
    ).merge(future, on="row_pos", how="left")
    outcome_cols = [c for c in assoc.columns if c.startswith("next_")]
    assoc_rows = []
    for y in outcome_cols:
        data = assoc[assoc["has_next_report"].fillna(False)].copy()
        data[y] = pd.to_numeric(data[y], errors="coerce")
        valid = data[y].notna() & data["observed_out_of_proxy"].notna()
        data = data[valid]
        if len(data) == 0:
            continue
        in_proxy = data[data["observed_out_of_proxy"] <= 0.5][y]
        out_proxy = data[data["observed_out_of_proxy"] > 0.5][y]
        beta_fe, n_fe = residualized_beta(data, y, "observed_out_of_proxy", ["report_date", "mandate_cluster"])
        assoc_rows.append(
            {
                "outcome": y,
                "n": int(len(data)),
                "out_of_proxy_n": int((data["observed_out_of_proxy"] > 0.5).sum()),
                "in_proxy_mean": float(in_proxy.mean()) if len(in_proxy) else np.nan,
                "out_of_proxy_mean": float(out_proxy.mean()) if len(out_proxy) else np.nan,
                "raw_difference_out_minus_in": float(out_proxy.mean() - in_proxy.mean()) if len(in_proxy) and len(out_proxy) else np.nan,
                "date_cluster_fe_beta": beta_fe,
                "date_cluster_fe_n": n_fe,
            }
        )
    pd.DataFrame(assoc_rows).to_csv(out_dir / "out_of_proxy_future_outcome_association.csv", index=False)

    manifest = {
        "selected_root": str(selected_root),
        "config": str(config_path),
        "package_root": str(package_root),
        "split": args.split,
        "tolerance": args.tol,
        "n_test": int(len(positions)),
        "notes": [
            "Size-matched persistence uses the same candidate-menu construction and ranks candidates by L1 proximity to lagged self allocation.",
            "Oracle and size-matched hit-source decompositions are computed from regenerated candidate menus with the same configuration and seed.",
            "Exact RP selected-candidate source attribution is not recoverable from the saved row-level evaluation because selected candidate IDs/actions were not stored.",
            "Future outcome association is predictive and should not be interpreted as legal mandate validation or causal evidence.",
        ],
    }
    with open(out_dir / "diagnostic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote diagnostics to", out_dir)
    print(pd.read_csv(out_dir / "persistence_single_point_baselines_summary.csv").to_string(index=False))
    print(pd.read_csv(out_dir / "size_matched_persistence_frontier_summary.csv").to_string(index=False))
    print(pd.read_csv(out_dir / "oracle_menu_ceiling_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
