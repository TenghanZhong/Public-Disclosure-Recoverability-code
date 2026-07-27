from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rowlevel_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--frontier", type=float, default=0.20)
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _clean_frontier_value(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").round(8)


def _share_table(df: pd.DataFrame, stage: str, source_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (move_bin, source), g in df.groupby(["move_bin", source_col], dropna=False):
        total = int((df["move_bin"] == move_bin).sum())
        rows.append(
            {
                "stage": stage,
                "move_bin": move_bin,
                "source_bucket": source,
                "hit_n": int(len(g)),
                "move_stage_hit_n": total,
                "share_within_move_stage_hits": float(len(g) / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def source_attribution(row_df: pd.DataFrame, selected_df: pd.DataFrame, frontier: float) -> pd.DataFrame:
    out = []
    out.append(_share_table(row_df[row_df["oracle_cover"] > 0], "oracle_nearest", "oracle_nearest_source_bucket"))
    f = selected_df[_clean_frontier_value(selected_df["frontier"]).eq(round(frontier, 8)) & (selected_df["coverage"] > 0)]
    for model, g in f.groupby("model", dropna=False):
        out.append(_share_table(g, f"f{int(frontier * 100):02d}_{model}", "nearest_selected_source_bucket"))
    return pd.concat([x for x in out if not x.empty], ignore_index=True)


def summarize_direction(selected_df: pd.DataFrame, frontier: float) -> pd.DataFrame:
    f = selected_df[_clean_frontier_value(selected_df["frontier"]).eq(round(frontier, 8))].copy()
    metrics = [
        "coverage",
        "distance",
        "centroid_selected_cosine_to_true_move",
        "centroid_selected_asset_cosine_to_true_move",
        "centroid_selected_sign_agreement_to_true_move",
        "centroid_selected_move_l1_error",
        "top_selected_cosine_to_true_move",
        "top_selected_asset_cosine_to_true_move",
        "top_selected_move_l1_error",
        "nearest_selected_cosine_to_true_move",
        "nearest_selected_move_l1_error",
        "true_move_l1",
    ]
    rows = []
    for (model, move_bin), g in f.groupby(["model", "move_bin"], dropna=False):
        row = {"model": model, "move_bin": move_bin, "n": int(len(g)), "n_series": int(g["series_id"].nunique())}
        for m in metrics:
            row[m] = float(pd.to_numeric(g[m], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap_direction(selected_df: pd.DataFrame, frontier: float, n_boot: int, seed: int) -> pd.DataFrame:
    f = selected_df[_clean_frontier_value(selected_df["frontier"]).eq(round(frontier, 8))].copy()
    metrics = [
        "coverage",
        "distance",
        "centroid_selected_cosine_to_true_move",
        "centroid_selected_asset_cosine_to_true_move",
        "centroid_selected_move_l1_error",
        "top_selected_cosine_to_true_move",
        "top_selected_move_l1_error",
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for (model, move_bin), g in f.groupby(["model", "move_bin"], dropna=False):
        clusters = g["series_id"].astype(str).dropna().unique()
        if len(clusters) == 0:
            continue
        metric_values = {m: pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float) for m in metrics}
        cluster_to_idx = {
            c: np.flatnonzero(g["series_id"].astype(str).to_numpy() == c)
            for c in clusters
        }
        boot = {m: [] for m in metrics}
        for _ in range(n_boot):
            draw = rng.choice(clusters, size=len(clusters), replace=True)
            idx = np.concatenate([cluster_to_idx[c] for c in draw])
            for m in metrics:
                boot[m].append(float(np.nanmean(metric_values[m][idx])))
        for m in metrics:
            vals = np.asarray(boot[m], dtype=float)
            point = float(np.nanmean(metric_values[m]))
            rows.append(
                {
                    "model": model,
                    "move_bin": move_bin,
                    "metric": m,
                    "point_estimate": point,
                    "ci_low": float(np.nanquantile(vals, 0.025)),
                    "ci_high": float(np.nanquantile(vals, 0.975)),
                    "bootstrap_mean": float(np.nanmean(vals)),
                    "bootstrap_se": float(np.nanstd(vals, ddof=1)),
                    "n": int(len(g)),
                    "n_series": int(len(clusters)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rowlevel_dir = Path(args.rowlevel_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_df = pd.read_csv(rowlevel_dir / "transition_menu_row_level.csv")
    selected_df = pd.read_csv(rowlevel_dir / "transition_menu_frontier_row_level.csv")

    source = source_attribution(row_df, selected_df, args.frontier)
    direction = summarize_direction(selected_df, args.frontier)
    direction_ci = cluster_bootstrap_direction(selected_df, args.frontier, args.n_boot, args.seed)

    source.to_csv(out_dir / "transition_source_by_move_bin.csv", index=False)
    direction.to_csv(out_dir / "transition_direction_by_move_bin.csv", index=False)
    direction_ci.to_csv(out_dir / "transition_direction_bootstrap_ci.csv", index=False)

    print("Wrote:")
    print(out_dir / "transition_source_by_move_bin.csv")
    print(out_dir / "transition_direction_by_move_bin.csv")
    print(out_dir / "transition_direction_bootstrap_ci.csv")
    print(direction.to_string(index=False))


if __name__ == "__main__":
    main()
