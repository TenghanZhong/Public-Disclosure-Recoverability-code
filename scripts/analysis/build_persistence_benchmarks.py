"""Build persistence and candidate-menu benchmarks with cluster confidence intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_root", default="outputs/full_pipeline")
    parser.add_argument("--menu_ablation_root", default="outputs/menu_ablation")
    parser.add_argument("--persistence_dir", default="outputs/persistence_diagnostics")
    parser.add_argument("--transition_rowlevel_dir", default="outputs/transition_diagnostics")
    parser.add_argument("--out_dir", default="outputs/persistence_benchmarks")
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_series_id(df: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path, columns=["series_id"]).reset_index(drop=True)
    out = df.copy()
    out["series_id"] = panel.loc[out["row_pos"].to_numpy(dtype=int), "series_id"].astype(str).to_numpy()
    return out


def bootstrap_ci(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = ["coverage_pct", "distance_l1", "violation_pct"]
    rows = []
    for model, g in df.groupby("model_label", dropna=False):
        clusters = g["series_id"].astype(str).dropna().unique()
        cluster_to_idx = {c: np.flatnonzero(g["series_id"].astype(str).to_numpy() == c) for c in clusters}
        vals = {m: pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float) for m in metrics}
        for m in metrics:
            boot = []
            for _ in range(n_boot):
                draw = rng.choice(clusters, size=len(clusters), replace=True)
                idx = np.concatenate([cluster_to_idx[c] for c in draw])
                boot.append(float(np.nanmean(vals[m][idx])))
            b = np.asarray(boot, dtype=float)
            rows.append(
                {
                    "model_label": model,
                    "metric": m,
                    "point_estimate": float(np.nanmean(vals[m])),
                    "ci_low": float(np.nanquantile(b, 0.025)),
                    "ci_high": float(np.nanquantile(b, 0.975)),
                    "bootstrap_se": float(np.nanstd(b, ddof=1)),
                    "n": int(len(g)),
                    "n_series": int(len(clusters)),
                    "sample_scope": g["sample_scope"].iloc[0],
                }
            )
    return pd.DataFrame(rows)


def add_model_rows(rows: list[pd.DataFrame], label: str, df: pd.DataFrame, coverage: str, distance: str, violation: str, scope: str) -> None:
    out = pd.DataFrame(
        {
            "model_label": label,
            "series_id": df["series_id"].astype(str),
            "coverage_pct": pd.to_numeric(df[coverage], errors="coerce").astype(float) * 100.0,
            "distance_l1": pd.to_numeric(df[distance], errors="coerce").astype(float),
            "violation_pct": pd.to_numeric(df[violation], errors="coerce").astype(float) * 100.0,
            "sample_scope": scope,
        }
    )
    rows.append(out)


def load_eval_model_rows(root: Path, labels: dict[str, str], scope: str) -> list[pd.DataFrame]:
    cols = [
        "model",
        "split",
        "row_pos",
        "frontier_f20_recommendation_coverage_l1tol",
        "frontier_f20_recommendation_distance_to_set_l1",
        "frontier_f20_recommendation_violation_rate",
    ]
    df = pd.read_parquet(root / "evaluation_row_level.parquet", columns=cols)
    df = df[df["split"].astype(str).eq("test") & df["model"].isin(labels)].copy()
    df = add_series_id(df, root / "panel_model_ready.parquet")
    rows = []
    for model, label in labels.items():
        g = df[df["model"].eq(model)]
        if not g.empty:
            add_model_rows(
                rows,
                label,
                g,
                "frontier_f20_recommendation_coverage_l1tol",
                "frontier_f20_recommendation_distance_to_set_l1",
                "frontier_f20_recommendation_violation_rate",
                scope,
            )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []

    selected_root = Path(args.selected_root)
    rows.extend(
        load_eval_model_rows(
            selected_root,
            {
                "exogenous_polytope": "Mandate-aware RP, f20",
                "point_minimax_preference": "Point minimax, f20",
                "global_polytope": "Global no-mandate RP, f20",
                "mv_inverse_optimization_polytope": "Mean-variance IO, f20",
            },
            "full_test",
        )
    )

    menu_root = Path(args.menu_ablation_root)
    rows.extend(load_eval_model_rows(menu_root / "menu_no_lagged_self_candidates", {"exogenous_polytope": "No lagged self candidates, f20"}, "full_test"))
    rows.extend(load_eval_model_rows(menu_root / "menu_peer_train_only_candidates", {"exogenous_polytope": "Training peer candidates only, f20"}, "full_test"))

    persistence_dir = Path(args.persistence_dir)
    single = pd.read_csv(persistence_dir / "persistence_single_point_baselines_row_level.csv")
    for label, cov, dist, viol in [
        ("Bare lagged self", "bare_persistence_coverage", "bare_persistence_distance_l1", "bare_persistence_violation"),
        ("Mandate-filtered lagged self", "mandate_filtered_persistence_coverage", "mandate_filtered_persistence_distance_l1", "mandate_filtered_persistence_violation"),
        ("Projected lagged self", "projected_persistence_coverage", "projected_persistence_distance_l1", "projected_persistence_violation"),
    ]:
        g = single[single["split"].astype(str).eq("test")].copy()
        if "series_id" not in g.columns:
            g = add_series_id(g, selected_root / "panel_model_ready.parquet")
        add_model_rows(rows, label, g, cov, dist, viol, "full_test")

    size = pd.read_csv(persistence_dir / "size_matched_persistence_frontier_row_level.csv")
    g = size[size["frontier"].astype(str).eq("f20") & size["split"].astype(str).eq("test")].copy()
    if "series_id" not in g.columns:
        g = add_series_id(g, selected_root / "panel_model_ready.parquet")
    add_model_rows(rows, "Size-matched persistence, f20", g, "coverage", "distance_l1", "violation_rate", "full_test")

    trans = pd.read_csv(Path(args.transition_rowlevel_dir) / "transition_menu_frontier_row_level.csv")
    trans = trans[np.isclose(pd.to_numeric(trans["frontier"], errors="coerce"), 0.20)]
    for model, label in [
        ("oracle_upper_bound", "Transition oracle ceiling, f20"),
        ("transition_source_then_persistence", "Transition source ranker, f20"),
        ("transition_knn_retrieval", "Transition KNN ranker, f20"),
    ]:
        g = trans[trans["model"].astype(str).eq(model)].copy()
        if not g.empty:
            add_model_rows(rows, label, g, "coverage", "distance", "violation", "sample_5000")

    long = pd.concat(rows, ignore_index=True)
    ci = bootstrap_ci(long, args.n_boot, args.seed)
    pivot = (
        ci.pivot_table(
            index=["model_label", "sample_scope", "n", "n_series"],
            columns="metric",
            values=["point_estimate", "ci_low", "ci_high"],
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns = ["_".join([str(x) for x in c if x]) for c in pivot.columns.to_flat_index()]
    long.to_csv(out_dir / "persistence_benchmarks_row_level_long.csv", index=False)
    ci.to_csv(out_dir / "persistence_ladder_cluster_ci_long.csv", index=False)
    pivot.to_csv(out_dir / "persistence_ladder_with_ci.csv", index=False)
    print(pivot.to_string(index=False))


if __name__ == "__main__":
    main()
