"""Summarize supervised 20% recommendation-set results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.pipeline.build_calibration_traceability import cluster_bootstrap_ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row_level",
        default="outputs/supervised_ranker/supervised_ranker_f20_row_level.parquet",
    )
    parser.add_argument("--out", default="tables/supervised_ranker_f20_reported.csv")
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_level = pd.read_parquet(args.row_level)
    test = row_level[row_level["split"].astype(str).eq("test")].copy()
    metrics = ["choice_coverage", "distance_to_set_l1", "recommendation_violation_rate"]
    summary, _ = cluster_bootstrap_ci(
        test,
        cluster_col="series_id",
        model_col="model",
        metric_cols=metrics,
        primary_model="supervised_feasibility_filter_f20",
        baseline_models=["supervised_global_f20"],
        n_boot=args.n_boot,
        seed=args.seed,
    )

    labels = {
        "supervised_global_f20": "Supervised ranking, global menu, f20",
        "supervised_feasibility_filter_f20": "Supervised ranking, feasibility filter, f20",
    }
    rows = []
    for model, label in labels.items():
        g = test[test["model"].eq(model)]
        if g.empty:
            raise ValueError(f"Missing supervised model in row-level output: {model}")
        s = summary[summary["model"].eq(model)].set_index("metric")
        rows.append(
            {
                "model_label": label,
                "sample_scope": "full_test",
                "n": int(len(g)),
                "n_series": int(g["series_id"].astype(str).nunique()),
                "recommendation_fraction": float(g["recommendation_fraction"].iloc[0]),
                "point_estimate_coverage_pct": 100.0 * float(g["choice_coverage"].mean()),
                "ci_low_coverage_pct": 100.0 * float(s.loc["coverage", "ci_low"]),
                "ci_high_coverage_pct": 100.0 * float(s.loc["coverage", "ci_high"]),
                "point_estimate_distance_l1": float(g["distance_to_set_l1"].mean()),
                "ci_low_distance_l1": float(s.loc["distance", "ci_low"]),
                "ci_high_distance_l1": float(s.loc["distance", "ci_high"]),
                "point_estimate_violation_pct": 100.0 * float(g["recommendation_violation_rate"].mean()),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
