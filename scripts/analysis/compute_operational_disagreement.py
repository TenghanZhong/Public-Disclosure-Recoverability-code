"""Compute RP-with-preference-sets versus point-minimax operational disagreement.

RP with preference sets has only a small net coverage gain
over the point-minimax baseline. This script decomposes that net difference
into gross local coverage changes on the same test reports and frontier, then
adds series-cluster bootstrap intervals and preference-set diagnostics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROW_LEVEL = os.environ.get("MANDATE_ROW_LEVEL")
DEFAULT_PANEL = os.environ.get("MANDATE_PANEL")

CATEGORIES = ["both", "neither", "point_only", "rp_only"]


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}"


def fmt_num(x: float) -> str:
    return f"{x:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute operational disagreement between RP with preference sets and point-minimax."
    )
    parser.add_argument("--row-level", type=Path, default=DEFAULT_ROW_LEVEL)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rp-model", default="exogenous_polytope")
    parser.add_argument("--point-model", default="point_minimax_preference")
    parser.add_argument("--frontier-tag", default="f20")
    parser.add_argument("--out-tex", type=Path, default=Path("tables/operational_value_macros.tex"))
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("tables/operational_value_summary.csv"),
    )
    parser.add_argument(
        "--out-category-csv",
        type=Path,
        default=Path("tables/operational_value_by_churn_category.csv"),
    )
    parser.add_argument(
        "--out-bootstrap-csv",
        type=Path,
        default=Path("tables/operational_value_bootstrap_ci.csv"),
    )
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--expected-net-pp",
        type=float,
        default=2.59,
        help="Expected RP minus point coverage difference in percentage points.",
    )
    parser.add_argument("--tolerance-pp", type=float, default=0.05)
    args = parser.parse_args()
    if args.row_level is None:
        parser.error("Provide --row-level or set MANDATE_ROW_LEVEL.")
    if args.panel is None:
        parser.error("Provide --panel or set MANDATE_PANEL.")
    return args


def load_analysis_frame(args: argparse.Namespace) -> pd.DataFrame:
    coverage_col = f"frontier_{args.frontier_tag}_choice_coverage"
    distance_col = f"frontier_{args.frontier_tag}_distance_to_set_l1"
    cols = [
        "model",
        "split",
        "row_pos",
        "scope_id",
        "theta_interval_l1_upper_diameter",
        "regret_min_support",
        "regret_max_support",
        coverage_col,
        distance_col,
    ]

    df = pd.read_parquet(args.row_level, columns=cols)
    df = df[df["split"].eq(args.split) & df["model"].isin([args.rp_model, args.point_model])].copy()
    if df.empty:
        raise ValueError("No rows found for requested split/models.")

    counts = df.groupby("model")["row_pos"].nunique()
    for model in [args.rp_model, args.point_model]:
        if model not in counts:
            raise ValueError(f"Missing model in row-level file: {model}")

    cov = df.pivot(index="row_pos", columns="model", values=coverage_col)
    dist = df.pivot(index="row_pos", columns="model", values=distance_col)
    if cov[[args.rp_model, args.point_model]].isna().any().any():
        raise ValueError("Coverage pivot contains missing model/report pairs.")

    rp_meta = df[df["model"].eq(args.rp_model)].set_index("row_pos")
    rp_cov = cov[args.rp_model].astype(bool)
    point_cov = cov[args.point_model].astype(bool)
    out = pd.DataFrame(index=cov.index)
    out["rp_cov"] = rp_cov
    out["point_cov"] = point_cov
    out["rp_only"] = rp_cov & ~point_cov
    out["point_only"] = point_cov & ~rp_cov
    out["both"] = rp_cov & point_cov
    out["neither"] = ~rp_cov & ~point_cov
    out["category"] = np.select(
        [out["both"], out["rp_only"], out["point_only"]],
        ["both", "rp_only", "point_only"],
        default="neither",
    )
    out["theta_width"] = rp_meta.loc[out.index, "theta_interval_l1_upper_diameter"].astype(float).to_numpy()
    out["regret_range"] = (
        rp_meta.loc[out.index, "regret_max_support"].astype(float)
        - rp_meta.loc[out.index, "regret_min_support"].astype(float)
    ).to_numpy()
    out["rp_distance"] = dist[args.rp_model].astype(float)
    out["point_distance"] = dist[args.point_model].astype(float)

    panel = pd.read_parquet(args.panel, columns=["series_id", "split"])
    mapped = panel.iloc[out.index.to_numpy()]
    if not mapped["split"].eq(args.split).all():
        bad = mapped.loc[~mapped["split"].eq(args.split), "split"].value_counts(dropna=False).to_dict()
        raise ValueError(f"row_pos to panel split check failed: {bad}")
    out["series_id"] = mapped["series_id"].to_numpy()
    if out["series_id"].isna().any():
        raise ValueError("series_id contains missing values after row_pos mapping.")
    return out.reset_index(names="row_pos")


def summarize_frame(data: pd.DataFrame, split: str, frontier_tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_total = len(data)
    rp_coverage = float(data["rp_cov"].mean())
    point_coverage = float(data["point_cov"].mean())
    rp_only = float(data["rp_only"].mean())
    point_only = float(data["point_only"].mean())
    net = rp_coverage - point_coverage
    gross_net = rp_only - point_only
    if abs(net - gross_net) > 1e-12:
        raise AssertionError("Gross decomposition does not equal net coverage difference.")

    summary = pd.DataFrame(
        [
            {
                "split": split,
                "frontier_tag": frontier_tag,
                "n_reports": n_total,
                "n_series": int(data["series_id"].nunique()),
                "rp_coverage_pct": 100.0 * rp_coverage,
                "point_coverage_pct": 100.0 * point_coverage,
                "rp_only_pct": 100.0 * rp_only,
                "point_only_pct": 100.0 * point_only,
                "net_coverage_pp": 100.0 * net,
                "rp_distance_l1": float(data["rp_distance"].mean()),
                "point_distance_l1": float(data["point_distance"].mean()),
                "distance_diff": float(data["rp_distance"].mean() - data["point_distance"].mean()),
            }
        ]
    )

    category_rows = []
    for category in CATEGORIES:
        d = data[data["category"].eq(category)]
        category_rows.append(
            {
                "category": category,
                "n": int(len(d)),
                "share_pct": 100.0 * len(d) / n_total,
                "theta_width_mean": float(d["theta_width"].mean()),
                "regret_range_mean": float(d["regret_range"].mean()),
                "rp_distance_mean": float(d["rp_distance"].mean()),
                "point_distance_mean": float(d["point_distance"].mean()),
            }
        )
    category = pd.DataFrame(category_rows)
    return summary, category


def make_cluster_aggregates(data: pd.DataFrame) -> pd.DataFrame:
    work = data[["series_id", "theta_width", "regret_range", "rp_distance", "point_distance"]].copy()
    work["n"] = 1
    work["rp_cov_n"] = data["rp_cov"].astype(int)
    work["point_cov_n"] = data["point_cov"].astype(int)
    for category in CATEGORIES:
        flag = data["category"].eq(category).astype(int)
        work[f"{category}_n"] = flag
        for col in ["theta_width", "regret_range", "rp_distance", "point_distance"]:
            work[f"{col}_{category}_sum"] = data[col] * flag
    agg_cols = {col: "sum" for col in work.columns if col != "series_id"}
    return work.groupby("series_id", sort=False).agg(agg_cols)


def metrics_from_cluster_sums(sums: pd.Series) -> dict[str, float]:
    total_n = float(sums["n"])
    both = float(sums["both_n"])
    rp_only = float(sums["rp_only_n"])
    point_only = float(sums["point_only_n"])
    rp_cov = both + rp_only
    point_cov = both + point_only

    def mean_for(col: str, category: str) -> float:
        denom = float(sums[f"{category}_n"])
        if denom <= 0:
            return np.nan
        return float(sums[f"{col}_{category}_sum"]) / denom

    metrics = {
        "rp_coverage_pct": 100.0 * rp_cov / total_n,
        "point_coverage_pct": 100.0 * point_cov / total_n,
        "rp_only_pct": 100.0 * rp_only / total_n,
        "point_only_pct": 100.0 * point_only / total_n,
        "net_coverage_pp": 100.0 * (rp_cov - point_cov) / total_n,
        "theta_width_rp_only": mean_for("theta_width", "rp_only"),
        "theta_width_point_only": mean_for("theta_width", "point_only"),
        "theta_width_both": mean_for("theta_width", "both"),
        "regret_range_rp_only": mean_for("regret_range", "rp_only"),
        "regret_range_point_only": mean_for("regret_range", "point_only"),
        "regret_range_both": mean_for("regret_range", "both"),
    }
    metrics["theta_width_diff_rp_only_minus_point_only"] = (
        metrics["theta_width_rp_only"] - metrics["theta_width_point_only"]
    )
    metrics["theta_width_diff_rp_only_minus_both"] = metrics["theta_width_rp_only"] - metrics["theta_width_both"]
    metrics["regret_range_diff_rp_only_minus_point_only"] = (
        metrics["regret_range_rp_only"] - metrics["regret_range_point_only"]
    )
    metrics["regret_range_diff_rp_only_minus_both"] = metrics["regret_range_rp_only"] - metrics["regret_range_both"]
    metrics["point_distance_rp_only"] = mean_for("point_distance", "rp_only")
    metrics["rp_distance_point_only"] = mean_for("rp_distance", "point_only")
    metrics["rp_distance_both"] = mean_for("rp_distance", "both")
    metrics["point_distance_both"] = mean_for("point_distance", "both")
    metrics["contested_distance_gap"] = metrics["point_distance_rp_only"] - metrics["rp_distance_point_only"]
    return metrics


def bootstrap_ci(data: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    clusters = make_cluster_aggregates(data)
    point = metrics_from_cluster_sums(clusters.sum(axis=0))
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {k: [] for k in point}
    cluster_values = clusters.to_numpy()
    n_clusters = cluster_values.shape[0]
    columns = clusters.columns
    for _ in range(n_boot):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        sums = pd.Series(cluster_values[idx].sum(axis=0), index=columns)
        boot_metrics = metrics_from_cluster_sums(sums)
        for key, value in boot_metrics.items():
            values[key].append(value)

    rows = []
    for key, estimate in point.items():
        arr = np.asarray(values[key], dtype=float)
        arr = arr[np.isfinite(arr)]
        rows.append(
            {
                "metric": key,
                "estimate": estimate,
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "n_boot": n_boot,
                "n_series": int(n_clusters),
            }
        )
    return pd.DataFrame(rows)


def ci_lookup(ci: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    row = ci.loc[ci["metric"].eq(metric)]
    if row.empty:
        raise KeyError(metric)
    r = row.iloc[0]
    return float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])


def fmt_ci(ci: pd.DataFrame, metric: str, digits: int = 2) -> str:
    estimate, low, high = ci_lookup(ci, metric)
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def main() -> None:
    args = parse_args()
    data = load_analysis_frame(args)
    summary, category = summarize_frame(data, args.split, args.frontier_tag)
    ci = bootstrap_ci(data, args.n_boot, args.seed)

    net_pp = float(summary.loc[0, "net_coverage_pp"])
    if abs(net_pp - args.expected_net_pp) > args.tolerance_pp:
        raise ValueError(
            f"Net coverage difference {net_pp:.3f}pp does not match expected "
            f"{args.expected_net_pp:.3f}pp within {args.tolerance_pp:.3f}pp."
        )

    rp_only_pct = float(summary.loc[0, "rp_only_pct"])
    point_only_pct = float(summary.loc[0, "point_only_pct"])
    rp_only_regret = float(category.loc[category["category"].eq("rp_only"), "regret_range_mean"].iloc[0])
    point_only_regret = float(category.loc[category["category"].eq("point_only"), "regret_range_mean"].iloc[0])
    regret_diff_ci = fmt_ci(ci, "regret_range_diff_rp_only_minus_point_only", digits=3)
    theta_diff_ci = fmt_ci(ci, "theta_width_diff_rp_only_minus_point_only", digits=3)
    point_dist_rp_only = float(category.loc[category["category"].eq("rp_only"), "point_distance_mean"].iloc[0])
    rp_dist_point_only = float(category.loc[category["category"].eq("point_only"), "rp_distance_mean"].iloc[0])
    both_dist = float(category.loc[category["category"].eq("both"), "rp_distance_mean"].iloc[0])
    gross_pct = rp_only_pct + point_only_pct
    contested_gap_ci = fmt_ci(ci, "contested_distance_gap", digits=3)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_category_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_bootstrap_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    category.to_csv(args.out_category_csv, index=False)
    ci.to_csv(args.out_bootstrap_csv, index=False)

    paragraph = (
        "The small net gain over point-minimax hides local coverage changes. "
        f"At the 20\\% frontier, RP with preference sets covers {fmt_ci(ci, 'rp_only_pct')}\\% "
        "of test reports missed by point-minimax, while point-minimax covers "
        f"{fmt_ci(ci, 'point_only_pct')}\\% missed by RP with preference sets; the net difference "
        f"is {fmt_ci(ci, 'net_coverage_pp')} percentage points. "
        "These changes are not only churn: reports covered only by RP with preference sets have a wider "
        f"decision-level rationalization-gap range over the sampled preference set than reports covered only by point-minimax "
        f"({fmt_num(rp_only_regret)} versus {fmt_num(point_only_regret)}; difference "
        f"{regret_diff_ci}). The scope-level $\\ThetaSet_g$ diameter difference is smaller "
        f"({theta_diff_ci}). "
        "The distance pattern points in the same direction: in the contested region, point-minimax "
        f"misses set-only reports by {fmt_num(point_dist_rp_only)}, while RP with preference sets misses "
        f"point-only reports by {fmt_num(rp_dist_point_only)} "
        f"(gap {contested_gap_ci})."
    )

    args.out_tex.write_text(
        "\n".join(
            [
                "% Operational disagreement diagnostics",
                f"\\newcommand{{\\OperationalRpOnlyPct}}{{{rp_only_pct:.2f}}}",
                f"\\newcommand{{\\OperationalPointOnlyPct}}{{{point_only_pct:.2f}}}",
                f"\\newcommand{{\\OperationalNetCoveragePp}}{{{net_pp:.2f}}}",
                f"\\newcommand{{\\OperationalRpOnlyPctCi}}{{{fmt_ci(ci, 'rp_only_pct')}}}",
                f"\\newcommand{{\\OperationalPointOnlyPctCi}}{{{fmt_ci(ci, 'point_only_pct')}}}",
                f"\\newcommand{{\\OperationalNetCoveragePpCi}}{{{fmt_ci(ci, 'net_coverage_pp')}}}",
                f"\\newcommand{{\\OperationalRegretRangeRpOnly}}{{{fmt_num(rp_only_regret)}}}",
                f"\\newcommand{{\\OperationalRegretRangePointOnly}}{{{fmt_num(point_only_regret)}}}",
                f"\\newcommand{{\\OperationalRegretRangeDiffCi}}{{{regret_diff_ci}}}",
                f"\\newcommand{{\\OperationalThetaWidthDiffCi}}{{{theta_diff_ci}}}",
                f"\\newcommand{{\\OperationalGrossDisagreementPct}}{{{gross_pct:.2f}}}",
                f"\\newcommand{{\\OperationalPointDistRpOnly}}{{{fmt_num(point_dist_rp_only)}}}",
                f"\\newcommand{{\\OperationalRpDistPointOnly}}{{{fmt_num(rp_dist_point_only)}}}",
                f"\\newcommand{{\\OperationalBothDist}}{{{fmt_num(both_dist)}}}",
                f"\\newcommand{{\\OperationalContestedDistGapCi}}{{{contested_gap_ci}}}",
                f"\\newcommand{{\\OperationalDisagreementParagraph}}{{{paragraph}}}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(category.to_string(index=False))
    print(ci.to_string(index=False))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_category_csv}")
    print(f"Wrote {args.out_bootstrap_csv}")
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
