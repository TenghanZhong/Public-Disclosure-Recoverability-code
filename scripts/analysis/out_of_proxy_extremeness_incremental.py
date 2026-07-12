"""Incremental extremeness controls for the mandate-proxy exception signal.

This diagnostic asks whether the out-of-proxy indicator predicts next-report
style drift after controlling for current allocation extremeness and boundary
proximity.  Extremeness controls are computed from the training split only, so
test rows do not help define their own reference distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from out_of_proxy_predictive_validity import (
    DEFAULT_PACKAGE_ROOT,
    add_package,
    as_numeric_frame,
    build_outcome_panel,
    fit_residualized_ols,
    load_action_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected_root",
        default=str(DEFAULT_PACKAGE_ROOT / "outputs" / "full_pipeline"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(DEFAULT_PACKAGE_ROOT / "tables" / "out_of_proxy_extremeness_incremental"),
    )
    parser.add_argument("--package_root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--config", default=None)
    parser.add_argument("--bound_mode", default="exogenous")
    parser.add_argument("--split", default="test")
    parser.add_argument("--demean_iter", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1e-7)
    return parser.parse_args()


def _safe_width(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    width = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    return np.where(np.isfinite(width) & (width > 1e-10), width, 1.0)


def compute_boundary_metrics(
    actions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tol: float = 1e-7,
) -> pd.DataFrame:
    """Compute exception and boundary-distance controls for each row."""

    actions = np.asarray(actions, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = _safe_width(lower, upper)

    below = np.maximum(lower - actions, 0.0)
    above = np.maximum(actions - upper, 0.0)
    gap = below + above
    severity = gap.sum(axis=1)

    signed_margin = np.minimum(actions - lower, upper - actions)
    normalized_gap = gap / width
    normalized_margin = signed_margin / width
    distance_to_nearest_bound = np.minimum(np.abs(actions - lower), np.abs(upper - actions)).min(axis=1)
    normalized_distance_to_nearest_bound = (
        np.minimum(np.abs(actions - lower), np.abs(upper - actions)) / width
    ).min(axis=1)

    return pd.DataFrame(
        {
            "out_of_proxy": (severity > tol).astype(int),
            "exception_severity_l1": severity,
            "max_normalized_violation": normalized_gap.max(axis=1),
            "min_signed_margin_to_bound": signed_margin.min(axis=1),
            "min_normalized_signed_margin": normalized_margin.min(axis=1),
            "distance_to_nearest_bound": distance_to_nearest_bound,
            "normalized_distance_to_nearest_bound": normalized_distance_to_nearest_bound,
        }
    )


def training_cluster_reference(
    panel: pd.DataFrame,
    action_cols: list[str],
    cluster_col: str,
    split_col: str = "split",
    train_label: str = "train",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return training-only cluster means and standard deviations."""

    actions = as_numeric_frame(panel, action_cols)
    train = panel[split_col].astype(str).eq(train_label)
    if not train.any():
        raise ValueError("No training rows found for extremeness reference distribution.")

    train_actions = actions.loc[train]
    train_clusters = panel.loc[train, cluster_col]
    means = train_actions.groupby(train_clusters, sort=False).mean()
    stds = train_actions.groupby(train_clusters, sort=False).std(ddof=0).replace(0.0, np.nan)
    global_mean = train_actions.mean()
    global_std = train_actions.std(ddof=0).replace(0.0, np.nan)
    return means, stds, global_mean, global_std


def cluster_extremeness_metrics(
    panel: pd.DataFrame,
    action_cols: list[str],
    cluster_col: str = "mandate_cluster",
) -> pd.DataFrame:
    """Compute current allocation extremeness relative to training clusters."""

    actions = as_numeric_frame(panel, action_cols).to_numpy(dtype=float)
    means, stds, global_mean, global_std = training_cluster_reference(panel, action_cols, cluster_col)
    cluster_values = panel[cluster_col].to_numpy()

    mean_mat = np.empty_like(actions, dtype=float)
    std_mat = np.empty_like(actions, dtype=float)
    global_mean_arr = global_mean.to_numpy(dtype=float)
    global_std_arr = global_std.to_numpy(dtype=float)
    global_std_arr = np.where(np.isfinite(global_std_arr) & (global_std_arr > 1e-10), global_std_arr, 1.0)

    mean_lookup = {k: v.to_numpy(dtype=float) for k, v in means.iterrows()}
    std_lookup = {
        k: np.where(np.isfinite(v.to_numpy(dtype=float)) & (v.to_numpy(dtype=float) > 1e-10), v.to_numpy(dtype=float), global_std_arr)
        for k, v in stds.iterrows()
    }

    for r, cluster in enumerate(cluster_values):
        mean_vec = mean_lookup.get(cluster, global_mean_arr)
        std_vec = std_lookup.get(cluster, global_std_arr)
        mean_mat[r, :] = mean_vec
        std_mat[r, :] = std_vec

    centered = actions - mean_mat
    z = centered / std_mat
    asset_cols = [c for c in action_cols if c.startswith("asset_")]
    asset_idx = [action_cols.index(c) for c in asset_cols]
    overlay_idx = [j for j, c in enumerate(action_cols) if c not in asset_cols]

    out = pd.DataFrame(
        {
            "cluster_center_l1": np.abs(centered).sum(axis=1),
            "cluster_center_l2": np.sqrt((centered**2).sum(axis=1)),
            "cluster_z_abs_mean": np.nanmean(np.abs(z), axis=1),
            "cluster_z_l2": np.sqrt(np.nansum(z**2, axis=1)),
        }
    )
    if asset_idx:
        out["asset_cluster_center_l1"] = np.abs(centered[:, asset_idx]).sum(axis=1)
        out["asset_hhi"] = (actions[:, asset_idx] ** 2).sum(axis=1)
        out["asset_max_weight"] = actions[:, asset_idx].max(axis=1)
    if overlay_idx:
        out["overlay_cluster_center_l1"] = np.abs(centered[:, overlay_idx]).sum(axis=1)
    return out


def corr_summary(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    use_cols = [c for c in cols if c in df.columns]
    for c in use_cols:
        data = df[["out_of_proxy", c]].replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            continue
        rows.append(
            {
                "variable": c,
                "n": int(len(data)),
                "mean_in_proxy": float(data.loc[data["out_of_proxy"].eq(0), c].mean()),
                "mean_out_of_proxy": float(data.loc[data["out_of_proxy"].eq(1), c].mean()),
                "corr_with_out_of_proxy": float(data["out_of_proxy"].corr(data[c])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    selected_root = Path(args.selected_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    package_root = Path(args.package_root)
    config_path = Path(args.config) if args.config else package_root / "configs" / "main_32gb.yaml"

    action_violations, fit_mandate_bounds, read_yaml = add_package(package_root)
    cfg = read_yaml(config_path)
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    date_col = cfg.get("keys", {}).get("date_col", "report_date")
    ac = load_action_columns(selected_root)
    action_cols = ac["action_cols"]
    asset_cols = ac["asset_action_cols"]
    overlay_cols = ac["overlay_action_cols"]

    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode=args.bound_mode)
    actions = as_numeric_frame(panel, action_cols).to_numpy(dtype=float)
    boundary = compute_boundary_metrics(actions, bounds.lower, bounds.upper, tol=args.tol)
    # Match the local metric to the canonical violation flag.
    canonical_out = action_violations(actions, bounds.lower, bounds.upper, tol=args.tol).astype(int)
    if not np.array_equal(boundary["out_of_proxy"].to_numpy(dtype=int), canonical_out):
        mismatch = int(np.abs(boundary["out_of_proxy"].to_numpy(dtype=int) - canonical_out).sum())
        raise RuntimeError(f"Boundary metric mismatch against action_violations: {mismatch} rows.")

    extremeness = cluster_extremeness_metrics(panel, action_cols, cluster_col="mandate_cluster")
    data = build_outcome_panel(panel, action_cols, entity_col)
    data["row_pos"] = np.arange(len(panel))
    for frame in [boundary, extremeness]:
        for c in frame.columns:
            data[c] = frame[c].to_numpy()

    if str(args.split).lower() != "all":
        data = data[data["split"].astype(str).eq(args.split)].copy()

    base_controls = [
        c
        for c in [
            "style_drift_l1",
            "lag_style_drift_l1",
            "log_net_assets",
            "net_flow_3m_scaled",
            "lag_net_flow_3m_scaled",
        ]
        if c in data.columns
    ]
    extremeness_controls = [
        "cluster_center_l1",
        "cluster_z_abs_mean",
        "asset_hhi",
        "asset_max_weight",
    ]
    boundary_controls = [
        "normalized_distance_to_nearest_bound",
        "min_normalized_signed_margin",
        "exception_severity_l1",
        "max_normalized_violation",
    ]

    specs = [
        ("baseline_own_lag", base_controls),
        ("plus_cluster_center_l1", base_controls + ["cluster_center_l1"]),
        ("plus_standardized_extremeness", base_controls + ["cluster_center_l1", "cluster_z_abs_mean"]),
        ("plus_asset_concentration", base_controls + ["cluster_center_l1", "cluster_z_abs_mean", "asset_hhi", "asset_max_weight"]),
        ("plus_boundary_proximity", base_controls + ["cluster_center_l1", "cluster_z_abs_mean", "normalized_distance_to_nearest_bound", "min_normalized_signed_margin"]),
        ("plus_exception_severity", base_controls + ["cluster_center_l1", "cluster_z_abs_mean", "exception_severity_l1", "max_normalized_violation"]),
        ("all_extremeness_boundary_controls", base_controls + extremeness_controls + boundary_controls),
    ]

    rows = []
    for spec, controls in specs:
        controls = [c for c in controls if c in data.columns]
        res = fit_residualized_ols(
            data,
            "next_style_drift_l1",
            controls=controls,
            fe_cols=[date_col, "mandate_cluster"],
            entity_col=entity_col,
            date_col=date_col,
            n_iter=args.demean_iter,
        )
        res["spec"] = spec
        res["regressor"] = "binary_exception"
        rows.append(res)

    continuous_specs = [
        (
            "severity_focus_plus_extremeness",
            "exception_severity_l1",
            base_controls
            + extremeness_controls
            + ["normalized_distance_to_nearest_bound", "min_normalized_signed_margin"],
        ),
        (
            "max_normalized_violation_focus_plus_extremeness",
            "max_normalized_violation",
            base_controls
            + extremeness_controls
            + ["normalized_distance_to_nearest_bound", "min_normalized_signed_margin"],
        ),
    ]
    for spec, focus, controls in continuous_specs:
        focus_data = data.copy()
        focus_data["out_of_proxy"] = focus_data[focus]
        res = fit_residualized_ols(
            focus_data,
            "next_style_drift_l1",
            controls=[c for c in controls if c in focus_data.columns and c != focus],
            fe_cols=[date_col, "mandate_cluster"],
            entity_col=entity_col,
            date_col=date_col,
            n_iter=args.demean_iter,
        )
        res["spec"] = spec
        res["regressor"] = focus
        if "out_of_proxy_n" in res:
            res["out_of_proxy_n"] = int((focus_data[focus] > args.tol).sum())
        rows.append(res)

    regs = pd.DataFrame(rows)
    cols = ["spec"] + [c for c in regs.columns if c != "spec"]
    regs = regs[cols]
    regs.to_csv(out_dir / "out_of_proxy_extremeness_incremental_regressions.csv", index=False)

    summary_cols = [
        "cluster_center_l1",
        "cluster_z_abs_mean",
        "asset_hhi",
        "asset_max_weight",
        "normalized_distance_to_nearest_bound",
        "min_normalized_signed_margin",
        "exception_severity_l1",
        "max_normalized_violation",
    ]
    summary = corr_summary(data, summary_cols)
    summary.to_csv(out_dir / "out_of_proxy_extremeness_control_summary.csv", index=False)

    reported = regs[
        [
            "spec",
            "regressor",
            "n",
            "out_of_proxy_n",
            "beta",
            "two_way_cluster_se",
            "two_way_cluster_t",
            "two_way_cluster_p",
            "controls",
        ]
    ].copy()
    reported.to_csv(out_dir / "reported_extremeness_incremental.csv", index=False)

    manifest = {
        "selected_root": str(selected_root),
        "package_root": str(package_root),
        "config": str(config_path),
        "bound_mode": args.bound_mode,
        "split": args.split,
        "n_rows_split": int(len(data)),
        "out_of_proxy_rate": float(data["out_of_proxy"].mean()),
        "n_series": int(data[entity_col].nunique()),
        "n_report_dates": int(data[date_col].nunique()),
        "outcome": "next_style_drift_l1",
        "fixed_effects": [date_col, "mandate_cluster"],
        "clustered_standard_errors": [entity_col, date_col],
        "extremeness_controls": extremeness_controls,
        "boundary_controls": boundary_controls,
        "notes": [
            "Cluster-center extremeness controls are computed from training split actions only.",
            "Boundary metrics use the same inferred mandate-proxy bounds as the canonical out_of_proxy flag.",
            "Coefficients are predictive associations, not causal or legal-validity estimates.",
        ],
    }
    with open(out_dir / "out_of_proxy_extremeness_incremental_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(
        reported[
            ["spec", "regressor", "n", "beta", "two_way_cluster_se", "two_way_cluster_t", "two_way_cluster_p"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
