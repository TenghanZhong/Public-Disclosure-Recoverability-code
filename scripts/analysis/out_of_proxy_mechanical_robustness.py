"""Mechanical robustness checks for mandate-proxy exception signals.

Runs holdings-count exclusion, scaled-distance, and nonviolating-coordinate
drift diagnostics.
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

from out_of_proxy_extremeness_incremental import (  # noqa: E402
    cluster_extremeness_metrics,
    compute_boundary_metrics,
)
from out_of_proxy_predictive_validity import (  # noqa: E402
    DEFAULT_PACKAGE_ROOT,
    add_package,
    as_numeric_frame,
    build_outcome_panel,
    fit_residualized_ols,
    load_action_columns,
)


DEFAULT_SELECTED_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "full_pipeline"
EXPECTED_BASELINE_BETA = 0.052630930230068616
EXPECTED_BASELINE_T = 3.9996535833089992
BASELINE_TOL = 5e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_root", default=str(DEFAULT_SELECTED_ROOT))
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parents[2] / "tables" / "mechanical_robustness"),
    )
    parser.add_argument("--package_root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--config", default=None)
    parser.add_argument("--bound_mode", default="exogenous")
    parser.add_argument("--split", default="test")
    parser.add_argument("--demean_iter", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1e-7)
    return parser.parse_args()


def training_coordinate_scale(
    actions: np.ndarray,
    train_mask: np.ndarray,
    min_scale: float = 1e-8,
) -> np.ndarray:
    """Return training-only coordinate standard deviations with safe fallback."""

    actions = np.asarray(actions, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)
    if actions.ndim != 2:
        raise ValueError("actions must be a 2D array")
    if train_mask.shape[0] != actions.shape[0]:
        raise ValueError("train_mask length must match actions rows")
    if not train_mask.any():
        raise ValueError("At least one training row is required for scale estimation")

    scale = np.nanstd(actions[train_mask], axis=0, ddof=0)
    return np.where(np.isfinite(scale) & (scale > min_scale), scale, 1.0)


def boundary_metrics_subset(
    actions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    include_idx: Iterable[int],
    scale: np.ndarray | None = None,
    tol: float = 1e-7,
) -> pd.DataFrame:
    """Compute exception metrics after selecting a subset of coordinates."""

    actions = np.asarray(actions, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    idx = np.asarray(list(include_idx), dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("include_idx must contain at least one coordinate")
    if actions.shape != lower.shape or actions.shape != upper.shape:
        raise ValueError("actions, lower, and upper must have the same shape")

    below = np.maximum(lower[:, idx] - actions[:, idx], 0.0)
    above = np.maximum(actions[:, idx] - upper[:, idx], 0.0)
    gap = below + above
    severity = gap.sum(axis=1)
    out = pd.DataFrame(
        {
            "out_of_proxy": (severity > tol).astype(int),
            "severity_l1": severity,
            "max_violation": gap.max(axis=1),
        }
    )
    if scale is not None:
        scale = np.asarray(scale, dtype=float)
        if scale.ndim != 1 or scale.shape[0] != actions.shape[1]:
            raise ValueError("scale must be one value per action coordinate")
        safe_scale = np.where(np.isfinite(scale[idx]) & (scale[idx] > 1e-8), scale[idx], 1.0)
        out["severity_standardized"] = (gap / safe_scale).sum(axis=1)
        out["max_standardized_violation"] = (gap / safe_scale).max(axis=1)
    return out


def lead_action_matrix(
    panel: pd.DataFrame,
    actions: np.ndarray,
    entity_col: str,
    date_col: str = "report_date",
) -> np.ndarray:
    """Return next-report action matrix aligned to the original row order."""

    work = pd.DataFrame({"_orig_pos": np.arange(len(panel)), entity_col: panel[entity_col].to_numpy()})
    work[date_col] = panel[date_col].to_numpy() if date_col in panel.columns else np.arange(len(panel))
    action_df = pd.DataFrame(actions, index=panel.index)
    sorted_idx = work.sort_values([entity_col, date_col])["_orig_pos"].to_numpy()
    lead_sorted = action_df.iloc[sorted_idx].groupby(work.iloc[sorted_idx][entity_col].to_numpy(), sort=False).shift(-1)
    lead = np.full_like(actions, np.nan, dtype=float)
    lead[sorted_idx, :] = lead_sorted.to_numpy(dtype=float)
    return lead


def lag_action_matrix(
    panel: pd.DataFrame,
    actions: np.ndarray,
    entity_col: str,
    date_col: str = "report_date",
) -> np.ndarray:
    """Return previous-report action matrix aligned to the original row order."""

    work = pd.DataFrame({"_orig_pos": np.arange(len(panel)), entity_col: panel[entity_col].to_numpy()})
    work[date_col] = panel[date_col].to_numpy() if date_col in panel.columns else np.arange(len(panel))
    action_df = pd.DataFrame(actions, index=panel.index)
    sorted_idx = work.sort_values([entity_col, date_col])["_orig_pos"].to_numpy()
    lag_sorted = action_df.iloc[sorted_idx].groupby(work.iloc[sorted_idx][entity_col].to_numpy(), sort=False).shift(1)
    lag = np.full_like(actions, np.nan, dtype=float)
    lag[sorted_idx, :] = lag_sorted.to_numpy(dtype=float)
    return lag


def nonviolating_next_drift(
    actions: np.ndarray,
    lead_actions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tol: float = 1e-7,
) -> np.ndarray:
    """L1 next drift on coordinates not currently outside the proxy."""

    actions = np.asarray(actions, dtype=float)
    lead_actions = np.asarray(lead_actions, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if actions.shape != lead_actions.shape or actions.shape != lower.shape or actions.shape != upper.shape:
        raise ValueError("all inputs must share shape")

    violation = ((lower - actions) > tol) | ((actions - upper) > tol)
    valid_lead = np.isfinite(lead_actions).all(axis=1)
    drift = np.abs(lead_actions - actions)
    drift[violation] = 0.0
    out = drift.sum(axis=1)
    out[~valid_lead] = np.nan
    return out


def nonviolating_current_drift(
    actions: np.ndarray,
    lag_actions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tol: float = 1e-7,
) -> np.ndarray:
    """Current drift from lag to current on coordinates not currently outside the proxy."""

    valid_lag = np.isfinite(lag_actions).all(axis=1)
    violation = ((lower - actions) > tol) | ((actions - upper) > tol)
    drift = np.abs(actions - lag_actions)
    drift[violation] = 0.0
    out = drift.sum(axis=1)
    out[~valid_lag] = np.nan
    return out


def run_focus_regression(
    data: pd.DataFrame,
    outcome: str,
    focus_col: str,
    controls: list[str],
    entity_col: str,
    date_col: str,
    demean_iter: int,
) -> dict:
    """Run the project-standard FE regression using focus_col as regressor."""

    work = data.copy()
    work["out_of_proxy"] = pd.to_numeric(work[focus_col], errors="coerce")
    controls = [c for c in controls if c in work.columns and c != focus_col]
    res = fit_residualized_ols(
        work,
        outcome,
        controls=controls,
        fe_cols=[date_col, "mandate_cluster"],
        entity_col=entity_col,
        date_col=date_col,
        n_iter=demean_iter,
    )
    res["out_of_proxy_n"] = int((pd.to_numeric(work[focus_col], errors="coerce") > 1e-7).sum())
    res["regressor"] = focus_col
    res["controls"] = ",".join(controls)
    return res


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
    holdings_col = "n_holdings_scaled"
    if holdings_col not in action_cols:
        raise ValueError(f"{holdings_col} not found in action columns")

    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode=args.bound_mode)
    actions = as_numeric_frame(panel, action_cols).to_numpy(dtype=float)
    canonical_out = action_violations(actions, bounds.lower, bounds.upper, tol=args.tol).astype(int)

    all_idx = list(range(len(action_cols)))
    no_holdings_idx = [j for j, c in enumerate(action_cols) if c != holdings_col]
    scale = training_coordinate_scale(actions, panel["split"].astype(str).eq("train").to_numpy())
    all_metrics = boundary_metrics_subset(actions, bounds.lower, bounds.upper, all_idx, scale=scale, tol=args.tol)
    if not np.array_equal(all_metrics["out_of_proxy"].to_numpy(dtype=int), canonical_out):
        mismatch = int(np.abs(all_metrics["out_of_proxy"].to_numpy(dtype=int) - canonical_out).sum())
        raise RuntimeError(f"All-coordinate exception mismatch against canonical flag: {mismatch} rows")

    no_holdings_metrics = boundary_metrics_subset(
        actions,
        bounds.lower,
        bounds.upper,
        no_holdings_idx,
        scale=scale,
        tol=args.tol,
    )
    base_boundary = compute_boundary_metrics(actions, bounds.lower, bounds.upper, tol=args.tol)
    extremeness = cluster_extremeness_metrics(panel, action_cols, cluster_col="mandate_cluster")
    lead_actions = lead_action_matrix(panel, actions, entity_col, date_col)
    lag_actions = lag_action_matrix(panel, actions, entity_col, date_col)

    data = build_outcome_panel(panel, action_cols, entity_col)
    data["row_pos"] = np.arange(len(panel))
    data["exception_all"] = all_metrics["out_of_proxy"].to_numpy()
    data["severity_all_l1"] = all_metrics["severity_l1"].to_numpy()
    data["severity_all_std"] = all_metrics["severity_standardized"].to_numpy()
    data["exception_no_holdings"] = no_holdings_metrics["out_of_proxy"].to_numpy()
    data["severity_no_holdings_l1"] = no_holdings_metrics["severity_l1"].to_numpy()
    data["severity_no_holdings_std"] = no_holdings_metrics["severity_standardized"].to_numpy()
    data["next_nonviolating_drift_l1"] = nonviolating_next_drift(
        actions,
        lead_actions,
        bounds.lower,
        bounds.upper,
        tol=args.tol,
    )
    data["current_nonviolating_drift_l1"] = nonviolating_current_drift(
        actions,
        lag_actions,
        bounds.lower,
        bounds.upper,
        tol=args.tol,
    )
    data["lag_nonviolating_drift_l1"] = data.groupby(entity_col, sort=False)["current_nonviolating_drift_l1"].shift(1)

    for c in base_boundary.columns:
        data[c] = base_boundary[c].to_numpy()
    for c in extremeness.columns:
        data[c] = extremeness[c].to_numpy()

    if str(args.split).lower() != "all":
        data = data[data["split"].astype(str).eq(args.split)].copy()

    common_controls = [
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
        c
        for c in [
            "cluster_center_l1",
            "cluster_z_abs_mean",
            "asset_hhi",
            "asset_max_weight",
            "normalized_distance_to_nearest_bound",
            "min_normalized_signed_margin",
        ]
        if c in data.columns
    ]
    nonviol_controls = [
        c
        for c in [
            "current_nonviolating_drift_l1",
            "lag_nonviolating_drift_l1",
            "log_net_assets",
            "net_flow_3m_scaled",
            "lag_net_flow_3m_scaled",
        ]
        if c in data.columns
    ]

    specs = [
        ("baseline_binary_all", "next_style_drift_l1", "exception_all", common_controls),
        ("no_holdings_binary", "next_style_drift_l1", "exception_no_holdings", common_controls),
        ("severity_all_l1", "next_style_drift_l1", "severity_all_l1", common_controls + extremeness_controls),
        (
            "severity_no_holdings_l1",
            "next_style_drift_l1",
            "severity_no_holdings_l1",
            common_controls + extremeness_controls,
        ),
        ("severity_all_standardized", "next_style_drift_l1", "severity_all_std", common_controls + extremeness_controls),
        (
            "severity_no_holdings_standardized",
            "next_style_drift_l1",
            "severity_no_holdings_std",
            common_controls + extremeness_controls,
        ),
        (
            "nonviolating_drift_binary_all",
            "next_nonviolating_drift_l1",
            "exception_all",
            nonviol_controls,
        ),
        (
            "nonviolating_drift_severity_l1",
            "next_nonviolating_drift_l1",
            "severity_all_l1",
            nonviol_controls + extremeness_controls,
        ),
        (
            "nonviolating_drift_severity_standardized",
            "next_nonviolating_drift_l1",
            "severity_all_std",
            nonviol_controls + extremeness_controls,
        ),
    ]

    rows = []
    for spec, outcome, focus, controls in specs:
        res = run_focus_regression(data, outcome, focus, controls, entity_col, date_col, args.demean_iter)
        res["spec"] = spec
        rows.append(res)

    regs = pd.DataFrame(rows)
    baseline = regs.loc[regs["spec"].eq("baseline_binary_all")].iloc[0]
    beta_diff = abs(float(baseline["beta"]) - EXPECTED_BASELINE_BETA)
    t_diff = abs(float(baseline["two_way_cluster_t"]) - EXPECTED_BASELINE_T)
    if beta_diff > BASELINE_TOL or t_diff > BASELINE_TOL:
        raise RuntimeError(
            "Baseline row differs from the reference risk-signal estimate: "
            f"beta={float(baseline['beta']):.12f}, t={float(baseline['two_way_cluster_t']):.12f}; "
            f"expected beta={EXPECTED_BASELINE_BETA:.12f}, t={EXPECTED_BASELINE_T:.12f}. "
            "Check selected_root, bounds, sample filter, tolerance, and clustering implementation."
        )
    ordered_cols = ["spec", "regressor", "outcome"] + [c for c in regs.columns if c not in {"spec", "regressor", "outcome"}]
    regs = regs[ordered_cols]
    regs.to_csv(out_dir / "mechanical_robustness_regressions.csv", index=False)

    metric_cols = [
        "exception_all",
        "severity_all_l1",
        "severity_all_std",
        "exception_no_holdings",
        "severity_no_holdings_l1",
        "severity_no_holdings_std",
        "next_nonviolating_drift_l1",
    ]
    metric_summary = []
    for col in metric_cols:
        s = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        metric_summary.append(
            {
                "metric": col,
                "n": int(len(s)),
                "mean": float(s.mean()),
                "p50": float(s.quantile(0.50)),
                "p90": float(s.quantile(0.90)),
                "p99": float(s.quantile(0.99)),
                "positive_rate": float((s > args.tol).mean()),
            }
        )
    pd.DataFrame(metric_summary).to_csv(out_dir / "mechanical_metric_summary.csv", index=False)

    gap = np.maximum(bounds.lower - actions, 0.0) + np.maximum(actions - bounds.upper, 0.0)
    test_mask = data["row_pos"].to_numpy(dtype=int)
    driver_rows = []
    for j, col in enumerate(action_cols):
        viol = gap[test_mask, j] > args.tol
        driver_rows.append(
            {
                "coordinate": col,
                "violation_count": int(viol.sum()),
                "share_of_exception_rows": float(viol.sum() / max((gap[test_mask].sum(axis=1) > args.tol).sum(), 1)),
                "mean_positive_gap": float(gap[test_mask, j][viol].mean()) if viol.any() else 0.0,
            }
        )
    pd.DataFrame(driver_rows).sort_values("violation_count", ascending=False).to_csv(
        out_dir / "coordinate_violation_drivers.csv",
        index=False,
    )

    reported = regs[
        [
            "spec",
            "regressor",
            "outcome",
            "n",
            "out_of_proxy_n",
            "beta",
            "two_way_cluster_se",
            "two_way_cluster_t",
            "two_way_cluster_p",
            "controls",
        ]
    ].copy()
    reported.to_csv(out_dir / "reported_mechanical_robustness.csv", index=False)

    manifest = {
        "selected_root": str(selected_root),
        "package_root": str(package_root),
        "config": str(config_path),
        "bound_mode": args.bound_mode,
        "split": args.split,
        "n_rows_split": int(len(data)),
        "n_series": int(data[entity_col].nunique()),
        "n_report_dates": int(data[date_col].nunique()),
        "holdings_coordinate": holdings_col,
        "training_scale_source": "coordinate standard deviation on training split only",
        "standard_errors": ["series_id", "report_date"],
        "fixed_effects": [date_col, "mandate_cluster"],
        "notes": [
            "No-holdings robustness ignores n_holdings_scaled in exception and severity construction but keeps the selected mandate-proxy bounds fixed.",
            "Standardized severity divides coordinate gaps by training-only action-coordinate standard deviations.",
            "Nonviolating drift excludes coordinates currently outside the mandate-proxy bounds before summing next-report L1 drift.",
            "Coefficients are predictive associations, not causal or legal-validity estimates.",
        ],
    }
    with open(out_dir / "mechanical_robustness_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(reported[["spec", "regressor", "outcome", "n", "beta", "two_way_cluster_t", "two_way_cluster_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
