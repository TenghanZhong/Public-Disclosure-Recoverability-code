"""Estimate boundary-signal regressions with a common bound constructor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from out_of_proxy_extremeness_incremental import (  # noqa: E402
    cluster_extremeness_metrics,
    compute_boundary_metrics,
)
from out_of_proxy_predictive_validity import (  # noqa: E402
    derivative_leverage_mask,
    fit_residualized_ols,
)
from boundary_inference_checks import (  # noqa: E402
    add_package,
    as_numeric,
    build_regression_panel,
    lead_lag_actions,
    load_action_columns,
    wild_cluster_bootstrap_p,
    zscore_on_mask,
)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_ROOT = DEFAULT_REPO_ROOT
DEFAULT_SELECTED_ROOT = DEFAULT_REPO_ROOT / "outputs" / "full_pipeline"
DEFAULT_CONFIG = DEFAULT_REPO_ROOT / "configs" / "main_32gb.yaml"
DEFAULT_OUT_DIR = DEFAULT_REPO_ROOT / "tables" / "boundary_regressions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_root", default=str(DEFAULT_SELECTED_ROOT))
    parser.add_argument("--package_root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--bound_mode", default="exogenous")
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--iqr_floor", type=float, default=1e-8)
    parser.add_argument("--demean_iter", type=int, default=12)
    parser.add_argument("--wild_boot", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def _cluster_controls(data: pd.DataFrame, controls: list[str]) -> list[str]:
    return [c for c in controls if c in data.columns]


def _run_focus(
    data: pd.DataFrame,
    outcome: str,
    focus: str,
    controls: list[str],
    entity_col: str,
    date_col: str,
    demean_iter: int,
    winsorize: bool = False,
) -> dict:
    work = data.copy()
    work["out_of_proxy"] = pd.to_numeric(work[focus], errors="coerce")
    return fit_residualized_ols(
        work,
        outcome,
        controls=_cluster_controls(work, [c for c in controls if c != focus]),
        fe_cols=[date_col, "mandate_cluster"],
        entity_col=entity_col,
        date_col=date_col,
        n_iter=demean_iter,
        winsorize=winsorize,
    )


def _add_group_drift(
    reg: pd.DataFrame,
    panel: pd.DataFrame,
    cols: list[str],
    prefix: str,
    entity_col: str,
    date_col: str,
) -> None:
    actions = as_numeric(panel, cols)
    lag, lead = lead_lag_actions(panel, actions, entity_col, date_col)
    cur = np.abs(actions.to_numpy(dtype=float) - lag).sum(axis=1)
    nxt = np.abs(lead - actions.to_numpy(dtype=float)).sum(axis=1)
    cur[~np.isfinite(lag).all(axis=1)] = np.nan
    nxt[~np.isfinite(lead).all(axis=1)] = np.nan
    reg[f"{prefix}_drift_l1"] = cur
    reg[f"next_{prefix}_drift_l1"] = nxt
    order = panel.sort_values([entity_col, date_col]).index
    tmp = reg.loc[order, [entity_col, f"{prefix}_drift_l1"]].copy()
    reg.loc[order, f"lag_{prefix}_drift_l1"] = (
        tmp.groupby(entity_col, sort=False)[f"{prefix}_drift_l1"].shift(1).to_numpy()
    )


def _main_design(
    data: pd.DataFrame,
    outcome: str,
    focus: str,
    controls: list[str],
    entity_col: str,
    date_col: str,
    demean_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from boundary_inference_checks import regression_design

    design = data.copy()
    design["out_of_proxy"] = pd.to_numeric(design[focus], errors="coerce")
    y_tilde, x_tilde, date_codes, _x_cols, _used = regression_design(
        design,
        outcome,
        "out_of_proxy",
        _cluster_controls(design, [c for c in controls if c != focus]),
        entity_col,
        date_col,
        demean_iter,
    )
    return y_tilde, x_tilde, date_codes


def _summarize_row(
    label: str,
    regressor_label: str,
    outcome_label: str,
    result: dict,
    family: str,
) -> dict:
    return {
        "specification": label,
        "regressor_outcome": regressor_label,
        "outcome": outcome_label,
        "n": int(result.get("n", 0)),
        "coef": float(result.get("beta", np.nan)),
        "two_way_cluster_t": float(result.get("two_way_cluster_t", np.nan)),
        "two_way_cluster_p": float(result.get("two_way_cluster_p", np.nan)),
        "series_cluster_t": float(result.get("series_cluster_t", np.nan)),
        "family": family,
        "controls": result.get("controls", ""),
    }


def main() -> None:
    args = parse_args()
    selected_root = Path(args.selected_root)
    package_root = Path(args.package_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_mandate_bounds, read_yaml = add_package(package_root)
    cfg = read_yaml(Path(args.config))
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    date_col = cfg.get("keys", {}).get("date_col", "report_date")
    action_meta = load_action_columns(selected_root)
    action_cols = action_meta["action_cols"]
    asset_cols = action_meta["asset_action_cols"]
    overlay_cols = action_meta["overlay_action_cols"]

    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    sorted_panel = panel.sort_values([entity_col, date_col]).index.to_numpy()
    sorted_ok = bool(np.array_equal(sorted_panel, np.arange(len(panel))))
    bounds = fit_mandate_bounds(
        panel,
        action_cols,
        asset_cols,
        overlay_cols,
        entity_col,
        cfg,
        mode=args.bound_mode,
    )
    actions = as_numeric(panel, action_cols).to_numpy(dtype=float)
    gap = np.maximum(bounds.lower - actions, 0.0) + np.maximum(actions - bounds.upper, 0.0)

    reg, meta = build_regression_panel(
        panel,
        action_cols,
        bounds.lower,
        bounds.upper,
        entity_col,
        date_col,
        args.tol,
        args.iqr_floor,
    )
    reg["row_pos"] = np.arange(len(panel))
    _add_group_drift(reg, panel, asset_cols, "asset_block", entity_col, date_col)
    _add_group_drift(reg, panel, overlay_cols, "overlay_block", entity_col, date_col)

    boundary = compute_boundary_metrics(actions, bounds.lower, bounds.upper, tol=args.tol)
    extremeness = cluster_extremeness_metrics(panel, action_cols, cluster_col="mandate_cluster")
    for frame in [boundary, extremeness]:
        for col in frame.columns:
            reg[col] = frame[col].to_numpy()
    # Use one binary exception definition across specifications.
    reg["exception"] = (gap.sum(axis=1) > args.tol).astype(int)
    reg["outside_proxy_distance_l1"] = gap.sum(axis=1)

    for col in ["derivative_pct", "allows_or_discusses_derivatives", "leverage"]:
        if col in panel.columns:
            reg[col] = pd.to_numeric(panel[col], errors="coerce")

    split_mask = reg["split"].astype(str).eq(args.split)
    scaled_col = zscore_on_mask(reg, "outside_proxy_distance_iqr_mean", split_mask)
    test = reg.loc[split_mask].copy()

    common = [
        "style_drift_l1",
        "lag_style_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]
    boundary_controls = common + [
        "cluster_center_l1",
        "cluster_z_abs_mean",
        "normalized_distance_to_nearest_bound",
        "min_normalized_signed_margin",
    ]
    generic_distance_controls = common + [
        "cluster_center_l1",
        "cluster_z_abs_mean",
        "asset_hhi",
        "asset_max_weight",
        "normalized_distance_to_nearest_bound",
        "min_normalized_signed_margin",
    ]
    inside_controls = [
        "current_nonviolating_drift_l1",
        "lag_nonviolating_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]
    asset_controls = [
        "asset_block_drift_l1",
        "lag_asset_block_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]
    overlay_controls = [
        "overlay_block_drift_l1",
        "lag_overlay_block_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]

    row_specs = [
        (
            "Preferred exception signal",
            "Exception $\\rightarrow$ next drift",
            "next_style_drift_l1",
            "exception",
            common,
            "main",
            False,
        ),
        (
            "Boundary-proximity controls",
            "Exception $\\rightarrow$ next drift",
            "next_style_drift_l1",
            "exception",
            boundary_controls,
            "extremeness",
            False,
        ),
        (
            "Winsorized outcome and controls (1--99%)",
            "Exception $\\rightarrow$ next drift",
            "next_style_drift_l1",
            "exception",
            common,
            "robustness",
            True,
        ),
        (
            "Outside-proxy distance",
            "Distance outside proxy $\\rightarrow$ next drift",
            "next_style_drift_l1",
            "outside_proxy_distance_l1",
            common,
            "distance",
            False,
        ),
        (
            "Distance with extremeness controls",
            "Distance outside proxy $\\rightarrow$ next drift",
            "next_style_drift_l1",
            "outside_proxy_distance_l1",
            generic_distance_controls,
            "distance",
            False,
        ),
        (
            "Distance scaled by training variation",
            "Scaled distance $\\rightarrow$ next drift",
            "next_style_drift_l1",
            scaled_col,
            common,
            "distance",
            False,
        ),
        (
            "Drift on coordinates still inside proxy",
            "Scaled distance $\\rightarrow$ drift inside proxy",
            "next_nonviolating_drift_l1",
            scaled_col,
            inside_controls,
            "mechanical",
            False,
        ),
        (
            "Binary with distance controls",
            "Exception $\\rightarrow$ next drift",
            "next_style_drift_l1",
            "exception",
            common + ["outside_proxy_distance_l1"],
            "distance",
            False,
        ),
        (
            "Asset-block drift outcome",
            "Exception $\\rightarrow$ next asset drift",
            "next_asset_block_drift_l1",
            "exception",
            asset_controls,
            "grouped_outcome",
            False,
        ),
        (
            "Overlay-block drift outcome",
            "Exception $\\rightarrow$ next overlay drift",
            "next_overlay_block_drift_l1",
            "exception",
            overlay_controls,
            "grouped_outcome",
            False,
        ),
    ]

    rows: list[dict] = []
    full_results: list[dict] = []
    for spec, label, outcome, focus, controls, family, winsorize in row_specs:
        res = _run_focus(
            test,
            outcome,
            focus,
            controls,
            entity_col,
            date_col,
            args.demean_iter,
            winsorize=winsorize,
        )
        full = dict(res)
        full.update({"specification": spec, "focus": focus, "family": family})
        full_results.append(full)
        rows.append(_summarize_row(spec, label, outcome, res, family))

    drop_mask = derivative_leverage_mask(test)
    drop_test = test.loc[~drop_mask].copy()
    drop_res = _run_focus(
        drop_test,
        "next_style_drift_l1",
        "exception",
        common,
        entity_col,
        date_col,
        args.demean_iter,
    )
    full = dict(drop_res)
    full.update(
        {
            "specification": "Drop derivative/leverage proxy funds",
            "focus": "exception",
            "family": "sample_restriction",
        }
    )
    full_results.append(full)
    rows.append(
        _summarize_row(
            "Drop derivative/leverage proxy funds",
            "Exception $\\rightarrow$ next drift",
            "next_style_drift_l1",
            drop_res,
            "sample_restriction",
        )
    )

    # Coordinate-exclusion diagnostics.
    coord_rows = []
    for j, col in enumerate(action_cols):
        coord_flag = (np.delete(gap, j, axis=1).sum(axis=1) > args.tol).astype(int)
        coord_data = test.copy()
        coord_data["exception_excluding_coordinate"] = coord_flag[coord_data["row_pos"].to_numpy(dtype=int)]
        res = _run_focus(
            coord_data,
            "next_style_drift_l1",
            "exception_excluding_coordinate",
            common,
            entity_col,
            date_col,
            args.demean_iter,
        )
        coord_rows.append(
            {
                "excluded_coordinate": col,
                "n": int(res.get("n", 0)),
                "coef": float(res.get("beta", np.nan)),
                "two_way_cluster_t": float(res.get("two_way_cluster_t", np.nan)),
                "two_way_cluster_p": float(res.get("two_way_cluster_p", np.nan)),
            }
        )

    # Wild-cluster checks for selected specifications.
    wild_rows = []
    wild_specs = [
        ("Preferred exception signal", "next_style_drift_l1", "exception", common),
        ("Distance scaled by training variation", "next_style_drift_l1", scaled_col, common),
        ("Drift on coordinates still inside proxy", "next_nonviolating_drift_l1", scaled_col, inside_controls),
    ]
    for idx, (spec, outcome, focus, controls) in enumerate(wild_specs):
        y_tilde, x_tilde, date_codes = _main_design(
            test,
            outcome,
            focus,
            controls,
            entity_col,
            date_col,
            args.demean_iter,
        )
        wild = wild_cluster_bootstrap_p(
            y_tilde,
            x_tilde,
            date_codes,
            args.wild_boot,
            args.seed + idx,
        )
        wild["specification"] = spec
        wild_rows.append(wild)

    reported = pd.DataFrame(rows)
    reported.to_csv(out_dir / "reported_boundary_regressions.csv", index=False)
    pd.DataFrame(full_results).to_csv(out_dir / "full_boundary_regressions.csv", index=False)
    pd.DataFrame(coord_rows).to_csv(out_dir / "coordinate_exclusion.csv", index=False)
    pd.DataFrame(wild_rows).to_csv(out_dir / "wild_bootstrap.csv", index=False)

    manifest = {
        "selected_root": str(selected_root),
        "package_root": str(package_root),
        "config": str(Path(args.config)),
        "bound_mode": args.bound_mode,
        "bounds_constructor": "mandate_setirl.bounds.fit_mandate_bounds",
        "split": args.split,
        "panel_sorted_by_series_date": sorted_ok,
        "n_rows_split": int(len(test)),
        "regression_sample_n_preferred": int(reported.loc[reported["specification"].eq("Preferred exception signal"), "n"].iloc[0]),
        "outside_proxy_share_split": float(test["exception"].mean()),
        "n_series_split": int(test[entity_col].nunique()),
        "n_report_dates_split": int(test[date_col].nunique()),
        "standardized_distance": "mean coordinate boundary gap divided by training-split IQR, then z-scored on the test split",
        "coordinate_exclusion": "recompute the binary exception after omitting one coordinate from the boundary gap sum",
        "wild_boot_draws": args.wild_boot,
        "seed": args.seed,
    }
    with open(out_dir / "boundary_regressions_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(reported.to_string(index=False))
    print(pd.DataFrame(wild_rows)[["specification", "date_cluster_t", "wild_cluster_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
