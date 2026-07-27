"""Robustness grid for the mandate-proxy exception signal.

This script extends out_of_proxy_predictive_validity.py. It tests whether the
out-of-proxy indicator and exception
severity predict next style-drift outcomes under alternative outcome
definitions, coordinate exclusions, and bound variants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from out_of_proxy_predictive_validity import (
    DEFAULT_PACKAGE_ROOT,
    add_package,
    as_numeric_frame,
    build_outcome_panel,
    derivative_leverage_mask,
    fit_residualized_ols,
    load_action_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--selected_root", default=str(repo_root / "outputs" / "full_pipeline"))
    parser.add_argument("--out_dir", default=str(repo_root / "outputs" / "risk_signal_robustness"))
    parser.add_argument("--package_root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--demean_iter", type=int, default=12)
    parser.add_argument("--winsor_low", type=float, default=0.01)
    parser.add_argument("--winsor_high", type=float, default=0.99)
    return parser.parse_args()


def compute_proxy_arrays(panel: pd.DataFrame, action_cols: list[str], bounds, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = as_numeric_frame(panel, action_cols).to_numpy(dtype=float)
    lower_gap = np.maximum(bounds.lower - actions, 0.0)
    upper_gap = np.maximum(actions - bounds.upper, 0.0)
    gap = lower_gap + upper_gap
    severity = gap.sum(axis=1)
    out = (severity > tol).astype(int)
    return out, severity, gap


def add_grouped_drift(data: pd.DataFrame, panel: pd.DataFrame, cols: list[str], prefix: str, entity_col: str, date_col: str) -> None:
    work = panel.sort_values([entity_col, date_col]).copy()
    actions = as_numeric_frame(work, cols)
    lag = actions.groupby(work[entity_col], sort=False).shift(1)
    lead = actions.groupby(work[entity_col], sort=False).shift(-1)
    current = (actions - lag).abs().sum(axis=1)
    nxt = (lead - actions).abs().sum(axis=1)
    current.loc[lag.isna().any(axis=1)] = np.nan
    nxt.loc[lead.isna().any(axis=1)] = np.nan
    tmp = pd.DataFrame(
        {
            f"{prefix}_drift_l1": current,
            f"next_{prefix}_drift_l1": nxt,
        },
        index=work.index,
    ).sort_index()
    tmp[f"lag_{prefix}_drift_l1"] = tmp[f"{prefix}_drift_l1"].groupby(panel[entity_col], sort=False).shift(1)
    data[f"{prefix}_drift_l1"] = tmp[f"{prefix}_drift_l1"]
    data[f"lag_{prefix}_drift_l1"] = tmp[f"lag_{prefix}_drift_l1"]
    data[f"next_{prefix}_drift_l1"] = tmp[f"next_{prefix}_drift_l1"]


def run_reg(df: pd.DataFrame, outcome: str, controls: list[str], label: dict, entity_col: str, date_col: str, n_iter: int) -> dict:
    res = fit_residualized_ols(
        df,
        outcome,
        controls=[c for c in controls if c in df.columns],
        fe_cols=[date_col, "mandate_cluster"],
        entity_col=entity_col,
        date_col=date_col,
        n_iter=n_iter,
    )
    res.update(label)
    return res


def main() -> None:
    args = parse_args()
    selected_root = Path(args.selected_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    package_root = Path(args.package_root)
    config_path = Path(args.config) if args.config else package_root / "configs" / "main_32gb.yaml"

    action_violations, fit_mandate_bounds, read_yaml = add_package(package_root)
    cfg_base = read_yaml(config_path)
    entity_col = cfg_base.get("keys", {}).get("entity_col", "series_id")
    date_col = cfg_base.get("keys", {}).get("date_col", "report_date")
    ac = load_action_columns(selected_root)
    action_cols = ac["action_cols"]
    asset_cols = ac["asset_action_cols"]
    overlay_cols = ac["overlay_action_cols"]

    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    data = build_outcome_panel(panel, action_cols, entity_col)
    data["row_pos"] = np.arange(len(panel))
    add_grouped_drift(data, panel, asset_cols, "asset_block", entity_col, date_col)
    add_grouped_drift(data, panel, overlay_cols, "overlay_block", entity_col, date_col)
    if str(args.split).lower() != "all":
        data = data[data["split"].astype(str).eq(args.split)].copy()
    common_controls = [c for c in ["log_net_assets", "net_flow_3m_scaled", "lag_net_flow_3m_scaled"] if c in data.columns]
    style_controls = [c for c in ["style_drift_l1", "lag_style_drift_l1"] + common_controls if c in data.columns]
    rows: list[dict] = []

    bound_variants: list[tuple[str, str, dict]] = [("main_exogenous", "exogenous", cfg_base)]
    for name, q in [("strict_peer_upper_099", 0.99), ("loose_peer_upper_0999", 0.999)]:
        cfg = json.loads(json.dumps(cfg_base))
        cfg.setdefault("bounds", {})
        cfg["bounds"]["peer_upper_q"] = q
        bound_variants.append((name, "exogenous", cfg))
    bound_variants.append(("global_bounds", "global", cfg_base))

    variant_arrays = {}
    for variant_name, mode, cfg in bound_variants:
        bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode=mode)
        out, severity, gap = compute_proxy_arrays(panel, action_cols, bounds)
        variant_arrays[variant_name] = (out, severity, gap)
        reg_data = data.copy()
        reg_data["out_of_proxy"] = out[reg_data["row_pos"].to_numpy(dtype=int)]
        rows.append(
            run_reg(
                reg_data,
                "next_style_drift_l1",
                style_controls,
                {"family": "indicator_bound_variant", "variant": variant_name, "regressor": "out_of_proxy"},
                entity_col,
                date_col,
                args.demean_iter,
            )
        )

    base_out, base_severity, base_gap = variant_arrays["main_exogenous"]
    base = data.copy()
    idx = base["row_pos"].to_numpy(dtype=int)
    base["out_of_proxy"] = base_out[idx]
    base["exception_severity"] = base_severity[idx]
    rows.append(
        {
            **fit_residualized_ols(
                base,
                "next_style_drift_l1",
                controls=style_controls,
                fe_cols=[date_col, "mandate_cluster"],
                entity_col=entity_col,
                date_col=date_col,
                n_iter=args.demean_iter,
                winsorize=True,
                winsor_low=args.winsor_low,
                winsor_high=args.winsor_high,
            ),
            "family": "winsorized_yx_1_99",
            "variant": "main_exogenous",
            "regressor": "out_of_proxy",
        }
    )

    severity_data = base.copy()
    severity_data["out_of_proxy"] = severity_data["exception_severity"]
    rows.append(
        run_reg(
            severity_data,
            "next_style_drift_l1",
            style_controls,
            {"family": "severity", "variant": "main_exogenous", "regressor": "exception_severity"},
            entity_col,
            date_col,
            args.demean_iter,
        )
    )

    for outcome, controls in [
        ("next_asset_block_drift_l1", [c for c in ["asset_block_drift_l1", "lag_asset_block_drift_l1"] + common_controls if c in base.columns]),
        ("next_overlay_block_drift_l1", [c for c in ["overlay_block_drift_l1", "lag_overlay_block_drift_l1"] + common_controls if c in base.columns]),
    ]:
        rows.append(
            run_reg(
                base,
                outcome,
                controls,
                {"family": "grouped_drift_outcome", "variant": "main_exogenous", "regressor": "out_of_proxy"},
                entity_col,
                date_col,
                args.demean_iter,
            )
        )

    drop_mask = derivative_leverage_mask(base)
    if drop_mask.any() and (~drop_mask).sum() > 100:
        rows.append(
            run_reg(
                base.loc[~drop_mask].copy(),
                "next_style_drift_l1",
                style_controls,
                {"family": "drop_derivative_leverage", "variant": "main_exogenous", "regressor": "out_of_proxy"},
                entity_col,
                date_col,
                args.demean_iter,
            )
        )

    coord_rows = []
    for j, col in enumerate(action_cols):
        flag = (np.delete(base_gap, j, axis=1).sum(axis=1) > 1e-7).astype(int)
        coord_data = data.copy()
        coord_data["out_of_proxy"] = flag[coord_data["row_pos"].to_numpy(dtype=int)]
        res = run_reg(
            coord_data,
            "next_style_drift_l1",
            style_controls,
            {"family": "coordinate_exclusion", "variant": f"exclude_{col}", "regressor": "out_of_proxy_excluding_coordinate"},
            entity_col,
            date_col,
            args.demean_iter,
        )
        coord_rows.append(res)
        rows.append(res)

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "out_of_proxy_risk_signal_robustness_grid.csv", index=False)
    pd.DataFrame(coord_rows).to_csv(out_dir / "out_of_proxy_coordinate_exclusion_grid.csv", index=False)
    manifest = {
        "selected_root": str(selected_root),
        "package_root": str(package_root),
        "config": str(config_path),
        "split": args.split,
        "n_rows_split": int(len(data)),
        "bound_variants": [v[0] for v in bound_variants],
        "notes": [
            "Grid specifications vary outcomes, coordinates, and bound settings.",
            "Coefficients are predictive associations, not legal-validity or causal estimates.",
            "The severity row uses total L1 distance outside the inferred mandate-proxy box as the regressor.",
        ],
    }
    with open(out_dir / "out_of_proxy_robustness_grid_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(grid[["family", "variant", "outcome", "n", "beta", "two_way_cluster_t", "two_way_cluster_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
