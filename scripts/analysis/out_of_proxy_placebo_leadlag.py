"""Placebo and lead-lag diagnostics for the mandate-proxy exception signal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    residualize_on_fixed_effects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--selected_root", default=str(repo_root / "outputs" / "full_pipeline"))
    parser.add_argument("--out_dir", default=str(repo_root / "outputs" / "placebo_leadlag"))
    parser.add_argument("--package_root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_perm", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demean_iter", type=int, default=12)
    return parser.parse_args()


def permute_within_groups(df: pd.DataFrame, value_col: str, group_cols: list[str], rng: np.random.Generator) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx))
        vals = df.loc[idx_arr, value_col].to_numpy(copy=True)
        out.loc[idx_arr] = rng.permutation(vals)
    return out


def fwl_beta(y: np.ndarray, x: np.ndarray, controls: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if controls is None or controls.size == 0:
        y_res = y - np.nanmean(y)
        x_res = x - np.nanmean(x)
    else:
        C = np.asarray(controls, dtype=float)
        C = C[:, np.isfinite(C).all(axis=0)]
        if C.size == 0:
            y_res = y - np.nanmean(y)
            x_res = x - np.nanmean(x)
        else:
            pinv = np.linalg.pinv(C.T @ C)
            y_res = y - C @ (pinv @ (C.T @ y))
            x_res = x - C @ (pinv @ (C.T @ x))
    denom = float(x_res @ x_res)
    if denom <= 1e-14:
        return np.nan
    return float(x_res @ y_res / denom)


def build_design_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], str, str]:
    selected_root = Path(args.selected_root)
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
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
    actions = as_numeric_frame(panel, action_cols).to_numpy(dtype=float)
    out_of_proxy = action_violations(actions, bounds.lower, bounds.upper, tol=1e-7).astype(int)

    data = build_outcome_panel(panel, action_cols, entity_col)
    data["row_pos"] = np.arange(len(panel))
    data["out_of_proxy"] = out_of_proxy
    if str(args.split).lower() != "all":
        data = data[data["split"].astype(str).eq(args.split)].copy()
    common = [c for c in ["log_net_assets", "net_flow_3m_scaled", "lag_net_flow_3m_scaled"] if c in data.columns]
    return data, common, entity_col, date_col


def prepare_placebo_frame(data: pd.DataFrame, common_controls: list[str], entity_col: str, date_col: str, n_iter: int):
    controls = [c for c in ["style_drift_l1", "lag_style_drift_l1"] + common_controls if c in data.columns]
    needed = ["next_style_drift_l1", "out_of_proxy", date_col, "mandate_cluster", entity_col] + controls
    work = data[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    for c in ["next_style_drift_l1", "out_of_proxy"] + controls:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    y = work["next_style_drift_l1"].to_numpy(dtype=float)
    x = work["out_of_proxy"].to_numpy(dtype=float)
    C = work[controls].to_numpy(dtype=float) if controls else np.empty((len(work), 0))
    fe_codes = [pd.factorize(work[c].to_numpy(), sort=False)[0] for c in [date_col, "mandate_cluster"]]
    y_fe, mat_fe = residualize_on_fixed_effects(y, np.column_stack([x, C]), fe_codes, n_iter)
    x_fe = mat_fe[:, 0]
    c_fe = mat_fe[:, 1:] if mat_fe.shape[1] > 1 else np.empty((len(work), 0))
    real_beta_fast = fwl_beta(y_fe, x_fe, c_fe)
    return work, y_fe, c_fe, fe_codes, real_beta_fast, controls


def residualize_vector_by_fe(x: np.ndarray, fe_codes: list[np.ndarray], n_iter: int) -> np.ndarray:
    dummy_y = np.zeros(len(x), dtype=float)
    _, x_res = residualize_on_fixed_effects(dummy_y, np.asarray(x, dtype=float)[:, None], fe_codes, n_iter)
    return x_res[:, 0]


def run_permutation_placebo(
    work: pd.DataFrame,
    y_fe: np.ndarray,
    c_fe: np.ndarray,
    fe_codes: list[np.ndarray],
    group_cols: list[str],
    n_perm: int,
    seed: int,
    n_iter: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_perm):
        x_perm = permute_within_groups(work, "out_of_proxy", group_cols, rng).to_numpy(dtype=float)
        x_perm_fe = residualize_vector_by_fe(x_perm, fe_codes, n_iter)
        beta = fwl_beta(y_fe, x_perm_fe, c_fe)
        rows.append({"permutation": b, "placebo_beta": beta})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, common_controls, entity_col, date_col = build_design_data(args)

    lead_lag_rows = []
    lead_lag_specs = [
        ("future_next_style_drift", "next_style_drift_l1", [c for c in ["style_drift_l1", "lag_style_drift_l1"] + common_controls if c in data.columns]),
        ("current_style_drift", "style_drift_l1", [c for c in ["lag_style_drift_l1"] + common_controls if c in data.columns]),
        ("past_style_drift", "lag_style_drift_l1", common_controls),
    ]
    for label, outcome, controls in lead_lag_specs:
        res = fit_residualized_ols(
            data,
            outcome,
            controls=controls,
            fe_cols=[date_col, "mandate_cluster"],
            entity_col=entity_col,
            date_col=date_col,
            n_iter=args.demean_iter,
        )
        res["diagnostic"] = label
        lead_lag_rows.append(res)
    lead_lag = pd.DataFrame(lead_lag_rows)

    work, y_fe, c_fe, fe_codes, real_beta_fast, controls = prepare_placebo_frame(
        data, common_controls, entity_col, date_col, args.demean_iter
    )
    placebo = run_permutation_placebo(
        work,
        y_fe,
        c_fe,
        fe_codes,
        [date_col, "mandate_cluster"],
        args.n_perm,
        args.seed,
        args.demean_iter,
    )
    real_beta = float(lead_lag.loc[lead_lag["diagnostic"].eq("future_next_style_drift"), "beta"].iloc[0])
    placebo_betas = placebo["placebo_beta"].to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "real_beta_clustered_regression": real_beta,
                "real_beta_fast_fwl": real_beta_fast,
                "placebo_mean": float(np.nanmean(placebo_betas)),
                "placebo_sd": float(np.nanstd(placebo_betas, ddof=1)),
                "placebo_p01": float(np.nanquantile(placebo_betas, 0.01)),
                "placebo_p025": float(np.nanquantile(placebo_betas, 0.025)),
                "placebo_p50": float(np.nanquantile(placebo_betas, 0.50)),
                "placebo_p975": float(np.nanquantile(placebo_betas, 0.975)),
                "placebo_p99": float(np.nanquantile(placebo_betas, 0.99)),
                "share_placebo_ge_real": float(np.nanmean(placebo_betas >= real_beta_fast)),
                "n_perm": int(args.n_perm),
                "n_regression_rows": int(len(work)),
            }
        ]
    )
    lead_lag.to_csv(out_dir / "out_of_proxy_lead_lag_diagnostics.csv", index=False)
    placebo.to_csv(out_dir / "out_of_proxy_permutation_placebo_betas.csv", index=False)
    summary.to_csv(out_dir / "out_of_proxy_permutation_placebo_summary.csv", index=False)
    manifest = {
        "selected_root": args.selected_root,
        "split": args.split,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "placebo_group_cols": [date_col, "mandate_cluster"],
        "lead_lag_outcomes": [x[1] for x in lead_lag_specs],
        "controls_for_future_placebo": controls,
        "notes": [
            "Permutation placebo shuffles out_of_proxy within report_date by mandate_cluster cells.",
            "Placebo beta uses the same fixed effects and controls via fast FWL residualization.",
            "Lead-lag rows use two-way clustered SEs from out_of_proxy_predictive_validity.py.",
        ],
    }
    with open(out_dir / "out_of_proxy_placebo_leadlag_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print("\nLEAD-LAG")
    print(lead_lag[["diagnostic", "outcome", "n", "beta", "two_way_cluster_t", "two_way_cluster_p"]].to_string(index=False))
    print("\nPLACEBO")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
