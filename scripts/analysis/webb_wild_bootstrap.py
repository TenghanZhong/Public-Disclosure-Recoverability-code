"""Webb six-point date-cluster wild bootstrap checks.

The regression sample and mandate bounds are built with the same current
package constructor as `boundary_inference_checks.py`. This script adds
Webb six-point weights and date-cluster size diagnostics for three boundary
specifications.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_ROOT = DEFAULT_REPO_ROOT
DEFAULT_SELECTED_ROOT = DEFAULT_REPO_ROOT / "outputs" / "full_pipeline"
DEFAULT_CONFIG = DEFAULT_REPO_ROOT / "configs" / "main_32gb.yaml"
DEFAULT_OUT_DIR = DEFAULT_REPO_ROOT / "tables" / "webb_wild_bootstrap"

sys.path.insert(0, str(DEFAULT_REPO_ROOT / "scripts"))

from boundary_inference_checks import (  # noqa: E402
    build_regression_panel,
    fit_residualized_ols,
    load_action_columns,
    regression_design,
    zscore_on_mask,
)


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
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def add_package(package_root: Path):
    sys.path.insert(0, str(package_root))
    from mandate_setirl.bounds import fit_mandate_bounds
    from mandate_setirl.utils import read_yaml

    return fit_mandate_bounds, read_yaml


def panel_sorted_by_series_date(panel: pd.DataFrame, entity_col: str, date_col: str) -> bool:
    order = panel.sort_values([entity_col, date_col], kind="mergesort").index.to_numpy()
    return bool(np.array_equal(order, np.arange(len(panel))))


def webb_six_point_values() -> tuple[np.ndarray, np.ndarray]:
    values = np.array(
        [
            -math.sqrt(3.0 / 2.0),
            -1.0,
            -math.sqrt(1.0 / 2.0),
            math.sqrt(1.0 / 2.0),
            1.0,
            math.sqrt(3.0 / 2.0),
        ],
        dtype=float,
    )
    probs = np.full(6, 1.0 / 6.0, dtype=float)
    return values, probs


def date_cluster_se(
    X: np.ndarray,
    residual: np.ndarray,
    xtx_inv: np.ndarray,
    cluster_codes: np.ndarray,
    focus: int,
) -> float:
    valid = cluster_codes >= 0
    codes = cluster_codes[valid]
    if len(codes) == 0:
        return np.nan
    n_groups = int(codes.max()) + 1
    xu = X[valid] * residual[valid, None]
    summed = np.zeros((n_groups, X.shape[1]), dtype=float)
    np.add.at(summed, codes, xu)
    meat = summed.T @ summed
    vcov = xtx_inv @ meat @ xtx_inv
    n = X.shape[0]
    k = X.shape[1]
    if n_groups > 1:
        vcov *= (n_groups / (n_groups - 1)) * ((n - 1) / max(n - k, 1))
    var = float(vcov[focus, focus])
    return math.sqrt(var) if var >= 0 else np.nan


def webb_wild_cluster_bootstrap_p(
    y_tilde: np.ndarray,
    X_tilde: np.ndarray,
    cluster_codes: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Restricted-null wild cluster bootstrap with Webb six-point weights."""
    rng = np.random.default_rng(seed)
    focus = 0
    x_full = X_tilde
    xtx_inv = np.linalg.pinv(x_full.T @ x_full)
    beta_hat = xtx_inv @ (x_full.T @ y_tilde)
    residual_full = y_tilde - x_full @ beta_hat
    se_hat = date_cluster_se(x_full, residual_full, xtx_inv, cluster_codes, focus)
    t_hat = float(beta_hat[focus] / se_hat) if se_hat > 0 else np.nan

    if x_full.shape[1] > 1:
        x_rest = x_full[:, 1:]
        beta_rest = np.linalg.pinv(x_rest.T @ x_rest) @ (x_rest.T @ y_tilde)
        fitted_null = x_rest @ beta_rest
        resid_null = y_tilde - fitted_null
    else:
        fitted_null = np.zeros_like(y_tilde)
        resid_null = y_tilde.copy()

    valid = cluster_codes >= 0
    n_clusters = int(cluster_codes[valid].max()) + 1 if valid.any() else 0
    values, probs = webb_six_point_values()
    boot_t = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        weights = rng.choice(values, size=n_clusters, p=probs)
        y_star = fitted_null + resid_null * weights[cluster_codes]
        beta_star = xtx_inv @ (x_full.T @ y_star)
        resid_star = y_star - x_full @ beta_star
        se_star = date_cluster_se(x_full, resid_star, xtx_inv, cluster_codes, focus)
        boot_t[b] = beta_star[focus] / se_star if se_star > 0 else np.nan
    boot_t = boot_t[np.isfinite(boot_t)]
    p_value = float((1.0 + np.sum(np.abs(boot_t) >= abs(t_hat))) / (1.0 + len(boot_t)))
    return {
        "date_cluster_t": t_hat,
        "webb_p": p_value,
        "webb_draws": int(len(boot_t)),
        "webb_weight_mean": float(np.sum(values * probs)),
        "webb_weight_variance": float(np.sum((values**2) * probs)),
    }


def run_fit(
    df: pd.DataFrame,
    outcome: str,
    regressor: str,
    controls: list[str],
    entity_col: str,
    date_col: str,
    demean_iter: int,
    label: str,
) -> dict:
    work = df.copy()
    work["out_of_proxy"] = pd.to_numeric(work[regressor], errors="coerce")
    result = fit_residualized_ols(
        work,
        outcome,
        controls=[c for c in controls if c in work.columns and c != regressor],
        fe_cols=[date_col, "mandate_cluster"],
        entity_col=entity_col,
        date_col=date_col,
        n_iter=demean_iter,
    )
    result["spec"] = label
    result["regressor"] = regressor
    return result


def summarize_date_clusters(data: pd.DataFrame, date_col: str, spec: str) -> dict:
    counts = data.groupby(date_col, dropna=False).size().astype(float)
    return {
        "spec": spec,
        "n_date_clusters": int(len(counts)),
        "date_cluster_size_min": int(counts.min()),
        "date_cluster_size_p25": float(counts.quantile(0.25)),
        "date_cluster_size_median": float(counts.median()),
        "date_cluster_size_p75": float(counts.quantile(0.75)),
        "date_cluster_size_max": int(counts.max()),
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
    sorted_ok = panel_sorted_by_series_date(panel, entity_col, date_col)
    if not sorted_ok:
        raise RuntimeError("panel_model_ready.parquet is not sorted by series/date; rebuild lag controls before running.")

    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode=args.bound_mode)
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
    split_mask = reg["split"].astype(str).eq(args.split)
    zcol = zscore_on_mask(reg, "outside_proxy_distance_iqr_mean", split_mask)
    test = reg.loc[split_mask].copy()

    controls_main = [
        "style_drift_l1",
        "lag_style_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]
    controls_inside = [
        "current_nonviolating_drift_l1",
        "lag_nonviolating_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]
    specs = [
        ("binary_exception", "next_style_drift_l1", "exception", controls_main),
        ("scaled_distance", "next_style_drift_l1", zcol, controls_main),
        ("scaled_distance_to_nonviolating_drift", "next_nonviolating_drift_l1", zcol, controls_inside),
    ]

    rows: list[dict] = []
    cluster_rows: list[dict] = []
    for i, (label, outcome, regressor, controls) in enumerate(specs):
        result = run_fit(test, outcome, regressor, controls, entity_col, date_col, args.demean_iter, label)
        if regressor == "exception":
            design_df = test.copy()
            design_df["out_of_proxy"] = pd.to_numeric(design_df["exception"], errors="coerce")
            y_tilde, X_tilde, date_codes, _x_cols, data = regression_design(
                design_df,
                outcome,
                "out_of_proxy",
                controls,
                entity_col,
                date_col,
                args.demean_iter,
            )
        else:
            y_tilde, X_tilde, date_codes, _x_cols, data = regression_design(
                test,
                outcome,
                regressor,
                controls,
                entity_col,
                date_col,
                args.demean_iter,
            )
        result.update(webb_wild_cluster_bootstrap_p(y_tilde, X_tilde, date_codes, args.wild_boot, args.seed + i))
        rows.append(result)
        cluster_rows.append(summarize_date_clusters(data, date_col, label))

    out = pd.DataFrame(rows)
    clusters = pd.DataFrame(cluster_rows)
    out.to_csv(out_dir / "webb_wild_regressions.csv", index=False)
    clusters.to_csv(out_dir / "date_cluster_sizes.csv", index=False)
    meta.update(
        {
            "selected_root": str(selected_root),
            "package_root": str(package_root),
            "config": str(Path(args.config)),
            "bound_mode": args.bound_mode,
            "bounds_constructor": "mandate_setirl.bounds.fit_mandate_bounds",
            "split": args.split,
            "panel_sorted_by_series_date": sorted_ok,
            "webb_draws_requested": int(args.wild_boot),
            "seed": int(args.seed),
            "webb_weights": "six-point Webb weights with mean zero and unit variance",
        }
    )
    with open(out_dir / "webb_wild_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print("\nWEBB REGRESSIONS")
    print(out[["spec", "outcome", "n", "beta", "two_way_cluster_t", "date_cluster_t", "webb_p"]].to_string(index=False))
    print("\nDATE CLUSTER SIZES")
    print(clusters.to_string(index=False))


if __name__ == "__main__":
    main()
