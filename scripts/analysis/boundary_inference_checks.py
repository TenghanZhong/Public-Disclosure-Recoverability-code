"""Boundary-distance regressions with date-cluster wild bootstrap inference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from out_of_proxy_predictive_validity import (  # noqa: E402
    fit_residualized_ols,
    residualize_on_fixed_effects,
)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_ROOT = DEFAULT_REPO_ROOT
DEFAULT_SELECTED_ROOT = DEFAULT_REPO_ROOT / "outputs" / "full_pipeline"
DEFAULT_CONFIG = DEFAULT_REPO_ROOT / "configs" / "main_32gb.yaml"
DEFAULT_OUT_DIR = DEFAULT_REPO_ROOT / "tables" / "boundary_inference_checks"


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


def add_package(package_root: Path):
    sys.path.insert(0, str(package_root))
    from mandate_setirl.bounds import fit_mandate_bounds
    from mandate_setirl.utils import read_yaml

    return fit_mandate_bounds, read_yaml


def load_action_columns(selected_root: Path) -> dict:
    with open(selected_root / "action_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def as_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce")


def lead_lag_actions(
    panel: pd.DataFrame,
    actions: pd.DataFrame,
    entity_col: str,
    date_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    order = panel.sort_values([entity_col, date_col]).index.to_numpy()
    lag = pd.DataFrame(index=panel.index, columns=actions.columns, dtype=float)
    lead = pd.DataFrame(index=panel.index, columns=actions.columns, dtype=float)
    sorted_actions = actions.loc[order]
    sorted_entity = panel.loc[order, entity_col]
    lag.loc[order] = sorted_actions.groupby(sorted_entity, sort=False).shift(1).to_numpy()
    lead.loc[order] = sorted_actions.groupby(sorted_entity, sort=False).shift(-1).to_numpy()
    return lag.to_numpy(dtype=float), lead.to_numpy(dtype=float)


def training_iqr_scale(actions: np.ndarray, train_mask: np.ndarray, floor: float) -> np.ndarray:
    q75 = np.nanquantile(actions[train_mask], 0.75, axis=0)
    q25 = np.nanquantile(actions[train_mask], 0.25, axis=0)
    iqr = q75 - q25
    return np.where(np.isfinite(iqr) & (iqr > floor), iqr, 1.0)


def build_regression_panel(
    panel: pd.DataFrame,
    action_cols: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    entity_col: str,
    date_col: str,
    tol: float,
    iqr_floor: float,
) -> tuple[pd.DataFrame, dict]:
    actions_df = as_numeric(panel, action_cols)
    actions = actions_df.to_numpy(dtype=float)
    lag_actions, lead_actions = lead_lag_actions(panel, actions_df, entity_col, date_col)
    lag_valid = np.isfinite(lag_actions).all(axis=1)
    next_valid = np.isfinite(lead_actions).all(axis=1)
    gap = np.maximum(lower - actions, 0.0) + np.maximum(actions - upper, 0.0)
    train_mask = panel["split"].astype(str).eq("train").to_numpy()
    scale_iqr = training_iqr_scale(actions, train_mask, iqr_floor)

    out = panel[[entity_col, "split", "mandate_cluster", date_col]].copy()
    out["style_drift_l1"] = np.abs(actions - lag_actions).sum(axis=1)
    out.loc[~lag_valid, "style_drift_l1"] = np.nan
    out["lag_style_drift_l1"] = out.groupby(entity_col, sort=False)["style_drift_l1"].shift(1)
    out["next_style_drift_l1"] = np.abs(lead_actions - actions).sum(axis=1)
    out.loc[~next_valid, "next_style_drift_l1"] = np.nan

    out["exception"] = (gap.sum(axis=1) > tol).astype(int)
    out["outside_proxy_distance_l1"] = gap.sum(axis=1)
    out["outside_proxy_distance_iqr_mean"] = (gap / scale_iqr).mean(axis=1)

    violation_coord = gap > tol
    next_drift = np.abs(lead_actions - actions)
    next_drift[violation_coord] = 0.0
    out["next_nonviolating_drift_l1"] = next_drift.sum(axis=1)
    out.loc[~next_valid, "next_nonviolating_drift_l1"] = np.nan

    current_nonviolating = np.abs(actions - lag_actions)
    current_nonviolating[violation_coord] = 0.0
    out["current_nonviolating_drift_l1"] = current_nonviolating.sum(axis=1)
    out.loc[~lag_valid, "current_nonviolating_drift_l1"] = np.nan
    out["lag_nonviolating_drift_l1"] = out.groupby(entity_col, sort=False)[
        "current_nonviolating_drift_l1"
    ].shift(1)

    for col in ["log_net_assets", "net_flow_3m_scaled", "lag_net_flow_3m_scaled"]:
        if col in panel.columns:
            out[col] = pd.to_numeric(panel[col], errors="coerce")

    meta = {
        "test_outside_proxy_share": float(out.loc[out["split"].astype(str).eq("test"), "exception"].mean()),
        "n_action_cols": int(len(action_cols)),
        "iqr_floor": float(iqr_floor),
    }
    return out, meta


def zscore_on_mask(df: pd.DataFrame, col: str, mask: pd.Series) -> str:
    vals = pd.to_numeric(df.loc[mask, col], errors="coerce")
    mean = float(vals.mean())
    sd = float(vals.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError(f"Cannot z-score {col}: invalid standard deviation")
    zcol = f"{col}_z"
    df[zcol] = (pd.to_numeric(df[col], errors="coerce") - mean) / sd
    return zcol


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
    res = fit_residualized_ols(
        work,
        outcome,
        controls=[c for c in controls if c in work.columns and c != regressor],
        fe_cols=[date_col, "mandate_cluster"],
        entity_col=entity_col,
        date_col=date_col,
        n_iter=demean_iter,
    )
    res["spec"] = label
    res["regressor"] = regressor
    focus = pd.to_numeric(work["out_of_proxy"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    unique = set(focus.dropna().unique())
    res["positive_regressor_n"] = int((focus > 1e-7).sum())
    if not unique.issubset({0, 1}):
        res["out_of_proxy_n"] = np.nan
    return res


def regression_design(
    df: pd.DataFrame,
    outcome: str,
    regressor: str,
    controls: list[str],
    entity_col: str,
    date_col: str,
    demean_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    x_cols = [regressor] + [c for c in controls if c in df.columns and c != regressor]
    needed = [outcome] + x_cols + [date_col, "mandate_cluster", entity_col]
    data = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    for c in [outcome] + x_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(subset=[outcome] + x_cols + [date_col, "mandate_cluster", entity_col])
    y = data[outcome].to_numpy(dtype=float)
    X = data[x_cols].to_numpy(dtype=float)
    fe_codes = [
        pd.factorize(data[date_col].to_numpy(), sort=False)[0],
        pd.factorize(data["mandate_cluster"].to_numpy(), sort=False)[0],
    ]
    y_tilde, X_tilde = residualize_on_fixed_effects(y, X, fe_codes, demean_iter)
    date_codes = pd.factorize(data[date_col].to_numpy(), sort=False)[0]
    return y_tilde, X_tilde, date_codes, np.asarray(x_cols), data


def wild_cluster_bootstrap_p(
    y_tilde: np.ndarray,
    X_tilde: np.ndarray,
    cluster_codes: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Rademacher wild cluster bootstrap for H0 beta_focus = 0.

    Controls and fixed effects are already residualized.  The restricted null
    keeps all non-focus regressors and excludes the first column.
    """
    rng = np.random.default_rng(seed)
    focus = 0
    X_full = X_tilde
    XtX_inv = np.linalg.pinv(X_full.T @ X_full)
    beta_hat = XtX_inv @ (X_full.T @ y_tilde)
    residual_full = y_tilde - X_full @ beta_hat
    se_hat = _date_cluster_se(X_full, residual_full, XtX_inv, cluster_codes, focus)
    t_hat = float(beta_hat[focus] / se_hat) if se_hat > 0 else np.nan

    if X_full.shape[1] > 1:
        X_rest = X_full[:, 1:]
        beta_rest = np.linalg.pinv(X_rest.T @ X_rest) @ (X_rest.T @ y_tilde)
        fitted_null = X_rest @ beta_rest
        resid_null = y_tilde - fitted_null
    else:
        fitted_null = np.zeros_like(y_tilde)
        resid_null = y_tilde.copy()

    valid = cluster_codes >= 0
    n_clusters = int(cluster_codes[valid].max()) + 1 if valid.any() else 0
    boot_t = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        weights = rng.choice(np.array([-1.0, 1.0]), size=n_clusters)
        y_star = fitted_null + resid_null * weights[cluster_codes]
        beta_star = XtX_inv @ (X_full.T @ y_star)
        resid_star = y_star - X_full @ beta_star
        se_star = _date_cluster_se(X_full, resid_star, XtX_inv, cluster_codes, focus)
        boot_t[b] = beta_star[focus] / se_star if se_star > 0 else np.nan
    boot_t = boot_t[np.isfinite(boot_t)]
    p = float((1.0 + np.sum(np.abs(boot_t) >= abs(t_hat))) / (1.0 + len(boot_t)))
    return {
        "date_cluster_t": t_hat,
        "wild_cluster_p": p,
        "wild_boot_draws": int(len(boot_t)),
        "wild_cluster_type": "Rademacher date-cluster restricted-null bootstrap",
    }


def _date_cluster_se(
    X: np.ndarray,
    residual: np.ndarray,
    XtX_inv: np.ndarray,
    cluster_codes: np.ndarray,
    focus: int,
) -> float:
    valid = cluster_codes >= 0
    codes = cluster_codes[valid]
    if len(codes) == 0:
        return np.nan
    n_groups = int(codes.max()) + 1
    Xu = X[valid] * residual[valid, None]
    summed = np.zeros((n_groups, X.shape[1]), dtype=float)
    np.add.at(summed, codes, Xu)
    meat = summed.T @ summed
    vcov = XtX_inv @ meat @ XtX_inv
    n = X.shape[0]
    k = X.shape[1]
    if n_groups > 1:
        vcov *= (n_groups / (n_groups - 1)) * ((n - 1) / max(n - k, 1))
    var = float(vcov[focus, focus])
    return math.sqrt(var) if var >= 0 else np.nan


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

    ac = load_action_columns(selected_root)
    action_cols = ac["action_cols"]
    asset_cols = ac["asset_action_cols"]
    overlay_cols = ac["overlay_action_cols"]
    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    bounds = fit_mandate_bounds(
        panel,
        action_cols,
        asset_cols,
        overlay_cols,
        entity_col,
        cfg,
        mode=args.bound_mode,
    )
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
        ("binary_exception_main_bounds", "next_style_drift_l1", "exception", controls_main),
        ("scaled_distance_main_bounds", "next_style_drift_l1", zcol, controls_main),
        ("scaled_distance_to_inside_coordinate_drift_main_bounds", "next_nonviolating_drift_l1", zcol, controls_inside),
    ]
    rows = []
    for idx, (label, outcome, regressor, controls) in enumerate(specs):
        res = run_fit(test, outcome, regressor, controls, entity_col, date_col, args.demean_iter, label)
        if regressor == "exception":
            design_df = test.copy()
            design_df["out_of_proxy"] = pd.to_numeric(design_df["exception"], errors="coerce")
            y_tilde, X_tilde, date_codes, x_cols, data = regression_design(
                design_df,
                outcome,
                "out_of_proxy",
                controls,
                entity_col,
                date_col,
                args.demean_iter,
            )
        else:
            y_tilde, X_tilde, date_codes, x_cols, data = regression_design(
                test,
                outcome,
                regressor,
                controls,
                entity_col,
                date_col,
                args.demean_iter,
            )
        wild = wild_cluster_bootstrap_p(
            y_tilde,
            X_tilde,
            date_codes,
            args.wild_boot,
            args.seed + idx,
        )
        res.update(wild)
        rows.append(res)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "boundary_wild_regressions.csv", index=False)
    meta.update(
        {
            "selected_root": str(selected_root),
            "package_root": str(package_root),
            "config": str(Path(args.config)),
            "bound_mode": args.bound_mode,
            "bounds_constructor": "mandate_setirl.bounds.fit_mandate_bounds",
            "split": args.split,
            "wild_boot_draws_requested": args.wild_boot,
            "seed": args.seed,
            "standardized_distance": "mean coordinate boundary gap divided by training-split IQR, then z-scored on the test split",
            "inside_coordinate_drift": "next-report L1 drift after excluding coordinates outside the proxy at the current report",
        }
    )
    with open(out_dir / "boundary_wild_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(
        out[
            [
                "spec",
                "outcome",
                "n",
                "beta",
                "two_way_cluster_t",
                "two_way_cluster_p",
                "date_cluster_t",
                "wild_cluster_p",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
