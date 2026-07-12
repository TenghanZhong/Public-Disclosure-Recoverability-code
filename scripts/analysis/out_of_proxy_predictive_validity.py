"""Predictive-validity diagnostics for the inferred mandate proxy.

The script rebuilds the selected exogenous mandate-proxy bounds, marks
disclosed allocations outside those bounds, and tests whether this flag
predicts next-report outcomes.  It reports raw distribution checks plus
fixed-effect regressions with one-way series and two-way series/date clustered
standard errors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - scipy is expected but not required.
    scipy_stats = None


DEFAULT_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_ROOT = DEFAULT_PACKAGE_ROOT / "outputs" / "full_pipeline"
DEFAULT_OUT_DIR = DEFAULT_PACKAGE_ROOT / "outputs" / "predictive_validity"


OUTCOME_MAP = {
    "next_style_drift_l1": {
        "source": "style_drift_l1",
        "controls": ["style_drift_l1", "lag_style_drift_l1"],
    },
    "next_RETURN_AFTR_FEES_AND_EXPENSES": {
        "source": "RETURN_AFTR_FEES_AND_EXPENSES",
        "controls": ["RETURN_AFTR_FEES_AND_EXPENSES", "lag_RETURN_AFTR_FEES_AND_EXPENSES"],
    },
    "next_RETURN_B4_FEES_AND_EXPENSES": {
        "source": "RETURN_B4_FEES_AND_EXPENSES",
        "controls": ["RETURN_B4_FEES_AND_EXPENSES", "lag_RETURN_B4_FEES_AND_EXPENSES"],
    },
    "next_STDV_AFTR_FEES_AND_EXPENSES": {
        "source": "STDV_AFTR_FEES_AND_EXPENSES",
        "controls": ["STDV_AFTR_FEES_AND_EXPENSES", "lag_STDV_AFTR_FEES_AND_EXPENSES"],
    },
    "next_STDV_B4_FEES_AND_EXPENSES": {
        "source": "STDV_B4_FEES_AND_EXPENSES",
        "controls": ["STDV_B4_FEES_AND_EXPENSES", "lag_STDV_B4_FEES_AND_EXPENSES"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected_root",
        default=str(DEFAULT_SELECTED_ROOT),
        help="Selected-specification result directory.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for diagnostic output files.",
    )
    parser.add_argument(
        "--package_root",
        default=str(DEFAULT_PACKAGE_ROOT),
        help="Repository root.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config YAML. Defaults to <package_root>/configs/main_32gb.yaml.",
    )
    parser.add_argument("--bound_mode", default="exogenous")
    parser.add_argument("--split", default="test", help="Panel split to analyze, or 'all'.")
    parser.add_argument("--winsor_low", type=float, default=0.01)
    parser.add_argument("--winsor_high", type=float, default=0.99)
    parser.add_argument("--demean_iter", type=int, default=12)
    return parser.parse_args()


def add_package(package_root: Path):
    sys.path.insert(0, str(package_root))
    from mandate_setirl.bounds import action_violations, fit_mandate_bounds
    from mandate_setirl.utils import read_yaml

    return action_violations, fit_mandate_bounds, read_yaml


def load_action_columns(selected_root: Path) -> dict:
    with open(selected_root / "action_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def as_numeric_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    return df[list(cols)].apply(pd.to_numeric, errors="coerce")


def build_outcome_panel(panel: pd.DataFrame, action_cols: list[str], entity_col: str) -> pd.DataFrame:
    date_col = "report_date" if "report_date" in panel.columns else None
    sort_cols = [entity_col] + ([date_col] if date_col else [])
    work = panel.sort_values(sort_cols).copy()
    out = work[[entity_col, "split", "mandate_cluster"]].copy()
    if date_col:
        out[date_col] = work[date_col]
    else:
        out["report_date"] = np.arange(len(work))

    actions = as_numeric_frame(work, action_cols)
    lag_actions = actions.groupby(work[entity_col], sort=False).shift(1)
    lead_actions = actions.groupby(work[entity_col], sort=False).shift(-1)
    out["style_drift_l1"] = (actions - lag_actions).abs().sum(axis=1)
    out.loc[lag_actions.isna().any(axis=1), "style_drift_l1"] = np.nan
    out["lag_style_drift_l1"] = out.groupby(entity_col, sort=False)["style_drift_l1"].shift(1)
    out["next_style_drift_l1"] = (lead_actions - actions).abs().sum(axis=1)
    out.loc[lead_actions.isna().any(axis=1), "next_style_drift_l1"] = np.nan

    for c in [
        "RETURN_AFTR_FEES_AND_EXPENSES",
        "RETURN_B4_FEES_AND_EXPENSES",
        "STDV_AFTR_FEES_AND_EXPENSES",
        "STDV_B4_FEES_AND_EXPENSES",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
        "derivative_pct",
        "allows_or_discusses_derivatives",
        "leverage",
    ]:
        if c in work.columns:
            out[c] = pd.to_numeric(work[c], errors="coerce")

    for c in [
        "RETURN_AFTR_FEES_AND_EXPENSES",
        "RETURN_B4_FEES_AND_EXPENSES",
        "STDV_AFTR_FEES_AND_EXPENSES",
        "STDV_B4_FEES_AND_EXPENSES",
    ]:
        if c in out.columns:
            out[f"lag_{c}"] = out.groupby(entity_col, sort=False)[c].shift(1)
            out[f"next_{c}"] = out.groupby(entity_col, sort=False)[c].shift(-1)

    return out.sort_index()


def normal_p_value(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))


def p_value_from_t(t_stat: float, df: int | None) -> float:
    if not np.isfinite(t_stat):
        return np.nan
    if scipy_stats is not None and df is not None and df > 1:
        return float(2.0 * scipy_stats.t.sf(abs(t_stat), df))
    return normal_p_value(t_stat)


def demean_matrix_by_codes(mat: np.ndarray, codes: np.ndarray) -> np.ndarray:
    valid = codes >= 0
    if not valid.any():
        return mat
    n_groups = int(codes[valid].max()) + 1
    counts = np.bincount(codes[valid], minlength=n_groups).astype(float)
    for j in range(mat.shape[1]):
        sums = np.bincount(codes[valid], weights=mat[valid, j], minlength=n_groups)
        means = sums / np.maximum(counts, 1.0)
        mat[valid, j] -= means[codes[valid]]
    return mat


def residualize_on_fixed_effects(
    y: np.ndarray,
    X: np.ndarray,
    fe_codes: list[np.ndarray],
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    mat = np.column_stack([y, X]).astype(float, copy=True)
    mat -= np.nanmean(mat, axis=0, keepdims=True)
    for _ in range(n_iter):
        for codes in fe_codes:
            mat = demean_matrix_by_codes(mat, codes)
    return mat[:, 0], mat[:, 1:]


def cluster_vcov(
    X: np.ndarray,
    residual: np.ndarray,
    XtX_inv: np.ndarray,
    codes: np.ndarray,
    finite_sample: bool = True,
) -> tuple[np.ndarray, int]:
    valid = codes >= 0
    codes = codes[valid]
    Xu = X[valid] * residual[valid, None]
    n_groups = int(codes.max()) + 1 if len(codes) else 0
    if n_groups <= 1:
        return np.full((X.shape[1], X.shape[1]), np.nan), n_groups
    summed = np.zeros((n_groups, X.shape[1]), dtype=float)
    np.add.at(summed, codes, Xu)
    meat = summed.T @ summed
    vcov = XtX_inv @ meat @ XtX_inv
    if finite_sample:
        n = X.shape[0]
        k = X.shape[1]
        correction = (n_groups / max(n_groups - 1, 1)) * ((n - 1) / max(n - k, 1))
        vcov *= correction
    return vcov, n_groups


def fit_residualized_ols(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    fe_cols: list[str],
    entity_col: str,
    date_col: str,
    n_iter: int,
    winsorize: bool = False,
    winsor_low: float = 0.01,
    winsor_high: float = 0.99,
) -> dict:
    x_cols = ["out_of_proxy"] + controls
    needed = []
    for c in [outcome] + x_cols + fe_cols + [entity_col, date_col]:
        if c in df.columns and c not in needed:
            needed.append(c)
    data = df[needed].copy()
    for c in [outcome] + x_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome] + x_cols + fe_cols + [entity_col, date_col])
    if len(data) < 100 or data["out_of_proxy"].nunique() < 2:
        return {
            "outcome": outcome,
            "n": int(len(data)),
            "beta": np.nan,
            "series_cluster_se": np.nan,
            "series_cluster_p": np.nan,
            "two_way_cluster_se": np.nan,
            "two_way_cluster_p": np.nan,
            "controls": ",".join(controls),
        }

    if winsorize:
        for c in [outcome] + [v for v in controls if v != "out_of_proxy"]:
            lo, hi = data[c].quantile([winsor_low, winsor_high])
            data[c] = data[c].clip(lo, hi)

    y = data[outcome].to_numpy(dtype=float)
    X = data[x_cols].to_numpy(dtype=float)
    fe_codes = [pd.factorize(data[c].to_numpy(), sort=False)[0] for c in fe_cols]
    y_tilde, X_tilde = residualize_on_fixed_effects(y, X, fe_codes, n_iter)
    XtX = X_tilde.T @ X_tilde
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X_tilde.T @ y_tilde)
    residual = y_tilde - X_tilde @ beta
    focus_idx = 0

    series_codes = pd.factorize(data[entity_col].to_numpy(), sort=False)[0]
    date_codes = pd.factorize(data[date_col].to_numpy(), sort=False)[0]
    pair_codes = pd.factorize(pd.MultiIndex.from_arrays([data[entity_col], data[date_col]]), sort=False)[0]

    vcov_series, n_series = cluster_vcov(X_tilde, residual, XtX_inv, series_codes)
    vcov_date, n_dates = cluster_vcov(X_tilde, residual, XtX_inv, date_codes)
    vcov_pair, n_pairs = cluster_vcov(X_tilde, residual, XtX_inv, pair_codes)
    vcov_tw = vcov_series + vcov_date - vcov_pair

    series_var = float(vcov_series[focus_idx, focus_idx])
    tw_var = float(vcov_tw[focus_idx, focus_idx])
    series_se = math.sqrt(series_var) if series_var >= 0 else np.nan
    tw_se = math.sqrt(tw_var) if tw_var >= 0 else np.nan
    series_t = float(beta[focus_idx] / series_se) if series_se and np.isfinite(series_se) else np.nan
    tw_t = float(beta[focus_idx] / tw_se) if tw_se and np.isfinite(tw_se) else np.nan
    tw_df = max(min(n_series, n_dates) - 1, 1)

    y_var = np.dot(y_tilde, y_tilde)
    r2 = 1.0 - float(np.dot(residual, residual) / y_var) if y_var > 0 else np.nan
    return {
        "outcome": outcome,
        "n": int(len(data)),
        "out_of_proxy_n": int(data["out_of_proxy"].sum()),
        "beta": float(beta[focus_idx]),
        "series_cluster_se": series_se,
        "series_cluster_t": series_t,
        "series_cluster_p": p_value_from_t(series_t, max(n_series - 1, 1)),
        "two_way_cluster_se": tw_se,
        "two_way_cluster_t": tw_t,
        "two_way_cluster_p": p_value_from_t(tw_t, tw_df),
        "n_series_clusters": int(n_series),
        "n_date_clusters": int(n_dates),
        "n_series_date_clusters": int(n_pairs),
        "within_r2": r2,
        "controls": ",".join(controls),
    }


def winsorized_mean(s: pd.Series, low: float, high: float) -> float:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.nan
    lo, hi = vals.quantile([low, high])
    return float(vals.clip(lo, hi).mean())


def distribution_checks(df: pd.DataFrame, outcomes: list[str], low: float, high: float) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for label, group in [("all", df), ("in_proxy", df[df["out_of_proxy"] == 0]), ("out_of_proxy", df[df["out_of_proxy"] == 1])]:
            vals = pd.to_numeric(group[outcome], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row = {"outcome": outcome, "group": label, "n": int(vals.shape[0])}
            if vals.empty:
                rows.append(row)
                continue
            qs = vals.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
            row.update(
                {
                    "mean": float(vals.mean()),
                    "median": float(qs.loc[0.50]),
                    "p1": float(qs.loc[0.01]),
                    "p5": float(qs.loc[0.05]),
                    "p95": float(qs.loc[0.95]),
                    "p99": float(qs.loc[0.99]),
                    "std": float(vals.std()),
                    "winsorized_mean_1_99": winsorized_mean(vals, low, high),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def raw_differences(df: pd.DataFrame, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        data = df[[outcome, "out_of_proxy"]].replace([np.inf, -np.inf], np.nan).dropna()
        if data.empty:
            continue
        in_vals = pd.to_numeric(data.loc[data["out_of_proxy"] == 0, outcome], errors="coerce")
        out_vals = pd.to_numeric(data.loc[data["out_of_proxy"] == 1, outcome], errors="coerce")
        rows.append(
            {
                "outcome": outcome,
                "n": int(len(data)),
                "out_of_proxy_n": int((data["out_of_proxy"] == 1).sum()),
                "in_proxy_mean": float(in_vals.mean()),
                "out_of_proxy_mean": float(out_vals.mean()),
                "raw_difference_out_minus_in": float(out_vals.mean() - in_vals.mean()),
                "in_proxy_median": float(in_vals.median()),
                "out_of_proxy_median": float(out_vals.median()),
                "median_difference_out_minus_in": float(out_vals.median() - in_vals.median()),
            }
        )
    return pd.DataFrame(rows)


def derivative_leverage_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "allows_or_discusses_derivatives" in df.columns:
        mask |= pd.to_numeric(df["allows_or_discusses_derivatives"], errors="coerce").fillna(0) > 0
    if "leverage" in df.columns:
        mask |= pd.to_numeric(df["leverage"], errors="coerce").fillna(0) > 0
    if "derivative_pct" in df.columns:
        mask |= pd.to_numeric(df["derivative_pct"], errors="coerce").fillna(0) > 0.10
    return mask


def main():
    args = parse_args()
    selected_root = Path(args.selected_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    package_root = Path(args.package_root)
    config_path = Path(args.config) if args.config else package_root / "configs" / "main_32gb.yaml"

    action_violations, fit_mandate_bounds, read_yaml = add_package(package_root)
    cfg = read_yaml(config_path)
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    ac = load_action_columns(selected_root)
    action_cols = ac["action_cols"]
    asset_cols = ac["asset_action_cols"]
    overlay_cols = ac["overlay_action_cols"]

    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode=args.bound_mode)
    actions = as_numeric_frame(panel, action_cols).to_numpy(dtype=float)
    out_of_proxy = action_violations(actions, bounds.lower, bounds.upper, tol=1e-7).astype(int)

    data = build_outcome_panel(panel, action_cols, entity_col)
    data["row_pos"] = np.arange(len(panel))
    data["out_of_proxy"] = out_of_proxy
    if str(args.split).lower() != "all":
        data = data[data["split"].astype(str).eq(args.split)].copy()
    date_col = "report_date"
    fe_cols = ["report_date", "mandate_cluster"]
    common_controls = [c for c in ["log_net_assets", "net_flow_3m_scaled", "lag_net_flow_3m_scaled"] if c in data.columns]
    outcomes = [c for c in OUTCOME_MAP if c in data.columns]

    raw = raw_differences(data, outcomes)
    raw.to_csv(out_dir / "out_of_proxy_raw_differences.csv", index=False)

    dist = distribution_checks(
        data,
        outcomes
        + [
            "RETURN_AFTR_FEES_AND_EXPENSES",
            "RETURN_B4_FEES_AND_EXPENSES",
            "STDV_AFTR_FEES_AND_EXPENSES",
            "STDV_B4_FEES_AND_EXPENSES",
        ],
        args.winsor_low,
        args.winsor_high,
    )
    dist.to_csv(out_dir / "return_unit_outlier_checks.csv", index=False)

    rows = []
    for outcome in outcomes:
        spec_controls = [c for c in OUTCOME_MAP[outcome]["controls"] + common_controls if c in data.columns]
        rows.append(
            {
                "spec": "fe_only",
                **fit_residualized_ols(
                    data,
                    outcome,
                    controls=[],
                    fe_cols=fe_cols,
                    entity_col=entity_col,
                    date_col=date_col,
                    n_iter=args.demean_iter,
                ),
            }
        )
        rows.append(
            {
                "spec": "own_lag_controls",
                **fit_residualized_ols(
                    data,
                    outcome,
                    controls=spec_controls,
                    fe_cols=fe_cols,
                    entity_col=entity_col,
                    date_col=date_col,
                    n_iter=args.demean_iter,
                ),
            }
        )
        rows.append(
            {
                "spec": "own_lag_controls_winsorized_yx_1_99",
                **fit_residualized_ols(
                    data,
                    outcome,
                    controls=spec_controls,
                    fe_cols=fe_cols,
                    entity_col=entity_col,
                    date_col=date_col,
                    n_iter=args.demean_iter,
                    winsorize=True,
                    winsor_low=args.winsor_low,
                    winsor_high=args.winsor_high,
                ),
            }
        )
        drop_mask = derivative_leverage_mask(data)
        if drop_mask.any() and (~drop_mask).sum() > 100:
            rows.append(
                {
                    "spec": "own_lag_controls_drop_derivative_leverage_proxy",
                    **fit_residualized_ols(
                        data.loc[~drop_mask].copy(),
                        outcome,
                        controls=spec_controls,
                        fe_cols=fe_cols,
                        entity_col=entity_col,
                        date_col=date_col,
                        n_iter=args.demean_iter,
                    ),
                }
            )

    regs = pd.DataFrame(rows)
    regs.to_csv(out_dir / "out_of_proxy_predictive_validity_regressions.csv", index=False)

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
        "outcomes": outcomes,
        "fixed_effects": fe_cols,
        "common_controls": common_controls,
        "winsor_low": args.winsor_low,
        "winsor_high": args.winsor_high,
        "notes": [
            "Coefficients are predictive associations, not legal-validity or causal estimates.",
            "Two-way clustered SEs cluster by series_id and report_date.",
            "Own-lag controls include the current outcome proxy and its one-period lag when available.",
        ],
    }
    with open(out_dir / "out_of_proxy_predictive_validity_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(regs[["spec", "outcome", "n", "beta", "two_way_cluster_se", "two_way_cluster_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
