"""Boundary-distance and filing-timing diagnostics.

Runs boundary-distance, nonviolating-coordinate drift, filing-timing, and
date-cluster diagnostics without rebuilding recommendation candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from out_of_proxy_predictive_validity import fit_residualized_ols  # noqa: E402


DEFAULT_SELECTED_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "full_pipeline"
)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "main_32gb.yaml"
)


@dataclass(frozen=True)
class FastBounds:
    lower: np.ndarray
    upper: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_root", default=str(DEFAULT_SELECTED_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--out_dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "tables"
            / "boundary_timing_diagnostics"
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--demean_iter", type=int, default=12)
    parser.add_argument("--iqr_floor", type=float, default=1e-8)
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_action_columns(selected_root: Path) -> dict:
    with open(selected_root / "action_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def as_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    return df[list(cols)].apply(pd.to_numeric, errors="coerce")


def _quantile_table(
    df: pd.DataFrame,
    action_cols: list[str],
    q_low: float,
    q_high: float,
    group_col: str | None = None,
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    x = df[action_cols].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if group_col is None:
        return x.quantile(q_low).astype(float), x.quantile(q_high).astype(float)
    work = pd.concat([df[[group_col]].reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    return work.groupby(group_col)[action_cols].quantile(q_low), work.groupby(group_col)[action_cols].quantile(q_high)


def fast_exogenous_bounds(
    panel: pd.DataFrame,
    action_cols: list[str],
    asset_cols: list[str],
    overlay_cols: list[str],
    cfg: dict,
) -> FastBounds:
    """Vectorized selected exogenous bounds.

    Mirrors mandate_setirl.bounds.fit_mandate_bounds for mode="exogenous" when
    ADV soft caps are disabled in the selected specification.
    """

    bcfg = cfg.get("bounds", {})
    train = panel.loc[panel["split"].astype(str).eq("train")].copy()
    if train.empty:
        train = panel.copy()
    g_low, g_high = _quantile_table(
        train,
        action_cols,
        float(bcfg.get("global_lower_q", 0.005)),
        float(bcfg.get("global_upper_q", 0.995)),
    )
    for c in asset_cols:
        g_low[c] = max(0.0, min(float(g_low[c]), 0.25))
        g_high[c] = min(1.0, max(float(g_high[c]), 0.05))
    for c in overlay_cols:
        if c != "n_holdings_scaled":
            g_low[c] = min(float(g_low[c]), 0.0)
            g_high[c] = max(float(g_high[c]), 0.01)

    lower = np.tile(g_low.to_numpy(dtype=float), (len(panel), 1))
    upper = np.tile(g_high.to_numpy(dtype=float), (len(panel), 1))

    if "mandate_cluster" in panel.columns:
        p_low, p_high = _quantile_table(
            train.assign(mandate_cluster=train["mandate_cluster"]),
            action_cols,
            float(bcfg.get("peer_lower_q", 0.05)),
            float(bcfg.get("peer_upper_q", 0.99)),
            "mandate_cluster",
        )
        clusters = panel["mandate_cluster"].to_numpy()
        cluster_to_pos = {cl: np.flatnonzero(clusters == cl) for cl in np.unique(clusters)}
        for cl, pos in cluster_to_pos.items():
            if cl not in p_low.index:
                continue
            lower[pos] = np.maximum(
                lower[pos], p_low.loc[cl, action_cols].to_numpy(dtype=float)
            )
            upper[pos] = np.minimum(
                upper[pos], p_high.loc[cl, action_cols].to_numpy(dtype=float)
            )

    return clean_bounds(lower, upper, action_cols, asset_cols, cfg)


def clean_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    action_cols: list[str],
    asset_cols: list[str],
    cfg: dict,
) -> FastBounds:
    lower = np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.nan_to_num(upper, nan=1.0, posinf=1.0, neginf=1.0)
    min_width = float(cfg.get("bounds", {}).get("min_width", 0.01))
    asset_set = set(asset_cols)
    asset_idx = []
    for j, c in enumerate(action_cols):
        if c in asset_set:
            asset_idx.append(j)
            lower[:, j] = np.clip(lower[:, j], 0.0, 1.0)
            upper[:, j] = np.clip(upper[:, j], 0.0, 1.0)
        elif c == "n_holdings_scaled":
            lower[:, j] = np.clip(lower[:, j], -8.0, 8.0)
            upper[:, j] = np.clip(upper[:, j], -8.0, 8.0)
        else:
            lower[:, j] = np.clip(lower[:, j], -1.0, 2.0)
            upper[:, j] = np.clip(upper[:, j], -1.0, 2.0)

        bad = upper[:, j] < lower[:, j] + min_width
        mid = (upper[:, j] + lower[:, j]) / 2.0
        lower[bad, j] = mid[bad] - min_width / 2.0
        upper[bad, j] = mid[bad] + min_width / 2.0
        if c in asset_set:
            lower[:, j] = np.clip(lower[:, j], 0.0, 1.0)
            upper[:, j] = np.clip(upper[:, j], lower[:, j] + 1e-8, 1.0)

    for j, c in enumerate(action_cols):
        if c in asset_set:
            lower[:, j] = 0.0
        elif c != "n_holdings_scaled":
            lower[:, j] = np.minimum(lower[:, j], 0.0)

    if asset_idx:
        idx = np.asarray(asset_idx, dtype=int)
        upper_sum = upper[:, idx].sum(axis=1)
        bad = upper_sum < 1.0 - 1e-10
        if bad.any():
            scale = (1.0 + 1e-8) / np.maximum(upper_sum[bad], 1e-12)
            upper[np.ix_(bad, idx)] = np.minimum(1.0, upper[np.ix_(bad, idx)] * scale[:, None])
            still_bad = upper[:, idx].sum(axis=1) < 1.0 - 1e-8
            if still_bad.any():
                upper[np.ix_(still_bad, idx)] = 1.0
    return FastBounds(lower=lower, upper=upper)


def action_violations(actions: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float) -> np.ndarray:
    return ((actions < lower - tol) | (actions > upper + tol)).any(axis=-1)


def boundary_gap(actions: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum(lower - actions, 0.0) + np.maximum(actions - upper, 0.0)


def training_iqr_scale(actions: np.ndarray, train_mask: np.ndarray, floor: float) -> np.ndarray:
    q75 = np.nanquantile(actions[train_mask], 0.75, axis=0)
    q25 = np.nanquantile(actions[train_mask], 0.25, axis=0)
    iqr = q75 - q25
    return np.where(np.isfinite(iqr) & (iqr > floor), iqr, 1.0)


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


def build_regression_panel(
    panel: pd.DataFrame,
    actions: np.ndarray,
    lag_actions: np.ndarray,
    lead_actions: np.ndarray,
    gap: np.ndarray,
    scale_iqr: np.ndarray,
    tol: float,
    entity_col: str,
    date_col: str,
) -> pd.DataFrame:
    out = panel[[entity_col, "split", "mandate_cluster", date_col]].copy()
    next_valid = np.isfinite(lead_actions).all(axis=1)
    lag_valid = np.isfinite(lag_actions).all(axis=1)
    out["style_drift_l1"] = np.abs(actions - lag_actions).sum(axis=1)
    out.loc[~lag_valid, "style_drift_l1"] = np.nan
    out["lag_style_drift_l1"] = out.groupby(entity_col, sort=False)["style_drift_l1"].shift(1)
    out["next_style_drift_l1"] = np.abs(lead_actions - actions).sum(axis=1)
    out.loc[~next_valid, "next_style_drift_l1"] = np.nan
    out["exception"] = (gap.sum(axis=1) > tol).astype(int)
    out["distance_l1"] = gap.sum(axis=1)
    robust_scaled = (gap / scale_iqr).mean(axis=1)
    out["distance_iqr_mean"] = robust_scaled

    violation_coord = gap > tol
    next_drift = np.abs(lead_actions - actions)
    next_drift[violation_coord] = 0.0
    out["next_nonviolating_drift_l1"] = next_drift.sum(axis=1)
    out.loc[~next_valid, "next_nonviolating_drift_l1"] = np.nan
    current_nonviolating = np.abs(actions - lag_actions)
    current_nonviolating[violation_coord] = 0.0
    out["current_nonviolating_drift_l1"] = current_nonviolating.sum(axis=1)
    out.loc[~lag_valid, "current_nonviolating_drift_l1"] = np.nan
    out["lag_nonviolating_drift_l1"] = out.groupby(entity_col, sort=False)["current_nonviolating_drift_l1"].shift(1)

    for c in ["log_net_assets", "net_flow_3m_scaled", "lag_net_flow_3m_scaled"]:
        if c in panel.columns:
            out[c] = pd.to_numeric(panel[c], errors="coerce")

    if "filing_date" in panel.columns:
        out["filing_date"] = pd.to_datetime(panel["filing_date"], errors="coerce")
    else:
        out["filing_date"] = pd.NaT
    out["report_date"] = pd.to_datetime(panel[date_col], errors="coerce")
    out["next_report_date"] = out.groupby(entity_col, sort=False)["report_date"].shift(-1)
    out["filing_before_next_report"] = out["filing_date"].notna() & out["next_report_date"].notna() & (out["filing_date"] <= out["next_report_date"])
    out["filing_delay_days"] = (out["filing_date"] - out["report_date"]).dt.days
    out["filing_to_next_report_days"] = (out["next_report_date"] - out["filing_date"]).dt.days
    return out


def zscore_on_sample(df: pd.DataFrame, col: str, mask: pd.Series) -> str:
    vals = pd.to_numeric(df.loc[mask, col], errors="coerce")
    sd = float(vals.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise ValueError(f"Cannot z-score {col}: zero or invalid standard deviation")
    mean = float(vals.mean())
    zcol = f"{col}_z"
    df[zcol] = (pd.to_numeric(df[col], errors="coerce") - mean) / sd
    return zcol


def run_regression(
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
    focus = pd.to_numeric(df[regressor], errors="coerce").replace([np.inf, -np.inf], np.nan)
    res["positive_regressor_n"] = int((focus > 1e-7).sum())
    if not set(focus.dropna().unique()).issubset({0, 1}):
        res["out_of_proxy_n"] = np.nan
    return res


def main() -> None:
    args = parse_args()
    selected_root = Path(args.selected_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_yaml(Path(args.config))
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    date_col = cfg.get("keys", {}).get("date_col", "report_date")
    ac = load_action_columns(selected_root)
    action_cols = ac["action_cols"]
    asset_cols = ac["asset_action_cols"]
    overlay_cols = ac["overlay_action_cols"]

    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    actions_df = as_numeric(panel, action_cols)
    actions = actions_df.to_numpy(dtype=float)
    bounds = fast_exogenous_bounds(panel, action_cols, asset_cols, overlay_cols, cfg)
    gap = boundary_gap(actions, bounds.lower, bounds.upper)
    scale = training_iqr_scale(actions, panel["split"].astype(str).eq("train").to_numpy(), args.iqr_floor)
    lag_actions, lead_actions = lead_lag_actions(panel, actions_df, entity_col, date_col)
    reg = build_regression_panel(
        panel,
        actions,
        lag_actions,
        lead_actions,
        gap,
        scale,
        args.tol,
        entity_col,
        date_col,
    )
    split_mask = reg["split"].astype(str).eq(args.split)
    zcol = zscore_on_sample(reg, "distance_iqr_mean", split_mask)

    controls_main = [
        "style_drift_l1",
        "lag_style_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]
    controls_nonviol = [
        "current_nonviolating_drift_l1",
        "lag_nonviolating_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]
    rows = []
    test = reg.loc[split_mask].copy()
    rows.append(run_regression(test, "next_style_drift_l1", "exception", controls_main, entity_col, date_col, args.demean_iter, "binary_exception"))
    rows.append(run_regression(test, "next_style_drift_l1", zcol, controls_main, entity_col, date_col, args.demean_iter, "iqr_scaled_distance_per_sd"))
    rows.append(run_regression(test, "next_nonviolating_drift_l1", zcol, controls_nonviol, entity_col, date_col, args.demean_iter, "iqr_scaled_distance_to_nonviolating_drift"))
    available = test.loc[test["filing_before_next_report"].fillna(False)].copy()
    rows.append(run_regression(available, "next_style_drift_l1", "exception", controls_main, entity_col, date_col, args.demean_iter, "binary_exception_filing_before_next"))
    rows.append(run_regression(available, "next_style_drift_l1", zcol, controls_main, entity_col, date_col, args.demean_iter, "iqr_scaled_distance_filing_before_next"))

    reg_out = pd.DataFrame(rows)
    reg_out.to_csv(out_dir / "boundary_timing_regressions.csv", index=False)

    timing = {
        "n_test_rows": int(split_mask.sum()),
        "n_test_rows_with_next": int(test["next_report_date"].notna().sum()),
        "n_filing_before_next": int(test["filing_before_next_report"].fillna(False).sum()),
        "share_filing_before_next_among_next": float(
            test.loc[test["next_report_date"].notna(), "filing_before_next_report"].mean()
        ),
        "median_filing_delay_days": float(test["filing_delay_days"].median()),
        "median_filing_to_next_report_days": float(test["filing_to_next_report_days"].median()),
        "n_report_date_clusters_regression_sample": int(
            test.loc[test["next_style_drift_l1"].notna(), date_col].nunique()
        ),
        "n_series_regression_sample": int(
            test.loc[test["next_style_drift_l1"].notna(), entity_col].nunique()
        ),
        "out_of_proxy_rate_test": float(test["exception"].mean()),
        "distance_iqr_mean_sd_test": float(test["distance_iqr_mean"].std(ddof=0)),
    }
    pd.DataFrame([timing]).to_csv(out_dir / "filing_timing_and_sample_summary.csv", index=False)

    metric_summary = []
    for col in ["distance_l1", "distance_iqr_mean", zcol, "next_style_drift_l1", "next_nonviolating_drift_l1"]:
        vals = pd.to_numeric(test[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        metric_summary.append(
            {
                "metric": col,
                "n": int(vals.shape[0]),
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=0)),
                "p50": float(vals.quantile(0.50)),
                "p90": float(vals.quantile(0.90)),
                "p99": float(vals.quantile(0.99)),
            }
        )
    pd.DataFrame(metric_summary).to_csv(out_dir / "boundary_timing_metric_summary.csv", index=False)

    manifest = {
        "selected_root": str(selected_root),
        "config": str(Path(args.config)),
        "split": args.split,
        "bounds": "vectorized exogenous global plus peer quantile bounds matching selected config; ADV soft caps disabled",
        "coordinate_scaling": "training-split IQR with safe floor; coefficient uses the z-scored test-split distance",
        "standard_errors": "two-way clustered by fund series and report date",
        "fixed_effects": ["report_date", "mandate_cluster"],
        "notes": [
            "The script does not regenerate candidate menus.",
            "Filing restriction keeps reports whose current N-PORT filing date is on or before the next report date.",
            "Nonviolating drift excludes coordinates outside the proxy at the current report.",
        ],
    }
    with open(out_dir / "boundary_timing_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(reg_out[["spec", "outcome", "n", "beta", "two_way_cluster_t", "two_way_cluster_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
