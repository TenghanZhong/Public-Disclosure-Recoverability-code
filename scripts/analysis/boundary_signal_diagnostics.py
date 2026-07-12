"""Additional diagnostics for the mandate-proxy excursion signal.

The script reuses the same selected-bound constructor as Table 1, then runs
three compact checks:

1. fund-series fixed effects;
2. lead-lag symmetry for future, current, and past drift;
3. test-split top-decile lift for next-report drift.
"""

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
from out_of_proxy_predictive_validity import fit_residualized_ols  # noqa: E402
from run_boundary_regressions import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUT_DIR,
    DEFAULT_PACKAGE_ROOT,
    DEFAULT_SELECTED_ROOT,
)
from boundary_inference_checks import (  # noqa: E402
    add_package,
    as_numeric,
    build_regression_panel,
    load_action_columns,
    zscore_on_mask,
)


DEFAULT_OUT = DEFAULT_OUT_DIR.parent / "boundary_signal_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_root", default=str(DEFAULT_SELECTED_ROOT))
    parser.add_argument("--package_root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--split", default="test")
    parser.add_argument("--bound_mode", default="exogenous")
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--iqr_floor", type=float, default=1e-8)
    parser.add_argument("--demean_iter", type=int, default=12)
    parser.add_argument("--top_share", type=float, default=0.10)
    return parser.parse_args()


def _cluster_controls(data: pd.DataFrame, controls: list[str]) -> list[str]:
    return [c for c in controls if c in data.columns]


def _fit(
    data: pd.DataFrame,
    outcome: str,
    focus: str,
    controls: list[str],
    fe_cols: list[str],
    entity_col: str,
    date_col: str,
    demean_iter: int,
) -> dict:
    work = data.copy()
    work["out_of_proxy"] = pd.to_numeric(work[focus], errors="coerce")
    return fit_residualized_ols(
        work,
        outcome,
        controls=_cluster_controls(work, [c for c in controls if c != focus]),
        fe_cols=fe_cols,
        entity_col=entity_col,
        date_col=date_col,
        n_iter=demean_iter,
    )


def _top_decile_lift(
    data: pd.DataFrame,
    score: str,
    outcome: str = "next_style_drift_l1",
    top_share: float = 0.10,
) -> dict:
    cols = [score, outcome]
    work = data[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) == 0:
        return {
            "score": score,
            "n": 0,
            "top_n": 0,
            "top_share": top_share,
            "precision_top_decile_outcome": np.nan,
            "baseline_top_decile_rate": np.nan,
            "lift": np.nan,
            "mean_outcome_top_score": np.nan,
            "mean_outcome_all": np.nan,
        }
    k = max(1, int(np.ceil(top_share * len(work))))
    threshold = work[outcome].quantile(1.0 - top_share)
    work["high_outcome"] = work[outcome] >= threshold
    ranked = work.sort_values(score, ascending=False, kind="mergesort").head(k)
    precision = float(ranked["high_outcome"].mean())
    baseline = float(work["high_outcome"].mean())
    return {
        "score": score,
        "n": int(len(work)),
        "top_n": int(k),
        "top_share": top_share,
        "outcome_top_decile_threshold": float(threshold),
        "precision_top_decile_outcome": precision,
        "baseline_top_decile_rate": baseline,
        "lift": float(precision / baseline) if baseline > 0 else np.nan,
        "mean_outcome_top_score": float(ranked[outcome].mean()),
        "mean_outcome_all": float(work[outcome].mean()),
        "mean_ratio": float(ranked[outcome].mean() / work[outcome].mean())
        if float(work[outcome].mean()) != 0
        else np.nan,
    }


def build_selected_test_panel(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    selected_root = Path(args.selected_root)
    package_root = Path(args.package_root)
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

    reg, _meta = build_regression_panel(
        panel,
        action_cols,
        bounds.lower,
        bounds.upper,
        entity_col,
        date_col,
        args.tol,
        args.iqr_floor,
    )
    boundary = compute_boundary_metrics(actions, bounds.lower, bounds.upper, tol=args.tol)
    extremeness = cluster_extremeness_metrics(panel, action_cols, cluster_col="mandate_cluster")
    for frame in [boundary, extremeness]:
        for col in frame.columns:
            reg[col] = frame[col].to_numpy()
    reg["exception"] = (gap.sum(axis=1) > args.tol).astype(int)
    reg["outside_proxy_distance_l1"] = gap.sum(axis=1)

    split_mask = reg["split"].astype(str).eq(args.split)
    scaled_col = zscore_on_mask(reg, "outside_proxy_distance_iqr_mean", split_mask)
    test = reg.loc[split_mask].copy()
    meta = {
        "selected_root": str(selected_root),
        "package_root": str(package_root),
        "config": str(Path(args.config)),
        "bound_mode": args.bound_mode,
        "split": args.split,
        "panel_sorted_by_series_date": sorted_ok,
        "entity_col": entity_col,
        "date_col": date_col,
        "scaled_distance_col": scaled_col,
        "n_split": int(len(test)),
        "n_series_split": int(test[entity_col].nunique()),
        "n_report_dates_split": int(test[date_col].nunique()),
        "outside_proxy_share_split": float(test["exception"].mean()),
    }
    return test, meta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test, meta = build_selected_test_panel(args)
    entity_col = meta["entity_col"]
    date_col = meta["date_col"]
    scaled_col = meta["scaled_distance_col"]
    common = [
        "style_drift_l1",
        "lag_style_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]

    regression_rows = []
    specs = [
        (
            "preferred_cluster_date_fe",
            "next_style_drift_l1",
            "exception",
            common,
            [date_col, "mandate_cluster"],
        ),
        (
            "fund_series_and_date_fe_binary",
            "next_style_drift_l1",
            "exception",
            common,
            [date_col, entity_col],
        ),
        (
            "fund_series_and_date_fe_scaled_distance",
            "next_style_drift_l1",
            scaled_col,
            common,
            [date_col, entity_col],
        ),
        (
            "lead_lag_future_scaled_distance",
            "next_style_drift_l1",
            scaled_col,
            common,
            [date_col, "mandate_cluster"],
        ),
        (
            "lead_lag_current_scaled_distance",
            "style_drift_l1",
            scaled_col,
            ["lag_style_drift_l1", "log_net_assets", "net_flow_3m_scaled", "lag_net_flow_3m_scaled"],
            [date_col, "mandate_cluster"],
        ),
        (
            "lead_lag_past_scaled_distance",
            "lag_style_drift_l1",
            scaled_col,
            ["log_net_assets", "net_flow_3m_scaled", "lag_net_flow_3m_scaled"],
            [date_col, "mandate_cluster"],
        ),
    ]
    for name, outcome, focus, controls, fe_cols in specs:
        res = _fit(
            test,
            outcome,
            focus,
            controls,
            fe_cols,
            entity_col,
            date_col,
            args.demean_iter,
        )
        regression_rows.append(
            {
                "diagnostic": name,
                "outcome": outcome,
                "focus": focus,
                "fixed_effects": ",".join(fe_cols),
                "n": int(res.get("n", 0)),
                "coef": float(res.get("beta", np.nan)),
                "two_way_cluster_t": float(res.get("two_way_cluster_t", np.nan)),
                "two_way_cluster_p": float(res.get("two_way_cluster_p", np.nan)),
                "series_cluster_t": float(res.get("series_cluster_t", np.nan)),
                "controls": res.get("controls", ""),
            }
        )

    lift_scores = [
        scaled_col,
        "exception",
        "lag_style_drift_l1",
        "cluster_center_l1",
        "cluster_z_abs_mean",
    ]
    lift_rows = [
        _top_decile_lift(test, score=s, top_share=args.top_share)
        for s in lift_scores
        if s in test.columns
    ]

    pd.DataFrame(regression_rows).to_csv(out_dir / "fund_fe_leadlag_diagnostics.csv", index=False)
    pd.DataFrame(lift_rows).to_csv(out_dir / "top_decile_lift_diagnostics.csv", index=False)
    manifest = dict(meta)
    manifest["top_share"] = args.top_share
    with open(out_dir / "boundary_signal_diagnostics_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print("\nREGRESSIONS")
    print(pd.DataFrame(regression_rows).to_string(index=False))
    print("\nTOP DECILE LIFT")
    print(pd.DataFrame(lift_rows).to_string(index=False))


if __name__ == "__main__":
    main()
