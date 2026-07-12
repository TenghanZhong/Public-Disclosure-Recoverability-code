#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mandate_setirl.bounds import action_violations, fit_mandate_bounds


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "calibration_traceability"
DEFAULT_BOOTSTRAP_SPECS = {
    "baseline": PROJECT_ROOT / "outputs" / "full_pipeline",
    "selected_bounds": PROJECT_ROOT / "outputs" / "selected_bounds",
    "signed_exposure": PROJECT_ROOT / "outputs" / "signed_exposure",
}
DEFAULT_SELECTION_SPECS = {
    "baseline": PROJECT_ROOT / "outputs" / "full_pipeline",
    "eight_clusters": PROJECT_ROOT / "outputs" / "eight_clusters",
    "upper_quantile_0995": PROJECT_ROOT / "outputs" / "upper_quantile_0995",
    "selected_bounds": PROJECT_ROOT / "outputs" / "selected_bounds",
}
DEFAULT_SELECTED_ROOT = PROJECT_ROOT / "outputs" / "selected_bounds"


CORE_METRIC_MAP = {
    "coverage": "choice_coverage",
    "recommendation_coverage": "recommendation_choice_coverage",
    "mandate_violation": "point_pred_mandate_violation_ex_observed",
    "distance": "distance_to_set_l1",
    "asset_only_distance": "distance_to_set_asset_only_l1",
    "overlay_only_distance": "distance_to_set_overlay_only_l1",
    "risk_overlay_distance": "distance_to_set_risk_overlay_l1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build calibration, bootstrap, and traceability outputs.")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--selected_root", default=str(DEFAULT_SELECTED_ROOT))
    p.add_argument("--selected_config", default=str(PROJECT_ROOT / "configs" / "main_32gb.yaml"))
    p.add_argument("--primary_model", default="exogenous_polytope")
    p.add_argument("--skip_bootstrap", action="store_true")
    p.add_argument("--skip_audit", action="store_true")
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _metric_columns(df: pd.DataFrame) -> list[str]:
    out = [v for v in CORE_METRIC_MAP.values() if v in df.columns]
    for col in df.columns:
        if col.startswith("frontier_f") and (
            col.endswith("_choice_coverage")
            or col.endswith("_distance_to_set_l1")
            or col.endswith("_violation_rate")
        ):
            out.append(col)
    return list(dict.fromkeys(out))


def _friendly_metric_name(column: str) -> str:
    reverse = {v: k for k, v in CORE_METRIC_MAP.items()}
    if column in reverse:
        return reverse[column]
    if column.startswith("frontier_f"):
        return column.replace("frontier_", "frontier@")
    return column


def attach_panel_metadata(eval_df: pd.DataFrame, panel: pd.DataFrame, entity_col: str = "series_id") -> pd.DataFrame:
    if "row_pos" not in eval_df.columns:
        raise ValueError("evaluation rows must contain row_pos")
    if entity_col not in panel.columns:
        raise ValueError(f"panel is missing entity column {entity_col!r}")
    row_pos = pd.to_numeric(eval_df["row_pos"], errors="raise").astype(int).to_numpy()
    if row_pos.max(initial=-1) >= len(panel) or row_pos.min(initial=0) < 0:
        raise ValueError("row_pos values are outside panel bounds")
    out = eval_df.copy()
    panel_reset = panel.reset_index(drop=True)
    out[entity_col] = panel_reset[entity_col].astype(str).to_numpy()[row_pos]
    if "report_date" in panel_reset.columns:
        out["report_date"] = panel_reset["report_date"].to_numpy()[row_pos]
    if "mandate_cluster" in panel_reset.columns:
        out["mandate_cluster"] = panel_reset["mandate_cluster"].to_numpy()[row_pos]
    return out


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    cluster_col: str,
    model_col: str,
    metric_cols: list[str],
    primary_model: str,
    baseline_models: list[str],
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters = pd.Index(sorted(df[cluster_col].dropna().astype(str).unique()))
    if clusters.empty:
        raise ValueError("No bootstrap clusters found")
    cluster_to_pos = {c: i for i, c in enumerate(clusters)}
    sample_idx = np.random.default_rng(seed).integers(0, len(clusters), size=(n_boot, len(clusters)))

    summaries: list[dict[str, Any]] = []
    draws_by_model_metric: dict[tuple[str, str], np.ndarray] = {}
    point_by_model_metric: dict[tuple[str, str], float] = {}
    models = sorted(df[model_col].dropna().astype(str).unique())

    work = df[[cluster_col, model_col] + metric_cols].copy()
    work[cluster_col] = work[cluster_col].astype(str)
    work[model_col] = work[model_col].astype(str)

    for model in models:
        mdf = work[work[model_col].eq(model)]
        for col in metric_cols:
            vals = pd.to_numeric(mdf[col], errors="coerce")
            valid = vals.notna()
            if not valid.any():
                continue
            tmp = pd.DataFrame({
                cluster_col: mdf.loc[valid, cluster_col].to_numpy(),
                "value": vals.loc[valid].to_numpy(dtype=float),
            })
            grouped = tmp.groupby(cluster_col)["value"].agg(["sum", "count"])
            sums = np.zeros(len(clusters), dtype=float)
            counts = np.zeros(len(clusters), dtype=float)
            for cluster_id, row in grouped.iterrows():
                pos = cluster_to_pos.get(str(cluster_id))
                if pos is not None:
                    sums[pos] = float(row["sum"])
                    counts[pos] = float(row["count"])
            point = float(sums.sum() / counts.sum()) if counts.sum() else np.nan
            boot_num = sums[sample_idx].sum(axis=1)
            boot_den = counts[sample_idx].sum(axis=1)
            draws = np.divide(boot_num, boot_den, out=np.full(n_boot, np.nan), where=boot_den > 0)
            metric_name = _friendly_metric_name(col)
            summaries.append({
                "model": model,
                "metric": metric_name,
                "source_column": col,
                "point_estimate": point,
                "bootstrap_mean": float(np.nanmean(draws)),
                "bootstrap_se": float(np.nanstd(draws, ddof=1)),
                "ci_low": float(np.nanquantile(draws, 0.025)),
                "ci_high": float(np.nanquantile(draws, 0.975)),
                "n_boot": int(n_boot),
                "n_clusters": int(len(clusters)),
                "n_obs_nonmissing": int(counts.sum()),
            })
            draws_by_model_metric[(model, metric_name)] = draws
            point_by_model_metric[(model, metric_name)] = point

    diffs: list[dict[str, Any]] = []
    for baseline in baseline_models:
        for metric_name in sorted({m for _, m in draws_by_model_metric}):
            key_main = (primary_model, metric_name)
            key_base = (baseline, metric_name)
            if key_main not in draws_by_model_metric or key_base not in draws_by_model_metric:
                continue
            draws = draws_by_model_metric[key_main] - draws_by_model_metric[key_base]
            point = point_by_model_metric[key_main] - point_by_model_metric[key_base]
            diffs.append({
                "comparison": f"{primary_model}_minus_{baseline}",
                "primary_model": primary_model,
                "baseline_model": baseline,
                "metric": metric_name,
                "point_estimate": float(point),
                "bootstrap_mean": float(np.nanmean(draws)),
                "bootstrap_se": float(np.nanstd(draws, ddof=1)),
                "ci_low": float(np.nanquantile(draws, 0.025)),
                "ci_high": float(np.nanquantile(draws, 0.975)),
                "n_boot": int(n_boot),
                "n_clusters": int(len(clusters)),
            })
    return pd.DataFrame(summaries), pd.DataFrame(diffs)


def run_bootstrap_for_root(
    spec_name: str,
    root: Path,
    out_dir: Path,
    n_boot: int,
    seed: int,
    primary_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_path = root / "evaluation_row_level.parquet"
    panel_path = root / "panel_model_ready.parquet"
    if not eval_path.exists() or not panel_path.exists():
        raise FileNotFoundError(f"Missing evaluation or panel parquet under {root}")
    eval_df = pd.read_parquet(eval_path)
    panel = pd.read_parquet(panel_path)
    eval_df = attach_panel_metadata(eval_df, panel)
    eval_df = eval_df[eval_df["split"].astype(str).eq("test")].copy()
    metric_cols = _metric_columns(eval_df)
    baselines = [m for m in sorted(eval_df["model"].astype(str).unique()) if m != primary_model]
    summary, diffs = cluster_bootstrap_ci(
        eval_df,
        cluster_col="series_id",
        model_col="model",
        metric_cols=metric_cols,
        primary_model=primary_model,
        baseline_models=baselines,
        n_boot=n_boot,
        seed=seed,
    )
    summary.insert(0, "spec", spec_name)
    summary.insert(1, "root", str(root))
    diffs.insert(0, "spec", spec_name)
    diffs.insert(1, "root", str(root))
    spec_dir = out_dir / "bootstrap_by_spec" / spec_name
    spec_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(spec_dir / "bootstrap_ci_model_metrics.csv", index=False)
    diffs.to_csv(spec_dir / "bootstrap_ci_model_differences.csv", index=False)
    del eval_df, panel
    gc.collect()
    return summary, diffs


def build_calibration_selection_table(selection_specs: dict[str, Path], out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, root in selection_specs.items():
        path = root / "main_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["model"].astype(str).eq("exogenous_polytope") & df["split"].astype(str).isin(["calib", "test"])].copy()
        for _, row in df.iterrows():
            d = row.to_dict()
            d.update({"spec": name, "root": str(root)})
            rows.append(d)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No calibration selection metrics found")
    calib = table[table["split"].astype(str).eq("calib")].copy()
    violation_col = "point_pred_mandate_violation_rate_ex_observed"
    calib["_valid_zero_violation"] = pd.to_numeric(calib[violation_col], errors="coerce").fillna(1.0) <= 1e-12
    calib["_selection_rank"] = np.nan
    valid = calib[calib["_valid_zero_violation"]].copy()
    sort_cols = ["observed_feasible_under_bounds", "choice_coverage", "distance_to_set_l1_mean"]
    valid["_neg_distance"] = -pd.to_numeric(valid["distance_to_set_l1_mean"], errors="coerce")
    valid = valid.sort_values(
        ["observed_feasible_under_bounds", "choice_coverage", "_neg_distance"],
        ascending=[False, False, False],
    )
    selected = valid.iloc[0]["spec"] if not valid.empty else None
    table["selected_by_calibration_rule"] = table["spec"].eq(selected)
    table["selection_rule"] = (
        "calibration split only: require zero point-prediction mandate violation; "
        "choose highest observed feasible-under-bounds rate; break ties by coverage and then lower distance"
    )
    table.to_csv(out_dir / "calibration_selection_grid.csv", index=False)
    md = [
        "# Calibration selection",
        "",
        "The selected specification is chosen using the calibration split only.",
        "Rule: require zero mandate violation for the exogenous model; choose the candidate with the highest observed feasible-under-bounds rate; use coverage and lower distance only as tie-breakers.",
        "",
        f"Selected spec: {selected}",
        "",
        "The rule prioritizes feasible-set coverage before recommendation metrics.",
    ]
    (out_dir / "calibration_selection_rule.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return table


def preference_intervals_long(pref: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    point_cols = [c for c in pref.columns if c.startswith("theta_point_")]
    for _, row in pref.iterrows():
        base = {
            "model": row.get("model", ""),
            "scope_id": row.get("scope_id", ""),
            "scope_type": row.get("scope_type", ""),
            "n_train_obs": row.get("n_train_obs", np.nan),
            "n_constraints": row.get("n_constraints", np.nan),
            "eps_bound": row.get("eps_bound", np.nan),
            "support_thetas": row.get("support_thetas", np.nan),
            "theta_interval_l1_upper_diameter": row.get("theta_interval_l1_upper_diameter", np.nan),
        }
        for col in point_cols:
            feature = col[len("theta_point_") :]
            rows.append({
                **base,
                "feature": feature,
                "theta_point": row.get(f"theta_point_{feature}", np.nan),
                "theta_min": row.get(f"theta_min_{feature}", np.nan),
                "theta_max": row.get(f"theta_max_{feature}", np.nan),
                "theta_width": row.get(f"theta_width_{feature}", np.nan),
            })
    return pd.DataFrame(rows)


def binding_constraint_summary(
    actions: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    action_cols: list[str],
    near_frac: float = 0.01,
    tol: float = 1e-7,
) -> pd.DataFrame:
    X = actions[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rows: list[dict[str, Any]] = []
    for j, col in enumerate(action_cols):
        width = np.maximum(upper[:, j] - lower[:, j], tol)
        near_tol = np.maximum(tol, near_frac * width)
        low_slack = X[:, j] - lower[:, j]
        up_slack = upper[:, j] - X[:, j]
        lower_viol = low_slack < -tol
        upper_viol = up_slack < -tol
        rows.append({
            "action_col": col,
            "n": int(len(X)),
            "lower_violation_n": int(lower_viol.sum()),
            "upper_violation_n": int(upper_viol.sum()),
            "any_violation_n": int((lower_viol | upper_viol).sum()),
            "lower_violation_share": float(lower_viol.mean()) if len(X) else np.nan,
            "upper_violation_share": float(upper_viol.mean()) if len(X) else np.nan,
            "any_violation_share": float((lower_viol | upper_viol).mean()) if len(X) else np.nan,
            "lower_binding_share": float(((low_slack >= -tol) & (low_slack <= near_tol)).mean()) if len(X) else np.nan,
            "upper_binding_share": float(((up_slack >= -tol) & (up_slack <= near_tol)).mean()) if len(X) else np.nan,
            "avg_lower_slack": float(np.nanmean(low_slack)) if len(X) else np.nan,
            "avg_upper_slack": float(np.nanmean(up_slack)) if len(X) else np.nan,
            "avg_width": float(np.nanmean(width)) if len(X) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["any_violation_n", "upper_binding_share"], ascending=[False, False])


def _row_constraint_explanation(
    action: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    action_cols: list[str],
    tol: float = 1e-7,
    max_items: int = 5,
) -> tuple[str, str]:
    violations = []
    bindings = []
    width = np.maximum(upper - lower, tol)
    for col, val, lo, hi, wid in zip(action_cols, action, lower, upper, width):
        if val < lo - tol:
            violations.append((col, f"{col}: below lower by {lo - val:.4g}"))
        elif val > hi + tol:
            violations.append((col, f"{col}: above upper by {val - hi:.4g}"))
        else:
            low_slack = val - lo
            up_slack = hi - val
            if min(low_slack, up_slack) <= max(tol, 0.01 * wid):
                side = "lower" if low_slack <= up_slack else "upper"
                slack = min(low_slack, up_slack)
                bindings.append((col, f"{col}: near {side} bound, slack {slack:.4g}"))
    return "; ".join(x[1] for x in violations[:max_items]), "; ".join(x[1] for x in bindings[:max_items])


def _scope_top_preferences(pref_long: pd.DataFrame, scope_id: Any, max_items: int = 5) -> str:
    scoped = pref_long[pref_long["scope_id"].astype(str).eq(str(scope_id))].copy()
    if scoped.empty:
        return ""
    scoped["abs_theta_point"] = pd.to_numeric(scoped["theta_point"], errors="coerce").abs()
    scoped = scoped.sort_values(["abs_theta_point", "theta_width"], ascending=[False, False]).head(max_items)
    return "; ".join(
        f"{r.feature}: [{float(r.theta_min):.3g}, {float(r.theta_max):.3g}], point {float(r.theta_point):.3g}"
        for r in scoped.itertuples(index=False)
        if pd.notna(r.theta_point)
    )


def build_auditability_evidence(root: Path, config_path: Path, out_dir: Path, primary_model: str) -> None:
    audit_dir = out_dir / "auditability"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cfg = _read_yaml(config_path)
    panel = pd.read_parquet(root / "panel_model_ready.parquet")
    action_meta = _read_json(root / "action_columns.json")
    action_cols = list(action_meta["action_cols"])
    asset_cols = list(action_meta.get("asset_action_cols", []))
    overlay_cols = list(action_meta.get("overlay_action_cols", []))
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")

    test_mask = panel["split"].astype(str).eq("test").to_numpy()
    actions = panel[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    test_binding = binding_constraint_summary(actions.loc[test_mask], bounds.lower[test_mask], bounds.upper[test_mask], action_cols)
    test_binding.to_csv(audit_dir / "binding_constraints_test.csv", index=False)

    observed_viol = action_violations(actions.to_numpy(dtype=float), bounds.lower, bounds.upper)
    infeasible_mask = test_mask & observed_viol
    if infeasible_mask.any():
        drivers = binding_constraint_summary(actions.loc[infeasible_mask], bounds.lower[infeasible_mask], bounds.upper[infeasible_mask], action_cols)
        drivers["driver_share_among_infeasible_rows"] = drivers["any_violation_n"] / max(int(infeasible_mask.sum()), 1)
    else:
        drivers = pd.DataFrame(columns=list(test_binding.columns) + ["driver_share_among_infeasible_rows"])
    drivers.to_csv(audit_dir / "violation_drivers_observed_infeasible.csv", index=False)

    pref_path = root / f"scope_preference_summary_{primary_model}.csv"
    pref = pd.read_csv(pref_path)
    pref_long = preference_intervals_long(pref)
    pref_long.to_csv(audit_dir / "preference_intervals_long.csv", index=False)
    pref_summary = (
        pref_long.groupby("feature")
        .agg(
            n_scopes=("scope_id", "nunique"),
            theta_point_mean=("theta_point", "mean"),
            theta_min_mean=("theta_min", "mean"),
            theta_max_mean=("theta_max", "mean"),
            theta_width_mean=("theta_width", "mean"),
            theta_width_median=("theta_width", "median"),
            theta_width_max=("theta_width", "max"),
        )
        .reset_index()
        .sort_values("theta_width_mean", ascending=False)
    )
    pref_summary.to_csv(audit_dir / "preference_interval_summary.csv", index=False)

    eval_df = pd.read_parquet(root / "evaluation_row_level.parquet")
    eval_df = attach_panel_metadata(eval_df, panel)
    ev = eval_df[eval_df["split"].astype(str).eq("test") & eval_df["model"].astype(str).eq(primary_model)].copy()
    ev["observed_feasible_under_bounds"] = ev["observed_feasible_under_bounds"].astype(bool)
    ev["choice_coverage"] = ev["choice_coverage"].astype(bool)
    ev["distance_to_set_l1"] = pd.to_numeric(ev["distance_to_set_l1"], errors="coerce")

    case_specs: list[tuple[str, pd.DataFrame, str]] = [
        ("covered_and_feasible", ev[ev["observed_feasible_under_bounds"] & ev["choice_coverage"]].sort_values("distance_to_set_l1"), "Observed action is feasible and rationalized by the learned set."),
        ("preference_rejected_feasible", ev[ev["observed_feasible_under_bounds"] & ~ev["choice_coverage"]].sort_values("distance_to_set_l1", ascending=False), "Observed action is mandate-feasible but outside the learned preference set."),
        ("mandate_infeasible", ev[~ev["observed_feasible_under_bounds"]].sort_values("distance_to_set_l1", ascending=False), "Observed action violates the exogenous mandate bounds."),
    ]
    panel_reset = panel.reset_index(drop=True)
    case_rows: list[dict[str, Any]] = []
    for case_type, candidates, interpretation in case_specs:
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        row_pos = int(row["row_pos"])
        action = actions.iloc[row_pos].to_numpy(dtype=float)
        violations, bindings = _row_constraint_explanation(action, bounds.lower[row_pos], bounds.upper[row_pos], action_cols)
        case_rows.append({
            "case_type": case_type,
            "interpretation": interpretation,
            "row_pos": row_pos,
            "series_id": panel_reset.loc[row_pos, "series_id"],
            "series_name": panel_reset.loc[row_pos, "series_name"] if "series_name" in panel_reset.columns else "",
            "report_date": panel_reset.loc[row_pos, "report_date"] if "report_date" in panel_reset.columns else "",
            "scope_id": row.get("scope_id", ""),
            "mandate_cluster": panel_reset.loc[row_pos, "mandate_cluster"] if "mandate_cluster" in panel_reset.columns else "",
            "choice_coverage": bool(row["choice_coverage"]),
            "observed_feasible_under_bounds": bool(row["observed_feasible_under_bounds"]),
            "distance_to_set_l1": row.get("distance_to_set_l1", np.nan),
            "asset_only_distance": row.get("distance_to_set_asset_only_l1", np.nan),
            "overlay_only_distance": row.get("distance_to_set_overlay_only_l1", np.nan),
            "risk_overlay_distance": row.get("distance_to_set_risk_overlay_l1", np.nan),
            "constraint_violations": violations,
            "near_binding_constraints": bindings,
            "scope_top_preference_intervals": _scope_top_preferences(pref_long, row.get("scope_id", "")),
        })
    cases = pd.DataFrame(case_rows)
    cases.to_csv(audit_dir / "auditability_case_studies.csv", index=False)

    model_cols = [
        "choice_coverage",
        "observed_feasible_under_bounds",
        "point_pred_mandate_violation_ex_observed",
        "action_set_violation_rate",
        "recommendation_action_set_violation_rate_ex_observed",
        "distance_to_set_l1",
        "distance_to_set_asset_only_l1",
        "distance_to_set_overlay_only_l1",
    ]
    existing = [c for c in model_cols if c in eval_df.columns]
    model_summary = (
        eval_df[eval_df["split"].astype(str).eq("test")]
        .groupby("model")[existing]
        .mean(numeric_only=True)
        .reset_index()
    )
    model_summary.to_csv(audit_dir / "model_level_violation_and_distance_summary.csv", index=False)

    lines = [
        "# Auditability evidence",
        "",
        "The analysis uses saved row-level outputs and reconstructed exogenous mandate bounds.",
        "Candidate recommendation actions are not persisted dimension-by-dimension, so dimension-level drivers are reported for observed action feasibility; model-level recommendation violation rates are reported from evaluation outputs.",
        "",
        "Key files:",
        "- preference_intervals_long.csv",
        "- preference_interval_summary.csv",
        "- binding_constraints_test.csv",
        "- violation_drivers_observed_infeasible.csv",
        "- auditability_case_studies.csv",
        "- model_level_violation_and_distance_summary.csv",
    ]
    (audit_dir / "README_auditability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "primary_model": args.primary_model,
        "bootstrap_specs": {k: str(v) for k, v in DEFAULT_BOOTSTRAP_SPECS.items()},
        "selection_specs": {k: str(v) for k, v in DEFAULT_SELECTION_SPECS.items()},
        "selected_root": args.selected_root,
        "selected_config": args.selected_config,
    }
    (out_dir / "calibration_traceability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.skip_bootstrap:
        all_summary = []
        all_diffs = []
        for i, (spec_name, root) in enumerate(DEFAULT_BOOTSTRAP_SPECS.items()):
            summary, diffs = run_bootstrap_for_root(
                spec_name=spec_name,
                root=root,
                out_dir=out_dir,
                n_boot=args.n_boot,
                seed=args.seed + i,
                primary_model=args.primary_model,
            )
            all_summary.append(summary)
            all_diffs.append(diffs)
        summary_df = pd.concat(all_summary, ignore_index=True)
        diff_df = pd.concat(all_diffs, ignore_index=True)
        summary_df.to_csv(out_dir / "bootstrap_ci_model_metrics.csv", index=False)
        diff_df.to_csv(out_dir / "bootstrap_ci_model_differences.csv", index=False)
        summary_df[summary_df["metric"].isin(CORE_METRIC_MAP.keys())].to_csv(out_dir / "bootstrap_ci_core_metrics.csv", index=False)
        summary_df[summary_df["metric"].astype(str).str.startswith("frontier@")].to_csv(out_dir / "bootstrap_ci_frontier_metrics.csv", index=False)

    selection = build_calibration_selection_table(DEFAULT_SELECTION_SPECS, out_dir)

    if not args.skip_audit:
        build_auditability_evidence(
            root=Path(args.selected_root),
            config_path=Path(args.selected_config),
            out_dir=out_dir,
            primary_model=args.primary_model,
        )

    print(f"Calibration and traceability outputs saved to: {out_dir}")
    print("Calibration selection:")
    cols = [
        "spec",
        "split",
        "model",
        "choice_coverage",
        "observed_feasible_under_bounds",
        "point_pred_mandate_violation_rate_ex_observed",
        "distance_to_set_l1_mean",
        "selected_by_calibration_rule",
    ]
    print(selection[[c for c in cols if c in selection.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
