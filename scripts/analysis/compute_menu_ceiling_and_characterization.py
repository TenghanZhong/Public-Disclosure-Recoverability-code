#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _load_core_modules(code_root: Path):
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from mandate_setirl.bounds import action_violations, fit_mandate_bounds
    from mandate_setirl.candidates import CandidateGenerator

    return action_violations, fit_mandate_bounds, CandidateGenerator


def _pct(x: float) -> float:
    return 100.0 * float(x)


def _fmt_pct_ci(mean: float, lo: float, hi: float, digits: int = 2) -> str:
    return f"{_pct(mean):.{digits}f} [{_pct(lo):.{digits}f}, {_pct(hi):.{digits}f}]"


def _fmt_num_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _bootstrap_metric(
    data: pd.DataFrame,
    group_col: str,
    metric_fn,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    grouped = [g.copy() for _, g in data.groupby(group_col, sort=False)]
    point = metric_fn(data)
    if n_boot <= 0 or len(grouped) == 0:
        return point, {k: (np.nan, np.nan) for k in point}
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {k: [] for k in point}
    n = len(grouped)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = pd.concat([grouped[i] for i in idx], ignore_index=True)
        row = metric_fn(sample)
        for k, v in row.items():
            vals[k].append(float(v))
    intervals = {
        k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975)))
        for k, v in vals.items()
    }
    return point, intervals


def _bootstrap_oracle_fast(data: pd.DataFrame, n_boot: int, seed: int) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    point = _oracle_metric(data)
    if n_boot <= 0:
        return point, {k: (np.nan, np.nan) for k in point}
    g = data.assign(
        oracle_sum=data["oracle_coverable"].astype(float),
        rp20_sum=data["rp_f20_covered"].astype(float),
        rp50_sum=data["rp_f50_covered"].astype(float),
        min_l1_sum=pd.to_numeric(data["oracle_min_l1"], errors="coerce").fillna(0.0),
        min_l1_count=np.isfinite(pd.to_numeric(data["oracle_min_l1"], errors="coerce")).astype(float),
        n_row=1.0,
    ).groupby("series_id", sort=False)[["n_row", "oracle_sum", "rp20_sum", "rp50_sum", "min_l1_sum", "min_l1_count"]].sum()
    arr = g.to_numpy(dtype=float)
    n_series = arr.shape[0]
    rng = np.random.default_rng(seed)
    vals: dict[str, list[float]] = {k: [] for k in point if k != "oracle_min_l1_median"}
    for _ in range(n_boot):
        sample = arr[rng.integers(0, n_series, size=n_series)].sum(axis=0)
        n_row, oracle_sum, rp20_sum, rp50_sum, min_l1_sum, min_l1_count = sample
        oracle = oracle_sum / n_row
        rp20 = rp20_sum / n_row
        rp50 = rp50_sum / n_row
        vals["oracle_coverage"].append(oracle)
        vals["rp_f20_coverage"].append(rp20)
        vals["rp_f50_coverage"].append(rp50)
        vals["conditional_hit_f20"].append(rp20 / oracle if oracle > 0 else np.nan)
        vals["conditional_hit_f50"].append(rp50 / oracle if oracle > 0 else np.nan)
        vals["oracle_min_l1_mean"].append(min_l1_sum / min_l1_count if min_l1_count > 0 else np.nan)
    intervals = {k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))) for k, v in vals.items()}
    intervals["oracle_min_l1_median"] = (np.nan, np.nan)
    return point, intervals


def _bootstrap_feature_diff_fast(data: pd.DataFrame, feature: str, n_boot: int, seed: int) -> tuple[dict[str, float], tuple[float, float]]:
    point = _category_metric(data, feature)
    if n_boot <= 0:
        return point, (np.nan, np.nan)
    d = data[data["churn_category"].isin(["set_only", "point_only"])].copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d["set_count"] = d["churn_category"].eq("set_only").astype(float)
    d["point_count"] = d["churn_category"].eq("point_only").astype(float)
    d["set_sum"] = d[feature].where(d["churn_category"].eq("set_only"), 0.0)
    d["point_sum"] = d[feature].where(d["churn_category"].eq("point_only"), 0.0)
    g = d.groupby("series_id", sort=False)[["set_count", "point_count", "set_sum", "point_sum"]].sum()
    arr = g.to_numpy(dtype=float)
    n_series = arr.shape[0]
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        set_count, point_count, set_sum, point_sum = arr[rng.integers(0, n_series, size=n_series)].sum(axis=0)
        vals.append((set_sum / set_count) - (point_sum / point_count) if set_count > 0 and point_count > 0 else np.nan)
    return point, (float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975)))


def _oracle_metric(df: pd.DataFrame) -> dict[str, float]:
    oracle = float(df["oracle_coverable"].mean())
    rp20 = float(df["rp_f20_covered"].mean())
    rp50 = float(df["rp_f50_covered"].mean())
    return {
        "oracle_coverage": oracle,
        "rp_f20_coverage": rp20,
        "rp_f50_coverage": rp50,
        "conditional_hit_f20": rp20 / oracle if oracle > 0 else np.nan,
        "conditional_hit_f50": rp50 / oracle if oracle > 0 else np.nan,
        "oracle_min_l1_mean": float(df["oracle_min_l1"].mean()),
        "oracle_min_l1_median": float(df["oracle_min_l1"].median()),
    }


def _category_metric(df: pd.DataFrame, feature: str) -> dict[str, float]:
    rp = df[df["churn_category"].eq("set_only")]
    pt = df[df["churn_category"].eq("point_only")]
    return {
        f"{feature}_set_only": float(rp[feature].mean()),
        f"{feature}_point_only": float(pt[feature].mean()),
        f"{feature}_diff": float(rp[feature].mean() - pt[feature].mean()),
    }


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _make_row_diagnostics(
    *,
    cfg: dict[str, Any],
    panel: pd.DataFrame,
    eval_df: pd.DataFrame,
    action_cols: list[str],
    asset_cols: list[str],
    overlay_cols: list[str],
    entity_col: str,
    code_root: Path,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    action_violations, fit_mandate_bounds, CandidateGenerator = _load_core_modules(code_root)

    tau = float(
        cfg.get("evaluation", {}).get(
            "coverage_l1_tolerance",
            cfg.get("evaluation", {}).get("recommendation_hit_l1_tol", 0.05),
        )
    )
    main_mode = str(cfg.get("bounds", {}).get("main_mode", "exogenous"))
    rng = np.random.default_rng(int(cfg.get("sample", {}).get("random_seed", seed)))
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode=main_mode)
    candgen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, rng, bounds)

    positions = np.flatnonzero(panel["split"].to_numpy() == "test")
    n_candidates = int(cfg.get("candidates", {}).get("n_candidates", 40))
    chunk_size = int(cfg.get("candidates", {}).get("chunk_size", 128))

    records: list[pd.DataFrame] = []
    for start in range(0, len(positions), chunk_size):
        pos = positions[start : start + chunk_size]
        batch = candgen.generate_for_positions(pos, n_candidates=n_candidates)
        actions = batch.actions
        lower, upper = bounds.bounds_for_positions(pos)
        violations = action_violations(actions, lower, upper)
        obs = actions[:, [0], :]
        dist = np.abs(actions[:, 1:, :] - obs).sum(axis=2)
        valid = ~violations[:, 1:]
        dist_valid = np.where(valid, dist, np.nan)
        finite = np.isfinite(dist_valid).any(axis=1)
        min_l1 = np.full(len(pos), np.nan)
        min_l1[finite] = np.nanmin(dist_valid[finite], axis=1)
        width = upper - lower
        asset_idx = [action_cols.index(c) for c in asset_cols if c in action_cols]
        overlay_idx = [action_cols.index(c) for c in overlay_cols if c in action_cols]
        frame = pd.DataFrame({
            "row_pos": pos.astype(int),
            "oracle_min_l1": min_l1,
            "oracle_coverable": np.isfinite(min_l1) & (min_l1 <= tau),
            "oracle_candidate_count": valid.sum(axis=1).astype(int),
            "observed_feasible_recomputed": ~action_violations(actions[:, 0, :], lower, upper),
            "mandate_width_mean": width.mean(axis=1),
            "asset_bound_width_mean": width[:, asset_idx].mean(axis=1) if asset_idx else np.nan,
            "overlay_bound_width_mean": width[:, overlay_idx].mean(axis=1) if overlay_idx else np.nan,
        })
        records.append(frame)
    oracle_df = pd.concat(records, ignore_index=True)

    panel_cols = [
        "series_id",
        "report_date",
        "mandate_cluster",
        "vix",
        "hy_oas",
        "n_holdings_scaled",
    ]
    lag_cols = [f"lag_{c}" for c in action_cols if f"lag_{c}" in panel.columns]
    keep_cols = list(dict.fromkeys([c for c in panel_cols + action_cols + lag_cols if c in panel.columns]))
    features = panel.loc[oracle_df["row_pos"].to_numpy(), keep_cols].reset_index(drop=True)
    features = _safe_numeric(features, [c for c in keep_cols if c not in {"series_id", "report_date"}])

    action_mat = features[[c for c in action_cols if c in features.columns]].to_numpy(dtype=float)
    lag_mat = np.zeros_like(action_mat)
    for j, c in enumerate([c for c in action_cols if c in features.columns]):
        lc = f"lag_{c}"
        lag_mat[:, j] = pd.to_numeric(features[lc], errors="coerce").fillna(features[c]).to_numpy(dtype=float) if lc in features.columns else features[c].to_numpy(dtype=float)
    asset_present = [c for c in asset_cols if c in features.columns]
    overlay_present = [c for c in overlay_cols if c in features.columns]
    risk_overlay = [c for c in ["short_pct", "derivative_pct", "restricted_pct", "non_us_pct"] if c in features.columns]
    features["style_drift_l1"] = np.abs(action_mat - lag_mat).sum(axis=1)
    features["asset_class_count_gt_5pct"] = (features[asset_present].to_numpy(dtype=float) > 0.05).sum(axis=1) if asset_present else np.nan
    features["asset_concentration_max"] = features[asset_present].max(axis=1) if asset_present else np.nan
    features["overlay_intensity"] = features[overlay_present].abs().sum(axis=1) if overlay_present else np.nan
    features["risk_overlay_intensity"] = features[risk_overlay].abs().sum(axis=1) if risk_overlay else np.nan

    primary = eval_df[
        eval_df["split"].astype(str).eq("test")
        & eval_df["model"].astype(str).eq("exogenous_polytope")
    ].copy()
    point = eval_df[
        eval_df["split"].astype(str).eq("test")
        & eval_df["model"].astype(str).eq("point_minimax_preference")
    ].copy()
    cols = [
        "row_pos",
        "frontier_f20_choice_coverage",
        "frontier_f50_choice_coverage",
        "frontier_f20_distance_to_set_l1",
        "frontier_f50_distance_to_set_l1",
        "regret_max_support",
        "regret_min_support",
        "theta_interval_l1_upper_diameter",
    ]
    primary = primary[[c for c in cols if c in primary.columns]].rename(columns={
        "frontier_f20_choice_coverage": "rp_f20_covered",
        "frontier_f50_choice_coverage": "rp_f50_covered",
        "frontier_f20_distance_to_set_l1": "rp_f20_distance",
        "frontier_f50_distance_to_set_l1": "rp_f50_distance",
        "regret_max_support": "rp_regret_max_support",
        "regret_min_support": "rp_regret_min_support",
        "theta_interval_l1_upper_diameter": "rp_theta_diameter",
    })
    point = point[[c for c in cols if c in point.columns]].rename(columns={
        "frontier_f20_choice_coverage": "point_f20_covered",
        "frontier_f50_choice_coverage": "point_f50_covered",
        "frontier_f20_distance_to_set_l1": "point_f20_distance",
        "frontier_f50_distance_to_set_l1": "point_f50_distance",
        "regret_max_support": "point_regret_max_support",
        "regret_min_support": "point_regret_min_support",
        "theta_interval_l1_upper_diameter": "point_theta_diameter",
    })

    out = oracle_df.merge(features, left_index=True, right_index=True)
    out = out.merge(primary, on="row_pos", how="left").merge(point, on="row_pos", how="left")
    out["rp_gap_range"] = out["rp_regret_max_support"].astype(float) - out["rp_regret_min_support"].astype(float)
    out["rp_f20_covered"] = out["rp_f20_covered"].astype(bool)
    out["point_f20_covered"] = out["point_f20_covered"].astype(bool)
    out["rp_f50_covered"] = out["rp_f50_covered"].astype(bool)
    out["point_f50_covered"] = out["point_f50_covered"].astype(bool)
    out["churn_category"] = np.select(
        [
            out["rp_f20_covered"] & out["point_f20_covered"],
            out["rp_f20_covered"] & ~out["point_f20_covered"],
            ~out["rp_f20_covered"] & out["point_f20_covered"],
        ],
        ["both", "set_only", "point_only"],
        default="neither",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute menu-ceiling and set-vs-point characterization diagnostics.")
    parser.add_argument("--results_root", default=os.environ.get("MANDATE_RESULTS_ROOT"))
    parser.add_argument("--config", default=os.environ.get("MANDATE_CONFIG"))
    parser.add_argument("--code_root", default=os.environ.get("MANDATE_CODE_ROOT"))
    parser.add_argument("--out_dir", default="tables")
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--recompute", action="store_true", help="Regenerate row-level menu diagnostics even if the parquet already exists.")
    args = parser.parse_args()
    if args.results_root is None:
        parser.error("Provide --results_root or set MANDATE_RESULTS_ROOT.")
    if args.config is None:
        parser.error("Provide --config or set MANDATE_CONFIG.")
    if args.code_root is None:
        parser.error("Provide --code_root or set MANDATE_CODE_ROOT.")

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(results_root / "action_columns.json", "r", encoding="utf-8") as f:
        action_info = json.load(f)
    action_cols = list(action_info["action_cols"])
    asset_cols = list(action_info["asset_action_cols"])
    overlay_cols = list(action_info["overlay_action_cols"])
    entity_col = str(cfg.get("keys", {}).get("entity_col", "series_id"))
    panel = pd.read_parquet(results_root / "panel_model_ready.parquet").reset_index(drop=True)
    eval_df = pd.read_parquet(results_root / "evaluation_row_level.parquet")

    row_path = out_dir / "menu_ceiling_set_characterization_row_level.parquet"
    if row_path.exists() and not args.recompute:
        row_diag = pd.read_parquet(row_path)
    else:
        row_diag = _make_row_diagnostics(
            cfg=cfg,
            panel=panel,
            eval_df=eval_df,
            action_cols=action_cols,
            asset_cols=asset_cols,
            overlay_cols=overlay_cols,
            entity_col=entity_col,
            code_root=Path(args.code_root),
            n_boot=args.n_boot,
            seed=args.seed,
        )
        row_diag.to_parquet(row_path, index=False)

    oracle_point, oracle_ci = _bootstrap_oracle_fast(row_diag, args.n_boot, args.seed)
    oracle_rows = []
    for key, value in oracle_point.items():
        lo, hi = oracle_ci[key]
        oracle_rows.append({"metric": key, "estimate": value, "ci_low": lo, "ci_high": hi})
    oracle_summary = pd.DataFrame(oracle_rows)
    oracle_summary.to_csv(out_dir / "menu_ceiling_summary.csv", index=False)

    category = row_diag.groupby("churn_category", sort=False).agg(
        n=("row_pos", "size"),
        share=("row_pos", lambda x: len(x) / len(row_diag)),
        oracle_coverable=("oracle_coverable", "mean"),
        oracle_min_l1=("oracle_min_l1", "mean"),
        style_drift_l1=("style_drift_l1", "mean"),
        risk_overlay_intensity=("risk_overlay_intensity", "mean"),
        asset_class_count_gt_5pct=("asset_class_count_gt_5pct", "mean"),
        asset_concentration_max=("asset_concentration_max", "mean"),
        mandate_width_mean=("mandate_width_mean", "mean"),
        rp_gap_range=("rp_gap_range", "mean"),
        vix=("vix", "mean"),
        rp_f20_distance=("rp_f20_distance", "mean"),
        point_f20_distance=("point_f20_distance", "mean"),
    ).reset_index()
    category.to_csv(out_dir / "set_point_characterization_by_category.csv", index=False)

    feature_list = [
        "oracle_min_l1",
        "style_drift_l1",
        "risk_overlay_intensity",
        "asset_class_count_gt_5pct",
        "asset_concentration_max",
        "mandate_width_mean",
        "vix",
        "rp_f20_distance",
        "point_f20_distance",
    ]
    diff_rows = []
    for feature in feature_list:
        point, ci = _bootstrap_feature_diff_fast(
            row_diag[row_diag["churn_category"].isin(["set_only", "point_only"])],
            feature,
            args.n_boot,
            args.seed + 17,
        )
        diff_rows.append({
            "feature": feature,
            "set_only_mean": point[f"{feature}_set_only"],
            "point_only_mean": point[f"{feature}_point_only"],
            "difference": point[f"{feature}_diff"],
            "ci_low": ci[0],
            "ci_high": ci[1],
        })
    diff = pd.DataFrame(diff_rows)
    diff.to_csv(out_dir / "set_point_characterization_differences.csv", index=False)

    oracle_lookup = oracle_summary.set_index("metric")
    table_path = out_dir / "tab_menu_ceiling.tex"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\caption{Menu ceiling for recommendation coverage}\n")
        f.write("\\label{tab:menu_ceiling}\n")
        f.write("\\centering\n\\small\n")
        f.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lc@{}}\n")
        f.write("\\toprule\nMetric & Estimate [95\\% CI] \\\\\n\\midrule\n")
        for metric, label in [
            ("oracle_coverage", "Oracle menu coverage"),
            ("rp_f20_coverage", "RP coverage, 20\\% frontier"),
            ("conditional_hit_f20", "Conditional hit rate, 20\\%"),
            ("rp_f50_coverage", "RP coverage, 50\\% frontier"),
            ("conditional_hit_f50", "Conditional hit rate, 50\\%"),
        ]:
            row = oracle_lookup.loc[metric]
            f.write(f"{label} & {_fmt_pct_ci(row['estimate'], row['ci_low'], row['ci_high'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular*}\n")
        f.write("\\begin{flushleft}\n")
        f.write("\\footnotesize Notes: Oracle menu coverage is the share of test reports for which at least one mandate-feasible generated candidate allocation in $\\Creco_{it}\\cap\\Amand_{it}$, excluding the disclosed allocation itself, lies within the fixed L1 threshold $\\tau=0.05$ of the disclosed allocation. Conditional hit rate divides model coverage by this menu ceiling. Intervals are series-cluster bootstrap intervals.\n")
        f.write("\\end{flushleft}\n\\end{table}\n")

    char_table_path = out_dir / "tab_set_point_characterization.tex"
    char = category.set_index("churn_category")
    with open(char_table_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\caption{Where the set model and point baseline disagree}\n")
        f.write("\\label{tab:set_point_characterization}\n")
        f.write("\\centering\n\\small\n")
        f.write("\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("Category & Share & Oracle L1 & Gap & RP dist. & Point dist. \\\\\n")
        f.write("\\midrule\n")
        for key, label in [("set_only", "Set only"), ("point_only", "Point only")]:
            row = char.loc[key]
            f.write(
                f"{label} & {_pct(row['share']):.2f} & "
                f"{row['oracle_min_l1']:.3f} & {row['rp_gap_range']:.3f} & "
                f"{row['rp_f20_distance']:.3f} & {row['point_f20_distance']:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular*}\n")
        f.write("\\begin{flushleft}\n")
        f.write("\\footnotesize Notes: Rows partition the 20\\% frontier cases where mandate-aware RP and the point-minimax baseline disagree. Share is the percentage of test reports in the category. Oracle L1 is the nearest mandate-feasible generated menu candidate to the disclosed allocation, excluding the disclosed allocation itself. Gap is the rationalization-gap range over the sampled preference set. Distances are L1 distances from each model's 20\\% recommendation set to the disclosed allocation.\n")
        f.write("\\end{flushleft}\n\\end{table}\n")

    combined_path = out_dir / "tab_menu_ceiling_set_value.tex"
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\caption{Menu ceiling and set-versus-point disagreement on the test split}\n")
        f.write("\\label{tab:menu_ceiling_set_value}\n")
        f.write("\\centering\n\\small\n")
        f.write("\\begin{tabular*}{0.48\\textwidth}{@{\\extracolsep{\\fill}}lc@{}}\n")
        f.write("\\toprule\n\\multicolumn{2}{c}{Panel A: menu ceiling} \\\\\n")
        f.write("\\midrule\nMetric & Estimate [95\\% CI] \\\\\n\\midrule\n")
        for metric, label in [
            ("oracle_coverage", "Oracle menu coverage"),
            ("rp_f20_coverage", "RP coverage, 20\\% frontier"),
            ("conditional_hit_f20", "Conditional hit rate, 20\\%"),
            ("rp_f50_coverage", "RP coverage, 50\\% frontier"),
            ("conditional_hit_f50", "Conditional hit rate, 50\\%"),
        ]:
            row = oracle_lookup.loc[metric]
            f.write(f"{label} & {_fmt_pct_ci(row['estimate'], row['ci_low'], row['ci_high'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular*}\n\\hfill\n")
        f.write("\\begin{tabular*}{0.48\\textwidth}{@{\\extracolsep{\\fill}}lccccc@{}}\n")
        f.write("\\toprule\n\\multicolumn{6}{c}{Panel B: disagreement at the 20\\% frontier} \\\\\n")
        f.write("\\midrule\nCategory & Share & Oracle L1 & Gap & RP dist. & Point dist. \\\\\n\\midrule\n")
        for key, label in [("set_only", "Set only"), ("point_only", "Point only")]:
            row = char.loc[key]
            f.write(
                f"{label} & {_pct(row['share']):.2f} & "
                f"{row['oracle_min_l1']:.3f} & {row['rp_gap_range']:.3f} & "
                f"{row['rp_f20_distance']:.3f} & {row['point_f20_distance']:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular*}\n")
        f.write("\\begin{flushleft}\n")
        f.write("\\footnotesize Notes: Panel A defines oracle menu coverage as the share of test reports for which at least one mandate-feasible generated candidate in $\\Creco_{it}\\cap\\Amand_{it}$, excluding the disclosed allocation itself, lies within $\\tau=0.05$ of the disclosed allocation. Conditional hit rate divides model coverage by this ceiling. Brackets in Panel A are series-cluster bootstrap intervals. Panel B partitions reports where mandate-aware RP and the point-minimax baseline disagree. Oracle L1 is the nearest mandate-feasible generated candidate distance, Gap is the rationalization-gap range over the sampled preference set, and distances are to each model's 20\\% recommendation set.\n")
        f.write("\\end{flushleft}\n\\end{table*}\n")

    print("Wrote:")
    for path in [
        out_dir / "menu_ceiling_summary.csv",
        out_dir / "set_point_characterization_by_category.csv",
        out_dir / "set_point_characterization_differences.csv",
        out_dir / "tab_menu_ceiling.tex",
        out_dir / "tab_set_point_characterization.tex",
        out_dir / "tab_menu_ceiling_set_value.tex",
    ]:
        print(path)


if __name__ == "__main__":
    main()
