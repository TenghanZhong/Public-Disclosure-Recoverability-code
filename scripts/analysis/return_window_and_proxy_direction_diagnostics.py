"""Return-window checks and proxy-direction diagnostics.

Checks return-field repetition and next-allocation movement toward the proxy.
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


DEFAULT_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SELECTED_ROOT = DEFAULT_PACKAGE_ROOT / "outputs" / "full_pipeline"
DEFAULT_OUT_DIR = DEFAULT_PACKAGE_ROOT / "outputs" / "return_window_direction"

RETURN_WINDOW_COLS = [
    "RETURN_AFTR_FEES_AND_EXPENSES",
    "RETURN_B4_FEES_AND_EXPENSES",
    "STDV_AFTR_FEES_AND_EXPENSES",
    "STDV_B4_FEES_AND_EXPENSES",
]


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
        help="Output directory for diagnostic CSV files.",
    )
    parser.add_argument(
        "--package_root",
        default=str(DEFAULT_PACKAGE_ROOT),
        help="Repository root.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--bound_mode", default="exogenous")
    parser.add_argument("--same_tol", type=float, default=1e-10)
    parser.add_argument("--n_boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_package(package_root: Path):
    sys.path.insert(0, str(package_root))
    from mandate_setirl.bounds import action_violations, fit_mandate_bounds
    from mandate_setirl.utils import bounded_simplex_project, read_yaml

    return action_violations, fit_mandate_bounds, bounded_simplex_project, read_yaml


def load_action_columns(selected_root: Path) -> dict:
    with open(selected_root / "action_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def as_numeric_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    return df[list(cols)].apply(pd.to_numeric, errors="coerce")


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    data = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return np.nan
    return float(data.iloc[:, 0].corr(data.iloc[:, 1]))


def return_window_checks(panel: pd.DataFrame, entity_col: str, date_col: str, out_dir: Path, same_tol: float):
    work = panel.sort_values([entity_col, date_col]).copy()
    work[date_col] = pd.to_datetime(work[date_col])
    rows = []
    split_rows = []
    transition_rows = []
    run_rows = []
    example_rows = []
    for col in [c for c in RETURN_WINDOW_COLS if c in work.columns]:
        vals = pd.to_numeric(work[col], errors="coerce")
        nxt = vals.groupby(work[entity_col], sort=False).shift(-1)
        next_date = work[date_col].groupby(work[entity_col], sort=False).shift(-1)
        data = pd.DataFrame(
            {
                entity_col: work[entity_col].astype(str),
                "split": work["split"].astype(str),
                "report_date": work[date_col],
                "next_report_date": next_date,
                "current": vals,
                "next": nxt,
            }
        ).dropna(subset=["current", "next", "next_report_date"])
        data["abs_change"] = (data["next"] - data["current"]).abs()
        data["same_next"] = data["abs_change"] <= same_tol
        data["gap_days"] = (data["next_report_date"] - data["report_date"]).dt.days
        data["transition_month"] = data["report_date"].dt.month.astype(int)
        rows.append(
            {
                "variable": col,
                "n_pairs": int(len(data)),
                "same_next_rate": float(data["same_next"].mean()),
                "corr_current_next": safe_corr(data["current"], data["next"]),
                "median_abs_change": float(data["abs_change"].median()),
                "p95_abs_change": float(data["abs_change"].quantile(0.95)),
                "p99_abs_change": float(data["abs_change"].quantile(0.99)),
                "current_median": float(data["current"].median()),
                "next_median": float(data["next"].median()),
                "current_mean": float(data["current"].mean()),
                "next_mean": float(data["next"].mean()),
                "current_min": float(data["current"].min()),
                "current_max": float(data["current"].max()),
                "next_min": float(data["next"].min()),
                "next_max": float(data["next"].max()),
            }
        )
        for split, g in data.groupby("split", dropna=False):
            split_rows.append(
                {
                    "variable": col,
                    "split": split,
                    "n_pairs": int(len(g)),
                    "same_next_rate": float(g["same_next"].mean()),
                    "corr_current_next": safe_corr(g["current"], g["next"]),
                    "median_abs_change": float(g["abs_change"].median()),
                    "p95_abs_change": float(g["abs_change"].quantile(0.95)),
                }
            )
        for month, g in data.groupby("transition_month", dropna=False):
            transition_rows.append(
                {
                    "variable": col,
                    "transition_month": int(month),
                    "n_pairs": int(len(g)),
                    "same_next_rate": float(g["same_next"].mean()),
                    "median_abs_change": float(g["abs_change"].median()),
                    "p95_abs_change": float(g["abs_change"].quantile(0.95)),
                }
            )
        # Run-length check: repeated values within each series reveal reporting-window reuse.
        run_lengths = []
        for sid, g in work[[entity_col, date_col, col]].dropna(subset=[col]).groupby(entity_col, sort=False):
            x = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            dates = pd.to_datetime(g[date_col]).to_numpy()
            if len(x) == 0:
                continue
            start = 0
            for i in range(1, len(x) + 1):
                if i == len(x) or not np.isclose(x[i], x[start], rtol=0.0, atol=same_tol):
                    run_len = i - start
                    run_lengths.append(run_len)
                    if run_len >= 4 and len(example_rows) < 24:
                        example_rows.append(
                            {
                                "variable": col,
                                "series_id": str(sid),
                                "run_length": int(run_len),
                                "start_date": str(pd.Timestamp(dates[start]).date()),
                                "end_date": str(pd.Timestamp(dates[i - 1]).date()),
                                "repeated_value": float(x[start]),
                            }
                        )
                    start = i
        if run_lengths:
            arr = np.asarray(run_lengths, dtype=float)
            run_rows.append(
                {
                    "variable": col,
                    "n_runs": int(len(arr)),
                    "mean_run_length": float(arr.mean()),
                    "median_run_length": float(np.median(arr)),
                    "p90_run_length": float(np.quantile(arr, 0.90)),
                    "p99_run_length": float(np.quantile(arr, 0.99)),
                    "share_runs_ge_4": float((arr >= 4).mean()),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "return_window_same_next_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out_dir / "return_window_same_next_by_split.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(out_dir / "return_window_same_next_by_transition_month.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out_dir / "return_window_repeated_value_runs.csv", index=False)
    pd.DataFrame(example_rows).to_csv(out_dir / "return_window_repeated_value_examples.csv", index=False)


def project_action(v: np.ndarray, lower: np.ndarray, upper: np.ndarray, asset_idx: np.ndarray, overlay_idx: np.ndarray, bounded_simplex_project) -> np.ndarray:
    x = np.asarray(v, dtype=float).copy()
    if len(asset_idx):
        x[asset_idx] = bounded_simplex_project(x[asset_idx], lower[asset_idx], upper[asset_idx], target_sum=1.0)
    if len(overlay_idx):
        x[overlay_idx] = np.clip(x[overlay_idx], lower[overlay_idx], upper[overlay_idx])
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def l1(a: np.ndarray, b: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    if idx is None:
        return np.abs(a - b).sum(axis=-1)
    return np.abs(a[..., idx] - b[..., idx]).sum(axis=-1)


def cosine(a: np.ndarray, b: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    if idx is not None:
        a = a[:, idx]
        b = b[:, idx]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(a.shape[0], np.nan, dtype=float)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def sign_agree(a: np.ndarray, b: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    if idx is not None:
        a = a[:, idx]
        b = b[:, idx]
    active = (np.abs(a) > 1e-10) | (np.abs(b) > 1e-10)
    denom = active.sum(axis=1)
    out = np.full(a.shape[0], np.nan, dtype=float)
    ok = denom > 0
    out[ok] = ((np.sign(a) == np.sign(b)) & active).sum(axis=1)[ok] / denom[ok]
    return out


def bootstrap_ci(df: pd.DataFrame, metrics: list[str], entity_col: str, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clusters = df[entity_col].astype(str).dropna().unique()
    rows = []
    point = {m: float(pd.to_numeric(df[m], errors="coerce").mean()) for m in metrics}
    by_cluster = {sid: g for sid, g in df.groupby(df[entity_col].astype(str), sort=False)}
    for m in metrics:
        draws = []
        for _ in range(n_boot):
            sample_ids = rng.choice(clusters, size=len(clusters), replace=True)
            vals = [pd.to_numeric(by_cluster[sid][m], errors="coerce").to_numpy(dtype=float) for sid in sample_ids]
            arr = np.concatenate(vals) if vals else np.array([], dtype=float)
            draws.append(float(np.nanmean(arr)) if arr.size else np.nan)
        lo, hi = np.nanquantile(draws, [0.025, 0.975])
        rows.append({"metric": m, "mean": point[m], "ci_low": float(lo), "ci_high": float(hi), "n_boot": int(n_boot), "n_clusters": int(len(clusters))})
    return pd.DataFrame(rows)


def summarize_direction(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [
        "current_distance_to_proxy",
        "next_distance_to_current_proxy",
        "distance_reduction",
        "distance_reduction_frac",
        "distance_decreased",
        "returned_inside_current_proxy",
        "positive_dot_to_proxy",
        "cosine_to_proxy",
        "asset_cosine_to_proxy",
        "overlay_cosine_to_proxy",
        "sign_agreement_to_proxy",
        "next_style_drift_l1",
    ]
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for keys, g in grouped:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)} if group_cols else {}
        row["n"] = int(len(g))
        row["n_series"] = int(g["series_id"].nunique())
        for m in metrics:
            row[m] = float(pd.to_numeric(g[m], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def direction_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "current_distance_to_proxy",
        "next_distance_to_current_proxy",
        "distance_reduction",
        "distance_reduction_frac",
        "cosine_to_proxy",
        "asset_cosine_to_proxy",
        "overlay_cosine_to_proxy",
        "next_style_drift_l1",
    ]
    qs = df[metrics].quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T.reset_index()
    qs = qs.rename(columns={"index": "metric", 0.01: "p01", 0.05: "p05", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.95: "p95", 0.99: "p99"})
    return qs


def sizable_violation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [0.01, 0.05, 0.10, 0.25, 0.50]:
        g = df[df["current_distance_to_proxy"] >= threshold]
        rows.append(
            {
                "current_distance_threshold": threshold,
                "n": int(len(g)),
                "n_series": int(g["series_id"].nunique()),
                "distance_reduction": float(g["distance_reduction"].mean()),
                "distance_decreased": float(g["distance_decreased"].mean()),
                "returned_inside_current_proxy": float(g["returned_inside_current_proxy"].mean()),
                "positive_dot_to_proxy": float(g["positive_dot_to_proxy"].mean()),
                "cosine_to_proxy": float(g["cosine_to_proxy"].mean()),
                "asset_cosine_to_proxy": float(g["asset_cosine_to_proxy"].mean()),
                "overlay_cosine_to_proxy": float(g["overlay_cosine_to_proxy"].mean()),
            }
        )
    return pd.DataFrame(rows)


def direction_decomposition(panel: pd.DataFrame, action_cols: list[str], asset_cols: list[str], overlay_cols: list[str], entity_col: str, bounds, action_violations, bounded_simplex_project, out_dir: Path, n_boot: int, seed: int):
    work = panel.sort_values([entity_col, "report_date"]).copy()
    next_index = work.groupby(entity_col, sort=False).shift(-1).index
    # Construct next actions by coordinate block.
    actions_sorted = as_numeric_frame(work, action_cols)
    next_actions_sorted = actions_sorted.groupby(work[entity_col], sort=False).shift(-1)
    has_next_sorted = ~next_actions_sorted.isna().any(axis=1)
    # Return to original row order.
    actions = panel[action_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    next_actions = next_actions_sorted.sort_index().to_numpy(dtype=float)
    has_next = has_next_sorted.sort_index().to_numpy(dtype=bool)
    lower, upper = bounds.lower, bounds.upper
    out_flag = action_violations(actions, lower, upper, tol=1e-7)
    sample = np.flatnonzero(out_flag & has_next)
    asset_idx = np.array([action_cols.index(c) for c in asset_cols], dtype=int)
    overlay_idx = np.array([action_cols.index(c) for c in overlay_cols], dtype=int)
    proj_current = np.zeros((len(sample), len(action_cols)), dtype=float)
    proj_next_to_current = np.zeros_like(proj_current)
    for ii, pos in enumerate(sample):
        proj_current[ii] = project_action(actions[pos], lower[pos], upper[pos], asset_idx, overlay_idx, bounded_simplex_project)
        proj_next_to_current[ii] = project_action(next_actions[pos], lower[pos], upper[pos], asset_idx, overlay_idx, bounded_simplex_project)
    current = actions[sample]
    nxt = next_actions[sample]
    to_proxy = proj_current - current
    move = nxt - current
    cur_dist = l1(current, proj_current)
    next_dist = l1(nxt, proj_next_to_current)
    reduction = cur_dist - next_dist
    direction = pd.DataFrame(
        {
            "row_pos": sample.astype(int),
            "series_id": panel.loc[sample, entity_col].astype(str).to_numpy(),
            "report_date": panel.loc[sample, "report_date"].to_numpy(),
            "split": panel.loc[sample, "split"].astype(str).to_numpy(),
            "mandate_cluster": panel.loc[sample, "mandate_cluster"].to_numpy(),
            "current_distance_to_proxy": cur_dist,
            "next_distance_to_current_proxy": next_dist,
            "distance_reduction": reduction,
            "distance_reduction_frac": reduction / np.maximum(cur_dist, 1e-12),
            "distance_decreased": (reduction > 1e-10).astype(float),
            "returned_inside_current_proxy": (next_dist <= 1e-7).astype(float),
            "positive_dot_to_proxy": (np.sum(move * to_proxy, axis=1) > 1e-10).astype(float),
            "cosine_to_proxy": cosine(move, to_proxy),
            "asset_cosine_to_proxy": cosine(move, to_proxy, asset_idx),
            "overlay_cosine_to_proxy": cosine(move, to_proxy, overlay_idx),
            "sign_agreement_to_proxy": sign_agree(move, to_proxy),
            "next_style_drift_l1": l1(nxt, current),
            "asset_current_distance_to_proxy": l1(current, proj_current, asset_idx),
            "asset_next_distance_to_current_proxy": l1(nxt, proj_next_to_current, asset_idx),
            "overlay_current_distance_to_proxy": l1(current, proj_current, overlay_idx),
            "overlay_next_distance_to_current_proxy": l1(nxt, proj_next_to_current, overlay_idx),
        }
    )
    direction["distance_direction"] = np.where(direction["distance_reduction"] > 1e-10, "toward_proxy", np.where(direction["distance_reduction"] < -1e-10, "away_from_proxy", "unchanged"))
    direction.to_csv(out_dir / "proxy_direction_row_level.csv", index=False)
    summarize_direction(direction, []).to_csv(out_dir / "proxy_direction_summary_all.csv", index=False)
    summarize_direction(direction, ["split"]).to_csv(out_dir / "proxy_direction_summary_by_split.csv", index=False)
    summarize_direction(direction, ["mandate_cluster"]).to_csv(out_dir / "proxy_direction_summary_by_cluster.csv", index=False)
    direction_quantiles(direction).to_csv(out_dir / "proxy_direction_quantiles.csv", index=False)
    sizable_violation_summary(direction).to_csv(out_dir / "proxy_direction_sizable_violation_summary.csv", index=False)
    boot_metrics = [
        "distance_reduction",
        "distance_reduction_frac",
        "distance_decreased",
        "returned_inside_current_proxy",
        "positive_dot_to_proxy",
        "cosine_to_proxy",
        "asset_cosine_to_proxy",
        "overlay_cosine_to_proxy",
    ]
    bootstrap_ci(direction, boot_metrics, "series_id", n_boot=n_boot, seed=seed).to_csv(out_dir / "proxy_direction_cluster_bootstrap_ci.csv", index=False)
    # Compact diagnostic table.
    direction.drop(columns=["row_pos"]).head(5000).to_csv(out_dir / "proxy_direction_row_sample.csv", index=False)


def main():
    args = parse_args()
    selected_root = Path(args.selected_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    package_root = Path(args.package_root)
    config_path = Path(args.config) if args.config else package_root / "configs" / "main_32gb.yaml"
    action_violations, fit_mandate_bounds, bounded_simplex_project, read_yaml = add_package(package_root)
    cfg = read_yaml(config_path)
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    date_col = cfg.get("keys", {}).get("date_col", "report_date")
    ac = load_action_columns(selected_root)
    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    return_window_checks(panel, entity_col, date_col, out_dir, args.same_tol)
    bounds = fit_mandate_bounds(
        panel,
        ac["action_cols"],
        ac["asset_action_cols"],
        ac["overlay_action_cols"],
        entity_col,
        cfg,
        mode=args.bound_mode,
    )
    direction_decomposition(
        panel,
        ac["action_cols"],
        ac["asset_action_cols"],
        ac["overlay_action_cols"],
        entity_col,
        bounds,
        action_violations,
        bounded_simplex_project,
        out_dir,
        args.n_boot,
        args.seed,
    )
    manifest = {
        "selected_root": str(selected_root),
        "package_root": str(package_root),
        "config": str(config_path),
        "bound_mode": args.bound_mode,
        "same_tol": args.same_tol,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "outputs": sorted(p.name for p in out_dir.glob("*.csv")),
        "notes": [
            "Return-window checks diagnose repeated disclosure fields; they are not used as main outcomes.",
            "Direction metrics are computed for out-of-proxy rows with a next report.",
            "The projection target is the nearest point inside the current report's inferred mandate proxy under the candidate projection rule.",
        ],
    }
    with open(out_dir / "return_window_direction_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(pd.read_csv(out_dir / "return_window_same_next_summary.csv").to_string(index=False))
    print(pd.read_csv(out_dir / "proxy_direction_summary_all.csv").to_string(index=False))
    print(pd.read_csv(out_dir / "proxy_direction_cluster_bootstrap_ci.csv").to_string(index=False))


if __name__ == "__main__":
    main()
