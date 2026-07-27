"""Candidate-menu and drift diagnostics.

Runs duplicate-candidate, conditional-ranking, and buy-and-hold drift checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from out_of_proxy_predictive_validity import fit_residualized_ols  # noqa: E402
from recoverability_tau_menu_checks import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_PACKAGE_ROOT,
    DEFAULT_SELECTED_ROOT,
    add_package_paths,
    lag_action_matrix,
    move_bins,
    transition_config,
)
from boundary_inference_checks import (  # noqa: E402
    add_package,
    as_numeric,
    build_regression_panel,
    zscore_on_mask,
)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = DEFAULT_REPO_ROOT / "tables" / "candidate_drift_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_root", default=str(DEFAULT_SELECTED_ROOT))
    parser.add_argument("--package_root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", default="7,13,21,42,57")
    parser.add_argument("--max_rows", type=int, default=2000)
    parser.add_argument("--n_candidates", type=int, default=40)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--transition_magnitude_count", type=int, default=12)
    parser.add_argument("--transition_stress_count", type=int, default=6)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--frontier", type=float, default=0.20)
    parser.add_argument("--dedupe_decimals", type=int, default=6)
    parser.add_argument("--iqr_floor", type=float, default=1e-8)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--demean_iter", type=int, default=12)
    parser.add_argument("--skip_candidates", action="store_true")
    parser.add_argument("--skip_buyhold", action="store_true")
    return parser.parse_args()


def l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=-1)


def load_action_columns(selected_root: Path) -> dict:
    with open(selected_root / "action_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def is_transition_source(source: object) -> bool:
    return str(source).startswith("transition")


def rounded_key(action: np.ndarray, decimals: int) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(action, dtype=float), decimals))


def unique_representatives(
    actions: np.ndarray,
    sources: np.ndarray,
    decimals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return first unique action vectors and their original entry indices."""
    seen: set[tuple[float, ...]] = set()
    keep: list[int] = []
    for idx, action in enumerate(actions):
        key = rounded_key(action, decimals)
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    keep_idx = np.asarray(keep, dtype=int)
    return actions[keep_idx], sources[keep_idx], keep_idx


def order_persistence(actions: np.ndarray, lag_action: np.ndarray) -> np.ndarray:
    lag_dist = l1(actions, lag_action)
    return np.lexsort((np.arange(len(actions)), lag_dist))


def order_transition_source(actions: np.ndarray, sources: np.ndarray, lag_action: np.ndarray) -> np.ndarray:
    lag_dist = l1(actions, lag_action)
    priority = np.asarray([0 if is_transition_source(s) else 1 for s in sources], dtype=float)
    return np.lexsort((np.arange(len(actions)), lag_dist, priority))


def selected_min_distance(
    actions: np.ndarray,
    order: np.ndarray,
    observed_action: np.ndarray,
    frontier: float,
) -> tuple[float, int]:
    if len(actions) == 0:
        return np.nan, 0
    k = max(1, int(np.ceil(frontier * len(actions))))
    selected = actions[order[:k]]
    return float(np.min(l1(selected, observed_action))), int(k)


def summarize_rates(row_df: pd.DataFrame, tau: float) -> pd.DataFrame:
    group_specs = [
        ("overall", []),
        ("move_bin", ["move_bin"]),
        ("proxy_status", ["observed_in_proxy"]),
        ("move_bin_x_proxy_status", ["move_bin", "observed_in_proxy"]),
    ]
    metric_pairs = [
        ("entry_support", "entry_oracle_min_l1"),
        ("unique_support", "unique_oracle_min_l1"),
        ("entry_persistence_cov", "entry_persistence_min_l1"),
        ("unique_persistence_cov", "unique_persistence_min_l1"),
        ("entry_transition_cov", "entry_transition_min_l1"),
        ("unique_transition_cov", "unique_transition_min_l1"),
    ]
    rows: list[dict] = []
    for group_name, cols in group_specs:
        grouped = [((), row_df)] if not cols else row_df.groupby(cols, dropna=False)
        for keys, g in grouped:
            if cols and not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, object] = {"group": group_name, "n": int(len(g)), "tau": float(tau)}
            for c, v in zip(cols, keys if cols else ()):
                row[c] = v
            for label, col in metric_pairs:
                hit = pd.to_numeric(g[col], errors="coerce") <= tau
                row[label] = float(hit.mean())
            for prefix in ["entry", "unique"]:
                support = float(row[f"{prefix}_support"])
                for ranker in ["persistence", "transition"]:
                    cov = float(row[f"{prefix}_{ranker}_cov"])
                    row[f"{prefix}_{ranker}_conditional_on_support"] = (
                        cov / support if support > 0 else np.nan
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_unique_counts(row_df: pd.DataFrame) -> pd.DataFrame:
    group_specs = [
        ("overall", []),
        ("move_bin", ["move_bin"]),
        ("proxy_status", ["observed_in_proxy"]),
        ("move_bin_x_proxy_status", ["move_bin", "observed_in_proxy"]),
    ]
    rows: list[dict] = []
    for group_name, cols in group_specs:
        grouped = [((), row_df)] if not cols else row_df.groupby(cols, dropna=False)
        for keys, g in grouped:
            if cols and not isinstance(keys, tuple):
                keys = (keys,)
            vals = pd.to_numeric(g["unique_feasible_count"], errors="coerce")
            dup = pd.to_numeric(g["duplicate_rate_feasible"], errors="coerce")
            row: dict[str, object] = {
                "group": group_name,
                "n": int(len(g)),
                "unique_mean": float(vals.mean()),
                "unique_median": float(vals.median()),
                "unique_p25": float(vals.quantile(0.25)),
                "unique_p75": float(vals.quantile(0.75)),
                "entry_mean": float(pd.to_numeric(g["feasible_entry_count"], errors="coerce").mean()),
                "duplicate_rate_mean": float(dup.mean()),
                "duplicate_rate_p75": float(dup.quantile(0.75)),
            }
            for c, v in zip(cols, keys if cols else ()):
                row[c] = v
            rows.append(row)
    return pd.DataFrame(rows)


def average_over_seeds(df: pd.DataFrame, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = {c: v for c, v in zip(group_cols, keys)}
        row["n_seeds"] = int(g["seed"].nunique())
        for c in value_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_min"] = float(vals.min())
            row[f"{c}_max"] = float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows)


def run_unique_candidate_diagnostics(args: argparse.Namespace, out_dir: Path) -> dict:
    selected_root = Path(args.selected_root)
    package_root = Path(args.package_root)
    add_package_paths(package_root)
    from mandate_setirl.bounds import action_violations, fit_mandate_bounds
    from mandate_setirl.candidates import CandidateGenerator
    from mandate_setirl.utils import read_yaml

    cfg = transition_config(read_yaml(Path(args.config)), args)
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    action_meta = load_action_columns(selected_root)
    action_cols = action_meta["action_cols"]
    asset_cols = action_meta["asset_action_cols"]
    overlay_cols = action_meta["overlay_action_cols"]
    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    actions_all = panel[action_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lag_actions = lag_action_matrix(panel, action_cols)
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
    split_pos = np.flatnonzero(panel["split"].astype(str).eq(args.split).to_numpy())
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    generator = CandidateGenerator(
        panel,
        action_cols,
        asset_cols,
        overlay_cols,
        entity_col,
        cfg,
        np.random.default_rng(seeds[0] if seeds else 7),
        bounds,
    )
    row_parts: list[pd.DataFrame] = []
    rate_parts: list[pd.DataFrame] = []
    count_parts: list[pd.DataFrame] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if args.max_rows and args.max_rows < len(split_pos):
            positions = np.sort(rng.choice(split_pos, size=args.max_rows, replace=False))
        else:
            positions = split_pos.copy()
        generator.rng = np.random.default_rng(seed)
        rows: list[dict] = []
        for start in range(0, len(positions), args.chunk_size):
            batch_pos = positions[start : start + args.chunk_size]
            print(f"unique candidate seed {seed}: rows {start + 1}-{start + len(batch_pos)} of {len(positions)}", flush=True)
            batch = generator.generate_for_positions(batch_pos, n_candidates=args.n_candidates)
            lower, upper = bounds.bounds_for_positions(batch_pos)
            cand_out = action_violations(batch.actions, lower, upper, tol=1e-7)
            observed_out = action_violations(actions_all[batch_pos], lower, upper, tol=1e-7)
            for bi, row_pos in enumerate(batch_pos):
                obs = actions_all[row_pos]
                lag = lag_actions[row_pos]
                rec_idx = np.arange(batch.actions.shape[1])
                rec_idx = rec_idx[rec_idx != batch.observed_index]
                feasible_idx = rec_idx[~cand_out[bi, rec_idx]]
                feasible_actions = batch.actions[bi, feasible_idx]
                feasible_sources = batch.sources[bi, feasible_idx].astype(str)
                unique_actions, unique_sources, _keep_idx = unique_representatives(
                    feasible_actions,
                    feasible_sources,
                    args.dedupe_decimals,
                )
                if len(feasible_actions) == 0:
                    entry_oracle = unique_oracle = np.nan
                    entry_pers = unique_pers = np.nan
                    entry_trans = unique_trans = np.nan
                    entry_pers_k = unique_pers_k = entry_trans_k = unique_trans_k = 0
                    duplicate_rate = np.nan
                else:
                    entry_oracle = float(np.min(l1(feasible_actions, obs)))
                    unique_oracle = float(np.min(l1(unique_actions, obs)))
                    duplicate_rate = 1.0 - len(unique_actions) / len(feasible_actions)
                    entry_pers, entry_pers_k = selected_min_distance(
                        feasible_actions,
                        order_persistence(feasible_actions, lag),
                        obs,
                        args.frontier,
                    )
                    unique_pers, unique_pers_k = selected_min_distance(
                        unique_actions,
                        order_persistence(unique_actions, lag),
                        obs,
                        args.frontier,
                    )
                    entry_trans, entry_trans_k = selected_min_distance(
                        feasible_actions,
                        order_transition_source(feasible_actions, feasible_sources, lag),
                        obs,
                        args.frontier,
                    )
                    unique_trans, unique_trans_k = selected_min_distance(
                        unique_actions,
                        order_transition_source(unique_actions, unique_sources, lag),
                        obs,
                        args.frontier,
                    )
                rows.append(
                    {
                        "seed": seed,
                        "row_pos": int(row_pos),
                        "series_id": str(panel.loc[row_pos, entity_col]),
                        "report_date": str(panel.loc[row_pos, "report_date"]),
                        "move_bin": move_bins(np.asarray([l1(obs, lag)]))[0],
                        "observed_in_proxy": bool(not observed_out[bi]),
                        "feasible_entry_count": int(len(feasible_actions)),
                        "unique_feasible_count": int(len(unique_actions)),
                        "duplicate_rate_feasible": float(duplicate_rate) if np.isfinite(duplicate_rate) else np.nan,
                        "entry_oracle_min_l1": entry_oracle,
                        "unique_oracle_min_l1": unique_oracle,
                        "entry_persistence_min_l1": entry_pers,
                        "unique_persistence_min_l1": unique_pers,
                        "entry_transition_min_l1": entry_trans,
                        "unique_transition_min_l1": unique_trans,
                        "entry_persistence_k": entry_pers_k,
                        "unique_persistence_k": unique_pers_k,
                        "entry_transition_k": entry_trans_k,
                        "unique_transition_k": unique_trans_k,
                    }
                )
        seed_df = pd.DataFrame(rows)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_df.to_csv(seed_dir / "unique_candidate_row_metrics.csv", index=False)
        rates = summarize_rates(seed_df, args.tau)
        counts = summarize_unique_counts(seed_df)
        rates["seed"] = seed
        counts["seed"] = seed
        rates.to_csv(seed_dir / "unique_candidate_rates.csv", index=False)
        counts.to_csv(seed_dir / "unique_candidate_counts.csv", index=False)
        row_parts.append(seed_df)
        rate_parts.append(rates)
        count_parts.append(counts)

    rate_by_seed = pd.concat(rate_parts, ignore_index=True)
    count_by_seed = pd.concat(count_parts, ignore_index=True)
    rate_summary = average_over_seeds(
        rate_by_seed,
        [
            "entry_support",
            "unique_support",
            "entry_persistence_cov",
            "unique_persistence_cov",
            "entry_transition_cov",
            "unique_transition_cov",
            "entry_persistence_conditional_on_support",
            "unique_persistence_conditional_on_support",
            "entry_transition_conditional_on_support",
            "unique_transition_conditional_on_support",
        ],
        ["group", "move_bin", "observed_in_proxy", "tau"],
    )
    count_summary = average_over_seeds(
        count_by_seed,
        [
            "unique_mean",
            "unique_median",
            "unique_p25",
            "unique_p75",
            "entry_mean",
            "duplicate_rate_mean",
            "duplicate_rate_p75",
        ],
        ["group", "move_bin", "observed_in_proxy"],
    )
    rate_by_seed.to_csv(out_dir / "unique_candidate_rates_by_seed.csv", index=False)
    count_by_seed.to_csv(out_dir / "unique_candidate_counts_by_seed.csv", index=False)
    rate_summary.to_csv(out_dir / "unique_candidate_rate_summary.csv", index=False)
    count_summary.to_csv(out_dir / "unique_candidate_count_summary.csv", index=False)
    pd.concat(row_parts, ignore_index=True).to_csv(out_dir / "unique_candidate_row_sample.csv", index=False)
    return {
        "candidate_rows_per_seed": int(args.max_rows if args.max_rows else len(split_pos)),
        "candidate_seeds": seeds,
        "candidate_split_rows": int(len(split_pos)),
    }


def _asset_to_ticker_from_package(package_root: Path) -> dict[str, str]:
    sys.path.insert(0, str(package_root))
    from mandate_setirl.features import _asset_to_ticker

    return _asset_to_ticker()


def lead_values(panel: pd.DataFrame, cols: list[str], entity_col: str, date_col: str) -> pd.DataFrame:
    order = panel.sort_values([entity_col, date_col]).index
    # Use positional columns because return benchmark mappings can repeat
    # the same source column across several asset buckets.
    out = pd.DataFrame(index=panel.index, columns=range(len(cols)), dtype=float)
    values = pd.DataFrame(
        {j: pd.to_numeric(panel.loc[order, col], errors="coerce").to_numpy(dtype=float) for j, col in enumerate(cols)},
        index=order,
    )
    entities = pd.Series(panel.loc[order, entity_col].to_numpy(), index=order)
    out.loc[order, :] = values.groupby(entities, sort=False).shift(-1).to_numpy()
    return out


def normalize_simplex(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).copy()
    row_sum = np.nansum(w, axis=1)
    out = np.full_like(w, np.nan, dtype=float)
    valid = np.isfinite(row_sum) & (row_sum > 1e-12)
    out[valid] = w[valid] / row_sum[valid, None]
    return out


def buyhold_next_weights(current_weights: np.ndarray, next_returns: np.ndarray) -> np.ndarray:
    gross = current_weights * (1.0 + next_returns)
    denom = np.nansum(gross, axis=1)
    out = np.full_like(gross, np.nan, dtype=float)
    valid = np.isfinite(denom) & (denom > 1e-12)
    out[valid] = gross[valid] / denom[valid, None]
    return out


def run_buyhold_adjusted_asset_drift(args: argparse.Namespace, out_dir: Path) -> dict:
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
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
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
    scaled_col = zscore_on_mask(reg, "outside_proxy_distance_iqr_mean", split_mask)

    mapping = _asset_to_ticker_from_package(package_root)
    return_cols: list[str] = []
    mapped_assets: list[str] = []
    for asset in asset_cols:
        ticker = mapping.get(asset)
        col = f"{str(ticker).lower()}_ret_63d" if ticker else None
        if col is None or col not in panel.columns:
            continue
        mapped_assets.append(asset)
        return_cols.append(col)
    if len(mapped_assets) < 2:
        raise RuntimeError("Not enough mapped asset return columns for buy-and-hold adjustment")

    current_w = normalize_simplex(as_numeric(panel, mapped_assets).to_numpy(dtype=float))
    next_w = normalize_simplex(lead_values(panel, mapped_assets, entity_col, date_col).to_numpy(dtype=float))
    next_returns = lead_values(panel, return_cols, entity_col, date_col).to_numpy(dtype=float)
    buyhold_w = buyhold_next_weights(current_w, next_returns)
    adjusted_drift = np.abs(next_w - buyhold_w).sum(axis=1)
    raw_asset_drift = np.abs(next_w - current_w).sum(axis=1)
    valid_adjusted = np.isfinite(adjusted_drift)
    valid_raw = np.isfinite(raw_asset_drift)
    reg["next_asset_buyhold_adjusted_drift_l1"] = adjusted_drift
    reg.loc[~valid_adjusted, "next_asset_buyhold_adjusted_drift_l1"] = np.nan
    reg["next_asset_raw_drift_l1"] = raw_asset_drift
    reg.loc[~valid_raw, "next_asset_raw_drift_l1"] = np.nan

    asset_actions = as_numeric(panel, mapped_assets).to_numpy(dtype=float)
    lag_asset = np.full_like(asset_actions, np.nan, dtype=float)
    for j, c in enumerate(mapped_assets):
        lag_col = f"lag_{c}"
        if lag_col in panel.columns:
            lag_asset[:, j] = pd.to_numeric(panel[lag_col], errors="coerce").to_numpy(dtype=float)
    reg["asset_style_drift_l1"] = np.abs(normalize_simplex(asset_actions) - normalize_simplex(lag_asset)).sum(axis=1)
    reg["lag_asset_style_drift_l1"] = reg.groupby(entity_col, sort=False)["asset_style_drift_l1"].shift(1)

    test = reg.loc[split_mask].copy()
    controls = [
        "asset_style_drift_l1",
        "lag_asset_style_drift_l1",
        "log_net_assets",
        "net_flow_3m_scaled",
        "lag_net_flow_3m_scaled",
    ]
    rows: list[dict] = []
    for label, focus in [
        ("binary_exception_buyhold_adjusted_asset_drift", "exception"),
        ("scaled_magnitude_buyhold_adjusted_asset_drift", scaled_col),
    ]:
        work = test.copy()
        work["out_of_proxy"] = pd.to_numeric(work[focus], errors="coerce")
        res = fit_residualized_ols(
            work,
            "next_asset_buyhold_adjusted_drift_l1",
            controls=[c for c in controls if c in work.columns and c != focus],
            fe_cols=[date_col, "mandate_cluster"],
            entity_col=entity_col,
            date_col=date_col,
            n_iter=args.demean_iter,
        )
        rows.append(
            {
                "spec": label,
                "outcome": "next_asset_buyhold_adjusted_drift_l1",
                "focus": focus,
                "n": int(res.get("n", 0)),
                "coef": float(res.get("beta", np.nan)),
                "two_way_cluster_t": float(res.get("two_way_cluster_t", np.nan)),
                "two_way_cluster_p": float(res.get("two_way_cluster_p", np.nan)),
                "series_cluster_t": float(res.get("series_cluster_t", np.nan)),
                "mean_outcome": float(work["next_asset_buyhold_adjusted_drift_l1"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "buyhold_adjusted_asset_drift_regressions.csv", index=False)
    summary = {
        "mapped_assets": mapped_assets,
        "return_columns": return_cols,
        "return_window": "next observed report row, 63-day ETF proxy return",
        "n_test_rows": int(split_mask.sum()),
        "n_valid_buyhold_adjusted_drift_test_rows": int(test["next_asset_buyhold_adjusted_drift_l1"].notna().sum()),
        "mean_buyhold_adjusted_asset_drift": float(test["next_asset_buyhold_adjusted_drift_l1"].mean()),
        "mean_raw_asset_drift": float(test["next_asset_raw_drift_l1"].mean()),
        "outside_proxy_share": float(test["exception"].mean()),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "buyhold_adjusted_asset_drift_summary.csv", index=False)
    meta.update(summary)
    return meta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "selected_root": str(Path(args.selected_root)),
        "package_root": str(Path(args.package_root)),
        "config": str(Path(args.config)),
        "out_dir": str(out_dir),
        "split": args.split,
        "tau": float(args.tau),
        "frontier": float(args.frontier),
        "dedupe_decimals": int(args.dedupe_decimals),
    }
    if not args.skip_candidates:
        manifest.update(run_unique_candidate_diagnostics(args, out_dir))
    if not args.skip_buyhold:
        manifest["buyhold"] = run_buyhold_adjusted_asset_drift(args, out_dir)
    with open(out_dir / "candidate_drift_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
