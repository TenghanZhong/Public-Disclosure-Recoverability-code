"""Recoverability tolerance and feasible-menu diagnostics.

Computes feasible-menu support and transition-ranked selection on repeated
test-split samples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_ROOT = DEFAULT_REPO_ROOT
DEFAULT_SELECTED_ROOT = DEFAULT_REPO_ROOT / "outputs" / "full_pipeline"
DEFAULT_CONFIG = DEFAULT_REPO_ROOT / "configs" / "main_32gb.yaml"
DEFAULT_OUT_DIR = DEFAULT_REPO_ROOT / "tables" / "recoverability_tau_menu_checks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_root", default=str(DEFAULT_SELECTED_ROOT))
    parser.add_argument("--package_root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", default="7,13,21,42,57,73,88,101,123,202")
    parser.add_argument("--max_rows", type=int, default=5000)
    parser.add_argument("--n_candidates", type=int, default=40)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--tolerances", default="0.05,0.10,0.25")
    parser.add_argument("--frontier", type=float, default=0.20)
    parser.add_argument("--transition_magnitude_count", type=int, default=12)
    parser.add_argument("--transition_stress_count", type=int, default=6)
    parser.add_argument("--row_level", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def add_package_paths(package_root: Path) -> None:
    sys.path.insert(0, str(package_root / "scripts"))
    sys.path.insert(0, str(package_root))


def l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=-1)


def move_bins(lag_l1: np.ndarray) -> np.ndarray:
    return np.asarray(
        pd.cut(
            lag_l1,
            [-np.inf, 0.05, 0.25, np.inf],
            labels=["stable_l1_le_005", "moderate_l1_005_025", "large_l1_gt_025"],
        ).astype(str)
    )


def lag_action_matrix(panel: pd.DataFrame, action_cols: list[str]) -> np.ndarray:
    lag = np.zeros((len(panel), len(action_cols)), dtype=float)
    for j, col in enumerate(action_cols):
        lag_col = f"lag_{col}"
        src = panel[lag_col] if lag_col in panel.columns else panel[col]
        lag[:, j] = (
            pd.to_numeric(src, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    return lag


def transition_config(base_cfg: dict, args: argparse.Namespace) -> dict:
    """Use the transition-menu configuration."""
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("candidates", {})
    cfg["candidates"].update(
        {
            "generator_mode": "transition_augmented",
            "include_observed_candidate": True,
            "prediction_exclude_observed": True,
            "same_date_peer_mode": "off",
            "allow_transductive_same_date": False,
            "n_candidates": args.n_candidates,
            "chunk_size": args.chunk_size,
            "peer_train_count": 2,
            "self_lag_exact_count": 1,
            "self_lag_projected_count": 1,
            "transition_knn_count": 0,
            "transition_small_count": 4,
            "transition_moderate_count": 12,
            "transition_large_count": 4,
            "transition_regime_count": 6,
            "transition_shrunk_count": 3,
            "transition_magnitude_count": args.transition_magnitude_count,
            "transition_stress_count": 4,
            "clean_frontier_count": 2,
            "raw_candidate_oversample_multiplier": 3,
            "dedupe_round_decimals": 6,
            "mover_routing_enabled": True,
            "source_quotas_by_predicted_move": {
                "stayer": {
                    "lag": 2,
                    "static_peer": 2,
                    "peer_small": 12,
                    "peer_moderate": 8,
                    "peer_large_stress": 2,
                    "regime": 6,
                    "shrunk": 5,
                    "clean": 2,
                },
                "moderate": {
                    "lag": 2,
                    "static_peer": 2,
                    "peer_small": 4,
                    "peer_moderate": 12,
                    "peer_large_stress": 8,
                    "regime": 6,
                    "shrunk": 3,
                    "clean": 2,
                },
                "large": {
                    "lag": 1,
                    "static_peer": 1,
                    "peer_small": 2,
                    "peer_moderate": 10,
                    "peer_large_stress": 14,
                    "regime": 6,
                    "shrunk": 3,
                    "clean": 2,
                },
            },
        }
    )
    cfg.setdefault("transition", {})
    cfg["transition"].update(
        {
            "enabled": True,
            "donor_split": "train",
            "min_move_l1": 0.005,
            "max_move_l1_quantile": 0.99,
            "scales": [0.5, 1.0],
            "same_cluster_first": True,
            "fallback_to_pooled": True,
            "retrieval_k": 16,
            "retrieval_state_weight": 0.7,
            "retrieval_lag_weight": 0.3,
            "regime_min_count": 50,
            "stress_move_quantile": 0.75,
            "direction_magnitudes": [0.08, 0.15, 0.30, 0.50],
            "mover_threshold": 0.05,
            "mover_prediction_k": 16,
        }
    )
    return cfg


def source_is_transition(source: object) -> bool:
    return str(source).startswith("transition")


def row_level_diagnostics(
    panel: pd.DataFrame,
    action_cols: list[str],
    entity_col: str,
    bounds,
    generator,
    positions: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    from mandate_setirl.bounds import action_violations

    actions = panel[action_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lag_actions = lag_action_matrix(panel, action_cols)
    lag_l1 = l1(actions, lag_actions)
    rows: list[dict] = []

    for start in range(0, len(positions), args.chunk_size):
        batch_pos = positions[start : start + args.chunk_size]
        print(f"candidate batch {start + 1}-{start + len(batch_pos)} of {len(positions)}", flush=True)
        batch = generator.generate_for_positions(batch_pos, n_candidates=args.n_candidates)
        lower, upper = bounds.bounds_for_positions(batch_pos)
        observed_out = action_violations(actions[batch_pos], lower, upper, tol=1e-7)
        cand_out = action_violations(batch.actions, lower, upper, tol=1e-7)
        for bi, row_pos in enumerate(batch_pos):
            rec_idx = np.arange(batch.actions.shape[1])
            rec_idx = rec_idx[rec_idx != batch.observed_index]
            feasible_mask = ~cand_out[bi, rec_idx]
            feasible_idx = rec_idx[feasible_mask]
            row = {
                "row_pos": int(row_pos),
                "seed_series_id": str(panel.loc[row_pos, entity_col]),
                "report_date": panel.loc[row_pos, "report_date"] if "report_date" in panel.columns else "",
                "move_bin": move_bins(np.asarray([lag_l1[row_pos]]))[0],
                "observed_in_proxy": bool(not observed_out[bi]),
                "lag_l1_to_obs": float(lag_l1[row_pos]),
                "candidate_count_excluding_observed": int(len(rec_idx)),
                "feasible_menu_size": int(len(feasible_idx)),
            }
            if len(feasible_idx) == 0:
                row.update(
                    {
                        "oracle_min_l1": np.nan,
                        "selected_f20_min_l1": np.nan,
                        "selected_f20_k": 0,
                        "duplicate_rate_rounded_1e6": np.nan,
                    }
                )
                rows.append(row)
                continue

            obs = actions[row_pos]
            lag = lag_actions[row_pos]
            feasible_actions = batch.actions[bi, feasible_idx]
            dist = l1(feasible_actions, obs)
            lag_dist = l1(feasible_actions, lag)
            sources = batch.sources[bi, feasible_idx].astype(str)
            source_priority = np.array([0 if source_is_transition(s) else 1 for s in sources], dtype=float)
            order = np.lexsort((np.arange(len(feasible_idx)), lag_dist, source_priority))
            k = max(1, int(np.ceil(args.frontier * len(feasible_idx))))
            selected_dist = dist[order[:k]]
            duplicate_rate = 1.0 - len({tuple(np.round(x, 6)) for x in feasible_actions}) / max(len(feasible_actions), 1)
            row.update(
                {
                    "oracle_min_l1": float(np.min(dist)),
                    "selected_f20_min_l1": float(np.min(selected_dist)),
                    "selected_f20_k": int(k),
                    "duplicate_rate_rounded_1e6": float(duplicate_rate),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_tau(row_df: pd.DataFrame, tolerances: list[float]) -> pd.DataFrame:
    group_specs = [
        ("overall", []),
        ("move_bin", ["move_bin"]),
        ("proxy_status", ["observed_in_proxy"]),
        ("move_bin_x_proxy_status", ["move_bin", "observed_in_proxy"]),
    ]
    out_rows: list[dict] = []
    for group_name, cols in group_specs:
        grouped = [((), row_df)] if not cols else row_df.groupby(cols, dropna=False)
        for keys, g in grouped:
            if cols and not isinstance(keys, tuple):
                keys = (keys,)
            base = {"group": group_name, "n": int(len(g))}
            for c, v in zip(cols, keys if cols else ()):
                base[c] = v
            for tau in tolerances:
                base_tau = dict(base)
                base_tau["tau"] = float(tau)
                base_tau["menu_support"] = float((pd.to_numeric(g["oracle_min_l1"], errors="coerce") <= tau).mean())
                base_tau["selected_20pct_coverage"] = float(
                    (pd.to_numeric(g["selected_f20_min_l1"], errors="coerce") <= tau).mean()
                )
                out_rows.append(base_tau)
    return pd.DataFrame(out_rows)


def summarize_menu_size(row_df: pd.DataFrame) -> pd.DataFrame:
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
            vals = pd.to_numeric(g["feasible_menu_size"], errors="coerce")
            row = {
                "group": group_name,
                "n": int(len(g)),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "p25": float(vals.quantile(0.25)),
                "p75": float(vals.quantile(0.75)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
            for c, v in zip(cols, keys if cols else ()):
                row[c] = v
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_seeds(seed_df: pd.DataFrame, metric_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, g in seed_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n_seeds"] = int(g["seed"].nunique())
        for metric in metric_cols:
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_min"] = float(np.nanmin(vals))
            row[f"{metric}_max"] = float(np.nanmax(vals))
            row[f"{metric}_p025"] = float(np.nanquantile(vals, 0.025))
            row[f"{metric}_p975"] = float(np.nanquantile(vals, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    selected_root = Path(args.selected_root)
    package_root = Path(args.package_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    add_package_paths(package_root)

    from mandate_setirl.bounds import fit_mandate_bounds
    from mandate_setirl.candidates import CandidateGenerator
    from mandate_setirl.utils import read_yaml
    from run_transition_menu_diagnostics import load_action_columns

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tolerances = [float(x.strip()) for x in args.tolerances.split(",") if x.strip()]
    base_cfg = read_yaml(Path(args.config))
    cfg = transition_config(base_cfg, args)
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    action_meta = load_action_columns(selected_root)
    action_cols = action_meta["action_cols"]
    asset_cols = action_meta["asset_action_cols"]
    overlay_cols = action_meta["overlay_action_cols"]
    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
    split_pos = np.flatnonzero(panel["split"].astype(str).eq(args.split).to_numpy())

    tau_seed_rows: list[pd.DataFrame] = []
    menu_seed_rows: list[pd.DataFrame] = []
    row_level_parts: list[pd.DataFrame] = []
    generator = CandidateGenerator(
        panel,
        action_cols,
        asset_cols,
        overlay_cols,
        entity_col,
        cfg,
        np.random.default_rng(seeds[0] if seeds else 42),
        bounds,
    )
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        tau_path = seed_dir / "recoverability_tau_by_group.csv"
        menu_path = seed_dir / "feasible_menu_size_by_group.csv"
        row_path = seed_dir / "recoverability_tau_row_metrics.csv"
        if args.resume and tau_path.exists() and menu_path.exists():
            print(f"loading existing seed {seed}", flush=True)
            tau_by_seed = pd.read_csv(tau_path)
            menu_by_seed = pd.read_csv(menu_path)
            tau_by_seed["seed"] = seed
            menu_by_seed["seed"] = seed
            tau_seed_rows.append(tau_by_seed)
            menu_seed_rows.append(menu_by_seed)
            if args.row_level and row_path.exists():
                row_level_parts.append(pd.read_csv(row_path))
            continue
        print(f"running seed {seed}", flush=True)
        rng = np.random.default_rng(seed)
        positions = split_pos
        if args.max_rows is not None and 0 < args.max_rows < len(split_pos):
            positions = np.sort(rng.choice(split_pos, size=args.max_rows, replace=False))
        generator.rng = np.random.default_rng(seed)
        row_df = row_level_diagnostics(panel, action_cols, entity_col, bounds, generator, positions, args)
        row_df["seed"] = seed
        tau_by_seed = summarize_tau(row_df, tolerances)
        tau_by_seed["seed"] = seed
        menu_by_seed = summarize_menu_size(row_df)
        menu_by_seed["seed"] = seed
        tau_by_seed.to_csv(tau_path, index=False)
        menu_by_seed.to_csv(menu_path, index=False)
        if args.row_level:
            row_df.to_csv(row_path, index=False)
        tau_seed_rows.append(tau_by_seed)
        menu_seed_rows.append(menu_by_seed)
        if args.row_level:
            row_level_parts.append(row_df)

    tau_by_seed_all = pd.concat(tau_seed_rows, ignore_index=True)
    menu_by_seed_all = pd.concat(menu_seed_rows, ignore_index=True)
    tau_summary = summarize_across_seeds(
        tau_by_seed_all,
        ["menu_support", "selected_20pct_coverage"],
        ["group", "move_bin", "observed_in_proxy", "tau"],
    )
    menu_summary = summarize_across_seeds(
        menu_by_seed_all,
        ["mean", "median", "p25", "p75", "min", "max"],
        ["group", "move_bin", "observed_in_proxy"],
    )

    tau_by_seed_all.to_csv(out_dir / "recoverability_tau_by_seed.csv", index=False)
    tau_summary.to_csv(out_dir / "recoverability_tau_summary.csv", index=False)
    menu_by_seed_all.to_csv(out_dir / "feasible_menu_size_by_seed.csv", index=False)
    menu_summary.to_csv(out_dir / "feasible_menu_size_summary.csv", index=False)
    if args.row_level and row_level_parts:
        pd.concat(row_level_parts, ignore_index=True).to_csv(out_dir / "recoverability_tau_row_sample.csv", index=False)

    manifest = {
        "selected_root": str(selected_root),
        "package_root": str(package_root),
        "config": str(Path(args.config)),
        "out_dir": str(out_dir),
        "split": args.split,
        "n_split_rows": int(len(split_pos)),
        "seeds": seeds,
        "max_rows_per_seed": int(args.max_rows) if args.max_rows is not None else None,
        "n_candidates": int(args.n_candidates),
        "frontier": float(args.frontier),
        "tolerances": tolerances,
        "definition": "Menu support and selected coverage are computed on feasible recommendation candidates after excluding the disclosed test allocation.",
    }
    with open(out_dir / "recoverability_tau_menu_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print("\nTAU SUMMARY")
    print(tau_summary.to_string(index=False))
    print("\nFEASIBLE MENU SIZE SUMMARY")
    print(menu_summary.to_string(index=False))


if __name__ == "__main__":
    main()
