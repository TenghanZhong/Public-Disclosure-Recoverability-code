from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mandate_setirl.bounds import fit_mandate_bounds
from mandate_setirl.candidates import CandidateGenerator
from mandate_setirl.utils import read_yaml
from run_observed_proxy_move_bin_diagnostic import (
    _candidate_diagnostics,
    _feasibility_rows,
    _positions,
    load_action_columns,
    summarize_group,
    transition_cfg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--selected_root", default=str(repo_root / "outputs" / "full_pipeline"))
    parser.add_argument("--config", default="configs/main_32gb.yaml")
    parser.add_argument("--out_dir", default=str(repo_root / "outputs" / "repeated_menu_diagnostics"))
    parser.add_argument("--seeds", default="7,13,21,42,57,73,88,101,123,202")
    parser.add_argument("--max_rows", type=int, default=5000)
    parser.add_argument("--n_candidates", type=int, default=40)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--transition_magnitude_count", type=int, default=12)
    parser.add_argument("--transition_stress_count", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _summarize_repeats(df: pd.DataFrame, metric_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n_seeds"] = int(g["seed"].nunique())
        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(vals))
            row[f"{m}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}_min"] = float(np.nanmin(vals))
            row[f"{m}_max"] = float(np.nanmax(vals))
            row[f"{m}_p025"] = float(np.nanquantile(vals, 0.025))
            row[f"{m}_p975"] = float(np.nanquantile(vals, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    selected_root = Path(args.selected_root)
    base_cfg = read_yaml(args.config)
    cfg = transition_cfg(base_cfg, args)
    action_meta = load_action_columns(selected_root)
    action_cols = action_meta["action_cols"]
    asset_cols = action_meta["asset_action_cols"]
    overlay_cols = action_meta["overlay_action_cols"]
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
    gen = CandidateGenerator(
        panel,
        action_cols,
        asset_cols,
        overlay_cols,
        entity_col,
        cfg,
        np.random.default_rng(args.seed if hasattr(args, "seed") else 42),
        bounds,
    )
    oracle_rows = []
    frontier_rows = []
    all_split_pos = np.flatnonzero(panel["split"].astype(str).eq("test").to_numpy())
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        oracle_path = seed_dir / "observed_proxy_move_bin_oracle_candidate_sample.csv"
        frontier_path = seed_dir / "observed_proxy_move_bin_frontier_candidate_sample.csv"
        if args.force or not (oracle_path.exists() and frontier_path.exists()):
            print(f"Running seed {seed}", flush=True)
            rng = np.random.default_rng(seed)
            sample_pos = _positions(panel, "test", args.max_rows, rng)
            sample_feas = _feasibility_rows(panel, action_cols, entity_col, bounds, sample_pos, "test")
            row_df, selected_df = _candidate_diagnostics(
                panel,
                action_cols,
                asset_cols,
                overlay_cols,
                entity_col,
                cfg,
                bounds,
                sample_pos,
                sample_feas,
                args,
                gen=gen,
            )
            oracle = summarize_group(
                row_df,
                ["move_bin", "observed_in_mandate_proxy"],
                ["oracle_cover", "oracle_min_l1", "duplicate_rate_rounded_1e6", "lag_l1_to_obs"],
            )
            frontier = summarize_group(
                selected_df,
                ["model", "frontier", "move_bin", "observed_in_mandate_proxy"],
                ["coverage", "distance", "violation"],
            )
            oracle.to_csv(oracle_path, index=False)
            frontier.to_csv(frontier_path, index=False)
            with open(seed_dir / "observed_proxy_move_bin_manifest.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "seed": seed,
                        "n_split_rows": int(len(all_split_pos)),
                        "n_candidate_sample_rows": int(len(sample_pos)),
                        "notes": [
                            "Shared transition-library diagnostic.",
                            "Observed candidate is excluded from oracle/frontier metrics.",
                        ],
                    },
                    f,
                    indent=2,
                )
        oracle = pd.read_csv(oracle_path)
        frontier = pd.read_csv(frontier_path)
        oracle["seed"] = seed
        frontier["seed"] = seed
        f20 = frontier[
            np.isclose(pd.to_numeric(frontier["frontier"], errors="coerce"), 0.20)
            & frontier["model"].astype(str).eq("transition_source_then_persistence")
        ].copy()
        oracle_rows.append(oracle)
        frontier_rows.append(f20)

    oracle_all = pd.concat(oracle_rows, ignore_index=True)
    frontier_all = pd.concat(frontier_rows, ignore_index=True)
    oracle_summary = _summarize_repeats(
        oracle_all,
        ["oracle_cover", "oracle_min_l1", "duplicate_rate_rounded_1e6", "lag_l1_to_obs"],
        ["move_bin", "observed_in_mandate_proxy"],
    )
    frontier_summary = _summarize_repeats(
        frontier_all,
        ["coverage", "distance", "violation"],
        ["move_bin", "observed_in_mandate_proxy"],
    )
    oracle_all.to_csv(out_dir / "menu_support_by_seed.csv", index=False)
    frontier_all.to_csv(out_dir / "transition_selection_by_seed.csv", index=False)
    oracle_summary.to_csv(out_dir / "repeated_menu_support_summary.csv", index=False)
    frontier_summary.to_csv(out_dir / "repeated_transition_selection_summary.csv", index=False)
    manifest = {
        "selected_root": args.selected_root,
        "config": args.config,
        "out_dir": str(out_dir),
        "seeds": seeds,
        "max_rows_per_seed": args.max_rows,
        "n_candidates": args.n_candidates,
        "frontier_model": "transition_source_then_persistence",
        "frontier": 0.20,
        "notes": [
            "Each seed draws a fresh diagnostic sample from the test split.",
            "Observed candidate is excluded by the underlying diagnostic script.",
            "Summary intervals are empirical 2.5 and 97.5 percentiles across repeated samples.",
        ],
    }
    with open(out_dir / "repeated_menu_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\nORACLE SUMMARY")
    print(oracle_summary.to_string(index=False))
    print("\nF20 TRANSITION SUMMARY")
    print(frontier_summary.to_string(index=False))


if __name__ == "__main__":
    main()
