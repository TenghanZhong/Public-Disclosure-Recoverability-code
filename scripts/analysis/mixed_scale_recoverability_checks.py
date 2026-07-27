"""Mixed-scale recoverability diagnostics.

Computes menu support under asset-block L1 distance and training-IQR-scaled
full-vector L1 distance. The script reuses the fitted candidate menus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recoverability_tau_menu_checks import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_PACKAGE_ROOT,
    DEFAULT_SELECTED_ROOT,
    add_package_paths,
    lag_action_matrix,
    move_bins,
    transition_config,
)


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = DEFAULT_REPO_ROOT / "tables" / "mixed_scale_recoverability_checks"


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
    parser.add_argument("--frontier", type=float, default=0.20)
    parser.add_argument("--full_tau", type=float, default=0.05)
    parser.add_argument("--asset_tau", type=float, default=0.05)
    parser.add_argument("--iqr_floor", type=float, default=1e-8)
    parser.add_argument("--transition_magnitude_count", type=int, default=12)
    parser.add_argument("--transition_stress_count", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).sum(axis=-1)


def scaled_l1(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.abs(a - b) / scale).sum(axis=-1)


def load_action_columns(selected_root: Path) -> dict:
    with open(selected_root / "action_columns.json", "r", encoding="utf-8") as f:
        return json.load(f)


def train_iqr_scales(
    panel: pd.DataFrame,
    action_cols: list[str],
    train_mask: np.ndarray,
    floor: float,
) -> np.ndarray:
    values = panel.loc[train_mask, action_cols].apply(pd.to_numeric, errors="coerce")
    q75 = values.quantile(0.75)
    q25 = values.quantile(0.25)
    scale = (q75 - q25).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float, copy=True)
    scale[(~np.isfinite(scale)) | (scale <= floor)] = 1.0
    return scale


def calibrated_scaled_tau(
    panel: pd.DataFrame,
    action_cols: list[str],
    train_mask: np.ndarray,
    scale: np.ndarray,
    full_tau: float,
) -> tuple[float, float]:
    actions = panel[action_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lag_actions = lag_action_matrix(panel, action_cols)
    train_idx = np.flatnonzero(train_mask)
    full_move = l1(actions[train_idx], lag_actions[train_idx])
    scaled_move = scaled_l1(actions[train_idx], lag_actions[train_idx], scale)
    valid = np.isfinite(full_move) & np.isfinite(scaled_move)
    if not np.any(valid):
        raise ValueError("No valid training rows for scaled tolerance calibration.")
    full_move = full_move[valid]
    scaled_move = scaled_move[valid]
    stable_share = float(np.mean(full_move <= full_tau))
    stable_share = min(max(stable_share, 0.0), 1.0)
    tau_scaled = float(np.quantile(scaled_move, stable_share))
    return tau_scaled, stable_share


def metric_threshold_table(full_tau: float, asset_tau: float, scaled_tau: float, stable_share: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "full_19_coordinate_l1",
                "tolerance": full_tau,
                "calibration": "prespecified full-vector tolerance",
                "training_stable_share": stable_share,
            },
            {
                "metric": "asset_block_l1",
                "tolerance": asset_tau,
                "calibration": "same asset-share L1 tolerance",
                "training_stable_share": stable_share,
            },
            {
                "metric": "training_iqr_scaled_full_l1",
                "tolerance": scaled_tau,
                "calibration": "training quantile matched to full-vector stable share",
                "training_stable_share": stable_share,
            },
        ]
    )


def summarize_seed(row_df: pd.DataFrame) -> pd.DataFrame:
    group_specs = [
        ("overall", []),
        ("move_bin", ["move_bin"]),
        ("move_bin_x_proxy_status", ["move_bin", "observed_in_proxy"]),
    ]
    metric_cols = {
        "full_19_coordinate_l1": "min_full_l1",
        "asset_block_l1": "min_asset_l1",
        "training_iqr_scaled_full_l1": "min_scaled_l1",
    }
    threshold_cols = {
        "full_19_coordinate_l1": "full_tau",
        "asset_block_l1": "asset_tau",
        "training_iqr_scaled_full_l1": "scaled_tau",
    }
    rows: list[dict] = []
    for group_name, cols in group_specs:
        grouped = [((), row_df)] if not cols else row_df.groupby(cols, dropna=False)
        for keys, g in grouped:
            if cols and not isinstance(keys, tuple):
                keys = (keys,)
            for metric, dist_col in metric_cols.items():
                threshold = float(g[threshold_cols[metric]].iloc[0])
                dist = pd.to_numeric(g[dist_col], errors="coerce")
                row: dict[str, object] = {
                    "group": group_name,
                    "metric": metric,
                    "n": int(len(g)),
                    "tolerance": threshold,
                    "menu_support": float((dist <= threshold).mean()),
                    "mean_min_distance": float(dist.mean()),
                    "median_min_distance": float(dist.median()),
                }
                for c, v in zip(cols, keys if cols else ()):
                    row[c] = v
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_seeds(seed_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["group", "metric", "move_bin", "observed_in_proxy"]
    rows: list[dict] = []
    for keys, g in seed_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        row["n_seeds"] = int(g["seed"].nunique())
        row["n_mean"] = float(pd.to_numeric(g["n"], errors="coerce").mean())
        for metric in ["menu_support", "mean_min_distance", "median_min_distance", "tolerance"]:
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_min"] = float(np.nanmin(vals))
            row[f"{metric}_max"] = float(np.nanmax(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def run_seed(
    seed: int,
    positions: np.ndarray,
    panel: pd.DataFrame,
    action_cols: list[str],
    asset_idx: np.ndarray,
    entity_col: str,
    bounds,
    generator,
    scale: np.ndarray,
    full_tau: float,
    asset_tau: float,
    scaled_tau: float,
    chunk_size: int,
    n_candidates: int,
    out_path: Path,
    reuse_existing: bool,
) -> pd.DataFrame:
    from mandate_setirl.bounds import action_violations

    if reuse_existing and out_path.exists():
        return pd.read_csv(out_path)

    actions = panel[action_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lag_actions = lag_action_matrix(panel, action_cols)
    lag_l1 = l1(actions, lag_actions)
    rows: list[dict] = []
    generator.rng = np.random.default_rng(seed)
    for start in range(0, len(positions), chunk_size):
        batch_pos = positions[start : start + chunk_size]
        print(f"seed {seed}: candidate batch {start + 1}-{start + len(batch_pos)} of {len(positions)}", flush=True)
        batch = generator.generate_for_positions(batch_pos, n_candidates=n_candidates)
        lower, upper = bounds.bounds_for_positions(batch_pos)
        observed_out = action_violations(actions[batch_pos], lower, upper, tol=1e-7)
        cand_out = action_violations(batch.actions, lower, upper, tol=1e-7)
        for bi, row_pos in enumerate(batch_pos):
            rec_idx = np.arange(batch.actions.shape[1])
            rec_idx = rec_idx[rec_idx != batch.observed_index]
            feasible_idx = rec_idx[~cand_out[bi, rec_idx]]
            row = {
                "seed": seed,
                "row_pos": int(row_pos),
                "series_id": str(panel.loc[row_pos, entity_col]),
                "report_date": panel.loc[row_pos, "report_date"] if "report_date" in panel.columns else "",
                "move_bin": move_bins(np.asarray([lag_l1[row_pos]]))[0],
                "observed_in_proxy": bool(not observed_out[bi]),
                "feasible_menu_size": int(len(feasible_idx)),
                "full_tau": float(full_tau),
                "asset_tau": float(asset_tau),
                "scaled_tau": float(scaled_tau),
            }
            if len(feasible_idx) == 0:
                row.update(
                    {
                        "min_full_l1": np.nan,
                        "min_asset_l1": np.nan,
                        "min_scaled_l1": np.nan,
                    }
                )
            else:
                obs = actions[row_pos]
                candidates = batch.actions[bi, feasible_idx]
                row.update(
                    {
                        "min_full_l1": float(np.min(l1(candidates, obs))),
                        "min_asset_l1": float(np.min(l1(candidates[:, asset_idx], obs[asset_idx]))),
                        "min_scaled_l1": float(np.min(scaled_l1(candidates, obs, scale))),
                    }
                )
            rows.append(row)
    row_df = pd.DataFrame(rows)
    row_df.to_csv(out_path, index=False)
    return row_df


def write_readme(out_dir: Path, manifest: dict, summary: pd.DataFrame) -> None:
    lines = [
        "# Mixed-scale recoverability diagnostics",
        "",
        "This diagnostic checks whether the recoverability boundary is an artifact of using a mixed 19-coordinate L1 distance.",
        "It uses the same transition-augmented candidate generator and selected result root as the main recoverability diagnostics.",
        "",
        "Metrics:",
        "- `full_19_coordinate_l1`: prespecified full action-vector L1 distance with tolerance 0.05.",
        "- `asset_block_l1`: asset-share block L1 distance with tolerance 0.05.",
        "- `training_iqr_scaled_full_l1`: full action-vector distance after dividing each coordinate gap by its training-split IQR; coordinates with near-zero IQR use scale 1.0.",
        "",
        f"Scaled tolerance is calibrated on the training split to match the full-vector stable-report share: `{manifest['training_full_tau_stable_share']:.6f}`.",
        f"Calibrated scaled tolerance: `{manifest['scaled_tau']:.6f}`.",
        "",
        "Support means:",
        "",
    ]
    reported = summary[
        summary["group"].eq("move_bin")
        & summary["metric"].isin(["full_19_coordinate_l1", "asset_block_l1", "training_iqr_scaled_full_l1"])
    ][["metric", "move_bin", "menu_support_mean"]].copy()
    lines.append(reported.to_markdown(index=False))
    lines.extend(
        [
            "",
            "Interpretation: the IQR-scaled full-vector check evaluates mixed-scale robustness. The asset-block check separates full action-profile support from asset-share support.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


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

    cfg = transition_config(read_yaml(Path(args.config)), args)
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    action_meta = load_action_columns(selected_root)
    action_cols = action_meta["action_cols"]
    asset_cols = action_meta["asset_action_cols"]
    overlay_cols = action_meta["overlay_action_cols"]
    asset_idx = np.asarray([action_cols.index(c) for c in asset_cols], dtype=int)
    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    train_mask = panel["split"].astype(str).eq("train").to_numpy()
    split_pos = np.flatnonzero(panel["split"].astype(str).eq(args.split).to_numpy())
    scale = train_iqr_scales(panel, action_cols, train_mask, args.iqr_floor)
    scaled_tau, stable_share = calibrated_scaled_tau(panel, action_cols, train_mask, scale, args.full_tau)
    thresholds = metric_threshold_table(args.full_tau, args.asset_tau, scaled_tau, stable_share)
    thresholds.to_csv(out_dir / "mixed_scale_metric_thresholds.csv", index=False)

    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
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

    seed_summaries: list[pd.DataFrame] = []
    for seed in seeds:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        positions = split_pos.copy()
        if args.max_rows is not None and 0 < args.max_rows < len(split_pos):
            positions = np.sort(rng.choice(split_pos, size=args.max_rows, replace=False))
        row_path = seed_dir / "mixed_scale_row_metrics.csv"
        row_df = pd.read_csv(row_path) if args.resume and row_path.exists() else run_seed(
            seed,
            positions,
            panel,
            action_cols,
            asset_idx,
            entity_col,
            bounds,
            generator,
            scale,
            args.full_tau,
            args.asset_tau,
            scaled_tau,
            args.chunk_size,
            args.n_candidates,
            row_path,
            args.resume,
        )
        seed_summary = summarize_seed(row_df)
        seed_summary["seed"] = seed
        seed_summary.to_csv(seed_dir / "mixed_scale_support_by_group.csv", index=False)
        seed_summaries.append(seed_summary)

    by_seed = pd.concat(seed_summaries, ignore_index=True)
    summary = summarize_across_seeds(by_seed)
    by_seed.to_csv(out_dir / "mixed_scale_support_by_seed.csv", index=False)
    summary.to_csv(out_dir / "mixed_scale_support_summary.csv", index=False)
    manifest = {
        "selected_root": str(selected_root),
        "package_root": str(package_root),
        "config": str(Path(args.config)),
        "out_dir": str(out_dir),
        "split": args.split,
        "n_split_rows": int(len(split_pos)),
        "seeds": seeds,
        "max_rows_per_seed": int(args.max_rows),
        "n_candidates": int(args.n_candidates),
        "full_tau": float(args.full_tau),
        "asset_tau": float(args.asset_tau),
        "scaled_tau": float(scaled_tau),
        "training_full_tau_stable_share": float(stable_share),
        "iqr_floor": float(args.iqr_floor),
        "small_iqr_scale": 1.0,
        "definition": "Menu support is the share of reports for which at least one feasible candidate is within the metric-specific tolerance of the held-out disclosed allocation.",
        "scaled_tolerance_rule": "Training split quantile of IQR-scaled adjacent-report moves, matched to the training share with full-vector L1 move <= full_tau.",
    }
    with open(out_dir / "mixed_scale_recoverability_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_readme(out_dir, manifest, summary)
    print(json.dumps(manifest, indent=2))
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
