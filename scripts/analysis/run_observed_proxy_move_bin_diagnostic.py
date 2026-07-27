from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mandate_setirl.bounds import action_violations, fit_mandate_bounds
from mandate_setirl.candidates import CandidateGenerator
from mandate_setirl.utils import read_yaml
from scripts.pipeline.run_transition_menu_diagnostics import (
    frontier_metrics,
    l1,
    load_action_columns,
    move_bins,
    summarize_group,
    transition_cfg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--selected_root", default=str(repo_root / "outputs" / "full_pipeline"))
    parser.add_argument("--config", default="configs/main_32gb.yaml")
    parser.add_argument("--out_dir", default=str(repo_root / "outputs" / "observed_proxy_by_move"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n_candidates", type=int, default=40)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--write_row_level", action="store_true")
    parser.add_argument("--skip_candidates", action="store_true")
    parser.add_argument("--transition_magnitude_count", type=int, default=12)
    parser.add_argument("--transition_stress_count", type=int, default=6)
    return parser.parse_args()


def _lag_action_matrix(panel: pd.DataFrame, action_cols: list[str]) -> np.ndarray:
    out = np.zeros((len(panel), len(action_cols)), dtype=float)
    for j, c in enumerate(action_cols):
        lag_col = f"lag_{c}"
        src = panel[lag_col] if lag_col in panel.columns else panel[c]
        out[:, j] = pd.to_numeric(src, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _positions(panel: pd.DataFrame, split: str, max_rows: int | None, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(panel["split"].astype(str).eq(split).to_numpy())
    if max_rows is not None and 0 < max_rows < len(pos):
        pos = np.sort(rng.choice(pos, size=max_rows, replace=False))
    return pos


def _feasibility_rows(
    panel: pd.DataFrame,
    action_cols: list[str],
    entity_col: str,
    bounds,
    positions: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    actions = panel[action_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lag_actions = _lag_action_matrix(panel, action_cols)
    lag_l1 = l1(actions, lag_actions)
    lower, upper = bounds.bounds_for_positions(positions)
    observed_out = action_violations(actions[positions], lower, upper, tol=1e-7)
    rows = pd.DataFrame(
        {
            "row_pos": positions.astype(int),
            "series_id": panel.loc[positions, entity_col].astype(str).to_numpy(),
            "report_date": panel.loc[positions, "report_date"].to_numpy() if "report_date" in panel.columns else "",
            "split": split_name,
            "move_bin": move_bins(lag_l1[positions]),
            "lag_l1_to_obs": lag_l1[positions],
            "observed_in_mandate_proxy": (~observed_out).astype(float),
            "observed_out_of_proxy": observed_out.astype(float),
        }
    )
    return rows


def _candidate_diagnostics(
    panel: pd.DataFrame,
    action_cols: list[str],
    asset_cols: list[str],
    overlay_cols: list[str],
    entity_col: str,
    cfg: dict,
    bounds,
    positions: np.ndarray,
    feasibility: pd.DataFrame,
    args: argparse.Namespace,
    gen: CandidateGenerator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if gen is None:
        rng = np.random.default_rng(args.seed)
        gen = CandidateGenerator(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, rng, bounds)
    actions_all = panel[action_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lag_actions_all = _lag_action_matrix(panel, action_cols)
    row_map = feasibility.set_index("row_pos")
    row_rows: list[dict] = []
    selected_rows: list[dict] = []
    fractions = [0.05, 0.10, 0.20, 0.30, 0.50]

    for start in range(0, len(positions), args.chunk_size):
        pos = positions[start : start + args.chunk_size]
        batch = gen.generate_for_positions(pos, n_candidates=args.n_candidates)
        lower, upper = bounds.bounds_for_positions(pos)
        cand_viol = action_violations(batch.actions, lower, upper, tol=1e-7)
        for bi, row_pos in enumerate(pos):
            obs = actions_all[row_pos]
            cand = batch.actions[bi]
            rec_idx = np.arange(cand.shape[0])
            rec_idx = rec_idx[rec_idx != batch.observed_index]
            dist = l1(cand[rec_idx], obs)
            viol = cand_viol[bi, rec_idx]
            sources = batch.sources[bi, rec_idx].astype(str)
            source_priority = np.array([0 if str(s).startswith("transition") else 1 for s in sources], dtype=float)
            donor_distance = np.nan_to_num(batch.donor_retrieval_distance[bi, rec_idx], nan=1e18)
            lag_actions = lag_actions_all[row_pos]
            lag_dist = l1(cand[rec_idx], lag_actions)
            orders = {
                "oracle_upper_bound": np.argsort(dist, kind="mergesort"),
                "size_matched_persistence": np.argsort(lag_dist, kind="mergesort"),
                "transition_source_then_persistence": np.lexsort((np.arange(len(rec_idx)), lag_dist, source_priority)),
                "transition_knn_retrieval": np.lexsort((np.arange(len(rec_idx)), donor_distance)),
            }
            base = row_map.loc[int(row_pos)].to_dict()
            nearest = int(np.argmin(dist))
            row_rows.append(
                {
                    **base,
                    "oracle_cover": float(dist[nearest] <= args.tol),
                    "oracle_min_l1": float(dist[nearest]),
                    "candidate_count": int(len(rec_idx)),
                    "duplicate_rate_rounded_1e6": float(
                        1.0 - len({tuple(np.round(x, 6)) for x in cand[rec_idx]}) / max(len(rec_idx), 1)
                    ),
                }
            )
            for model, order in orders.items():
                for m in frontier_metrics(dist, viol, sources, order, fractions, args.tol):
                    m.update(
                        {
                            **base,
                            "model": model,
                            "row_pos": int(row_pos),
                        }
                    )
                    selected_rows.append(m)
    return pd.DataFrame(row_rows), pd.DataFrame(selected_rows)


def main() -> None:
    args = parse_args()
    selected_root = Path(args.selected_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = read_yaml(args.config)
    cfg = transition_cfg(base_cfg, args)
    action_meta = load_action_columns(selected_root)
    action_cols = action_meta["action_cols"]
    asset_cols = action_meta["asset_action_cols"]
    overlay_cols = action_meta["overlay_action_cols"]
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")

    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
    rng = np.random.default_rng(args.seed)
    all_split_pos = np.flatnonzero(panel["split"].astype(str).eq(args.split).to_numpy())
    sample_pos = _positions(panel, args.split, args.max_rows, rng)

    full_feas = _feasibility_rows(panel, action_cols, entity_col, bounds, all_split_pos, args.split)
    sample_feas = _feasibility_rows(panel, action_cols, entity_col, bounds, sample_pos, args.split)

    full_summary = summarize_group(
        full_feas,
        ["move_bin", "observed_in_mandate_proxy"],
        ["observed_out_of_proxy", "lag_l1_to_obs"],
    )
    sample_summary = summarize_group(
        sample_feas,
        ["move_bin", "observed_in_mandate_proxy"],
        ["observed_out_of_proxy", "lag_l1_to_obs"],
    )
    full_summary.to_csv(out_dir / "observed_proxy_by_move_bin_full_split.csv", index=False)
    sample_summary.to_csv(out_dir / "observed_proxy_by_move_bin_candidate_sample.csv", index=False)
    if args.write_row_level:
        full_feas.to_csv(out_dir / "observed_proxy_row_level_full_split.csv", index=False)
        sample_feas.to_csv(out_dir / "observed_proxy_row_level_candidate_sample.csv", index=False)

    candidate_row_summary = pd.DataFrame()
    candidate_frontier_summary = pd.DataFrame()
    if not args.skip_candidates:
        row_df, selected_df = _candidate_diagnostics(
            panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, bounds, sample_pos, sample_feas, args
        )
        candidate_row_summary = summarize_group(
            row_df,
            ["move_bin", "observed_in_mandate_proxy"],
            ["oracle_cover", "oracle_min_l1", "duplicate_rate_rounded_1e6", "lag_l1_to_obs"],
        )
        candidate_frontier_summary = summarize_group(
            selected_df,
            ["model", "frontier", "move_bin", "observed_in_mandate_proxy"],
            ["coverage", "distance", "violation"],
        )
        candidate_row_summary.to_csv(out_dir / "observed_proxy_move_bin_oracle_candidate_sample.csv", index=False)
        candidate_frontier_summary.to_csv(out_dir / "observed_proxy_move_bin_frontier_candidate_sample.csv", index=False)
        if args.write_row_level:
            row_df.to_csv(out_dir / "observed_proxy_candidate_row_level.csv", index=False)
            selected_df.to_csv(out_dir / "observed_proxy_candidate_frontier_row_level.csv", index=False)

    manifest = {
        "selected_root": str(selected_root),
        "config": str(Path(args.config).resolve()),
        "out_dir": str(out_dir),
        "split": args.split,
        "n_split_rows": int(len(all_split_pos)),
        "n_candidate_sample_rows": int(len(sample_pos)),
        "max_rows": args.max_rows,
        "n_candidates": args.n_candidates,
        "tolerance": args.tol,
        "notes": [
            "observed_in_mandate_proxy is computed from the disclosed allocation against exogenous bounds.",
            "candidate diagnostics exclude the observed candidate from recommendation/oracle metrics.",
            "candidate diagnostics use the routed transition-menu configuration.",
        ],
    }
    with open(out_dir / "observed_proxy_move_bin_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print("\nFULL SPLIT OBSERVED FEASIBILITY")
    print(full_summary.to_string(index=False))
    print("\nCANDIDATE SAMPLE OBSERVED FEASIBILITY")
    print(sample_summary.to_string(index=False))
    if not candidate_row_summary.empty:
        print("\nCANDIDATE SAMPLE ORACLE BY OBSERVED FEASIBILITY")
        print(candidate_row_summary.to_string(index=False))
        print("\nCANDIDATE SAMPLE FRONTIER BY OBSERVED FEASIBILITY")
        print(candidate_frontier_summary.to_string(index=False))


if __name__ == "__main__":
    main()
