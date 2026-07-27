#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "robustness"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robustness experiments.")
    p.add_argument("--base_config", default=str(PROJECT_ROOT / "configs" / "main_32gb.yaml"))
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--only", nargs="*", default=None, help="Run only these variant names.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--rerun", action="store_true")
    return p.parse_args()


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def no_extra_ablation() -> dict[str, Any]:
    return {
        "ablation": {
            "run_no_mandate": False,
            "run_mv_inverse_optimization": False,
            "run_full_bounds_ablation": False,
            "run_point_irl": True,
        },
        "conformal": {"enabled": False},
        "flow_validation": {"enabled": False},
        "plots": {"enabled": False},
    }


def with_global_mv() -> dict[str, Any]:
    return {
        "ablation": {
            "run_no_mandate": True,
            "run_mv_inverse_optimization": True,
            "run_full_bounds_ablation": False,
            "run_point_irl": True,
        },
        "conformal": {"enabled": False},
        "flow_validation": {"enabled": False},
        "plots": {"enabled": False},
    }


def variant_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    specs.append({
        "name": "grouped_metrics_n40",
        "group": "grouped_metrics_clipped",
        "overrides": no_extra_ablation(),
        "notes": "support-mode setting with grouped L1 metrics and clipped action cleaning",
    })

    exact_common = deep_update(no_extra_ablation(), {
        "candidates": {"n_candidates": 20, "chunk_size": 64},
        "irl": {"max_eval_obs": 1000, "exact_max_candidates_per_obs": 20},
    })
    specs.append({
        "name": "support_subsample_n20_eval1000",
        "group": "exact_lp_subsample",
        "overrides": deep_update(exact_common, {"irl": {"membership_mode": "support"}}),
    })
    specs.append({
        "name": "exact_lp_subsample_n20_eval1000",
        "group": "exact_lp_subsample",
        "overrides": deep_update(exact_common, {"irl": {"membership_mode": "lp_exact"}}),
    })

    specs.append({
        "name": "action_drop_negative_derivative_short",
        "group": "action_cleaning",
        "overrides": deep_update(no_extra_ablation(), {
            "actions": {
                "summary_action_cleaning": {
                    "drop_rows": {
                        "enabled": True,
                        "columns": ["derivative_pct", "short_pct"],
                        "lower_bounds": {"derivative_pct": 0.0, "short_pct": 0.0},
                    }
                }
            }
        }),
        "notes": "drops rows where signed derivative/short exposures are negative instead of clipping them to zero",
    })
    specs.append({
        "name": "action_signed_split_derivative_short",
        "group": "action_cleaning",
        "overrides": deep_update(no_extra_ablation(), {
            "actions": {
                "summary_action_cleaning": {
                    "signed_split_cols": ["derivative_pct", "short_pct"],
                }
            }
        }),
        "notes": "splits signed derivative/short exposures into positive and negative-absolute overlay actions",
    })

    specs.append({
        "name": "candidate_n64_seed42",
        "group": "global_mv_benchmarks",
        "overrides": deep_update(with_global_mv(), {
            "sample": {"random_seed": 42},
            "candidates": {"n_candidates": 64, "chunk_size": 128},
        }),
    })
    specs.append({
        "name": "menu_no_lagged_self_candidates",
        "group": "candidate_menu_ablation",
        "overrides": deep_update(no_extra_ablation(), {
            "candidates": {
                "peer_train_frac": 0.9,
                "self_frac": 0.0,
                "dirichlet_frac": 0.0,
                "frontier_frac": 0.1,
            },
        }),
        "notes": "removes lagged self candidates; train-period peer and frontier candidates remain",
    })
    specs.append({
        "name": "menu_peer_train_only_candidates",
        "group": "candidate_menu_ablation",
        "overrides": deep_update(no_extra_ablation(), {
            "candidates": {
                "peer_train_frac": 1.0,
                "self_frac": 0.0,
                "dirichlet_frac": 0.0,
                "frontier_frac": 0.0,
            },
        }),
        "notes": "uses only train-period peer candidates after excluding lagged self and frontier candidates",
    })
    specs.append({
        "name": "cluster_k8",
        "group": "global_mv_benchmarks",
        "overrides": deep_update(with_global_mv(), {
            "clusters": {"n_clusters": 8},
        }),
    })
    specs.append({
        "name": "peer_upper_0995",
        "group": "global_mv_benchmarks",
        "overrides": deep_update(with_global_mv(), {
            "bounds": {"peer_upper_q": 0.995},
        }),
    })
    specs.append({
        "name": "time_split_train2022_calib2023",
        "group": "global_mv_benchmarks",
        "overrides": deep_update(with_global_mv(), {
            "sample": {"train_end": "2022-12-31", "calib_end": "2023-12-31"},
        }),
    })
    specs.append({
        "name": "selected_k8_peer_upper_0995",
        "group": "selected_specification",
        "overrides": deep_update(with_global_mv(), {
            "clusters": {"n_clusters": 8},
            "bounds": {"peer_upper_q": 0.995},
        }),
        "notes": "eight clusters and peer upper quantile 0.995",
    })
    specs.append({
        "name": "signed_split_global_mv_representative",
        "group": "signed_split_representative_global_mv",
        "overrides": deep_update(with_global_mv(), {
            "actions": {
                "summary_action_cleaning": {
                    "signed_split_cols": ["derivative_pct", "short_pct"],
                }
            }
        }),
        "notes": "global and mean-variance baselines under signed derivative and short exposure representation",
    })

    return specs


def write_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False, allow_unicode=False)


def load_base_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def command_for(config_path: Path, data_dir: Path, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_full_experiment.py"),
        "--config",
        str(config_path),
        "--data_dir",
        str(data_dir),
        "--out_dir",
        str(out_dir),
    ]


def summarize_result(spec: dict[str, Any], out_dir: Path, status: str, elapsed_sec: float | None, error: str = "") -> pd.DataFrame:
    metrics_path = out_dir / "main_metrics.csv"
    rows: list[dict[str, Any]] = []
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        for _, row in metrics.iterrows():
            d = row.to_dict()
            d.update({
                "variant": spec["name"],
                "experiment_group": spec["group"],
                "status": status,
                "elapsed_sec": elapsed_sec,
                "out_dir": str(out_dir),
                "error": error,
                "notes": spec.get("notes", ""),
            })
            rows.append(d)
    else:
        rows.append({
            "variant": spec["name"],
            "experiment_group": spec["group"],
            "status": status,
            "elapsed_sec": elapsed_sec,
            "out_dir": str(out_dir),
            "error": error or "main_metrics.csv missing",
            "notes": spec.get("notes", ""),
        })
    return pd.DataFrame(rows)


def write_derived_summaries(out_root: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    summary.to_csv(out_root / "robustness_summary.csv", index=False)
    split_col = summary["split"] if "split" in summary.columns else pd.Series("", index=summary.index)
    test = summary[split_col.astype(str).eq("test")].copy()
    primary_models = [
        "exogenous_polytope",
        "point_minimax_preference",
        "mv_inverse_optimization_polytope",
        "global_polytope",
    ]
    model_col = test["model"] if "model" in test.columns else pd.Series("", index=test.index)
    primary = test[model_col.astype(str).isin(primary_models)].copy()
    keep = [
        "variant", "experiment_group", "status", "model", "n", "choice_coverage",
        "recommendation_choice_coverage", "observed_feasible_under_bounds",
        "point_pred_mandate_violation_rate_ex_observed",
        "action_set_fraction_mean", "distance_to_set_l1_mean",
        "distance_to_set_asset_only_l1_mean", "distance_to_set_overlay_only_l1_mean",
        "distance_to_set_risk_overlay_l1_mean", "out_dir",
    ]
    keep = [c for c in keep if c in primary.columns]
    primary[keep].to_csv(out_root / "primary_test_summary.csv", index=False)

    exact = primary[primary["experiment_group"].eq("exact_lp_subsample")].copy()
    if not exact.empty:
        exact[keep].to_csv(out_root / "exact_lp_subsample_comparison.csv", index=False)

    action = primary[primary["experiment_group"].isin(["grouped_metrics_clipped", "action_cleaning"])].copy()
    if not action.empty:
        action[keep].to_csv(out_root / "action_cleaning_robustness_summary.csv", index=False)

    rep = primary[primary["experiment_group"].eq("global_mv_benchmarks")].copy()
    if not rep.empty:
        rep[keep].to_csv(out_root / "representative_global_mv_summary.csv", index=False)

    final_extra = primary[primary["experiment_group"].isin([
        "selected_specification",
        "signed_split_representative_global_mv",
    ])].copy()
    if not final_extra.empty:
        final_extra[keep].to_csv(out_root / "selected_specification_summary.csv", index=False)

    print("\nPrimary test summary:", flush=True)
    if keep:
        print(primary[keep].to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config)
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    config_dir = out_root / "configs"
    log_dir = out_root / "logs"
    status_dir = out_root / "status"
    for p in [config_dir, log_dir, status_dir]:
        p.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_config(base_config_path)
    specs = variant_specs()
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s["name"] in wanted]
        missing = sorted(wanted.difference({s["name"] for s in specs}))
        if missing:
            raise SystemExit(f"Unknown variants requested: {missing}")

    manifest_rows = []
    summary_frames: list[pd.DataFrame] = []

    for idx, spec in enumerate(specs, start=1):
        name = spec["name"]
        out_dir = out_root / name
        cfg_path = config_dir / f"{name}.yaml"
        cfg = deep_update(base_cfg, spec.get("overrides", {}))
        cfg["data_dir"] = str(data_dir).replace("\\", "/")
        cfg["out_dir"] = str(out_dir).replace("\\", "/")
        write_yaml(cfg_path, cfg)
        manifest_rows.append({
            "variant": name,
            "experiment_group": spec["group"],
            "config_path": str(cfg_path),
            "out_dir": str(out_dir),
            "overrides_json": json.dumps(spec.get("overrides", {}), sort_keys=True),
            "notes": spec.get("notes", ""),
        })

        print(f"\n[{idx}/{len(specs)}] {name} -> {out_dir}", flush=True)
        metrics_path = out_dir / "main_metrics.csv"
        if metrics_path.exists() and not args.rerun:
            print(f"Skipping completed variant: {name}", flush=True)
            summary_frames.append(summarize_result(spec, out_dir, "already_completed", None))
            write_derived_summaries(out_root, pd.concat(summary_frames, ignore_index=True))
            continue
        if args.dry_run:
            summary_frames.append(summarize_result(spec, out_dir, "dry_run", None))
            write_derived_summaries(out_root, pd.concat(summary_frames, ignore_index=True))
            continue

        cmd = command_for(cfg_path, data_dir, out_dir)
        stdout_path = log_dir / f"{name}.stdout.log"
        stderr_path = log_dir / f"{name}.stderr.log"
        status_path = status_dir / f"{name}.json"
        status_path.write_text(json.dumps({
            "variant": name,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command": cmd,
            "out_dir": str(out_dir),
        }, indent=2), encoding="utf-8")
        start = time.time()
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        with stdout_path.open("w", encoding="utf-8", errors="replace") as out_fh, stderr_path.open("w", encoding="utf-8", errors="replace") as err_fh:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=out_fh, stderr=err_fh, text=True, env=env)
        elapsed = time.time() - start
        status = "completed" if proc.returncode == 0 and metrics_path.exists() else "failed"
        error = "" if status == "completed" else f"returncode={proc.returncode}; metrics_exists={metrics_path.exists()}"
        status_path.write_text(json.dumps({
            "variant": name,
            "status": status,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "out_dir": str(out_dir),
            "error": error,
        }, indent=2), encoding="utf-8")
        print(f"{name}: {status} in {elapsed/60:.2f} min", flush=True)
        summary_frames.append(summarize_result(spec, out_dir, status, elapsed, error))
        write_derived_summaries(out_root, pd.concat(summary_frames, ignore_index=True))

    pd.DataFrame(manifest_rows).to_csv(out_root / "robustness_manifest.csv", index=False)
    if summary_frames:
        write_derived_summaries(out_root, pd.concat(summary_frames, ignore_index=True))
    print(f"\nRobustness outputs saved to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
