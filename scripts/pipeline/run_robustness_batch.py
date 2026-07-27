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
DEFAULT_DATA_DIR = Path(r"data\\raw")
DEFAULT_OUT_ROOT = Path("outputs/robustness_grid")
DEFAULT_REFERENCE_OUT = Path("outputs/full_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robustness variants for mandate set-valued IRL.")
    p.add_argument("--base_config", default=str(PROJECT_ROOT / "configs" / "main_32gb.yaml"))
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--reference_out", default=str(DEFAULT_REFERENCE_OUT))
    p.add_argument("--only", nargs="*", default=None, help="Run only these variant names.")
    p.add_argument("--dry_run", action="store_true", help="Write configs/manifest but do not run experiments.")
    p.add_argument("--rerun", action="store_true", help="Rerun even when main_metrics.csv already exists.")
    return p.parse_args()


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def variant_specs() -> list[dict[str, Any]]:
    common_ablation = {
        "ablation": {
            "run_no_mandate": False,
            "run_mv_inverse_optimization": False,
            "run_full_bounds_ablation": False,
            "run_point_irl": True,
        },
        "flow_validation": {"enabled": False},
        "plots": {"enabled": False},
    }

    specs: list[dict[str, Any]] = []

    # Candidate-count and seed grid; reuse the n=40, seed=42 reference run.
    for n in [32, 40, 64]:
        for seed in [7, 42, 123]:
            name = f"candidate_n{n}_seed{seed}"
            if n == 40 and seed == 42:
                specs.append({
                    "name": name,
                    "group": "candidate",
                    "reuse_reference": True,
                    "overrides": {},
                    "notes": "selected reference run",
                })
                continue
            specs.append({
                "name": name,
                "group": "candidate",
                "overrides": deep_update(common_ablation, {
                    "sample": {"random_seed": seed},
                    "candidates": {"n_candidates": n, "chunk_size": 128},
                }),
            })

    # One-change-at-a-time mandate/bound tightness robustness.
    bound_specs = [
        ("bounds_history_buffer_005", {"bounds": {"history_buffer": 0.05}}),
        ("bounds_history_buffer_015", {"bounds": {"history_buffer": 0.15}}),
        ("bounds_peer_lower_003", {"bounds": {"peer_lower_q": 0.03}}),
        ("bounds_peer_upper_0995", {"bounds": {"peer_upper_q": 0.995}}),
        ("bounds_no_adv_soft_caps", {"bounds": {"use_adv_soft_caps": False}}),
    ]
    for name, patch in bound_specs:
        specs.append({
            "name": name,
            "group": "bounds",
            "overrides": deep_update(common_ablation, patch),
        })

    # Scope/cluster robustness.
    for k in [8, 24]:
        specs.append({
            "name": f"cluster_k{k}",
            "group": "scope_cluster",
            "overrides": deep_update(common_ablation, {"clusters": {"n_clusters": k}}),
        })

    # Alternative time split.
    specs.append({
        "name": "time_split_train2022_calib2023",
        "group": "time_split",
        "overrides": deep_update(common_ablation, {
            "sample": {"train_end": "2022-12-31", "calib_end": "2023-12-31"}
        }),
    })

    # Additional checks.
    specs.append({
        "name": "optional_conformal_alpha010",
        "group": "optional_conformal",
        "overrides": deep_update(common_ablation, {"conformal": {"enabled": True, "alpha": 0.10}}),
    })
    specs.append({
        "name": "optional_flow_validation",
        "group": "optional_flow",
        "overrides": deep_update(common_ablation, {"flow_validation": {"enabled": True}}),
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
            if str(row.get("split", "")) not in {"calib", "test"}:
                continue
            d = row.to_dict()
            d.update({
                "variant": spec["name"],
                "robustness_group": spec["group"],
                "status": status,
                "elapsed_sec": elapsed_sec,
                "out_dir": str(out_dir),
                "error": error,
            })
            rows.append(d)
    else:
        rows.append({
            "variant": spec["name"],
            "robustness_group": spec["group"],
            "status": status,
            "elapsed_sec": elapsed_sec,
            "out_dir": str(out_dir),
            "error": error or "main_metrics.csv missing",
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config)
    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    reference_out = Path(args.reference_out)
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
    summary_frames = []

    for idx, spec in enumerate(specs, start=1):
        name = spec["name"]
        out_dir = reference_out if spec.get("reuse_reference") else out_root / name
        cfg_path = config_dir / f"{name}.yaml"
        cfg = deep_update(base_cfg, spec.get("overrides", {}))
        cfg["data_dir"] = str(data_dir).replace("\\", "/")
        cfg["out_dir"] = str(out_dir).replace("\\", "/")
        write_yaml(cfg_path, cfg)
        manifest_rows.append({
            "variant": name,
            "robustness_group": spec["group"],
            "config_path": str(cfg_path),
            "out_dir": str(out_dir),
            "reuse_reference": bool(spec.get("reuse_reference", False)),
            "overrides_json": json.dumps(spec.get("overrides", {}), sort_keys=True),
            "notes": spec.get("notes", ""),
        })

        print(f"\n[{idx}/{len(specs)}] {name} -> {out_dir}", flush=True)
        metrics_path = out_dir / "main_metrics.csv"
        if spec.get("reuse_reference"):
            status = "reused_main" if metrics_path.exists() else "missing_main"
            summary_frames.append(summarize_result(spec, out_dir, status, None))
            continue
        if metrics_path.exists() and not args.rerun:
            print(f"Skipping completed variant: {name}", flush=True)
            summary_frames.append(summarize_result(spec, out_dir, "already_completed", None))
            continue
        if args.dry_run:
            summary_frames.append(summarize_result(spec, out_dir, "dry_run", None))
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
        with stdout_path.open("w", encoding="utf-8", errors="replace") as out_fh, stderr_path.open("w", encoding="utf-8", errors="replace") as err_fh:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=out_fh, stderr=err_fh, text=True)
        elapsed = time.time() - start
        status = "completed" if proc.returncode == 0 and metrics_path.exists() else "failed"
        error = ""
        if status != "completed":
            error = f"returncode={proc.returncode}; metrics_exists={metrics_path.exists()}"
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
        pd.concat(summary_frames, ignore_index=True).to_csv(out_root / "robustness_summary.csv", index=False)

    pd.DataFrame(manifest_rows).to_csv(out_root / "robustness_manifest.csv", index=False)
    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        summary.to_csv(out_root / "robustness_summary.csv", index=False)
        primary = summary[
            summary.get("model", pd.Series(dtype=object)).astype(str).eq("exogenous_polytope")
            & summary.get("split", pd.Series(dtype=object)).astype(str).eq("test")
        ].copy()
        keep = [
            "variant", "robustness_group", "status", "n", "choice_coverage",
            "recommendation_choice_coverage", "observed_feasible_under_bounds",
            "point_pred_mandate_violation_rate_ex_observed",
            "action_set_fraction_mean", "out_dir",
        ]
        keep = [c for c in keep if c in primary.columns]
        primary[keep].to_csv(out_root / "robustness_primary_test_summary.csv", index=False)
        print("\nPrimary test summary:")
        print(primary[keep].to_string(index=False), flush=True)
    print(f"\nDone. Robustness outputs saved to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
