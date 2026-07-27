#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import os
# Avoid BLAS/threadpool hangs in constrained notebook or CI environments.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from mandate_setirl.utils import ensure_dir, read_yaml, set_seed, smart_write_table
from mandate_setirl.panel import build_panel, audit_panel
from mandate_setirl.bounds import fit_mandate_bounds, feasible_set_metrics
from mandate_setirl.candidates import CandidateGenerator
from mandate_setirl.features import fit_utility_feature_model
from mandate_setirl.irl import fit_set_valued_irl, evaluate_irl_model, summarize_metrics, summarize_frontier_metrics, save_preference_summary
from mandate_setirl.baselines import evaluate_simple_baselines, fit_mv_inverse_optimization_baseline
from mandate_setirl.flow import run_flow_validation


def parse_args():
    p = argparse.ArgumentParser(description="Run mandate-aware revealed-preference / set-valued IRL experiment")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--data_dir", default=None, help="Override data_dir in config")
    p.add_argument("--out_dir", default=None, help="Override out_dir in config")
    p.add_argument("--fast", action="store_true", help="Use reduced settings for a smoke run")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.fast:
        cfg.setdefault("candidates", {})["n_candidates"] = min(int(cfg.get("candidates", {}).get("n_candidates", 64)), 12)
        cfg.setdefault("candidates", {})["chunk_size"] = min(int(cfg.get("candidates", {}).get("chunk_size", 512)), 64)
        cfg.setdefault("irl", {})["max_train_obs_per_scope"] = min(int(cfg.get("irl", {}).get("max_train_obs_per_scope", 6000)), 80)
        cfg.setdefault("irl", {})["max_constraints_per_scope"] = min(int(cfg.get("irl", {}).get("max_constraints_per_scope", 50000)), 150)
        cfg.setdefault("irl", {})["max_eval_obs"] = min(int(cfg.get("irl", {}).get("max_eval_obs", 5000) or 5000), 100)
        cfg.setdefault("utility", {})["scaler_sample_obs"] = min(int(cfg.get("utility", {}).get("scaler_sample_obs", 20000)), 500)
        # Reduced settings for a schema and execution check.
        cfg.setdefault("ablation", {})["run_no_mandate"] = False
        cfg.setdefault("ablation", {})["run_mv_inverse_optimization"] = False
        cfg.setdefault("flow_validation", {})["enabled"] = False
    out_dir = ensure_dir(cfg["out_dir"])
    rng = set_seed(int(cfg.get("sample", {}).get("random_seed", 42)))

    print("[1/10] Building model-ready panel")
    pdta = build_panel(cfg, data_dir_override=cfg.get("data_dir"), out_dir=str(out_dir))
    audit_panel(pdta, out_dir)
    panel = pdta.panel.reset_index(drop=True)
    print(f"Panel rows: {len(panel):,}; actions: {len(pdta.action_cols)}; entities: {panel[pdta.entity_col].nunique():,}")

    print("[2/10] Fitting multiple feasible-bound models")
    bound_modes = list(cfg.get("bounds", {}).get("diagnostic_modes", ["global", "exogenous", "lagged_history", "full"]))
    bound_models = {}
    feasible_frames = []
    for mode in bound_modes:
        bm = fit_mandate_bounds(panel, pdta.action_cols, pdta.asset_action_cols, pdta.overlay_action_cols, pdta.entity_col, cfg, mode=mode)
        bound_models[mode] = bm
        feasible_frames.append(feasible_set_metrics(panel, bm, str(out_dir), name=mode))
    feasible_metrics = pd.concat(feasible_frames, ignore_index=True) if feasible_frames else pd.DataFrame()
    smart_write_table(feasible_metrics, out_dir / "mandate_feasible_set_metrics_all_modes.csv")
    print(feasible_metrics[feasible_metrics["split"].isin(["train", "calib", "test"])].to_string(index=False))

    main_bound_mode = str(cfg.get("bounds", {}).get("main_mode", "exogenous"))
    if main_bound_mode not in bound_models:
        bound_models[main_bound_mode] = fit_mandate_bounds(panel, pdta.action_cols, pdta.asset_action_cols, pdta.overlay_action_cols, pdta.entity_col, cfg, mode=main_bound_mode)
    main_bounds = bound_models[main_bound_mode]
    print(f"[3/10] Initializing candidate generator with main bound mode: {main_bound_mode}")
    candgen = CandidateGenerator(panel, pdta.action_cols, pdta.asset_action_cols, pdta.overlay_action_cols, pdta.entity_col, cfg, rng, main_bounds)

    print("[4/10] Fitting utility feature scaler")
    feature_model = fit_utility_feature_model(panel, candgen, cfg, rng)

    print("[5/10] Fitting mandate-aware polyhedral revealed-preference model")
    set_model = fit_set_valued_irl(panel, candgen, feature_model, cfg, pdta.entity_col, rng, model_name=f"{main_bound_mode}_polytope", out_dir=str(out_dir))

    eval_frames = []
    pred_frames = []

    # Optional split-conformal calibration of the near-optimality radius.
    # Nonconformity is the minimum support gap that includes the observed action.
    cfg_eval = copy.deepcopy(cfg)
    cfcfg = cfg.get("conformal", {})
    if cfcfg.get("enabled", False):
        print("[6/10] Calibrating set radius on calibration split")
        ev_probe, _ = evaluate_irl_model(panel, candgen, feature_model, set_model, cfg, split="calib", use_point=False, model_name=f"{main_bound_mode}_polytope_calib_probe", reference_bounds=main_bounds)
        if not ev_probe.empty and "regret_min_support" in ev_probe.columns:
            probe = ev_probe.copy()
            if cfcfg.get("feasible_only", True) and "observed_feasible_under_bounds" in probe.columns:
                probe = probe[probe["observed_feasible_under_bounds"].astype(bool)]
            scores = pd.to_numeric(probe["regret_min_support"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(scores):
                alpha = float(cfcfg.get("alpha", 0.10))
                q_level = min(1.0, max(0.0, np.ceil((len(scores) + 1) * (1 - alpha)) / max(len(scores), 1)))
                eta_conf = float(np.quantile(scores.to_numpy(dtype=float), q_level, method="higher"))
                base_eta = float(cfg.get("irl", {}).get("set_near_optimal_eta", 0.05))
                eta_final = max(base_eta, eta_conf) if cfcfg.get("never_shrink_eta", True) else eta_conf
                cfg_eval.setdefault("irl", {})["set_near_optimal_eta"] = eta_final
                smart_write_table(pd.DataFrame([{
                    "alpha": alpha,
                    "n_calibration_scores": int(len(scores)),
                    "q_level": q_level,
                    "base_eta": base_eta,
                    "eta_conformal": eta_conf,
                    "eta_final": eta_final,
                    "feasible_only": bool(cfcfg.get("feasible_only", True)),
                }]), out_dir / "conformal_eta_calibration.csv")

    for split in ["calib", "test"]:
        print(f"[6/10] Evaluating polyhedral set model on {split}")
        ev, pred = evaluate_irl_model(panel, candgen, feature_model, set_model, cfg_eval, split=split, use_point=False, model_name=f"{main_bound_mode}_polytope", reference_bounds=main_bounds)
        eval_frames.append(ev)
        pred_frames.append(pred)
        print(f"[6/10] Evaluating point baseline from same polytope on {split}")
        evp, predp = evaluate_irl_model(panel, candgen, feature_model, set_model, cfg_eval, split=split, use_point=True, model_name="point_minimax_preference", reference_bounds=main_bounds)
        eval_frames.append(evp)
        pred_frames.append(predp)

    if cfg.get("ablation", {}).get("run_mv_inverse_optimization", True):
        print("[7/10] Fitting mean-variance inverse-optimization baseline")
        mv_model = fit_mv_inverse_optimization_baseline(panel, candgen, feature_model, cfg, pdta.entity_col, rng)
        save_preference_summary(mv_model, str(out_dir), "mv_io_polytope")
        for split in ["calib", "test"]:
            evm, predm = evaluate_irl_model(panel, candgen, feature_model, mv_model, cfg, split=split, use_point=False, model_name="mv_inverse_optimization_polytope", reference_bounds=main_bounds)
            eval_frames.append(evm)
            pred_frames.append(predm)

    if cfg.get("ablation", {}).get("run_no_mandate", True):
        print("[8/10] Running no-mandate/global-bound ablation")
        global_bounds = bound_models.get("global") or fit_mandate_bounds(panel, pdta.action_cols, pdta.asset_action_cols, pdta.overlay_action_cols, pdta.entity_col, cfg, mode="global")
        candgen_global = CandidateGenerator(panel, pdta.action_cols, pdta.asset_action_cols, pdta.overlay_action_cols, pdta.entity_col, cfg, rng, global_bounds)
        feat_global = fit_utility_feature_model(panel, candgen_global, cfg, rng)
        global_model = fit_set_valued_irl(panel, candgen_global, feat_global, cfg, pdta.entity_col, rng, model_name="global_polytope", out_dir=str(out_dir))
        for split in ["calib", "test"]:
            evg, predg = evaluate_irl_model(panel, candgen_global, feat_global, global_model, cfg, split=split, use_point=False, model_name="global_polytope", reference_bounds=main_bounds)
            eval_frames.append(evg)
            pred_frames.append(predg)

    if cfg.get("ablation", {}).get("run_full_bounds_ablation", False) and main_bound_mode != "full":
        print("[8/10] Running full-bound operational ablation")
        full_bounds = bound_models.get("full") or fit_mandate_bounds(panel, pdta.action_cols, pdta.asset_action_cols, pdta.overlay_action_cols, pdta.entity_col, cfg, mode="full")
        candgen_full = CandidateGenerator(panel, pdta.action_cols, pdta.asset_action_cols, pdta.overlay_action_cols, pdta.entity_col, cfg, rng, full_bounds)
        feat_full = fit_utility_feature_model(panel, candgen_full, cfg, rng)
        full_model = fit_set_valued_irl(panel, candgen_full, feat_full, cfg, pdta.entity_col, rng, model_name="full_operational_polytope", out_dir=str(out_dir))
        for split in ["calib", "test"]:
            evf, predf = evaluate_irl_model(panel, candgen_full, feat_full, full_model, cfg, split=split, use_point=False, model_name="full_operational_polytope", reference_bounds=main_bounds)
            eval_frames.append(evf)
            pred_frames.append(predf)

    print("[9/10] Writing metrics and turnover-controlled flow validation")
    eval_df = pd.concat([x for x in eval_frames if x is not None and not x.empty], ignore_index=True) if eval_frames else pd.DataFrame()
    pred_df = pd.concat([x for x in pred_frames if x is not None and not x.empty], ignore_index=True) if pred_frames else pd.DataFrame()
    metrics = summarize_metrics(eval_df)
    frontier_metrics = summarize_frontier_metrics(eval_df)
    simple = evaluate_simple_baselines(panel, pdta.action_cols, pdta.entity_col, split="test")
    if not simple.empty:
        if metrics.empty:
            metrics = simple.copy()
        else:
            for c in metrics.columns:
                if c not in simple.columns:
                    simple[c] = pd.NA
            simple_aligned = simple[metrics.columns].dropna(axis=1, how="all")
            metrics_base = metrics.dropna(axis=1, how="all")
            # Reindex after dropping all-NA columns to avoid pandas' future concat warning.
            all_cols = list(dict.fromkeys(list(metrics_base.columns) + list(simple_aligned.columns)))
            metrics = pd.concat([metrics_base.reindex(columns=all_cols), simple_aligned.reindex(columns=all_cols)], ignore_index=True)
    smart_write_table(eval_df, out_dir / "evaluation_row_level.parquet")
    smart_write_table(pred_df, out_dir / "prediction_compatibility.parquet")
    smart_write_table(metrics, out_dir / "main_metrics.csv")
    smart_write_table(frontier_metrics, out_dir / "coverage_efficiency_frontier_metrics.csv")
    print(metrics.to_string(index=False))
    if not frontier_metrics.empty:
        print("\nCoverage-efficiency frontier:")
        print(frontier_metrics.head(40).to_string(index=False))

    if not pred_df.empty:
        primary_model_name = f"{main_bound_mode}_polytope"
        flow_pred = pred_df[pred_df["model"].eq(primary_model_name)].copy()
        flow = run_flow_validation(panel, flow_pred, cfg, pdta.entity_col, pdta.date_col, pdta.action_cols, str(out_dir))
        if not flow.empty:
            print(flow[flow["term"].astype(str).str.contains("compat", na=False)].head(20).to_string(index=False))

    if cfg.get("plots", {}).get("enabled", False):
        print("[10/10] Making figures", flush=True)
        pref_files = list(Path(out_dir).glob(f"scope_preference_summary_{main_bound_mode}_polytope.csv"))
        pref = pd.read_csv(pref_files[0]) if pref_files else None
        from mandate_setirl.plots import make_figures
        make_figures(metrics, pref, out_dir)
    print(f"Done. Results saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
    # Ensure a clean exit after all output streams are flushed.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
