from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from out_of_proxy_predictive_validity import (
    DEFAULT_PACKAGE_ROOT,
    add_package,
    as_numeric_frame,
    build_outcome_panel,
    load_action_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_root", default="outputs/full_pipeline")
    parser.add_argument("--transition_rowlevel", default="outputs/transition_diagnostics/transition_menu_row_level.csv")
    parser.add_argument("--out_dir", default="outputs/traceability_cases")
    parser.add_argument("--package_root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def top_violation_labels(action: np.ndarray, lower: np.ndarray, upper: np.ndarray, cols: list[str], k: int = 4) -> tuple[float, str]:
    gap = np.maximum(lower - action, 0.0) + np.maximum(action - upper, 0.0)
    severity = float(gap.sum())
    if severity <= 1e-12:
        return severity, ""
    order = np.argsort(-gap)[:k]
    labels = [f"{cols[j]}={gap[j]:.3f}" for j in order if gap[j] > 1e-12]
    return severity, "; ".join(labels)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_root = Path(args.selected_root)
    package_root = Path(args.package_root)
    config_path = Path(args.config) if args.config else package_root / "configs" / "main_32gb.yaml"
    action_violations, fit_mandate_bounds, read_yaml = add_package(package_root)
    cfg = read_yaml(config_path)
    entity_col = cfg.get("keys", {}).get("entity_col", "series_id")
    ac = load_action_columns(selected_root)
    action_cols = ac["action_cols"]
    asset_cols = ac["asset_action_cols"]
    overlay_cols = ac["overlay_action_cols"]
    panel = pd.read_parquet(selected_root / "panel_model_ready.parquet").reset_index(drop=True)
    bounds = fit_mandate_bounds(panel, action_cols, asset_cols, overlay_cols, entity_col, cfg, mode="exogenous")
    actions = as_numeric_frame(panel, action_cols).to_numpy(dtype=float)
    out_proxy = action_violations(actions, bounds.lower, bounds.upper, tol=1e-7)
    outcome = build_outcome_panel(panel, action_cols, entity_col)
    outcome["row_pos"] = np.arange(len(panel))
    outcome["out_of_proxy"] = out_proxy.astype(int)
    lower = bounds.lower
    upper = bounds.upper
    cards = []

    test = outcome[outcome["split"].astype(str).eq("test")].copy()
    risk_candidates = test[
        (test["out_of_proxy"].eq(1))
        & pd.to_numeric(test["next_style_drift_l1"], errors="coerce").notna()
    ].copy()
    if not risk_candidates.empty:
        risk_candidates["rank_score"] = pd.to_numeric(risk_candidates["next_style_drift_l1"], errors="coerce")
        r = risk_candidates.sort_values("rank_score", ascending=False).iloc[0]
        pos = int(r["row_pos"])
        severity, drivers = top_violation_labels(actions[pos], lower[pos], upper[pos], action_cols)
        cards.append(
            {
                "case_id": "out_of_proxy_future_drift",
                "row_pos": pos,
                "series_id": str(panel.loc[pos, entity_col]),
                "report_date": str(panel.loc[pos, "report_date"]),
                "mandate_cluster": panel.loc[pos, "mandate_cluster"] if "mandate_cluster" in panel.columns else "",
                "move_bin": "",
                "out_of_proxy": 1,
                "exception_severity_l1": severity,
                "top_exception_drivers": drivers,
                "current_style_drift_l1": float(r["style_drift_l1"]),
                "lag_style_drift_l1": float(r["lag_style_drift_l1"]) if pd.notna(r["lag_style_drift_l1"]) else np.nan,
                "next_style_drift_l1": float(r["next_style_drift_l1"]),
                "oracle_cover": np.nan,
                "oracle_min_l1": np.nan,
                "interpretation": "Out-of-proxy disclosed allocation followed by high next-report style drift.",
            }
        )

    transition_path = Path(args.transition_rowlevel)
    if transition_path.exists():
        transition = pd.read_csv(transition_path)
        transition["row_pos"] = pd.to_numeric(transition["row_pos"], errors="coerce").astype("Int64")
        transition = transition.dropna(subset=["row_pos"]).copy()
        transition["row_pos"] = transition["row_pos"].astype(int)
        transition["out_of_proxy"] = out_proxy[transition["row_pos"].to_numpy(dtype=int)].astype(int)
        unrecoverable = transition[
            transition["move_bin"].astype(str).eq("mover_l1_gt_025")
            & transition["out_of_proxy"].eq(0)
            & (pd.to_numeric(transition["oracle_cover"], errors="coerce") == 0)
        ].copy()
        if not unrecoverable.empty:
            unrecoverable["rank_score"] = pd.to_numeric(unrecoverable["oracle_min_l1"], errors="coerce")
            r = unrecoverable.sort_values("rank_score", ascending=False).iloc[0]
            pos = int(r["row_pos"])
            severity, drivers = top_violation_labels(actions[pos], lower[pos], upper[pos], action_cols)
            out_row = outcome.loc[pos]
            cards.append(
                {
                    "case_id": "in_proxy_large_mover_unrecoverable",
                    "row_pos": pos,
                    "series_id": str(panel.loc[pos, entity_col]),
                    "report_date": str(panel.loc[pos, "report_date"]),
                    "mandate_cluster": panel.loc[pos, "mandate_cluster"] if "mandate_cluster" in panel.columns else "",
                    "move_bin": str(r["move_bin"]),
                    "out_of_proxy": 0,
                    "exception_severity_l1": severity,
                    "top_exception_drivers": drivers,
                    "current_style_drift_l1": float(out_row["style_drift_l1"]) if pd.notna(out_row["style_drift_l1"]) else np.nan,
                    "lag_style_drift_l1": float(out_row["lag_style_drift_l1"]) if pd.notna(out_row["lag_style_drift_l1"]) else np.nan,
                    "next_style_drift_l1": float(out_row["next_style_drift_l1"]) if pd.notna(out_row["next_style_drift_l1"]) else np.nan,
                    "oracle_cover": float(r["oracle_cover"]),
                    "oracle_min_l1": float(r["oracle_min_l1"]),
                    "interpretation": "Large disclosed reallocation is inside the mandate proxy but no generated public-disclosure analogue is close.",
                }
            )

    cards_df = pd.DataFrame(cards)
    cards_df.to_csv(out_dir / "traceability_cases.csv", index=False)
    md_lines = ["# Traceability Cases", ""]
    for _, row in cards_df.iterrows():
        md_lines.append(f"## {row['case_id']}")
        for c in cards_df.columns:
            md_lines.append(f"- **{c}:** {row[c]}")
        md_lines.append("")
    (out_dir / "traceability_cases.md").write_text("\n".join(md_lines), encoding="utf-8")
    manifest = {
        "selected_root": str(selected_root),
        "transition_rowlevel": str(transition_path),
        "out_dir": str(out_dir),
        "n_cases": int(len(cards_df)),
        "notes": [
            "Cases summarize post-estimation examples.",
            "They should not be presented as randomly selected representative evidence.",
        ],
    }
    with open(out_dir / "traceability_cases_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(cards_df.to_string(index=False))


if __name__ == "__main__":
    main()
