from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir


def make_figures(metrics_df: pd.DataFrame, pref_summary: pd.DataFrame | None, out_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = ensure_dir(Path(out_dir) / "figures")
    if metrics_df is not None and not metrics_df.empty and {"action_set_fraction_mean", "choice_coverage", "model"}.issubset(metrics_df.columns):
        plt.figure(figsize=(7, 5))
        for _, row in metrics_df.iterrows():
            plt.scatter(row["action_set_fraction_mean"], row["choice_coverage"])
            plt.text(row["action_set_fraction_mean"], row["choice_coverage"], str(row["model"])[:20], fontsize=8)
        plt.xlabel("Average predicted action-set fraction")
        plt.ylabel("Observed choice coverage")
        plt.title("Coverage-efficiency tradeoff")
        plt.tight_layout()
        plt.savefig(fig_dir / "coverage_vs_set_size.png", dpi=180)
        plt.close()
    if pref_summary is not None and not pref_summary.empty:
        point_cols = [c for c in pref_summary.columns if c.startswith("theta_point_")]
        if point_cols:
            sample = pref_summary[pref_summary["scope_type"].ne("fallback")].head(5) if "scope_type" in pref_summary.columns else pref_summary.head(5)
            if not sample.empty:
                row = sample.iloc[0]
                names = [c.replace("theta_point_", "") for c in point_cols]
                means = np.array([row[c] for c in point_cols], dtype=float)
                lows = np.array([row.get(f"theta_min_{n}", np.nan) for n in names], dtype=float)
                highs = np.array([row.get(f"theta_max_{n}", np.nan) for n in names], dtype=float)
                valid = np.isfinite(means) & np.isfinite(lows) & np.isfinite(highs)
                if valid.any():
                    idx_valid = np.flatnonzero(valid)
                    order = idx_valid[np.argsort(means[idx_valid])[::-1]][:min(10, len(idx_valid))]
                    plt.figure(figsize=(8, 5))
                    y = np.arange(len(order))
                    left = np.maximum(means[order] - lows[order], 0)
                    right = np.maximum(highs[order] - means[order], 0)
                    plt.errorbar(means[order], y, xerr=[left, right], fmt="o")
                    plt.yticks(y, [names[i] for i in order])
                    plt.xlabel("Preference weight interval")
                    plt.title(f"Example preference intervals: {row.get('scope_id','scope')}")
                    plt.tight_layout()
                    plt.savefig(fig_dir / "preference_intervals_examples.png", dpi=180)
                    plt.close()
