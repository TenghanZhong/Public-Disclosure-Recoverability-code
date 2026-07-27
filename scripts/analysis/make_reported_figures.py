"""Generate evaluation figures from committed summary tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO_DIR / "tables"
FIG_DIR = REPO_DIR / "figures"

MOVE_ORDER = ["stayer_l1_le_005", "moderate_l1_005_025", "mover_l1_gt_025"]
MOVE_LABELS = {
    "stayer_l1_le_005": "Stable",
    "moderate_l1_005_025": "Moderate",
    "mover_l1_gt_025": "Large",
}
STATUS_LABELS = {1.0: "In proxy", 0.0: "Out"}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{libertine}\usepackage{newtxmath}",
            "font.family": "serif",
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.linewidth": 0.35,
            "patch.linewidth": 0.5,
        }
    )


def clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.22)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.035,
        label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontweight="bold",
        fontsize=9.0,
    )


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "pdf": str(pdf.relative_to(REPO_DIR)).replace("\\", "/"),
        "png": str(png.relative_to(REPO_DIR)).replace("\\", "/"),
    }


def load_recoverability_panel() -> pd.DataFrame:
    oracle = pd.read_csv(TABLE_DIR / "repeated_menu_support_summary.csv")
    f20 = pd.read_csv(TABLE_DIR / "repeated_transition_selection_summary.csv")
    key = ["move_bin", "observed_in_mandate_proxy"]
    cols_oracle = key + ["oracle_cover_mean", "oracle_cover_p025", "oracle_cover_p975"]
    cols_f20 = key + ["coverage_mean", "coverage_p025", "coverage_p975"]
    out = oracle[cols_oracle].merge(f20[cols_f20], on=key, how="inner")
    out["move_order"] = out["move_bin"].map({m: i for i, m in enumerate(MOVE_ORDER)})
    out["status_order"] = out["observed_in_mandate_proxy"].map({1.0: 0, 0.0: 1})
    return out.sort_values(["status_order", "move_order"]).reset_index(drop=True)


def make_risk_signal_forest() -> dict[str, str]:
    df = pd.read_csv(TABLE_DIR / "reported_risk_signal_table.csv")

    def panel_rows(specs: list[tuple[str, str]]) -> pd.DataFrame:
        rows = []
        for spec, label in specs:
            sub = df[df["specification"].eq(spec)].copy()
            if sub.empty:
                raise ValueError(f"Missing risk-signal figure row: {spec}")
            rows.append(sub.iloc[[0]].assign(label=label))
        out = pd.concat(rows, ignore_index=True).iloc[::-1].reset_index(drop=True)
        out["se"] = out["coef"].astype(float).abs() / out["clustered_t"].astype(float).abs()
        return out

    binary = panel_rows(
        [
            ("Preferred exception signal", "Main binary signal"),
            ("Boundary-proximity controls", "Boundary controls"),
            ("Winsorized outcome and controls (1--99%)", "Winsorized"),
            ("Binary with distance controls", "Add boundary magnitude"),
            ("Asset-block drift outcome", "Asset drift"),
            ("Overlay-block drift outcome", "Overlay drift"),
            ("Drop derivative/leverage proxy funds", "Exclude derivative/leveraged"),
        ]
    )
    scaled = panel_rows(
        [
            ("Distance scaled by training variation", "Next-report drift"),
            ("Drift on coordinates still inside proxy", "Inside-proxy drift"),
        ]
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.35, 3.55),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.3, 1.0]},
    )
    panels = [
        (axes[0], binary, "A. Binary exception", "Coefficient on binary exception"),
        (axes[1], scaled, "B. IQR-scaled boundary magnitude", "Coefficient on scaled boundary magnitude"),
    ]
    colors = ["#0072B2", "#D55E00"]
    for i, (ax, plot, title, xlabel) in enumerate(panels):
        beta = plot["coef"].astype(float).to_numpy()
        se = plot["se"].astype(float).to_numpy()
        lo = beta - 1.96 * se
        hi = beta + 1.96 * se
        y = np.arange(len(plot))
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.errorbar(
            beta,
            y,
            xerr=np.vstack([beta - lo, hi - beta]),
            fmt="o",
            color=colors[i],
            capsize=2.5,
            markersize=3.5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(plot["label"])
        ax.set_xlabel(xlabel)
        panel_label(ax, title)
        clean_axis(ax, grid_axis="x")
    return save_figure(fig, "fig_risk_signal_forest")


def make_recoverability_heatmap() -> dict[str, str]:
    df = load_recoverability_panel()
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.95), constrained_layout=True)
    specs = [
        ("oracle_cover_mean", "A. Menu support"),
        ("coverage_mean", r"B. Transition-candidate 20\% selection"),
    ]
    for ax, (col, title) in zip(axes, specs):
        mat = np.zeros((2, 3))
        for r, status in enumerate([1.0, 0.0]):
            for c, move in enumerate(MOVE_ORDER):
                val = df[(df["observed_in_mandate_proxy"] == status) & (df["move_bin"] == move)][col].iloc[0]
                mat[r, c] = float(val) * 100.0
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="equal")
        ax.set_xticks(range(3))
        ax.set_xticklabels([MOVE_LABELS[m] for m in MOVE_ORDER])
        ax.set_yticks(range(2))
        ax.set_yticklabels(["In proxy", "Out"])
        ax.tick_params(axis="both", labelsize=8.0)
        for r in range(2):
            for c in range(3):
                txt_color = "white" if mat[r, c] > 55 else "#111111"
                ax.text(c, r, f"{mat[r, c]:.1f}", ha="center", va="center", fontsize=8.2, color=txt_color)
        panel_label(ax, title)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.78, pad=0.03, aspect=28)
    cbar.set_label(r"Rate (\%)", fontsize=8.4)
    cbar.ax.tick_params(labelsize=7.8)
    return save_figure(fig, "fig_recoverability_heatmap")


def make_persistence_ladder() -> dict[str, str]:
    df = pd.read_csv(TABLE_DIR / "persistence_ladder_with_ci.csv")
    keep = [
        "Bare lagged self",
        "Projected lagged self",
        "Mandate-filtered lagged self",
        "Mandate-aware RP, f20",
        "Size-matched persistence, f20",
        "Transition source ranker, f20",
        "Transition oracle ceiling, f20",
    ]
    plot = df[df["model_label"].isin(keep)].copy()
    plot["order"] = plot["model_label"].map({m: i for i, m in enumerate(keep)})
    plot = plot.sort_values("order").iloc[::-1].reset_index(drop=True)
    labels = (
        plot["model_label"]
        .str.replace(", f20", r", 20\% set", regex=False)
        .str.replace("oracle ceiling", "menu upper bound", regex=False)
    )
    y = np.arange(len(plot))
    coverage = plot["point_estimate_coverage_pct"].astype(float).to_numpy()
    lo = plot["ci_low_coverage_pct"].astype(float).to_numpy()
    hi = plot["ci_high_coverage_pct"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(3.55, 2.55), constrained_layout=True)
    colors = ["#8da0cb" if "oracle" not in m.lower() else "#66c2a5" for m in plot["model_label"]]
    ax.barh(y, coverage, color=colors, edgecolor="#222222")
    ax.errorbar(coverage, y, xerr=np.vstack([coverage - lo, hi - coverage]), fmt="none", ecolor="#222222", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Coverage (\%)")
    ax.set_xlim(0, 55)
    panel_label(ax, "Persistence ladder")
    clean_axis(ax, grid_axis="x")
    return save_figure(fig, "fig_persistence_ladder")


def make_direction_source_figure() -> dict[str, str]:
    direction = pd.read_csv(TABLE_DIR / "transition_direction_bootstrap_ci.csv")
    metric = "centroid_selected_cosine_to_true_move"
    models = {
        "oracle_upper_bound": "Menu upper bd.",
        "size_matched_persistence": "Persistence",
        "transition_knn_retrieval": "KNN",
    }
    moves = ["moderate_l1_005_025", "mover_l1_gt_025"]
    dir_rows = direction[
        direction["metric"].eq(metric)
        & direction["model"].isin(models)
        & direction["move_bin"].isin(moves)
    ].copy()
    dir_rows["model_label"] = dir_rows["model"].map(models)
    dir_rows["move_label"] = dir_rows["move_bin"].map({"moderate_l1_005_025": "Moderate", "mover_l1_gt_025": "Large"})

    source = pd.read_csv(TABLE_DIR / "transition_source_by_move_bin.csv")
    source = source[source["stage"].eq("oracle_nearest") & source["move_bin"].isin(MOVE_ORDER)].copy()
    source["move_label"] = source["move_bin"].map(MOVE_LABELS)
    pivot = (
        source.pivot_table(
            index="move_label",
            columns="source_bucket",
            values="share_within_move_stage_hits",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex([MOVE_LABELS[m] for m in MOVE_ORDER])
        .fillna(0.0)
    )
    buckets = [
        "lag exact/projected",
        "peer small transition",
        "peer moderate transition",
        "peer large transition",
        "regime transition",
        "same-series small transition",
        "same-series moderate transition",
        "same-series large transition",
        "static peer",
    ]
    bucket_labels = {
        "lag exact/projected": "Lag",
        "peer small transition": "Peer small",
        "peer moderate transition": "Peer mod.",
        "peer large transition": "Peer large",
        "regime transition": "Regime",
        "same-series small transition": "Self small",
        "same-series moderate transition": "Self mod.",
        "same-series large transition": "Self large",
        "static peer": "Static peer",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.45), constrained_layout=True)

    ax = axes[0]
    width = 0.24
    x = np.arange(len(moves))
    for j, (model, label) in enumerate(models.items()):
        sub = dir_rows[dir_rows["model"].eq(model)].set_index("move_bin").reindex(moves)
        y = sub["point_estimate"].astype(float).to_numpy()
        lo = sub["ci_low"].astype(float).to_numpy()
        hi = sub["ci_high"].astype(float).to_numpy()
        pos = x + (j - 1) * width
        ax.bar(pos, y, width, label=label, edgecolor="#222222")
        ax.errorbar(pos, y, yerr=np.vstack([y - lo, hi - y]), fmt="none", ecolor="#222222", capsize=2)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Moderate", "Large"])
    ax.set_ylabel("Cosine with realized move")
    ax.set_ylim(-0.05, 0.95)
    ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=6.5)
    panel_label(ax, "A. Direction vs endpoint recovery")
    clean_axis(ax)

    ax = axes[1]
    x = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3", "#1b9e77"]
    for color, bucket in zip(colors, buckets):
        if bucket not in pivot.columns:
            continue
        vals = pivot[bucket].to_numpy() * 100.0
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, label=bucket_labels[bucket], color=color, edgecolor="#222222")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel(r"Menu upper-bound hit source share (\%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6.1, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.01))
    panel_label(ax, "B. Source attribution")
    clean_axis(ax)

    return save_figure(fig, "fig_direction_source_attribution")


def main() -> None:
    setup_style()
    outputs = {
        "risk_signal_forest": make_risk_signal_forest(),
        "recoverability_heatmap": make_recoverability_heatmap(),
        "persistence_ladder": make_persistence_ladder(),
        "direction_source_attribution": make_direction_source_figure(),
    }
    manifest = {
        "source_tables": [
            "tables/reported_risk_signal_table.csv",
            "tables/boundary_regressions/reported_boundary_regressions.csv",
            "tables/boundary_regressions/wild_bootstrap.csv",
            "tables/repeated_menu_support_summary.csv",
            "tables/repeated_transition_selection_summary.csv",
            "tables/persistence_ladder_with_ci.csv",
            "tables/transition_direction_bootstrap_ci.csv",
            "tables/transition_source_by_move_bin.csv",
        ],
        "figure_roles": {
            "core_figures": [
                "risk_signal_forest",
                "recoverability_heatmap",
            ],
            "additional_diagnostics": [
                "persistence_ladder",
                "direction_source_attribution",
            ],
            "description": "Core figures summarize risk signals and recoverability; additional figures summarize persistence and transition sources.",
        },
        "outputs": outputs,
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, paths in outputs.items():
        print(name)
        for kind, path in paths.items():
            print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
