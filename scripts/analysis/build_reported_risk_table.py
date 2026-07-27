"""Build the risk-signal summary table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables"
SOURCE_PATH = TABLES / "boundary_regressions" / "reported_boundary_regressions.csv"
OUT_PATH = TABLES / "reported_risk_signal_table.csv"


def _one(df: pd.DataFrame, spec: str) -> pd.Series:
    mask = df["specification"].eq(spec)
    rows = df.loc[mask].copy()
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row for {spec}, found {len(rows)}")
    return rows.iloc[0].copy()


def main() -> None:
    source = pd.read_csv(SOURCE_PATH)
    rows: list[dict] = []

    specs = [
        "Preferred exception signal",
        "Boundary-proximity controls",
        "Winsorized outcome and controls (1--99%)",
        "Outside-proxy distance",
        "Distance with extremeness controls",
        "Distance scaled by training variation",
        "Drift on coordinates still inside proxy",
        "Binary with distance controls",
        "Asset-block drift outcome",
        "Overlay-block drift outcome",
        "Drop derivative/leverage proxy funds",
    ]
    for order, spec in enumerate(specs, start=1):
        row = _one(source, spec)
        rows.append(
            {
                "table_order": order,
                "specification": spec,
                "regressor_outcome": row["regressor_outcome"],
                "n": int(row["n"]),
                "coef": float(row["coef"]),
                "clustered_t": float(row["two_way_cluster_t"]),
                "coef_display": f"{float(row['coef']):.4f}",
                "clustered_t_display": f"{float(row['two_way_cluster_t']):.2f}",
                "source_file": SOURCE_PATH.relative_to(ROOT).as_posix(),
                "source_selector": f"specification={spec}",
                "controls_or_notes": str(row.get("controls", "")),
            }
        )

    out = pd.DataFrame(rows).sort_values("table_order")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
