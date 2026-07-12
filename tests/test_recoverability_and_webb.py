from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from webb_wild_bootstrap import webb_six_point_values
from recoverability_tau_menu_checks import summarize_menu_size, summarize_tau


def test_webb_weights_have_zero_mean_and_unit_variance():
    values, probs = webb_six_point_values()
    assert math.isclose(float(probs.sum()), 1.0)
    assert abs(float(np.sum(values * probs))) < 1e-12
    assert math.isclose(float(np.sum((values**2) * probs)), 1.0)


def test_tau_coverage_is_monotone_in_tolerance():
    rows = pd.DataFrame(
        {
            "move_bin": ["stable_l1_le_005", "stable_l1_le_005", "large_l1_gt_025"],
            "observed_in_proxy": [True, False, True],
            "oracle_min_l1": [0.04, 0.08, 0.20],
            "selected_f20_min_l1": [0.06, 0.09, 0.30],
            "feasible_menu_size": [39, 20, 5],
        }
    )
    out = summarize_tau(rows, [0.05, 0.10, 0.25])
    overall = out[out["group"].eq("overall")].sort_values("tau")
    assert overall["menu_support"].is_monotonic_increasing
    assert overall["selected_20pct_coverage"].is_monotonic_increasing


def test_menu_size_summary_excludes_missing_groups_and_reports_quantiles():
    rows = pd.DataFrame(
        {
            "move_bin": ["stable_l1_le_005", "stable_l1_le_005", "large_l1_gt_025"],
            "observed_in_proxy": [True, False, True],
            "feasible_menu_size": [39, 20, 5],
        }
    )
    out = summarize_menu_size(rows)
    overall = out[out["group"].eq("overall")].iloc[0]
    assert overall["mean"] == (39 + 20 + 5) / 3
    assert overall["median"] == 20
    assert overall["p25"] == 12.5
    assert overall["p75"] == 29.5
