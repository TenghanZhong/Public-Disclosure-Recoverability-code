from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import smart_write_table


@dataclass
class BoundModel:
    action_cols: list[str]
    lower: np.ndarray
    upper: np.ndarray
    row_index: pd.Index
    asset_action_cols: list[str]
    overlay_action_cols: list[str]
    mode: str = "exogenous"

    def bounds_for_positions(self, positions: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray]:
        return self.lower[positions], self.upper[positions]


def _quantile_table(df: pd.DataFrame, action_cols: list[str], q_low: float, q_high: float, group_col: str | None = None):
    if group_col is None:
        lows = df[action_cols].quantile(q_low).astype(float)
        highs = df[action_cols].quantile(q_high).astype(float)
        return lows, highs
    lows = df.groupby(group_col)[action_cols].quantile(q_low)
    highs = df.groupby(group_col)[action_cols].quantile(q_high)
    return lows, highs


def fit_mandate_bounds(
    panel: pd.DataFrame,
    action_cols: list[str],
    asset_action_cols: list[str],
    overlay_action_cols: list[str],
    entity_col: str,
    cfg: dict[str, Any],
    mode: str = "exogenous",
) -> BoundModel:
    """Create row-level action bounds.

    Supported modes are global, exogenous, lagged_history, and full. The
    exogenous mode excludes each fund's own action history. The 'mandate' alias
    maps to cfg.bounds.main_mode.
    """
    bcfg = cfg.get("bounds", {})
    if mode == "mandate":
        mode = str(bcfg.get("main_mode", "exogenous"))
    valid = {"global", "exogenous", "lagged_history", "full"}
    if mode not in valid:
        raise ValueError(f"Unknown bounds mode {mode}; expected one of {sorted(valid)}")

    train = panel[panel["split"] == "train"].copy()
    if train.empty:
        train = panel.copy()
    X_train = train[action_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    g_low, g_high = _quantile_table(X_train, action_cols, float(bcfg.get("global_lower_q", 0.005)), float(bcfg.get("global_upper_q", 0.995)))
    for c in asset_action_cols:
        g_low[c] = max(0.0, min(float(g_low[c]), 0.25))
        g_high[c] = min(1.0, max(float(g_high[c]), 0.05))
    for c in overlay_action_cols:
        if c != "n_holdings_scaled":
            g_low[c] = min(float(g_low[c]), 0.0)
            g_high[c] = max(float(g_high[c]), 0.01)

    n = len(panel)
    lower = np.tile(g_low.to_numpy(dtype=float), (n, 1))
    upper = np.tile(g_high.to_numpy(dtype=float), (n, 1))

    if mode in {"exogenous", "full"}:
        lower, upper = _apply_peer_bounds(panel, train, action_cols, lower, upper, bcfg)
        lower, upper = _apply_adv_soft_caps(panel, action_cols, lower, upper, bcfg)

    if mode in {"lagged_history", "full"}:
        lower, upper = _apply_lagged_history_bounds(panel, action_cols, entity_col, lower, upper, bcfg)

    return _clean_bound_model(panel, action_cols, asset_action_cols, overlay_action_cols, lower, upper, cfg, mode)


def _apply_peer_bounds(panel, train, action_cols, lower, upper, bcfg):
    if "mandate_cluster" not in panel.columns:
        return lower, upper
    p_low, p_high = _quantile_table(
        train.assign(mandate_cluster=train["mandate_cluster"]),
        action_cols,
        float(bcfg.get("peer_lower_q", 0.05)),
        float(bcfg.get("peer_upper_q", 0.99)),
        "mandate_cluster",
    )
    clusters = panel["mandate_cluster"].to_numpy()
    for row_pos, cl in enumerate(clusters):
        if cl in p_low.index:
            lower[row_pos] = np.maximum(lower[row_pos], p_low.loc[cl, action_cols].to_numpy(dtype=float))
            upper[row_pos] = np.minimum(upper[row_pos], p_high.loc[cl, action_cols].to_numpy(dtype=float))
    return lower, upper


def _apply_lagged_history_bounds(panel, action_cols, entity_col, lower, upper, bcfg):
    """Apply strictly prior own-history quantile bounds, row by row."""
    if entity_col not in panel.columns:
        return lower, upper
    ql = float(bcfg.get("fund_lower_q", 0.05))
    qh = float(bcfg.get("fund_upper_q", 0.99))
    buf = float(bcfg.get("history_buffer", 0.10))
    min_obs = int(bcfg.get("history_min_obs", 3))
    date_col = bcfg.get("date_col") or ("report_date" if "report_date" in panel.columns else None)
    if date_col is None or date_col not in panel.columns:
        order = np.arange(len(panel))
    else:
        order = panel.sort_values([entity_col, date_col]).index.to_numpy()
    pos_by_index = {idx: i for i, idx in enumerate(panel.index)}
    sorted_panel = panel.loc[order]
    for _, g in sorted_panel.groupby(entity_col, sort=False):
        past = []
        for idx, row in g.iterrows():
            row_pos = pos_by_index[idx]
            if len(past) >= min_obs:
                hist = np.vstack(past)
                lo = np.quantile(hist, ql, axis=0) - buf
                hi = np.quantile(hist, qh, axis=0) + buf
                lower[row_pos] = np.maximum(lower[row_pos], lo)
                upper[row_pos] = np.minimum(upper[row_pos], hi)
            vals = pd.to_numeric(row[action_cols], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
            past.append(vals)
    return lower, upper


def _apply_adv_soft_caps(panel, action_cols, lower, upper, bcfg):
    if not bcfg.get("use_adv_soft_caps", False):
        return lower, upper
    col_to_idx = {c: i for i, c in enumerate(action_cols)}
    derivative_cols = ["derivative_pct", "derivative_pct_pos", "derivative_pct_neg_abs"]
    derivative_idx = [col_to_idx[c] for c in derivative_cols if c in col_to_idx]
    if derivative_idx and "allows_or_discusses_derivatives" in panel.columns:
        absent = pd.to_numeric(panel["allows_or_discusses_derivatives"], errors="coerce").fillna(0).to_numpy() <= 0
        cap = float(bcfg.get("no_derivative_keyword_cap", 0.05))
        for idx in derivative_idx:
            # Absence of a keyword does not imply a restriction.
            mask = absent & (upper[:, idx] <= max(cap * 4, cap + 0.10))
            upper[mask, idx] = np.minimum(upper[mask, idx], cap)
    short_cols = ["short_pct", "short_pct_pos", "short_pct_neg_abs"]
    short_idx = [col_to_idx[c] for c in short_cols if c in col_to_idx]
    if short_idx and "shorting" in panel.columns:
        absent = pd.to_numeric(panel["shorting"], errors="coerce").fillna(0).to_numpy() <= 0
        cap = float(bcfg.get("no_short_keyword_cap", 0.03))
        for idx in short_idx:
            mask = absent & (upper[:, idx] <= max(cap * 4, cap + 0.10))
            upper[mask, idx] = np.minimum(upper[mask, idx], cap)
    if "restricted_pct" in col_to_idx and "liquidity_or_redemption_constraint" in panel.columns:
        idx = col_to_idx["restricted_pct"]
        has_liq = pd.to_numeric(panel["liquidity_or_redemption_constraint"], errors="coerce").fillna(0).to_numpy() > 0
        cap = float(bcfg.get("liquidity_restricted_cap", 0.10))
        upper[has_liq, idx] = np.minimum(upper[has_liq, idx], cap)
    return lower, upper


def _clean_bound_model(panel, action_cols, asset_action_cols, overlay_action_cols, lower, upper, cfg, mode: str) -> BoundModel:
    lower = np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.nan_to_num(upper, nan=1.0, posinf=1.0, neginf=1.0)
    min_width = float(cfg.get("bounds", {}).get("min_width", 0.01))
    for j, c in enumerate(action_cols):
        if c in asset_action_cols:
            lower[:, j] = np.clip(lower[:, j], 0.0, 1.0)
            upper[:, j] = np.clip(upper[:, j], 0.0, 1.0)
        elif c == "n_holdings_scaled":
            lower[:, j] = np.clip(lower[:, j], -8.0, 8.0)
            upper[:, j] = np.clip(upper[:, j], -8.0, 8.0)
        else:
            lower[:, j] = np.clip(lower[:, j], -1.0, 2.0)
            upper[:, j] = np.clip(upper[:, j], -1.0, 2.0)
        bad = upper[:, j] < lower[:, j] + min_width
        mid = (upper[:, j] + lower[:, j]) / 2.0
        lower[bad, j] = mid[bad] - min_width / 2.0
        upper[bad, j] = mid[bad] + min_width / 2.0
        if c in asset_action_cols:
            lower[:, j] = np.clip(lower[:, j], 0.0, 1.0)
            upper[:, j] = np.clip(upper[:, j], lower[:, j] + 1e-8, 1.0)
    # Text-derived restrictions enter as upper bounds.
    asset_idx = [j for j, c in enumerate(action_cols) if c in asset_action_cols]
    for j, c in enumerate(action_cols):
        if c in asset_action_cols:
            lower[:, j] = 0.0
        elif c != "n_holdings_scaled":
            lower[:, j] = np.minimum(lower[:, j], 0.0)

    # Asset bounds must admit a simplex point: sum(lower) <= 1 <= sum(upper).
    if asset_idx:
        idx = np.asarray(asset_idx, dtype=int)
        upper_sum = upper[:, idx].sum(axis=1)
        bad = upper_sum < 1.0 - 1e-10
        if bad.any():
            # Enlarge asset caps proportionally.
            scale = (1.0 + 1e-8) / np.maximum(upper_sum[bad], 1e-12)
            upper[np.ix_(bad, idx)] = np.minimum(1.0, upper[np.ix_(bad, idx)] * scale[:, None])
            # Use broad caps if clipping leaves the row infeasible.
            still_bad = upper[:, idx].sum(axis=1) < 1.0 - 1e-8
            if still_bad.any():
                upper[np.ix_(still_bad, idx)] = 1.0
    return BoundModel(action_cols=action_cols, lower=lower, upper=upper, row_index=panel.index, asset_action_cols=asset_action_cols, overlay_action_cols=overlay_action_cols, mode=mode)


def action_violations(actions: np.ndarray, lower: np.ndarray, upper: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    a = np.asarray(actions, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if a.ndim == 3:
        lo = lo[:, None, :]
        hi = hi[:, None, :]
    return ((a < lo - tol) | (a > hi + tol)).any(axis=-1)


def feasible_set_metrics(panel: pd.DataFrame, bound_model: BoundModel, out_dir: str | None = None, name: str | None = None) -> pd.DataFrame:
    actions = panel[bound_model.action_cols].to_numpy(dtype=float)
    viol = action_violations(actions, bound_model.lower, bound_model.upper)
    width = bound_model.upper - bound_model.lower
    rows = []
    label = name or bound_model.mode
    for split in ["train", "calib", "test"]:
        mask = panel["split"].to_numpy() == split
        if mask.sum() == 0:
            continue
        rows.append({
            "bound_model": label,
            "split": split,
            "n": int(mask.sum()),
            "observed_action_coverage": float(1.0 - viol[mask].mean()),
            "avg_bound_width": float(np.nanmean(width[mask])),
            "median_bound_width": float(np.nanmedian(width[mask])),
        })
    for c_idx, c in enumerate(bound_model.action_cols):
        dim_viol = (actions[:, c_idx] < bound_model.lower[:, c_idx] - 1e-7) | (actions[:, c_idx] > bound_model.upper[:, c_idx] + 1e-7)
        rows.append({
            "bound_model": label,
            "split": f"dimension:{c}",
            "n": int(len(panel)),
            "observed_action_coverage": float(1.0 - dim_viol.mean()),
            "avg_bound_width": float(np.nanmean(width[:, c_idx])),
            "median_bound_width": float(np.nanmedian(width[:, c_idx])),
        })
    out = pd.DataFrame(rows)
    if out_dir:
        suffix = label.replace("/", "_")
        smart_write_table(out, f"{out_dir}/mandate_feasible_set_metrics_{suffix}.csv")
    return out
