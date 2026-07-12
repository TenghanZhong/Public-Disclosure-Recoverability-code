from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from .candidates import CandidateBatch, CandidateGenerator
from .utils import SimpleStandardScaler


@dataclass
class UtilityFeatureModel:
    feature_names: list[str]
    scaler: SimpleStandardScaler
    asset_return_panel: np.ndarray     # N x K, row-specific market-state proxy
    asset_risk_panel: np.ndarray       # N x K
    asset_drawdown_panel: np.ndarray   # N x K
    action_cols: list[str]
    asset_action_cols: list[str]
    overlay_action_cols: list[str]
    proxy_columns: dict[str, dict[str, str | None]]

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        B, M, D = raw_features.shape
        z = self.scaler.transform(raw_features.reshape(-1, D)).reshape(B, M, D)
        z = np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0)
        return z


def fit_utility_feature_model(
    panel: pd.DataFrame,
    candgen: CandidateGenerator,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> UtilityFeatureModel:
    feature_names = list(cfg.get("utility", {}).get("feature_names", []))
    if not feature_names:
        feature_names = [
            "ret_proxy", "neg_risk_proxy", "neg_drawdown_proxy", "flow_liquidity_fit", "neg_style_drift",
            "liquidity_score", "neg_derivative_use", "neg_short_use", "neg_restricted_use", "neg_non_us_exposure", "neg_compliance_interaction",
        ]
    asset_ret, asset_risk, asset_dd, proxy_cols = infer_asset_proxy_panels(panel, candgen.action_cols, candgen.asset_cols, cfg)
    tmp_model = UtilityFeatureModel(
        feature_names=feature_names,
        scaler=SimpleStandardScaler(),
        asset_return_panel=asset_ret,
        asset_risk_panel=asset_risk,
        asset_drawdown_panel=asset_dd,
        action_cols=candgen.action_cols,
        asset_action_cols=candgen.asset_cols,
        overlay_action_cols=candgen.overlay_cols,
        proxy_columns=proxy_cols,
    )
    train_pos = np.flatnonzero(panel["split"].to_numpy() == "train")
    if len(train_pos) == 0:
        train_pos = np.arange(len(panel))
    sample_obs = int(cfg.get("utility", {}).get("scaler_sample_obs", 20000))
    if len(train_pos) > sample_obs:
        train_pos = rng.choice(train_pos, size=sample_obs, replace=False)
    n_c = int(cfg.get("utility", {}).get("scaler_candidates_per_obs", 24))
    chunks = []
    for start in range(0, len(train_pos), 1024):
        pos = train_pos[start:start + 1024]
        batch = candgen.generate_for_positions(pos, n_candidates=n_c)
        raw = compute_raw_utility_features(panel, batch, tmp_model)
        chunks.append(raw.reshape(-1, raw.shape[-1]))
    X = np.vstack(chunks) if chunks else np.zeros((1, len(feature_names)))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    tmp_model.scaler.fit(X)
    tmp_model.scaler.scale_[tmp_model.scaler.scale_ < 1e-12] = 1.0
    return tmp_model


def _asset_to_ticker() -> dict[str, str]:
    return {
        "asset_EC": "SPY",
        "asset_STIV": "IEF",
        "asset_DBT": "LQD",
        "asset_DE": "UUP",
        "asset_EP": "GLD",
        "asset_ABS_MBS": "IEF",
        "asset_RA": "IWM",
        "asset_DIR": "SPY",
        "asset_DFE": "SPY",
        "asset_LON": "TLT",
        "asset_OTHER": "SPY",
    }


def infer_asset_proxy_panels(
    panel: pd.DataFrame,
    action_cols: list[str],
    asset_cols: list[str],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, str | None]]]:
    """Build row-specific return/risk/drawdown proxy panels.

    The previous implementation used one full-sample median vector, which was
    both state-insensitive and mildly look-ahead.  This version uses row-level
    market-state columns when present and fills missing values with train-period
    medians only.
    """
    n = len(panel)
    k = len(action_cols)
    ret = np.zeros((n, k), dtype=float)
    risk = np.zeros((n, k), dtype=float)
    dd = np.zeros((n, k), dtype=float)
    proxy_cols: dict[str, dict[str, str | None]] = {}
    train_mask = panel["split"].to_numpy() == "train" if "split" in panel.columns else np.ones(n, dtype=bool)
    mapping = _asset_to_ticker()
    default_ret = {
        "asset_EC": 0.06, "asset_STIV": 0.015, "asset_DBT": 0.025, "asset_DE": 0.02, "asset_EP": 0.035,
        "asset_ABS_MBS": 0.025, "asset_RA": 0.05, "asset_DIR": 0.04, "asset_DFE": 0.045, "asset_LON": 0.03, "asset_OTHER": 0.035,
    }
    default_risk = {
        "asset_EC": 0.16, "asset_STIV": 0.02, "asset_DBT": 0.06, "asset_DE": 0.07, "asset_EP": 0.18,
        "asset_ABS_MBS": 0.05, "asset_RA": 0.18, "asset_DIR": 0.16, "asset_DFE": 0.17, "asset_LON": 0.12, "asset_OTHER": 0.12,
    }
    for j, c in enumerate(action_cols):
        if c not in asset_cols:
            continue
        ticker = mapping.get(c, "SPY")
        ret_col = _find_col(panel, ticker, ["ret", "return"], cfg.get("utility", {}).get("return_window_preference", ["63", "21", "252"]))
        risk_col = _find_col(panel, ticker, ["vol", "rolling_vol", "ewma_vol"], cfg.get("utility", {}).get("vol_window_preference", ["63", "20", "21", "252"]))
        dd_col = _find_col(panel, ticker, ["drawdown", "dd"], ["63", "252", "21"])
        proxy_cols[c] = {"return": ret_col, "risk": risk_col, "drawdown": dd_col}
        ret[:, j] = _row_series_with_train_fill(panel, ret_col, train_mask, default_ret.get(c, 0.035))
        risk[:, j] = np.abs(_row_series_with_train_fill(panel, risk_col, train_mask, default_risk.get(c, 0.12)))
        dd[:, j] = np.abs(_row_series_with_train_fill(panel, dd_col, train_mask, default_risk.get(c, 0.12)))
    return ret, risk, dd, proxy_cols


def _find_col(panel: pd.DataFrame, ticker: str, keywords: list[str], windows: list[str]) -> str | None:
    cols = list(panel.columns)
    lower_map = {c: c.lower() for c in cols}
    ticker_l = ticker.lower()
    for w in windows:
        for c, lc in lower_map.items():
            if ticker_l in lc and str(w) in lc and any(k.lower() in lc for k in keywords):
                return c
    for c, lc in lower_map.items():
        if ticker_l in lc and any(k.lower() in lc for k in keywords):
            return c
    return None


def _row_series_with_train_fill(panel: pd.DataFrame, col: str | None, train_mask: np.ndarray, default: float) -> np.ndarray:
    if col is None or col not in panel.columns:
        return np.full(len(panel), default, dtype=float)
    x = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_vals = x.loc[train_mask]
    fill = float(train_vals.median(skipna=True)) if train_vals.notna().any() else default
    return x.fillna(fill).to_numpy(dtype=float)


def compute_raw_utility_features(panel: pd.DataFrame, batch: CandidateBatch, model: UtilityFeatureModel) -> np.ndarray:
    pos = batch.row_positions
    A = np.asarray(batch.actions, dtype=float)  # B x M x K
    B, M, K = A.shape
    names = model.feature_names
    raw_map: dict[str, np.ndarray] = {}

    # Row-specific linear proxy features.
    ret_proxy = np.einsum("bmk,bk->bm", A, model.asset_return_panel[pos])
    risk_proxy = np.einsum("bmk,bk->bm", A, np.abs(model.asset_risk_panel[pos]))
    drawdown_proxy = np.einsum("bmk,bk->bm", A, np.abs(model.asset_drawdown_panel[pos]))

    derivative = _gross_action_col(A, model.action_cols, "derivative_pct")
    short = _gross_action_col(A, model.action_cols, "short_pct")
    restricted = _action_col(A, model.action_cols, "restricted_pct")
    non_us = _action_col(A, model.action_cols, "non_us_pct")
    cash = _action_col(A, model.action_cols, "cash_proxy")
    stiv = _action_col(A, model.action_cols, "asset_STIV")

    liquidity_score = stiv + cash - restricted - 0.5 * derivative - 0.25 * non_us
    redemption_pressure = _panel_col(panel, pos, ["lag_redemption_flow_3m_scaled", "redemption_flow_3m_scaled", "lag_net_flow_3m_scaled"])
    flow_liquidity_fit = liquidity_score * np.maximum(redemption_pressure[:, None], 0.0) - 0.1 * np.abs(liquidity_score - 0.10)

    lag = np.zeros((B, K), dtype=float)
    panel_pos = panel.iloc[pos]
    for j, c in enumerate(model.action_cols):
        lag_col = f"lag_{c}"
        if lag_col in panel.columns:
            lag[:, j] = pd.to_numeric(panel_pos[lag_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            lag[:, j] = pd.to_numeric(panel_pos[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    style_drift = np.sum(np.abs(A - lag[:, None, :]), axis=2)

    drp = _panel_col(panel, pos, ["adv_total_drp_records", "total_drp_records", "adviser_drp_total", "adv_drp_total", "ia_regulatory_total"])
    drp = np.log1p(np.maximum(drp, 0.0))
    compliance_interaction = drp[:, None] * (derivative + short + restricted)

    raw_map["ret_proxy"] = ret_proxy
    raw_map["neg_risk_proxy"] = -risk_proxy
    raw_map["neg_drawdown_proxy"] = -drawdown_proxy
    raw_map["flow_liquidity_fit"] = flow_liquidity_fit
    raw_map["neg_style_drift"] = -style_drift
    raw_map["liquidity_score"] = liquidity_score
    raw_map["neg_derivative_use"] = -derivative
    raw_map["neg_short_use"] = -short
    raw_map["neg_restricted_use"] = -restricted
    raw_map["neg_non_us_exposure"] = -non_us
    raw_map["neg_compliance_interaction"] = -compliance_interaction

    out = np.zeros((B, M, len(names)), dtype=float)
    for d, name in enumerate(names):
        out[:, :, d] = raw_map.get(name, np.zeros((B, M)))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_scaled_utility_features(panel: pd.DataFrame, batch: CandidateBatch, model: UtilityFeatureModel) -> np.ndarray:
    return model.transform(compute_raw_utility_features(panel, batch, model))


def _action_col(A: np.ndarray, action_cols: list[str], name: str) -> np.ndarray:
    if name in action_cols:
        return A[:, :, action_cols.index(name)]
    return np.zeros(A.shape[:2], dtype=float)


def _gross_action_col(A: np.ndarray, action_cols: list[str], name: str) -> np.ndarray:
    if name in action_cols:
        return _action_col(A, action_cols, name)
    return _action_col(A, action_cols, f"{name}_pos") + _action_col(A, action_cols, f"{name}_neg_abs")


def _panel_col(panel: pd.DataFrame, pos: np.ndarray, candidates: list[str]) -> np.ndarray:
    for c in candidates:
        if c in panel.columns:
            return pd.to_numeric(panel.iloc[pos][c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.zeros(len(pos), dtype=float)
