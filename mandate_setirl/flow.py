from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import present_cols, smart_write_table


def add_future_flow_columns(panel: pd.DataFrame, entity_col: str, date_col: str, y_cols: list[str]) -> pd.DataFrame:
    df = panel.sort_values([entity_col, date_col]).copy()
    for y in y_cols:
        if y in df.columns:
            df[f"next_{y}"] = df.groupby(entity_col)[y].shift(-1)
    return df


def add_turnover_style_controls(panel: pd.DataFrame, entity_col: str, action_cols: list[str]) -> pd.DataFrame:
    df = panel.copy()
    A = df[action_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    lag = np.zeros_like(A)
    for j, c in enumerate(action_cols):
        lag_col = f"lag_{c}"
        if lag_col in df.columns:
            lag[:, j] = pd.to_numeric(df[lag_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            lag[:, j] = A[:, j]
    df["turnover_l1_action"] = np.abs(A - lag).sum(axis=1)
    # Train-period entity style mean.  Use train rows only to avoid look-ahead.
    train = df[df["split"].eq("train")]
    if len(train):
        means = train.groupby(entity_col)[action_cols].mean()
        fallback = train[action_cols].mean().to_numpy(dtype=float)
        style_dist = []
        for _, row in df.iterrows():
            ent = row[entity_col]
            if ent in means.index:
                base = means.loc[ent].to_numpy(dtype=float)
            else:
                base = fallback
            cur = pd.to_numeric(row[action_cols], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            style_dist.append(float(np.abs(cur - base).sum()))
        df["style_drift_train_mean_l1"] = style_dist
    else:
        df["style_drift_train_mean_l1"] = df["turnover_l1_action"]
    return df


def two_way_demean(df: pd.DataFrame, cols: list[str], entity_col: str, time_col: str, n_iter: int = 8) -> pd.DataFrame:
    X = df[cols].astype(float).copy().fillna(0.0)
    for _ in range(n_iter):
        X -= X.groupby(df[entity_col]).transform("mean")
        X -= X.groupby(df[time_col]).transform("mean")
        X += X.mean(axis=0)
    return X


def _ols_twfe(tmp: pd.DataFrame, y: str, x_cols: list[str], entity_col: str, date_col: str, spec: str) -> list[dict[str, Any]]:
    dm = two_way_demean(tmp, [y] + x_cols, entity_col, date_col)
    Y = dm[y].to_numpy(dtype=float)
    X = sm.add_constant(dm[x_cols].to_numpy(dtype=float), has_constant="add")
    model = sm.OLS(Y, X).fit(cov_type="HC1")
    names = ["const"] + x_cols
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "spec": spec,
            "dependent": y,
            "term": name,
            "coef": float(model.params[i]),
            "std_err": float(model.bse[i]),
            "p_value": float(model.pvalues[i]),
            "n": int(model.nobs),
            "r2": float(model.rsquared),
        })
    return rows


def residualize_twfe(tmp: pd.DataFrame, x: str, controls: list[str], entity_col: str, date_col: str) -> np.ndarray:
    if not controls:
        dm = two_way_demean(tmp, [x], entity_col, date_col)
        return dm[x].to_numpy(dtype=float)
    dm = two_way_demean(tmp, [x] + controls, entity_col, date_col)
    Y = dm[x].to_numpy(dtype=float)
    X = sm.add_constant(dm[controls].to_numpy(dtype=float), has_constant="add")
    try:
        fit = sm.OLS(Y, X).fit()
        return Y - fit.predict(X)
    except Exception:
        return Y


def run_flow_validation(
    panel: pd.DataFrame,
    pred_df: pd.DataFrame,
    cfg: dict[str, Any],
    entity_col: str,
    date_col: str,
    action_cols: list[str] | None = None,
    out_dir: str | None = None,
) -> pd.DataFrame:
    fcfg = cfg.get("flow_validation", {})
    if not fcfg.get("enabled", True):
        return pd.DataFrame()
    y_cols = list(fcfg.get("y_cols", ["net_flow_3m"]))
    df = panel.copy().reset_index(drop=True)
    df["row_pos"] = np.arange(len(df))
    if action_cols:
        df = add_turnover_style_controls(df, entity_col, action_cols)
    df = add_future_flow_columns(df, entity_col, date_col, y_cols)
    merged = pred_df.merge(df, on="row_pos", how="left", suffixes=("", "_panel"))
    base_controls = present_cols(merged, fcfg.get("controls", []))
    hard_controls = present_cols(merged, ["turnover_l1_action", "style_drift_train_mean_l1"])
    all_controls = list(dict.fromkeys(base_controls + hard_controls))
    rows = []
    min_obs = int(fcfg.get("min_obs", 1000))
    for y in y_cols:
        yy = f"next_{y}"
        if yy not in merged.columns:
            continue
        use_cols = [yy, entity_col, date_col, "compatibility_score"] + all_controls
        tmp = merged[use_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(tmp) < min_obs:
            continue
        for c in [yy, "compatibility_score"] + all_controls:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp = tmp.dropna(subset=[yy, "compatibility_score"] + all_controls)
        if len(tmp) < min_obs:
            continue
        # Controlled compatibility regression.
        reg_cols = ["compatibility_score"] + all_controls
        try:
            rows.extend(_ols_twfe(tmp, yy, reg_cols, entity_col, date_col, spec="compat_plus_turnover_style_controls"))
        except Exception as e:
            rows.append({"spec": "compat_plus_turnover_style_controls", "dependent": yy, "term": "ERROR", "error": str(e), "n": len(tmp)})
        # Residualized compatibility diagnostic.
        try:
            tmp2 = tmp.copy()
            tmp2["compatibility_residualized"] = residualize_twfe(tmp2, "compatibility_score", all_controls, entity_col, date_col)
            rows.extend(_ols_twfe(tmp2, yy, ["compatibility_residualized"] + all_controls, entity_col, date_col, spec="residualized_compatibility"))
        except Exception as e:
            rows.append({"spec": "residualized_compatibility", "dependent": yy, "term": "ERROR", "error": str(e), "n": len(tmp)})
        # Stability-subsample regression.
        if hard_controls:
            try:
                rows.extend(_ols_twfe(tmp, yy, hard_controls + base_controls, entity_col, date_col, spec="turnover_style_only"))
            except Exception as e:
                rows.append({"spec": "turnover_style_only", "dependent": yy, "term": "ERROR", "error": str(e), "n": len(tmp)})
    out = pd.DataFrame(rows)
    if out_dir:
        smart_write_table(out, f"{out_dir}/flow_validation_turnover_controlled.csv")
    return out
