from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .candidates import CandidateGenerator
from .features import UtilityFeatureModel
from .irl import IRLModel, make_scope_column, collect_pairwise_constraints
from .polytope import build_preference_polytope


def evaluate_simple_baselines(
    panel: pd.DataFrame,
    action_cols: list[str],
    entity_col: str,
    split: str = "test",
) -> pd.DataFrame:
    rows = []
    mask = panel["split"].to_numpy() == split
    pos = np.flatnonzero(mask)
    A = panel[action_cols].to_numpy(dtype=float)
    lag = np.zeros_like(A)
    for j, c in enumerate(action_cols):
        lag_col = f"lag_{c}"
        if lag_col in panel.columns:
            lag[:, j] = pd.to_numeric(panel[lag_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            lag[:, j] = A[:, j]
    if len(pos):
        dist = np.abs(A[pos] - lag[pos]).sum(axis=1)
        rows.append({"model": "persistence", "split": split, "n": len(pos), "point_l1_to_observed_mean": float(dist.mean()), "point_top1_observed": np.nan})
    train = panel[panel["split"] == "train"]
    if "mandate_cluster" in panel.columns and len(train):
        cluster_mean = train.groupby("mandate_cluster")[action_cols].mean()
        fallback = train[action_cols].mean().to_numpy(dtype=float)
        preds = []
        for p in pos:
            cl = panel.iloc[p].get("mandate_cluster", None)
            if cl in cluster_mean.index:
                preds.append(cluster_mean.loc[cl].to_numpy(dtype=float))
            else:
                preds.append(fallback)
        if preds:
            preds = np.vstack(preds)
            dist = np.abs(A[pos] - preds).sum(axis=1)
            rows.append({"model": "peer_cluster_train_mean", "split": split, "n": len(pos), "point_l1_to_observed_mean": float(dist.mean()), "point_top1_observed": np.nan})
    return pd.DataFrame(rows)


def fit_mv_inverse_optimization_baseline(
    panel: pd.DataFrame,
    candgen: CandidateGenerator,
    feature_model: UtilityFeatureModel,
    cfg: dict[str, Any],
    entity_col: str,
    rng: np.random.Generator,
) -> IRLModel:
    """Mean-variance inverse-optimization baseline as a two-feature polytope.

    This is the direct inverse-optimization risk-preference baseline:
    only return and risk tradeoffs are active.  It shares the same LP/polytope
    machinery for comparability.
    """
    d = len(feature_model.feature_names)
    try:
        active = [feature_model.feature_names.index("ret_proxy"), feature_model.feature_names.index("neg_risk_proxy")]
    except ValueError:
        active = [0, min(1, d - 1)]
    scope_series, _ = make_scope_column(panel, cfg, entity_col)
    panel = panel.copy()
    panel["_scope_id"] = scope_series.astype(str)
    train_mask = panel["split"].to_numpy() == "train"
    train_pos_all = np.flatnonzero(train_mask)
    scopes = sorted(panel.loc[train_pos_all, "_scope_id"].dropna().astype(str).unique().tolist())
    prefs = {}
    irl_cfg = cfg.get("irl", {})
    max_train = int(irl_cfg.get("max_train_obs_per_scope", 6000))
    min_train = int(irl_cfg.get("min_train_obs_per_scope", 60))
    max_constraints = int(irl_cfg.get("max_constraints_per_scope", 50000))
    eps_add = float(irl_cfg.get("eps_additive", 0.02))
    eps_mult = float(irl_cfg.get("eps_multiplier", 1.05))

    theta_lower = np.zeros(d, dtype=float)
    theta_upper = np.zeros(d, dtype=float)
    theta_upper[active] = 1.0

    def restrict_G(G: np.ndarray) -> np.ndarray:
        # Keep all columns so evaluation can reuse the common feature space,
        # but force all non mean-variance weights to zero via theta bounds.
        G2 = np.zeros_like(G)
        G2[:, active] = G[:, active]
        return G2

    fallback_positions = []
    for scope in tqdm(scopes, desc="fit mv_io polytope"):
        pos = np.flatnonzero(train_mask & (panel["_scope_id"].astype(str).to_numpy() == str(scope)))
        if len(pos) < min_train:
            continue
        if len(pos) > max_train:
            pos = rng.choice(pos, size=max_train, replace=False)
        fallback_positions.append(pos)
        G = restrict_G(collect_pairwise_constraints(pos, panel, candgen, feature_model, cfg))
        prefs[str(scope)] = build_preference_polytope(str(scope), G, len(pos), eps_additive=eps_add, eps_multiplier=eps_mult, max_constraints=max_constraints, rng=rng, theta_lower=theta_lower, theta_upper=theta_upper)

    if fallback_positions:
        pooled = np.concatenate(fallback_positions)
    else:
        pooled = train_pos_all if len(train_pos_all) else np.arange(len(panel))
    if len(pooled) > max_train:
        pooled = rng.choice(pooled, size=max_train, replace=False)
    G = restrict_G(collect_pairwise_constraints(pooled, panel, candgen, feature_model, cfg))
    prefs["__fallback__"] = build_preference_polytope("__fallback__", G, len(pooled), eps_additive=eps_add, eps_multiplier=eps_mult, max_constraints=max_constraints, rng=rng, theta_lower=theta_lower, theta_upper=theta_upper)
    return IRLModel(preferences=prefs, feature_names=feature_model.feature_names, scope_col="_scope_id", fallback_scope_id="__fallback__", estimator="mv_inverse_optimization_polytope")
