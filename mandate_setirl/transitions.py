from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is optional.
    cKDTree = None


@dataclass
class TransitionLibrary:
    donor_pos: np.ndarray
    donor_series_id: np.ndarray
    donor_date: np.ndarray
    donor_cluster: np.ndarray
    donor_lag: np.ndarray
    donor_action: np.ndarray
    donor_delta: np.ndarray
    donor_move_l1: np.ndarray
    donor_state: np.ndarray
    retrieval_X: np.ndarray
    panel_cluster: np.ndarray
    panel_lag: np.ndarray
    panel_retrieval_X: np.ndarray
    trees_by_cluster: dict[int, Any]
    local_index_by_cluster: dict[int, np.ndarray]
    trees_by_cluster_magnitude: dict[tuple[int, str], Any]
    local_index_by_cluster_magnitude: dict[tuple[int, str], np.ndarray]
    trees_by_magnitude: dict[str, Any]
    local_index_by_magnitude: dict[str, np.ndarray]
    stress_trees_by_cluster: dict[int, Any]
    stress_local_index_by_cluster: dict[int, np.ndarray]
    stress_tree: Any
    stress_local_index: np.ndarray
    pooled_tree: Any
    donor_pos_to_index: dict[int, int]
    cfg: dict[str, Any]

    def query(self, row_pos: int, k: int | None = None) -> np.ndarray:
        picked, _ = self.query_with_distance(row_pos, k=k)
        return picked

    def query_with_distance(self, row_pos: int, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        tcfg = self.cfg.get("transition", {})
        k = int(k or tcfg.get("retrieval_k", 16))
        same_first = bool(tcfg.get("same_cluster_first", True))
        fallback = bool(tcfg.get("fallback_to_pooled", True))
        cl = int(self.panel_cluster[row_pos])
        query_x = self.panel_retrieval_X[row_pos]
        picks: list[tuple[int, float]] = []
        if same_first and cl in self.trees_by_cluster:
            picks.extend(self._query_tree(self.trees_by_cluster[cl], self.local_index_by_cluster[cl], query_x, max(k * 2, k), row_pos))
        if fallback and len(picks) < k and self.pooled_tree is not None:
            pooled = self._query_tree(self.pooled_tree, np.arange(len(self.donor_pos)), query_x, max(k * 3, k), row_pos)
            seen = {idx for idx, _ in picks}
            for idx, dist in pooled:
                if idx not in seen:
                    picks.append((idx, dist))
                    seen.add(idx)
                if len(picks) >= k:
                    break
        picks = picks[:k]
        if not picks:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx = np.asarray([p[0] for p in picks], dtype=int)
        dist = np.asarray([p[1] for p in picks], dtype=float)
        return self.donor_pos[idx], dist

    def query_by_magnitude(self, row_pos: int, move_min: float, move_max: float, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0 or len(self.donor_pos) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        cl = int(self.panel_cluster[row_pos])
        query_x = self.panel_retrieval_X[row_pos]
        label = self._magnitude_label(move_min, move_max)
        picks: list[tuple[int, float]] = []
        key = (cl, label)
        if key in self.trees_by_cluster_magnitude:
            picks.extend(self._query_tree(self.trees_by_cluster_magnitude[key], self.local_index_by_cluster_magnitude[key], query_x, max(k * 2, k), row_pos))
        if len(picks) < k and label in self.trees_by_magnitude:
            seen = {idx for idx, _ in picks}
            pooled = self._query_tree(self.trees_by_magnitude[label], self.local_index_by_magnitude[label], query_x, max(k * 3, k), row_pos)
            for idx, dist in pooled:
                if idx not in seen:
                    picks.append((idx, dist))
                    seen.add(idx)
                if len(picks) >= k:
                    break
        picks = picks[:k]
        if not picks:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx = np.asarray([p[0] for p in picks], dtype=int)
        dist = np.asarray([p[1] for p in picks], dtype=float)
        return self.donor_pos[idx], dist

    def predict_move_probability_and_magnitude(self, row_pos: int, k: int | None = None) -> tuple[float, float]:
        tcfg = self.cfg.get("transition", {})
        k = int(k or tcfg.get("mover_prediction_k", tcfg.get("retrieval_k", 16)))
        picked, _ = self.query_with_distance(row_pos, k=k)
        idx = self.donor_indices_for(picked)
        if len(idx) == 0:
            return 0.0, 0.0
        moves = self.donor_move_l1[idx]
        p_move = float(np.mean(moves > float(tcfg.get("mover_threshold", 0.05))))
        pred_mag = float(np.nanmedian(moves))
        return p_move, pred_mag

    def query_stress(self, row_pos: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k <= 0 or len(self.donor_pos) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        tcfg = self.cfg.get("transition", {})
        q = float(tcfg.get("stress_move_quantile", 0.75))
        cl = int(self.panel_cluster[row_pos])
        query_x = self.panel_retrieval_X[row_pos]
        picks: list[tuple[int, float]] = []
        if cl in self.stress_trees_by_cluster:
            picks.extend(self._query_tree(self.stress_trees_by_cluster[cl], self.stress_local_index_by_cluster[cl], query_x, max(k * 2, k), row_pos))
        if len(picks) < k and self.stress_tree is not None:
            seen = {idx for idx, _ in picks}
            pooled = self._query_tree(self.stress_tree, self.stress_local_index, query_x, max(k * 3, k), row_pos)
            for idx, dist in pooled:
                if idx not in seen:
                    picks.append((idx, dist))
                    seen.add(idx)
                if len(picks) >= k:
                    break
        picks = picks[:k]
        if not picks:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx = np.asarray([p[0] for p in picks], dtype=int)
        dist = np.asarray([p[1] for p in picks], dtype=float)
        return self.donor_pos[idx], dist

    @staticmethod
    def _magnitude_label(move_min: float, move_max: float) -> str:
        if move_max <= 0.05:
            return "small"
        if move_min >= 0.25:
            return "large"
        return "moderate"

    def deltas_for(self, donor_positions: np.ndarray | list[int]) -> np.ndarray:
        idx = [self.donor_pos_to_index[int(p)] for p in donor_positions if int(p) in self.donor_pos_to_index]
        return self.donor_delta[np.asarray(idx, dtype=int)] if idx else np.zeros((0, self.donor_delta.shape[1]))

    def donor_indices_for(self, donor_positions: np.ndarray | list[int]) -> np.ndarray:
        return np.asarray([self.donor_pos_to_index[int(p)] for p in donor_positions if int(p) in self.donor_pos_to_index], dtype=int)

    def regime_deltas(self, row_pos: int, max_count: int = 4) -> list[np.ndarray]:
        if len(self.donor_delta) == 0 or max_count <= 0:
            return []
        cl = int(self.panel_cluster[row_pos])
        idx = np.flatnonzero(self.donor_cluster == cl)
        if len(idx) < int(self.cfg.get("transition", {}).get("regime_min_count", 25)):
            idx = np.arange(len(self.donor_delta))
        deltas = self.donor_delta[idx]
        moves = self.donor_move_l1[idx]
        out: list[np.ndarray] = []
        out.append(np.nanmedian(deltas, axis=0))
        if max_count > 1 and len(deltas) >= 3:
            out.append(np.nanquantile(deltas, 0.25, axis=0))
        if max_count > 2 and len(deltas) >= 3:
            out.append(np.nanquantile(deltas, 0.75, axis=0))
        if max_count > 3:
            for j in np.argsort(moves)[::-1]:
                cand = deltas[int(j)]
                if all(np.linalg.norm(cand - prev) > 1e-8 for prev in out):
                    out.append(cand)
                if len(out) >= max_count:
                    break
        return out[:max_count]

    def _query_tree(self, tree, local_idx: np.ndarray, query_x: np.ndarray, k: int, row_pos: int | None = None) -> list[tuple[int, float]]:
        if len(local_idx) == 0:
            return []
        kk = min(k, len(local_idx))
        if cKDTree is not None and tree is not None:
            dist, idx = tree.query(query_x, k=kk)
            dist = np.atleast_1d(dist).astype(float)
            idx = np.atleast_1d(idx).astype(int)
            out = []
            for i, d in zip(idx, dist):
                donor_i = int(local_idx[i])
                if row_pos is not None and int(self.donor_pos[donor_i]) == int(row_pos):
                    continue
                out.append((donor_i, float(d)))
            return out
        X = self.retrieval_X[local_idx]
        dist = np.sum((X - query_x[None, :]) ** 2, axis=1)
        order = np.argsort(dist)[:kk]
        out = []
        for i in order:
            donor_i = int(local_idx[i])
            if row_pos is not None and int(self.donor_pos[donor_i]) == int(row_pos):
                continue
            out.append((donor_i, float(dist[i])))
        return out


def lag_action_matrix(panel: pd.DataFrame, action_cols: list[str]) -> np.ndarray:
    out = np.zeros((len(panel), len(action_cols)), dtype=float)
    for j, c in enumerate(action_cols):
        lag_col = f"lag_{c}"
        src = lag_col if lag_col in panel.columns else c
        out[:, j] = pd.to_numeric(panel[src], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return out


def make_state_matrix(panel: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    tcfg = cfg.get("transition", {})
    cols = list(tcfg.get("state_columns", []))
    if not cols:
        preferred = [
            "vix", "hy_oas", "log_net_assets", "net_flow_3m_scaled", "lag_net_flow_3m_scaled",
            "RETURN_AFTR_FEES_AND_EXPENSES", "STDV_AFTR_FEES_AND_EXPENSES",
            "IS_ETF", "IS_INDEX", "IS_FUND_OF_FUND", "IS_NON_DIVERSIFIED",
        ]
        cols = [c for c in preferred if c in panel.columns]
    if not cols:
        return np.zeros((len(panel), 1), dtype=float), ["constant_state"]
    X = np.zeros((len(panel), len(cols)), dtype=float)
    train_mask = panel["split"].astype(str).eq(str(tcfg.get("donor_split", "train"))).to_numpy() if "split" in panel.columns else np.ones(len(panel), dtype=bool)
    for j, c in enumerate(cols):
        s = pd.to_numeric(panel[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        fill = float(s.loc[train_mask].median(skipna=True)) if s.loc[train_mask].notna().any() else 0.0
        X[:, j] = s.fillna(fill).to_numpy(dtype=float)
    return X, cols


def build_transition_library(
    panel: pd.DataFrame,
    action_cols: list[str],
    entity_col: str,
    date_col: str,
    cfg: dict[str, Any],
) -> TransitionLibrary:
    tcfg = cfg.get("transition", {})
    donor_split = str(tcfg.get("donor_split", "train"))
    A = panel[action_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    L = lag_action_matrix(panel, action_cols)
    delta = A - L
    move = np.abs(delta).sum(axis=1)
    split = panel["split"].astype(str).to_numpy() if "split" in panel.columns else np.full(len(panel), donor_split)
    valid = split == donor_split
    valid &= np.isfinite(move)
    valid &= move > float(tcfg.get("min_move_l1", 0.005))
    donor_candidates = np.flatnonzero(valid)
    if len(donor_candidates) > 0:
        q = float(tcfg.get("max_move_l1_quantile", 0.99))
        cap = np.quantile(move[donor_candidates], q)
        donor_candidates = donor_candidates[move[donor_candidates] <= cap]
    clusters = panel["mandate_cluster"].fillna(0).astype(int).to_numpy() if "mandate_cluster" in panel.columns else np.zeros(len(panel), dtype=int)
    state, _ = make_state_matrix(panel, cfg)
    donor_state = state[donor_candidates]
    center = np.nanmedian(donor_state, axis=0) if len(donor_candidates) else np.zeros(state.shape[1])
    scale = np.nanstd(donor_state, axis=0) if len(donor_candidates) else np.ones(state.shape[1])
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (state - center[None, :]) / scale[None, :]
    state_w = np.sqrt(float(tcfg.get("retrieval_state_weight", 0.7)))
    lag_w = np.sqrt(float(tcfg.get("retrieval_lag_weight", 0.3)))
    retrieval_all = np.hstack([state_w * z, lag_w * L])
    retrieval_donor = retrieval_all[donor_candidates]
    trees: dict[int, Any] = {}
    local: dict[int, np.ndarray] = {}
    for cl in np.unique(clusters[donor_candidates]) if len(donor_candidates) else []:
        idx = np.flatnonzero(clusters[donor_candidates] == cl)
        local[int(cl)] = idx
        trees[int(cl)] = cKDTree(retrieval_donor[idx]) if cKDTree is not None and len(idx) else None
    pooled_tree = cKDTree(retrieval_donor) if cKDTree is not None and len(retrieval_donor) else None
    mag_trees: dict[tuple[int, str], Any] = {}
    mag_local: dict[tuple[int, str], np.ndarray] = {}
    pooled_mag_trees: dict[str, Any] = {}
    pooled_mag_local: dict[str, np.ndarray] = {}
    move_donor = move[donor_candidates]
    mag_masks = {
        "small": move_donor < 0.05,
        "moderate": (move_donor >= 0.05) & (move_donor <= 0.25),
        "large": move_donor > 0.25,
    }
    for label, mask in mag_masks.items():
        idx_all = np.flatnonzero(mask)
        pooled_mag_local[label] = idx_all
        pooled_mag_trees[label] = cKDTree(retrieval_donor[idx_all]) if cKDTree is not None and len(idx_all) else None
        for cl in np.unique(clusters[donor_candidates]) if len(donor_candidates) else []:
            idx = np.flatnonzero((clusters[donor_candidates] == cl) & mask)
            mag_local[(int(cl), label)] = idx
            mag_trees[(int(cl), label)] = cKDTree(retrieval_donor[idx]) if cKDTree is not None and len(idx) else None
    stress_q = float(tcfg.get("stress_move_quantile", 0.75))
    stress_thresh = np.nanquantile(move_donor, stress_q) if len(move_donor) else np.inf
    stress_mask = move_donor >= stress_thresh
    stress_idx_all = np.flatnonzero(stress_mask)
    stress_tree = cKDTree(retrieval_donor[stress_idx_all]) if cKDTree is not None and len(stress_idx_all) else None
    stress_trees: dict[int, Any] = {}
    stress_local: dict[int, np.ndarray] = {}
    for cl in np.unique(clusters[donor_candidates]) if len(donor_candidates) else []:
        idx = np.flatnonzero((clusters[donor_candidates] == cl) & stress_mask)
        stress_local[int(cl)] = idx
        stress_trees[int(cl)] = cKDTree(retrieval_donor[idx]) if cKDTree is not None and len(idx) else None
    return TransitionLibrary(
        donor_pos=donor_candidates.astype(int),
        donor_series_id=panel[entity_col].astype(str).to_numpy()[donor_candidates] if entity_col in panel.columns else donor_candidates.astype(str),
        donor_date=panel[date_col].to_numpy()[donor_candidates] if date_col in panel.columns else donor_candidates.astype(object),
        donor_cluster=clusters[donor_candidates].astype(int),
        donor_lag=L[donor_candidates],
        donor_action=A[donor_candidates],
        donor_delta=delta[donor_candidates],
        donor_move_l1=move[donor_candidates],
        donor_state=donor_state,
        retrieval_X=retrieval_donor,
        panel_cluster=clusters.astype(int),
        panel_lag=L,
        panel_retrieval_X=retrieval_all,
        trees_by_cluster=trees,
        local_index_by_cluster=local,
        trees_by_cluster_magnitude=mag_trees,
        local_index_by_cluster_magnitude=mag_local,
        trees_by_magnitude=pooled_mag_trees,
        local_index_by_magnitude=pooled_mag_local,
        stress_trees_by_cluster=stress_trees,
        stress_local_index_by_cluster=stress_local,
        stress_tree=stress_tree,
        stress_local_index=stress_idx_all,
        pooled_tree=pooled_tree,
        donor_pos_to_index={int(p): i for i, p in enumerate(donor_candidates.astype(int))},
        cfg=cfg,
    )
