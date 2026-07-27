from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .bounds import BoundModel
from .transitions import TransitionLibrary, build_transition_library
from .utils import bounded_simplex_project


@dataclass
class CandidateBatch:
    row_positions: np.ndarray
    actions: np.ndarray      # B x M x K
    sources: np.ndarray      # B x M object/string
    donor_pos: np.ndarray | None = None
    donor_series_id: np.ndarray | None = None
    donor_date: np.ndarray | None = None
    donor_move_l1: np.ndarray | None = None
    same_series_donor: np.ndarray | None = None
    donor_retrieval_distance: np.ndarray | None = None
    post_projection_l1_from_lag: np.ndarray | None = None
    post_projection_l1_from_raw_transition_candidate: np.ndarray | None = None
    predicted_move_probability: np.ndarray | None = None
    predicted_move_magnitude: np.ndarray | None = None
    predicted_move_bin: np.ndarray | None = None
    observed_index: int = 0


class CandidateGenerator:
    """Generate counterfactual allocation menus.

    Menus combine peer, historical, transition, and optional synthetic
    allocations.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        action_cols: list[str],
        asset_action_cols: list[str],
        overlay_action_cols: list[str],
        entity_col: str,
        cfg: dict[str, Any],
        rng: np.random.Generator,
        bounds: BoundModel,
    ):
        self.panel = panel.reset_index(drop=True)
        self.action_cols = action_cols
        self.asset_cols = asset_action_cols
        self.overlay_cols = overlay_action_cols
        self.entity_col = entity_col
        self.cfg = cfg
        self.rng = rng
        self.bounds = bounds
        self.K = len(action_cols)
        self.asset_idx = np.array([action_cols.index(c) for c in asset_action_cols], dtype=int)
        self.overlay_idx = np.array([action_cols.index(c) for c in overlay_action_cols], dtype=int)
        include_obs = bool(cfg.get("candidates", {}).get("include_observed_candidate", True))
        if not include_obs:
            raise ValueError(
                "candidates.include_observed_candidate must remain true for revealed-preference constraints; "
                "point-prediction diagnostics exclude the observed reference candidate."
            )
        self.actions_all = self.panel[action_cols].to_numpy(dtype=float)
        self.lag_actions_all = self._lag_actions_matrix()
        self.date_col = cfg.get("keys", {}).get("date_col", "report_date")
        self._build_pools()
        self.generator_mode = str(cfg.get("candidates", {}).get("generator_mode", "standard")).lower()
        self.transition_library: TransitionLibrary | None = None
        if self.generator_mode == "transition_augmented":
            self.transition_library = build_transition_library(self.panel, self.action_cols, self.entity_col, self.date_col, self.cfg)

    def _lag_actions_matrix(self) -> np.ndarray:
        arr = np.zeros((len(self.panel), self.K), dtype=float)
        for j, c in enumerate(self.action_cols):
            lag = f"lag_{c}"
            if lag in self.panel.columns:
                arr[:, j] = pd.to_numeric(self.panel[lag], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            else:
                arr[:, j] = self.actions_all[:, j]
        return arr

    def _build_pools(self):
        train_mask = self.panel["split"].to_numpy() == "train"
        self.train_positions = np.flatnonzero(train_mask)
        if len(self.train_positions) == 0:
            self.train_positions = np.arange(len(self.panel))
        self.clusters = self.panel["mandate_cluster"].fillna(0).astype(int).to_numpy() if "mandate_cluster" in self.panel.columns else np.zeros(len(self.panel), dtype=int)
        self.dates = pd.to_datetime(self.panel[self.date_col]).to_numpy() if self.date_col in self.panel.columns else np.arange(len(self.panel))

        self.cluster_train_pools: dict[int, np.ndarray] = {}
        for cl in np.unique(self.clusters[self.train_positions]):
            pos = self.train_positions[self.clusters[self.train_positions] == cl]
            self.cluster_train_pools[int(cl)] = pos

        # Same-date test peers are transductive; the default excludes them.
        self.cluster_date_pools: dict[tuple[int, object], np.ndarray] = {}
        self.cluster_date_train_pools: dict[tuple[int, object], np.ndarray] = {}
        for cl in np.unique(self.clusters):
            idx_cl = np.flatnonzero(self.clusters == cl)
            for d in np.unique(self.dates[idx_cl]):
                pos = idx_cl[self.dates[idx_cl] == d]
                self.cluster_date_pools[(int(cl), d)] = pos
                pos_train = pos[self.panel["split"].to_numpy()[pos] == "train"] if "split" in self.panel.columns else pos
                self.cluster_date_train_pools[(int(cl), d)] = pos_train

        self.entity_pools_all: dict[str, np.ndarray] = {}
        self.entity_pools_train: dict[str, np.ndarray] = {}
        if self.entity_col in self.panel.columns:
            ent = self.panel[self.entity_col].astype(str).to_numpy()
            for e in np.unique(ent):
                self.entity_pools_all[str(e)] = np.flatnonzero(ent == e)
            for e in np.unique(ent[self.train_positions]):
                self.entity_pools_train[str(e)] = self.train_positions[ent[self.train_positions] == e]

    def generate_for_positions(self, positions: np.ndarray | list[int], n_candidates: int | None = None, bounds_override: BoundModel | None = None) -> CandidateBatch:
        positions = np.asarray(positions, dtype=int)
        if self.cfg.get("candidates", {}).get("include_observed_candidate", True) is False:
            raise ValueError(
                "include_observed_candidate=false is not supported for revealed-preference fitting; "
                "the observed allocation must be candidate 0 to construct compatibility inequalities. "
                "Use *_ex_observed metrics for prediction-style evaluation."
            )
        M = int(n_candidates or self.cfg.get("candidates", {}).get("n_candidates", 64))
        if M < 2:
            M = 2
        B = len(positions)
        actions = np.zeros((B, M, self.K), dtype=float)
        sources = np.empty((B, M), dtype=object)
        donor_pos = -np.ones((B, M), dtype=int)
        donor_series_id = np.empty((B, M), dtype=object)
        donor_date = np.empty((B, M), dtype=object)
        donor_move_l1 = np.full((B, M), np.nan, dtype=float)
        same_series_donor = np.zeros((B, M), dtype=bool)
        donor_retrieval_distance = np.full((B, M), np.nan, dtype=float)
        post_projection_l1_from_lag = np.full((B, M), np.nan, dtype=float)
        post_projection_l1_from_raw = np.full((B, M), np.nan, dtype=float)
        predicted_move_probability = np.full(B, np.nan, dtype=float)
        predicted_move_magnitude = np.full(B, np.nan, dtype=float)
        predicted_move_bin = np.empty(B, dtype=object)
        donor_series_id[:, :] = ""
        donor_date[:, :] = ""
        bmodel = bounds_override or self.bounds
        lower, upper = bmodel.bounds_for_positions(positions)
        actions[:, 0, :] = self.actions_all[positions]
        sources[:, 0] = "observed"
        for bi, pos in enumerate(positions):
            candidates = [actions[bi, 0].copy()]
            srcs = ["observed"]
            metas: list[dict[str, Any]] = [self._empty_meta(raw_action=actions[bi, 0].copy())]
            if self.generator_mode == "transition_augmented":
                p_move, pred_mag, pred_bin = self._predict_move_route(pos)
                predicted_move_probability[bi] = p_move
                predicted_move_magnitude[bi] = pred_mag
                predicted_move_bin[bi] = pred_bin
                selected = self._dedupe_and_select_records(
                    pos,
                    self._transition_augmented_actions(pos, bi, lower, upper),
                    target_count=M - 1,
                    lower=lower[bi],
                    upper=upper[bi],
                    predicted_move_bin=pred_bin,
                )
                for rec in selected:
                    candidates.append(rec["action"])
                    srcs.append(rec["source"])
                    metas.append(rec)
            else:
                n_remaining = M - 1
                counts = self._counts(n_remaining)
                for _ in range(counts["peer_same_date"]):
                    candidates.append(self._sample_peer_same_date(pos))
                    srcs.append("peer_same_date")
                    metas.append(self._empty_meta())
                for _ in range(counts["peer_train"]):
                    candidates.append(self._sample_peer_train(pos))
                    srcs.append("peer_train")
                    metas.append(self._empty_meta())
                for _ in range(counts["self"]):
                    candidates.append(self._sample_self_lagged(pos))
                    srcs.append("self_lagged")
                    metas.append(self._empty_meta())
                for _ in range(counts["dirichlet"]):
                    candidates.append(self._sample_dirichlet(bi, lower, upper))
                    srcs.append("dirichlet")
                    metas.append(self._empty_meta())
                for name, vec in self._frontier_actions(pos, bi, lower, upper):
                    if len(candidates) < M:
                        candidates.append(vec)
                        srcs.append(name)
                        metas.append(self._empty_meta(raw_action=vec.copy()))
            while len(candidates) < M:
                # Fill with real train peers first; synthetic candidates only if necessary.
                candidates.append(self._sample_peer_train(pos))
                srcs.append("peer_train_fill")
                metas.append(self._empty_meta())
            arr = np.vstack(candidates[:M])
            for m in range(1, M):
                arr[m] = self.project_action(arr[m], lower[bi], upper[bi])
            lag = self.lag_actions_all[pos]
            actions[bi] = arr
            sources[bi] = np.array(srcs[:M], dtype=object)
            for m, meta in enumerate(metas[:M]):
                donor_pos[bi, m] = int(meta.get("donor_pos", -1))
                donor_series_id[bi, m] = str(meta.get("donor_series_id", ""))
                donor_date[bi, m] = meta.get("donor_date", "")
                donor_move_l1[bi, m] = float(meta.get("donor_move_l1", np.nan))
                donor_retrieval_distance[bi, m] = float(meta.get("donor_retrieval_distance", np.nan))
                same_series_donor[bi, m] = bool(meta.get("same_series_donor", False))
                post_projection_l1_from_lag[bi, m] = float(np.abs(arr[m] - lag).sum())
                raw = meta.get("raw_action", None)
                if raw is not None:
                    post_projection_l1_from_raw[bi, m] = float(np.abs(arr[m] - np.asarray(raw, dtype=float)).sum())
        return CandidateBatch(
            row_positions=positions,
            actions=actions,
            sources=sources,
            donor_pos=donor_pos,
            donor_series_id=donor_series_id,
            donor_date=donor_date,
            donor_move_l1=donor_move_l1,
            same_series_donor=same_series_donor,
            donor_retrieval_distance=donor_retrieval_distance,
            post_projection_l1_from_lag=post_projection_l1_from_lag,
            post_projection_l1_from_raw_transition_candidate=post_projection_l1_from_raw,
            predicted_move_probability=predicted_move_probability,
            predicted_move_magnitude=predicted_move_magnitude,
            predicted_move_bin=predicted_move_bin,
            observed_index=0,
        )

    def _counts(self, n_remaining: int) -> dict[str, int]:
        ccfg = self.cfg.get("candidates", {})
        same_mode = str(ccfg.get("same_date_peer_mode", "off")).lower()
        peer_same_frac = float(ccfg.get("peer_same_date_frac", 0.0))
        if same_mode in {"off", "none", "false", "0"}:
            peer_same_frac = 0.0
        peer_same = int(round(n_remaining * peer_same_frac))
        peer_train = int(round(n_remaining * float(ccfg.get("peer_train_frac", 0.70))))
        n_self = int(round(n_remaining * float(ccfg.get("self_frac", 0.20))))
        n_dir = int(round(n_remaining * float(ccfg.get("dirichlet_frac", 0.0))))
        total = peer_same + peer_train + n_self + n_dir
        if total > n_remaining:
            # Reduce synthetic first, then same-date peers because those can be
            # transductive if explicitly enabled.
            over = total - n_remaining
            take = min(over, n_dir); n_dir -= take; over -= take
            take = min(over, peer_same); peer_same -= take; over -= take
            take = min(over, n_self); n_self -= take; over -= take
            peer_train = max(0, peer_train - over)
        n_front = max(0, n_remaining - peer_same - peer_train - n_self - n_dir)
        return {"peer_same_date": peer_same, "peer_train": peer_train, "self": n_self, "dirichlet": n_dir, "frontier": n_front}

    def _transition_augmented_actions(self, pos: int, bi: int, lower: np.ndarray, upper: np.ndarray):
        ccfg = self.cfg.get("candidates", {})
        tcfg = self.cfg.get("transition", {})
        out = []
        lag = self.lag_actions_all[pos].copy()
        for _ in range(int(ccfg.get("self_lag_exact_count", 1))):
            out.append(self._candidate_record("lag_exact", lag.copy(), raw_action=lag.copy()))
        for _ in range(int(ccfg.get("self_lag_projected_count", 1))):
            out.append(self._candidate_record("lag_projected", self.project_action(lag.copy(), lower[bi], upper[bi]), raw_action=lag.copy()))
        raw_mult = max(1, int(ccfg.get("raw_candidate_oversample_multiplier", 3)))
        for _ in range(int(ccfg.get("peer_train_count", 6)) * raw_mult):
            out.append(self._candidate_record("peer_static_train", self._sample_peer_train(pos)))
        if self.transition_library is not None:
            donor_pos, donor_dist = self.transition_library.query_with_distance(pos, k=int(tcfg.get("retrieval_k", ccfg.get("transition_knn_count", 16))))
            deltas = self.transition_library.deltas_for(donor_pos)
            scales = [float(x) for x in tcfg.get("scales", [0.5, 1.0])]
            knn_budget = int(ccfg.get("transition_knn_count", 16)) * raw_mult
            made = 0
            for d, dp, dd in zip(deltas, donor_pos, donor_dist):
                for lam in scales:
                    if made >= knn_budget:
                        break
                    raw = lag + lam * d
                    out.append(self._transition_record(pos, f"transition_knn_lam{lam:g}", raw, lower[bi], upper[bi], int(dp), float(dd)))
                    made += 1
                if made >= knn_budget:
                    break
            shrunk_budget = int(ccfg.get("transition_shrunk_count", 4)) * raw_mult
            made = 0
            for d, dp, dd in zip(deltas, donor_pos, donor_dist):
                if made >= shrunk_budget:
                    break
                raw = lag + 0.5 * d
                out.append(self._transition_record(pos, "transition_shrunk", raw, lower[bi], upper[bi], int(dp), float(dd)))
                made += 1
            small_budget = int(ccfg.get("transition_small_count", 0)) * raw_mult
            for d, dp, dd in self._magnitude_transition_iter(pos, 0.005, 0.05, small_budget):
                raw = lag + d
                out.append(self._transition_record(pos, "transition_small", raw, lower[bi], upper[bi], int(dp), float(dd)))
            moderate_budget = int(ccfg.get("transition_moderate_count", ccfg.get("transition_magnitude_count", 0))) * raw_mult
            for d, dp, dd in self._magnitude_transition_iter(pos, 0.05, 0.25, moderate_budget):
                raw = lag + d
                out.append(self._transition_record(pos, "transition_moderate", raw, lower[bi], upper[bi], int(dp), float(dd)))
            direction_mags = [float(x) for x in tcfg.get("direction_magnitudes", [0.08, 0.15, 0.30, 0.50])]
            large_budget = int(ccfg.get("transition_large_count", 0)) * raw_mult
            for n, (d, dp, dd) in enumerate(self._magnitude_transition_iter(pos, 0.25, np.inf, large_budget)):
                mag = direction_mags[n % len(direction_mags)]
                raw = lag + self._direction_scaled_delta(d, mag)
                out.append(self._transition_record(pos, f"transition_large_dir_m{mag:g}", raw, lower[bi], upper[bi], int(dp), float(dd)))
            stress_budget = int(ccfg.get("transition_stress_count", 0)) * raw_mult
            if stress_budget > 0:
                spos, sdist = self.transition_library.query_stress(pos, stress_budget)
                sdeltas = self.transition_library.deltas_for(spos)
                for n, (d, dp, dd) in enumerate(zip(sdeltas, spos, sdist)):
                    mag = direction_mags[n % len(direction_mags)]
                    raw = lag + self._direction_scaled_delta(d, mag)
                    out.append(self._transition_record(pos, f"transition_stress_dir_m{mag:g}", raw, lower[bi], upper[bi], int(dp), float(dd)))
            regime_budget = int(ccfg.get("transition_regime_count", 8)) * raw_mult
            for d in self.transition_library.regime_deltas(pos, max_count=regime_budget):
                raw = lag + d
                out.append(self._candidate_record("transition_regime_medoid", self.project_action(raw, lower[bi], upper[bi]), raw_action=raw))
        for name, vec in self._clean_mandate_frontier(pos, bi, lower, upper):
            out.append(self._candidate_record(name, vec, raw_action=vec.copy()))
        return out

    def _magnitude_transition_iter(self, pos: int, lo: float, hi: float, count: int):
        if self.transition_library is None or count <= 0:
            return []
        mpos, mdist = self.transition_library.query_by_magnitude(pos, lo, hi, count)
        mdeltas = self.transition_library.deltas_for(mpos)
        return list(zip(mdeltas, mpos, mdist))

    @staticmethod
    def _direction_scaled_delta(delta: np.ndarray, target_l1: float) -> np.ndarray:
        denom = float(np.abs(delta).sum())
        if denom <= 1e-12 or not np.isfinite(denom):
            return np.zeros_like(delta)
        return float(target_l1) * delta / denom

    def _predict_move_route(self, pos: int) -> tuple[float, float, str]:
        ccfg = self.cfg.get("candidates", {})
        if not bool(ccfg.get("mover_routing_enabled", False)) or self.transition_library is None:
            return np.nan, np.nan, "unrouted"
        k = int(self.cfg.get("transition", {}).get("mover_prediction_k", self.cfg.get("transition", {}).get("retrieval_k", 16)))
        p_move, pred_mag = self.transition_library.predict_move_probability_and_magnitude(pos, k=k)
        if pred_mag <= 0.05:
            return p_move, pred_mag, "stayer"
        if pred_mag <= 0.25:
            return p_move, pred_mag, "moderate"
        return p_move, pred_mag, "large"

    def _dedupe_and_select_records(self, pos: int, records: list[dict[str, Any]], target_count: int, lower: np.ndarray, upper: np.ndarray, predicted_move_bin: str = "unrouted") -> list[dict[str, Any]]:
        decimals = int(self.cfg.get("candidates", {}).get("dedupe_round_decimals", 6))
        seen: set[tuple[float, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for rec in records:
            action = self.project_action(np.asarray(rec["action"], dtype=float), lower, upper)
            key = tuple(np.round(action, decimals))
            if key in seen:
                continue
            seen.add(key)
            rec = dict(rec)
            rec["action"] = action
            deduped.append(rec)
        quotas = self._source_quotas(target_count, predicted_move_bin=predicted_move_bin)
        selected: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        for bucket in ["lag", "static_peer", "peer_small", "peer_moderate", "peer_large_stress", "regime", "shrunk", "clean"]:
            need = int(quotas.get(bucket, 0))
            if need <= 0:
                continue
            for idx, rec in enumerate(deduped):
                if idx in selected_ids or self._quota_bucket(str(rec.get("source", ""))) != bucket:
                    continue
                selected.append(rec)
                selected_ids.add(idx)
                if sum(self._quota_bucket(str(r.get("source", ""))) == bucket for r in selected) >= need:
                    break
        for idx, rec in enumerate(deduped):
            if len(selected) >= target_count:
                break
            if idx not in selected_ids:
                selected.append(rec)
                selected_ids.add(idx)
        return selected[:target_count]

    def _source_quotas(self, target_count: int, predicted_move_bin: str = "unrouted") -> dict[str, int]:
        ccfg = self.cfg.get("candidates", {})
        if bool(ccfg.get("mover_routing_enabled", False)):
            routed = ccfg.get("source_quotas_by_predicted_move", {})
            if isinstance(routed, dict) and str(predicted_move_bin) in routed:
                return {str(k): int(v) for k, v in routed[str(predicted_move_bin)].items()}
            default_routed = self._default_routed_quotas().get(str(predicted_move_bin))
            if default_routed is not None:
                base = dict(default_routed)
            else:
                base = None
        else:
            base = None
        explicit = ccfg.get("source_quotas")
        if base is not None:
            pass
        elif isinstance(explicit, dict):
            base = {str(k): int(v) for k, v in explicit.items()}
        else:
            base = {
            "lag": 2,
            "static_peer": 2,
            "peer_small": 4,
            "peer_moderate": 12,
            "peer_large_stress": 8,
            "regime": 6,
            "shrunk": 3,
            "clean": 2,
            }
        base_total = sum(base.values())
        if target_count == base_total:
            return base
        scaled = {k: int(np.floor(v * target_count / base_total)) for k, v in base.items()}
        while sum(scaled.values()) < target_count:
            for k in ["peer_moderate", "peer_large_stress", "regime", "peer_small", "shrunk", "lag", "static_peer", "clean"]:
                scaled[k] += 1
                if sum(scaled.values()) >= target_count:
                    break
        return scaled

    @staticmethod
    def _default_routed_quotas() -> dict[str, dict[str, int]]:
        return {
            "stayer": {
                "lag": 2,
                "static_peer": 2,
                "peer_small": 12,
                "peer_moderate": 8,
                "peer_large_stress": 2,
                "regime": 6,
                "shrunk": 5,
                "clean": 2,
            },
            "moderate": {
                "lag": 2,
                "static_peer": 2,
                "peer_small": 4,
                "peer_moderate": 12,
                "peer_large_stress": 8,
                "regime": 6,
                "shrunk": 3,
                "clean": 2,
            },
            "large": {
                "lag": 1,
                "static_peer": 1,
                "peer_small": 2,
                "peer_moderate": 10,
                "peer_large_stress": 14,
                "regime": 6,
                "shrunk": 3,
                "clean": 2,
            },
        }

    @staticmethod
    def _quota_bucket(source: str) -> str:
        s = str(source)
        if s.startswith("lag_"):
            return "lag"
        if s.startswith("peer_static") or s.startswith("peer_train"):
            return "static_peer"
        if s.startswith("transition_small"):
            return "peer_small"
        if s.startswith("transition_moderate"):
            return "peer_moderate"
        if s.startswith("transition_large") or s.startswith("transition_stress"):
            return "peer_large_stress"
        if s.startswith("transition_regime"):
            return "regime"
        if s.startswith("transition_shrunk"):
            return "shrunk"
        if s.startswith("clean_frontier"):
            return "clean"
        if s.startswith("transition_knn"):
            return "peer_moderate"
        return "other"

    def _empty_meta(self, raw_action=None) -> dict[str, Any]:
        return {
            "donor_pos": -1,
            "donor_series_id": "",
            "donor_date": "",
            "donor_move_l1": np.nan,
            "same_series_donor": False,
            "donor_retrieval_distance": np.nan,
            "raw_action": raw_action,
        }

    def _candidate_record(self, source: str, action: np.ndarray, raw_action=None, **meta) -> dict[str, Any]:
        rec = self._empty_meta(raw_action=raw_action)
        rec.update(meta)
        rec["source"] = source
        rec["action"] = action
        return rec

    def _transition_record(self, row_pos: int, source: str, raw: np.ndarray, lower: np.ndarray, upper: np.ndarray, donor_pos: int, donor_dist: float) -> dict[str, Any]:
        projected = self.project_action(raw, lower, upper)
        donor_idx = self.transition_library.donor_indices_for([donor_pos])[0] if self.transition_library is not None else -1
        row_series = str(self.panel.loc[row_pos, self.entity_col]) if self.entity_col in self.panel.columns else ""
        donor_series = str(self.transition_library.donor_series_id[donor_idx]) if donor_idx >= 0 else ""
        return self._candidate_record(
            source,
            projected,
            raw_action=raw.copy(),
            donor_pos=donor_pos,
            donor_series_id=donor_series,
            donor_date=self.transition_library.donor_date[donor_idx] if donor_idx >= 0 else "",
            donor_move_l1=float(self.transition_library.donor_move_l1[donor_idx]) if donor_idx >= 0 else np.nan,
            same_series_donor=bool(row_series and donor_series == row_series),
            donor_retrieval_distance=donor_dist,
        )

    def _sample_peer_same_date(self, pos: int) -> np.ndarray:
        ccfg = self.cfg.get("candidates", {})
        mode = str(ccfg.get("same_date_peer_mode", "off")).lower()
        if mode in {"off", "none", "false", "0"}:
            return self._sample_peer_train(pos)
        cl = int(self.clusters[pos])
        d = self.dates[pos]
        if mode in {"train_only", "train", "asof", "nontransductive"}:
            pool = self.cluster_date_train_pools.get((cl, d))
        elif mode in {"transductive", "all", "same_date_all"}:
            if not bool(ccfg.get("allow_transductive_same_date", False)):
                raise ValueError(
                    "same_date_peer_mode='transductive' requires "
                    "candidates.allow_transductive_same_date=true."
                )
            pool = self.cluster_date_pools.get((cl, d))
        else:
            raise ValueError(f"Unknown candidates.same_date_peer_mode={mode!r}")
        if pool is None or len(pool) <= 1:
            return self._sample_peer_train(pos)
        pool = pool[pool != pos]
        if len(pool) == 0:
            return self._sample_peer_train(pos)
        pick = int(self.rng.choice(pool))
        return self.actions_all[pick].copy()

    def _sample_peer_train(self, pos: int) -> np.ndarray:
        cl = int(self.clusters[pos])
        pool = self.cluster_train_pools.get(cl, self.train_positions)
        if pool is None or len(pool) == 0:
            pool = self.train_positions
        pick = int(self.rng.choice(pool)) if len(pool) else int(pos)
        return self.actions_all[pick].copy()

    def _sample_self_lagged(self, pos: int) -> np.ndarray:
        # Strictly earlier rows for the same fund, else current lag vector, else train peer.
        ent = str(self.panel.loc[pos, self.entity_col]) if self.entity_col in self.panel.columns else ""
        pool = self.entity_pools_all.get(ent)
        if pool is None or len(pool) == 0:
            return self.lag_actions_all[pos].copy()
        if self.date_col in self.panel.columns:
            d = self.dates[pos]
            prior = pool[self.dates[pool] < d]
        else:
            prior = pool[pool < pos]
        if len(prior) == 0:
            return self.lag_actions_all[pos].copy()
        pick = int(self.rng.choice(prior))
        return self.actions_all[pick].copy()

    def _sample_dirichlet(self, bi: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        ccfg = self.cfg.get("candidates", {})
        v = np.zeros(self.K, dtype=float)
        if len(self.asset_idx) > 0:
            alpha = float(ccfg.get("dirichlet_alpha", 0.7))
            w = self.rng.dirichlet(np.ones(len(self.asset_idx)) * alpha)
            v[self.asset_idx] = w
        for j in self.overlay_idx:
            lo = lower[bi, j]
            hi = upper[bi, j]
            if not np.isfinite(lo): lo = 0.0
            if not np.isfinite(hi) or hi <= lo: hi = lo + 0.01
            v[j] = self.rng.uniform(lo, hi)
        return v

    def _frontier_actions(self, pos: int, bi: int, lower: np.ndarray, upper: np.ndarray):
        out = []
        out.append(("frontier_style", self.lag_actions_all[pos].copy()))
        v = self.lag_actions_all[pos].copy()
        if len(self.asset_idx) > 0:
            v[self.asset_idx] = 0.0
            stiv = self._col_index_safe("asset_STIV")
            v[stiv if stiv is not None else self.asset_idx[0]] = 1.0
        for name in [
            "restricted_pct", "derivative_pct", "derivative_pct_pos", "derivative_pct_neg_abs",
            "short_pct", "short_pct_pos", "short_pct_neg_abs", "non_us_pct",
        ]:
            idx = self._col_index_safe(name)
            if idx is not None:
                v[idx] = lower[bi, idx]
        cash = self._col_index_safe("cash_proxy")
        if cash is not None:
            v[cash] = upper[bi, cash]
        out.append(("frontier_liquidity", v))
        v2 = self.lag_actions_all[pos].copy()
        if len(self.asset_idx) > 0:
            v2[self.asset_idx] = 0.0
            ec = self._col_index_safe("asset_EC")
            v2[ec if ec is not None else self.asset_idx[0]] = 1.0
        out.append(("frontier_equity", v2))
        v3 = self.lag_actions_all[pos].copy()
        if len(self.asset_idx) > 0:
            v3[self.asset_idx] = 0.0
            for col in ["asset_DBT", "asset_ABS_MBS", "asset_STIV"]:
                idx = self._col_index_safe(col)
                if idx is not None:
                    v3[idx] = 1.0
                    break
            if v3[self.asset_idx].sum() == 0:
                v3[self.asset_idx[0]] = 1.0
        out.append(("frontier_conservative", v3))
        return out

    def _clean_mandate_frontier(self, pos: int, bi: int, lower: np.ndarray, upper: np.ndarray):
        out = []
        cl = int(self.clusters[pos])
        pool = self.cluster_train_pools.get(cl, self.train_positions)
        if pool is None or len(pool) == 0:
            pool = self.train_positions
        base = np.nanmedian(self.actions_all[pool], axis=0) if len(pool) else np.zeros(self.K, dtype=float)
        out.append(("clean_frontier_centroid", self.project_action(base.copy(), lower[bi], upper[bi])))
        conservative = base.copy()
        if len(self.asset_idx) > 0:
            conservative[self.asset_idx] = 0.0
            for col in ["asset_STIV", "asset_DBT", "asset_ABS_MBS"]:
                idx = self._col_index_safe(col)
                if idx is not None:
                    conservative[idx] = 1.0
                    break
        out.append(("clean_frontier_conservative", self.project_action(conservative, lower[bi], upper[bi])))
        equity = base.copy()
        if len(self.asset_idx) > 0:
            equity[self.asset_idx] = 0.0
            ec = self._col_index_safe("asset_EC")
            equity[ec if ec is not None else self.asset_idx[0]] = 1.0
        out.append(("clean_frontier_equity", self.project_action(equity, lower[bi], upper[bi])))
        return out[: int(self.cfg.get("candidates", {}).get("clean_frontier_count", 3))]

    def _col_index_safe(self, name: str):
        try:
            return self.action_cols.index(name)
        except ValueError:
            return None

    def project_action(self, v: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=float).copy()
        if len(self.asset_idx) > 0:
            x_asset = x[self.asset_idx]
            lo = lower[self.asset_idx]
            hi = upper[self.asset_idx]
            x[self.asset_idx] = bounded_simplex_project(x_asset, lo, hi, target_sum=1.0)
        if len(self.overlay_idx) > 0:
            x[self.overlay_idx] = np.clip(x[self.overlay_idx], lower[self.overlay_idx], upper[self.overlay_idx])
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
