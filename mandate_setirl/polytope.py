from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linprog


@dataclass
class LinearPreferencePolytope:
    """Preference set for linear utility weights on a bounded simplex.

    The identified set is
        Theta = {theta_lower <= theta <= theta_upper,
                 1'theta = 1,
                 G theta <= eps_bound}.

    Rows of G are utility-feature differences phi(alt)-phi(obs).  The
    inequality G theta <= eps means the observed action rationalizes the
    candidate menu up to max-regret eps.  This is the polyhedral revealed-
    preference object; no rejection sampling is used by default.
    """

    scope_id: str
    G: np.ndarray
    eps_star: float
    eps_bound: float
    theta_point: np.ndarray
    theta_min: np.ndarray
    theta_max: np.ndarray
    support_thetas: np.ndarray
    n_constraints: int
    n_train_obs: int
    feasible: bool
    status: str = "ok"
    theta_lower: np.ndarray | None = None
    theta_upper: np.ndarray | None = None

    @property
    def d(self) -> int:
        return int(self.theta_point.shape[0])

    @property
    def interval_width(self) -> np.ndarray:
        return np.maximum(self.theta_max - self.theta_min, 0.0)

    @property
    def interval_l1_upper_diameter(self) -> float:
        return float(np.sum(self.interval_width))

    def A_ub(self) -> np.ndarray:
        return self.G

    def b_ub(self) -> np.ndarray:
        return np.full(self.G.shape[0], self.eps_bound, dtype=float)


def _simplex_eq(d: int) -> tuple[np.ndarray, np.ndarray]:
    return np.ones((1, d), dtype=float), np.array([1.0], dtype=float)


def _sanitize_theta_bounds(d: int, theta_lower=None, theta_upper=None) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(d, dtype=float) if theta_lower is None else np.asarray(theta_lower, dtype=float).copy()
    hi = np.ones(d, dtype=float) if theta_upper is None else np.asarray(theta_upper, dtype=float).copy()
    if lo.shape[0] != d or hi.shape[0] != d:
        raise ValueError("theta_lower/theta_upper must match feature dimension")
    lo = np.nan_to_num(lo, nan=0.0, posinf=0.0, neginf=0.0)
    hi = np.nan_to_num(hi, nan=1.0, posinf=1.0, neginf=0.0)
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    hi = np.maximum(hi, lo)
    # Fall back to the ordinary simplex when the coordinate bounds are infeasible.
    if lo.sum() > 1.0 + 1e-10 or hi.sum() < 1.0 - 1e-10:
        lo = np.zeros(d, dtype=float)
        hi = np.ones(d, dtype=float)
    return lo, hi


def _theta_bounds(d: int, theta_lower=None, theta_upper=None):
    lo, hi = _sanitize_theta_bounds(d, theta_lower, theta_upper)
    return [(float(lo[j]), float(hi[j])) for j in range(d)]


def _bounded_simplex_midpoint(d: int, theta_lower=None, theta_upper=None) -> np.ndarray:
    lo, hi = _sanitize_theta_bounds(d, theta_lower, theta_upper)
    x = lo.copy()
    rem = 1.0 - x.sum()
    cap = hi - lo
    if rem > 1e-12 and cap.sum() > 1e-12:
        x += rem * cap / cap.sum()
    s = x.sum()
    if s <= 0 or not np.isfinite(s):
        x = np.ones(d, dtype=float) / d
    else:
        x = x / s
    x = np.minimum(np.maximum(x, lo), hi)
    # final small correction if clipping perturbed the sum
    s = x.sum()
    if abs(s - 1.0) > 1e-8:
        rem = 1.0 - s
        if rem > 0:
            cap = hi - x
            if cap.sum() > 1e-12:
                x += rem * cap / cap.sum()
        else:
            room = x - lo
            if room.sum() > 1e-12:
                x += rem * room / room.sum()
    return x


def solve_minimax_epsilon(
    G: np.ndarray,
    *,
    eps_floor: float = 0.0,
    method: str = "highs",
    theta_lower=None,
    theta_upper=None,
) -> tuple[bool, float, np.ndarray, str]:
    """Solve min_{theta in bounded simplex, eps>=eps_floor} eps s.t. G theta <= eps."""
    G = np.asarray(G, dtype=float)
    if G.ndim != 2:
        raise ValueError("G must be 2D")
    n, d = G.shape
    c = np.zeros(d + 1, dtype=float)
    c[-1] = 1.0
    A_ub = np.hstack([G, -np.ones((n, 1), dtype=float)]) if n else np.zeros((0, d + 1))
    b_ub = np.zeros(n, dtype=float)
    A_eq = np.zeros((1, d + 1), dtype=float)
    A_eq[0, :d] = 1.0
    b_eq = np.array([1.0], dtype=float)
    bounds = _theta_bounds(d, theta_lower, theta_upper) + [(eps_floor, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
    if not res.success:
        theta = _bounded_simplex_midpoint(d, theta_lower, theta_upper)
        eps = float(max(eps_floor, np.max(G @ theta) if n else eps_floor))
        return False, eps, theta, str(res.message)
    return True, float(res.x[-1]), np.asarray(res.x[:d], dtype=float), str(res.message)


def solve_theta_lp(
    c: np.ndarray,
    G: np.ndarray,
    eps_bound: float,
    *,
    maximize: bool = False,
    method: str = "highs",
    theta_lower=None,
    theta_upper=None,
) -> tuple[bool, np.ndarray, float, str]:
    """Linear optimization over Theta."""
    G = np.asarray(G, dtype=float)
    c = np.asarray(c, dtype=float)
    d = c.shape[0]
    obj = -c if maximize else c
    A_ub = G if G.size else None
    b_ub = np.full(G.shape[0], eps_bound, dtype=float) if G.size else None
    A_eq, b_eq = _simplex_eq(d)
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=_theta_bounds(d, theta_lower, theta_upper), method=method)
    if not res.success:
        theta = _bounded_simplex_midpoint(d, theta_lower, theta_upper)
        return False, theta, float(np.nan), str(res.message)
    val = float(c @ res.x)
    return True, np.asarray(res.x, dtype=float), val, str(res.message)


def compute_coordinate_intervals(
    G: np.ndarray,
    eps_bound: float,
    *,
    method: str = "highs",
    theta_lower=None,
    theta_upper=None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    d = G.shape[1]
    lo, hi = _sanitize_theta_bounds(d, theta_lower, theta_upper)
    mins = np.zeros(d, dtype=float)
    maxs = np.zeros(d, dtype=float)
    support: list[np.ndarray] = []
    for j in range(d):
        c = np.zeros(d, dtype=float)
        c[j] = 1.0
        ok_min, th_min, val_min, _ = solve_theta_lp(c, G, eps_bound, maximize=False, method=method, theta_lower=lo, theta_upper=hi)
        ok_max, th_max, val_max, _ = solve_theta_lp(c, G, eps_bound, maximize=True, method=method, theta_lower=lo, theta_upper=hi)
        if ok_min and np.isfinite(val_min):
            mins[j] = max(float(lo[j]), min(float(hi[j]), val_min))
            support.append(th_min)
        else:
            mins[j] = float(lo[j])
        if ok_max and np.isfinite(val_max):
            maxs[j] = max(float(lo[j]), min(float(hi[j]), val_max))
            support.append(th_max)
        else:
            maxs[j] = float(hi[j])
    maxs = np.maximum(maxs, mins)
    return mins, maxs, support


def dedupe_thetas(thetas: Iterable[np.ndarray], *, tol: int = 8) -> np.ndarray:
    arrs = []
    seen = set()
    for th in thetas:
        x = np.asarray(th, dtype=float)
        if x.ndim != 1:
            continue
        s = float(x.sum())
        if s <= 0 or not np.isfinite(s):
            continue
        x = np.clip(x / s, 0.0, 1.0)
        key = tuple(np.round(x, tol))
        if key not in seen:
            seen.add(key)
            arrs.append(x)
    if not arrs:
        return np.empty((0, 0), dtype=float)
    return np.vstack(arrs)


def build_preference_polytope(
    scope_id: str,
    G: np.ndarray,
    n_train_obs: int,
    *,
    eps_additive: float = 0.02,
    eps_multiplier: float = 1.05,
    max_constraints: int | None = None,
    rng: np.random.Generator | None = None,
    method: str = "highs",
    theta_lower=None,
    theta_upper=None,
) -> LinearPreferencePolytope:
    """Fit a polyhedral identified set from pairwise revealed-preference inequalities."""
    G = np.asarray(G, dtype=float)
    G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
    if G.ndim != 2:
        raise ValueError("G must be 2D")
    d = G.shape[1]
    theta_lower_arr, theta_upper_arr = _sanitize_theta_bounds(d, theta_lower, theta_upper)
    if max_constraints is not None and max_constraints > 0 and G.shape[0] > max_constraints:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(np.arange(G.shape[0]), size=int(max_constraints), replace=False)
        G_fit = G[idx]
    else:
        G_fit = G
    feasible, eps_star, theta_point, status = solve_minimax_epsilon(G_fit, method=method, theta_lower=theta_lower_arr, theta_upper=theta_upper_arr)
    eps_bound = float(max(eps_star * eps_multiplier, eps_star + eps_additive))
    theta_min, theta_max, support = compute_coordinate_intervals(G_fit, eps_bound, method=method, theta_lower=theta_lower_arr, theta_upper=theta_upper_arr)
    support.append(theta_point)
    midpoint = _bounded_simplex_midpoint(d, theta_lower_arr, theta_upper_arr)
    if G_fit.shape[0] == 0 or np.all(G_fit @ midpoint <= eps_bound + 1e-8):
        support.append(midpoint)
    support_arr = dedupe_thetas(support)
    if support_arr.size == 0:
        support_arr = theta_point[None, :]
    return LinearPreferencePolytope(
        scope_id=str(scope_id),
        G=G_fit,
        eps_star=float(eps_star),
        eps_bound=eps_bound,
        theta_point=theta_point,
        theta_min=theta_min,
        theta_max=theta_max,
        support_thetas=support_arr,
        n_constraints=int(G_fit.shape[0]),
        n_train_obs=int(n_train_obs),
        feasible=bool(feasible),
        status=status,
        theta_lower=theta_lower_arr,
        theta_upper=theta_upper_arr,
    )


def candidate_is_near_optimal_exact(
    pref: LinearPreferencePolytope,
    F: np.ndarray,
    candidate_idx: int,
    eta: float,
    *,
    benchmark_mask: np.ndarray | None = None,
    method: str = "highs",
) -> bool:
    """Exact LP membership test for whether candidate can be near-optimal."""
    F = np.asarray(F, dtype=float)
    M, d = F.shape
    constraints = [pref.G]
    b_list = [np.full(pref.G.shape[0], pref.eps_bound, dtype=float)]
    if benchmark_mask is None:
        F_bench = F
    else:
        mask = np.asarray(benchmark_mask, dtype=bool)
        if mask.shape[0] != M:
            raise ValueError("benchmark_mask must have one entry per candidate")
        F_bench = F[mask]
        if F_bench.shape[0] == 0:
            F_bench = F
    action_G = F_bench - F[candidate_idx][None, :]
    constraints.append(action_G)
    b_list.append(np.full(F_bench.shape[0], eta, dtype=float))
    A_ub = np.vstack(constraints)
    b_ub = np.concatenate(b_list)
    A_eq, b_eq = _simplex_eq(d)
    res = linprog(
        np.zeros(d, dtype=float),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=_theta_bounds(d, pref.theta_lower, pref.theta_upper),
        method=method,
    )
    return bool(res.success)
