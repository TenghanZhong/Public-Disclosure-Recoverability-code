from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def smart_read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False, **kwargs)
    if suffix in {".feather"}:
        return pd.read_feather(path, **kwargs)
    raise ValueError(f"Unsupported file type: {path}")


def smart_write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            df.to_parquet(path, index=index)
        except Exception:
            df.to_csv(path.with_suffix(".csv"), index=index)
    elif suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        df.to_csv(path, index=index)
    else:
        df.to_csv(path.with_suffix(".csv"), index=index)


def present_cols(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    cols = set(df.columns)
    return [c for c in candidates if c in cols]


def coerce_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def percent_to_decimal(s: pd.Series, mode: str | bool = "auto") -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if mode == "auto":
        vals = np.abs(x.to_numpy(dtype=float))
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return x
        # Sparse percentage fields require upper-tail scale evidence.
        med = float(np.nanmedian(vals))
        p95 = float(np.nanpercentile(vals, 95))
        mx = float(np.nanmax(vals))
        if med > 1.5 or p95 > 1.5 or mx > 2.0:
            return x / 100.0
        return x
    if mode is True or str(mode).lower() in {"true", "1", "yes", "0_100"}:
        return x / 100.0
    return x


def safe_divide(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray, default: float = 0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(num, dtype=float) / np.asarray(den, dtype=float)
    out[~np.isfinite(out)] = default
    return out


def robust_zscore(s: pd.Series, clip: float = 8.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad < 1e-12:
        std = x.std(skipna=True)
        if not np.isfinite(std) or std < 1e-12:
            return pd.Series(np.zeros(len(x)), index=x.index)
        z = (x - x.mean(skipna=True)) / std
    else:
        z = 0.6745 * (x - med) / mad
    return z.clip(-clip, clip).fillna(0.0)



class SimpleStandardScaler:
    """Small sklearn-free standard scaler used for reproducible experiments.

    Implements the fit/transform API used by the experiment without importing
    sklearn.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("SimpleStandardScaler expects a 2D array")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("SimpleStandardScaler must be fitted before transform")
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return (X - self.mean_) / self.scale_

def winsorize_array(x: np.ndarray, low_q: float = 0.005, high_q: float = 0.995) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(y)
    if finite.sum() == 0:
        return np.zeros_like(y)
    lo = np.nanquantile(y[finite], low_q)
    hi = np.nanquantile(y[finite], high_q)
    y = np.clip(y, lo, hi)
    y[~np.isfinite(y)] = 0.0
    return y


def l1_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(a - b), axis=-1)


def bounded_simplex_project(v: np.ndarray, lower: np.ndarray, upper: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
    """Fast approximate projection onto {x: lower <= x <= upper, sum x = target_sum}.

    Clips to bounds and then redistributes surplus or deficit across coordinates
    with remaining capacity. Candidate generation calls this routine repeatedly;
    it does not solve an exact quadratic projection problem.
    """
    v = np.asarray(v, dtype=float)
    lower = np.nan_to_num(np.asarray(lower, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.nan_to_num(np.asarray(upper, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    n = len(v)
    if n == 0:
        return v
    lower = np.clip(lower, 0.0, 1.0)
    upper = np.clip(np.maximum(upper, lower + 1e-9), 0.0, 1.0)
    if lower.sum() > target_sum:
        lower = lower / max(lower.sum(), 1e-12) * target_sum
        upper = np.maximum(upper, lower + 1e-9)
    if upper.sum() < target_sum:
        upper = np.ones(n)
        lower = np.zeros(n)
    x = np.nan_to_num(v, nan=target_sum / n, posinf=target_sum / n, neginf=target_sum / n)
    x = np.clip(x, lower, upper)
    for _ in range(8):
        total = x.sum()
        diff = target_sum - total
        if abs(diff) <= 1e-10:
            break
        if diff > 0:
            cap = upper - x
            cap_sum = cap[cap > 1e-12].sum()
            if cap_sum <= 1e-12:
                break
            add = np.minimum(cap, diff * cap / cap_sum)
            x += add
        else:
            room = x - lower
            room_sum = room[room > 1e-12].sum()
            if room_sum <= 1e-12:
                break
            sub = np.minimum(room, (-diff) * room / room_sum)
            x -= sub
    total = x.sum()
    if total <= 0 or not np.isfinite(total):
        x = np.ones(n) / n * target_sum
    elif abs(total - target_sum) > 1e-6:
        # Normalize and clip the fallback candidate.
        x = np.clip(x / total * target_sum, lower, upper)
    return x


def dataframe_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def chunk_indices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)
