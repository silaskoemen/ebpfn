"""Conditional-coverage meter (spec §3.2).

Partition X label-agnostically (X only -- never on Y or on the real/decoy label),
then per cell compute MMD^2 between the 1-D Y | cell of real vs decoy with a
characteristic RBF kernel (median-heuristic bandwidth). Aggregate two ways:
mass-weighted mean and max over cells.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from ebpfn.priors import Dataset


class CellPartition:
    """A label-agnostic K-cell partition of X, fit on pooled covariates."""

    def __init__(self, n_cells: int, method: str = "kmeans", rng: np.random.Generator | None = None):
        self.n_cells = n_cells
        self.method = method
        self._seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else 0
        self._km: KMeans | None = None

    def fit(self, X: np.ndarray) -> "CellPartition":
        if self.method != "kmeans":
            raise NotImplementedError(f"partition method {self.method!r} not implemented")
        self._km = KMeans(n_clusters=self.n_cells, n_init=10, random_state=self._seed).fit(X)
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        if self._km is None:
            raise RuntimeError("CellPartition.fit must be called before assign")
        return self._km.predict(X)


def _median_bandwidth(y: np.ndarray, eps: float = 1e-8) -> float:
    """Median heuristic on 1-D Y: median of pairwise |yi - yj|."""
    if y.size > 1000:  # cap the O(m^2) heuristic
        y = np.random.default_rng(0).choice(y, size=1000, replace=False)
    diffs = np.abs(y[:, None] - y[None, :])
    med = np.median(diffs[np.triu_indices(y.size, k=1)])
    return float(max(med, eps))


def _rbf_mmd2(y1: np.ndarray, y2: np.ndarray, bandwidth: float) -> float:
    """Unbiased MMD^2 estimate with an RBF kernel on 1-D samples."""
    gamma = 1.0 / (2.0 * bandwidth**2)

    def k(a, b):
        return np.exp(-gamma * (a[:, None] - b[None, :]) ** 2)

    m, n = y1.size, y2.size
    kxx = k(y1, y1)
    kyy = k(y2, y2)
    kxy = k(y1, y2)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    term_x = kxx.sum() / (m * (m - 1))
    term_y = kyy.sum() / (n * (n - 1))
    term_xy = kxy.mean()
    return float(term_x + term_y - 2 * term_xy)


def per_cell_mmd(
    Dr: Dataset,
    Dd: Dataset,
    cells: CellPartition,
    bandwidth: str = "median",
    min_per_cell: int = 20,
    max_per_cell: int = 500,
    rng: np.random.Generator | None = None,
) -> dict[int, dict]:
    """cell_id -> {mmd2, bandwidth, n_real, n_decoy, mass}.

    Cells with fewer than `min_per_cell` points on either side are skipped.
    Bandwidth uses pooled Y within the cell (still label-agnostic on X).
    """
    if bandwidth != "median":
        raise NotImplementedError(f"bandwidth rule {bandwidth!r} not implemented")
    rng = rng or np.random.default_rng(0)
    lr = cells.assign(Dr.X)
    ld = cells.assign(Dd.X)
    total = Dr.n + Dd.n
    out: dict[int, dict] = {}
    for c in range(cells.n_cells):
        yr = Dr.Y[lr == c]
        yd = Dd.Y[ld == c]
        if yr.size < min_per_cell or yd.size < min_per_cell:
            continue
        if yr.size > max_per_cell:
            yr = rng.choice(yr, size=max_per_cell, replace=False)
        if yd.size > max_per_cell:
            yd = rng.choice(yd, size=max_per_cell, replace=False)
        bw = _median_bandwidth(np.concatenate([yr, yd]))
        out[c] = {
            "mmd2": _rbf_mmd2(yr, yd, bw),
            "bandwidth": bw,
            "n_real": int(yr.size),
            "n_decoy": int(yd.size),
            "mass": (yr.size + yd.size) / total,
        }
    return out


def aggregate(cell_mmd: dict[int, dict]) -> dict:
    """Mass-weighted mean and max over cells (spec §3.2)."""
    if not cell_mmd:
        return {"mean": float("nan"), "max": float("nan"), "n_cells": 0}
    vals = np.array([c["mmd2"] for c in cell_mmd.values()])
    mass = np.array([c["mass"] for c in cell_mmd.values()])
    mass = mass / mass.sum()
    return {"mean": float((vals * mass).sum()), "max": float(vals.max()), "n_cells": len(cell_mmd)}
