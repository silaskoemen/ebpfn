"""Joint distance + null band (spec §3.1).

For regression, OTDD reduces to p-Wasserstein on the joint (X, Y) with ground
cost d((x,y),(x',y'))^p = ||x-x'||^p + lambda*|y-y'|^p. With p=2 that is squared
Euclidean distance on the augmented vector z = [x, sqrt(lambda)*y], so sliced-
Wasserstein on z gives the headline s-OTDD; exact OT (POT emd2) is the d=2 check.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import ot
from ot.sliced import sliced_wasserstein_distance

from ebpfn.priors import Dataset

DistFn = Callable[[Dataset, Dataset], float]


def standardize_per_task(D: Dataset, eps: float = 1e-8) -> Dataset:
    """Per-task standardization so lambda=1 is principled (spec §3.1)."""
    X = (D.X - D.X.mean(axis=0)) / (D.X.std(axis=0) + eps)
    Y = (D.Y - D.Y.mean()) / (D.Y.std() + eps)
    return Dataset(X=X, Y=Y)


def _augment(D: Dataset, lam: float) -> np.ndarray:
    return np.hstack([D.X, np.sqrt(lam) * D.Y[:, None]])


def s_otdd(
    Da: Dataset,
    Db: Dataset,
    lam: float,
    n_proj: int,
    rng: np.random.Generator,
    p: int = 2,
    standardize: bool = True,
) -> float:
    """Sliced-Wasserstein s-OTDD on the joint (X, Y). Headline distance."""
    if standardize:
        Da, Db = standardize_per_task(Da), standardize_per_task(Db)
    Za, Zb = _augment(Da, lam), _augment(Db, lam)
    seed = int(rng.integers(0, 2**31 - 1))
    return float(sliced_wasserstein_distance(Za, Zb, n_projections=n_proj, p=p, seed=seed))


def exact_otdd(Da: Dataset, Db: Dataset, lam: float, p: int = 2, standardize: bool = True) -> float:
    """Exact p-Wasserstein via POT emd2 (d=2, small n reference)."""
    if standardize:
        Da, Db = standardize_per_task(Da), standardize_per_task(Db)
    Za, Zb = _augment(Da, lam), _augment(Db, lam)
    M = ot.dist(Za, Zb, metric="euclidean") ** p
    a = np.full(Za.shape[0], 1.0 / Za.shape[0])
    b = np.full(Zb.shape[0], 1.0 / Zb.shape[0])
    return float(ot.emd2(a, b, M) ** (1.0 / p))


def make_sotdd_fn(lam: float, n_proj: int, rng: np.random.Generator, p: int = 2) -> DistFn:
    """Bind s_otdd into a (Da, Db) -> float for recall/null-band use."""
    return lambda Da, Db: s_otdd(Da, Db, lam=lam, n_proj=n_proj, rng=rng, p=p)


def recall_to_cloud(D: Dataset, cloud: list[Dataset], dist_fn: DistFn) -> float:
    """min over a cloud: how well `cloud` recalls `D` (spec §3.1)."""
    return min(dist_fn(D, Dc) for Dc in cloud)


def cloud_recall(probe: list[Dataset], cloud: list[Dataset], dist_fn: DistFn) -> np.ndarray:
    """recall_to_cloud for each probe task -> array of per-task recalls."""
    return np.array([recall_to_cloud(D, cloud, dist_fn) for D in probe])


def null_band(
    real_probe: list[Dataset],
    real_ref: list[Dataset],
    dist_fn: DistFn,
    rng: np.random.Generator,
    alpha: float = 0.05,
    n_boot: int = 1000,
) -> dict:
    """Real-vs-real' null band (spec §3.1).

    Each held-out real probe task is recalled against a held-out real reference
    cloud; the null band is the bootstrap (1-alpha) interval of the mean recall.
    'OTDD says covered' == decoy mean recall inside this band.
    """
    recalls = cloud_recall(real_probe, real_ref, dist_fn)
    boot_means = np.array([rng.choice(recalls, size=recalls.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "recalls": recalls,
        "mean": float(recalls.mean()),
        "band_lo": float(lo),
        "band_hi": float(hi),
        "alpha": alpha,
    }


def inside_band(value: float, band: dict) -> bool:
    return band["band_lo"] <= value <= band["band_hi"]
