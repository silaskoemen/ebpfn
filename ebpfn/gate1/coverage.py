"""Prior coverage of real tasks (plans/gate1_revised.md §3.4).

How well does the prior's sample cloud cover a real task? Per task we sample a
d-matched prior cloud and take the k-NN-mean s-OTDD recall (mean of the k nearest
cloud distances) -- the coverage statistic the gate test correlates against
calibration. We also track the X-only sliced distance (the P(X) over-matching
diagnostic, §4) so feature-marginal match is separable from conditional
structure, and a per-d prior self-recall null band as the noise floor.

Reuses Gate-0 `ebpfn/distance.py` (s_otdd, standardization). Task and cloud are
subsampled to a common row count so the sliced distance compares equal-size
clouds; the task's true n is recorded elsewhere as the n confound covariate.
"""
from __future__ import annotations

import numpy as np
from ot.sliced import sliced_wasserstein_distance

from ebpfn.distance import DistFn, s_otdd
from ebpfn.gate1.config import CoverageConfig
from ebpfn.gate1.corpus import RealTask
from ebpfn.gate1.prior import MixturePrior
from ebpfn.priors import Dataset


def _subsample(D: Dataset, m: int, rng: np.random.Generator) -> Dataset:
    if D.n <= m:
        return D
    idx = rng.choice(D.n, size=m, replace=False)
    return Dataset(X=D.X[idx], Y=D.Y[idx])


def _standardize_x(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + eps)


def knn_mean_recall(D: Dataset, cloud: list[Dataset], dist_fn: DistFn, k: int) -> float:
    """Mean of the k smallest distances from D to the cloud (recall-style)."""
    dists = np.array([dist_fn(D, Dc) for Dc in cloud])
    k = min(k, dists.size)
    return float(np.sort(dists)[:k].mean())


def x_only_sliced(
    Da: Dataset, Db: Dataset, n_proj: int, rng: np.random.Generator, p: int = 2, standardize: bool = True
) -> float:
    """Sliced-Wasserstein between the X marginals only (P(X) over-matching diagnostic)."""
    Xa, Xb = (_standardize_x(Da.X), _standardize_x(Db.X)) if standardize else (Da.X, Db.X)
    seed = int(rng.integers(0, 2**31 - 1))
    return float(sliced_wasserstein_distance(Xa, Xb, n_projections=n_proj, p=p, seed=seed))


def _dist_fns(cfg: CoverageConfig, rng: np.random.Generator) -> tuple[DistFn, DistFn]:
    """(joint s-OTDD, X-only) distance functions bound to the config."""
    joint = lambda Da, Db: s_otdd(Da, Db, lam=cfg.lam, n_proj=cfg.n_proj, rng=rng, p=cfg.p, standardize=cfg.standardize)
    x_only = lambda Da, Db: x_only_sliced(Da, Db, cfg.n_proj, rng, p=cfg.p, standardize=cfg.standardize)
    return joint, x_only


def task_coverage(task: Dataset, prior: MixturePrior, d: int, cfg: CoverageConfig, rng: np.random.Generator) -> dict:
    """Joint and X-only k-NN-mean coverage of one task by a d-matched prior cloud."""
    cloud = prior.sample_cloud(cfg.cloud_n_tasks, cfg.cloud_n_rows, d, rng)
    m = min(cfg.cloud_n_rows, task.n)
    probe = _subsample(task, m, rng)
    cloud = [_subsample(Dc, m, rng) for Dc in cloud]
    joint_fn, x_only_fn = _dist_fns(cfg, rng)
    return {
        "coverage": knn_mean_recall(probe, cloud, joint_fn, cfg.k),
        "x_only_coverage": knn_mean_recall(probe, cloud, x_only_fn, cfg.k),
    }


def prior_self_null(prior: MixturePrior, d: int, cfg: CoverageConfig, rng: np.random.Generator) -> dict:
    """Prior self-recall band at dim d: the coverage a prior-typical task gets
    (the noise floor on the coverage axis). Bootstrap (1-alpha) band of the mean."""
    probe = prior.sample_cloud(cfg.cloud_n_tasks, cfg.cloud_n_rows, d, rng)
    ref = prior.sample_cloud(cfg.cloud_n_tasks, cfg.cloud_n_rows, d, rng)
    joint_fn, _ = _dist_fns(cfg, rng)
    recalls = np.array([knn_mean_recall(P, ref, joint_fn, cfg.k) for P in probe])
    boot = np.array([rng.choice(recalls, size=recalls.size, replace=True).mean() for _ in range(cfg.n_boot)])
    lo, hi = np.percentile(boot, [100 * cfg.null_alpha / 2, 100 * (1 - cfg.null_alpha / 2)])
    return {"d": d, "mean": float(recalls.mean()), "band_lo": float(lo), "band_hi": float(hi)}


def corpus_coverage(corpus: list[RealTask], prior: MixturePrior, cfg: CoverageConfig, rng: np.random.Generator) -> list[dict]:
    """Per-task coverage row for the whole corpus (joint + X-only + schema)."""
    rows = []
    for t in corpus:
        cov = task_coverage(t.data, prior, t.d, cfg, rng)
        rows.append({
            "source_did": t.source_did, "target": t.target,
            "n": t.n, "d": t.d, "learnability_r2": t.learnability_r2,
            **cov,
        })
    return rows


def corpus_null(corpus: list[RealTask], prior: MixturePrior, cfg: CoverageConfig, rng: np.random.Generator) -> dict[int, dict]:
    """Prior self-recall null band per unique d present in the corpus."""
    return {d: prior_self_null(prior, d, cfg, rng) for d in sorted({t.d for t in corpus})}
