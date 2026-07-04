"""Descriptor-space coverage of real tasks (plans/gate2.md §2).

Coverage is the Mahalanobis distance of a real task's conditional-structure
descriptor to the prior's descriptor cloud, with Ledoit-Wolf shrinkage (the
descriptor dimension is comparable to the cloud size, so a raw covariance is
ill-conditioned). The cloud is d-matched to each real task. The within-prior
independent prior-probe distance distribution is the null band: a real task is
"outside" when its distance exceeds the null's `outside_quantile`.

The critical Gate-2 design choice (the documented pushback): we test the variance
of the *coverage quantity*, not of the raw descriptor. A broad prior can spread
over the descriptor space and re-flatten the coverage null exactly as the joint
s-OTDD distance did in Gate-1 -- so `variance_check` runs BEFORE any calibration
number is touched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf

from ebpfn.gate1.prior import MixturePrior
from ebpfn.gate2.config import DescriptorConfig, Gate2Config, Gate2CoverageConfig
from ebpfn.gate2.descriptor import conditional_descriptor
from ebpfn.priors import Dataset


@dataclass
class DescriptorCloud:
    """A prior's descriptor cloud at one d: mean, shrunk precision, null distances."""

    d: int
    mean: np.ndarray
    precision: np.ndarray
    self_dists: np.ndarray  # independent prior-probe Mahalanobis distances (the null band)

    def distance(self, desc: np.ndarray) -> float:
        delta = desc - self.mean
        return float(np.sqrt(max(0.0, delta @ self.precision @ delta)))

    def null_quantile(self, q: float) -> float:
        return float(np.quantile(self.self_dists, q))


def build_cloud(prior: MixturePrior, d: int, desc_cfg: DescriptorConfig,
                cov_cfg: Gate2CoverageConfig, rng: np.random.Generator) -> DescriptorCloud:
    """Sample a d-matched prior cloud and fit its descriptor mean/precision.

    The null band is measured on an independent prior probe cloud. Using the same
    descriptors both to fit the covariance and to define the null makes the null
    too optimistic, especially after adding higher-dimensional spectral features.
    """
    tasks = prior.sample_cloud(cov_cfg.cloud_n_tasks, cov_cfg.cloud_n_rows, d, rng)
    descs = np.array([conditional_descriptor(t, desc_cfg, rng) for t in tasks])
    lw = LedoitWolf().fit(descs)
    mean = descs.mean(axis=0)
    prec = np.linalg.pinv(lw.covariance_)
    probe = prior.sample_cloud(cov_cfg.cloud_n_tasks, cov_cfg.cloud_n_rows, d, rng)
    probe_descs = np.array([conditional_descriptor(t, desc_cfg, rng) for t in probe])
    self_dists = np.array([np.sqrt(max(0.0, (row - mean) @ prec @ (row - mean))) for row in probe_descs])
    return DescriptorCloud(d=d, mean=mean, precision=prec, self_dists=self_dists)


def corpus_coverage(corpus, prior: MixturePrior, desc_cfg: DescriptorConfig,
                    cov_cfg: Gate2CoverageConfig, rng: np.random.Generator) -> list[dict]:
    """Per-task descriptor coverage of the whole corpus (clouds cached per d)."""
    clouds: dict[int, DescriptorCloud] = {}
    rows = []
    for t in corpus:
        if t.d not in clouds:
            clouds[t.d] = build_cloud(prior, t.d, desc_cfg, cov_cfg, rng)
        cloud = clouds[t.d]
        desc = conditional_descriptor(t.data, desc_cfg, rng)
        dist = cloud.distance(desc)
        thr = cloud.null_quantile(cov_cfg.outside_quantile)
        rows.append({
            "source_did": t.source_did, "target": t.target, "n": t.n, "d": t.d,
            "coverage": dist,  # Mahalanobis distance in descriptor space (higher = worse covered)
            "null_thr": thr,
            "null_median": float(np.median(cloud.self_dists)),
            "outside": bool(dist > thr),
        })
    return rows


def variance_check(coverage_rows: list[dict], cfg: Gate2Config) -> dict:
    """Part A go/no-go: does descriptor coverage discriminate real tasks at all?

    PASS requires both (a) enough real tasks fall outside the prior's self-null
    band, and (b) the real coverage distances sit materially above the null
    median. A FAIL means coverage-gating is dead regardless of calibration -- the
    Gate-1 failure mode, caught here before looking at any calibration number.
    """
    dist = np.array([r["coverage"] for r in coverage_rows])
    frac_outside = float(np.mean([r["outside"] for r in coverage_rows]))
    null_median = float(np.median([r["null_median"] for r in coverage_rows]))
    median_ratio = float(np.median(dist) / (null_median + 1e-12))
    passes = bool(frac_outside >= cfg.min_frac_outside and median_ratio >= cfg.min_median_ratio)
    return {
        "n_tasks": len(coverage_rows),
        "frac_outside": frac_outside,
        "median_ratio": median_ratio,
        "real_dist_median": float(np.median(dist)),
        "real_dist_iqr": float(np.subtract(*np.percentile(dist, [75, 25]))),
        "null_median": null_median,
        "min_frac_outside": cfg.min_frac_outside,
        "min_median_ratio": cfg.min_median_ratio,
        "passes": passes,
    }
