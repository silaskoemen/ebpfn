"""Coverage statistics: k-NN-mean recall mechanics, the X-only diagnostic, the
self-recall null band, and the semantic check that an off-distribution task
scores *higher* coverage distance than a prior-typical one. §3.4."""

from __future__ import annotations

import numpy as np
import pytest
from ebpfn.gate1 import CoverageConfig
from ebpfn.gate1 import PriorConfig
from ebpfn.gate1 import build_prior
from ebpfn.gate1 import knn_mean_recall
from ebpfn.gate1 import prior_self_null
from ebpfn.gate1 import task_coverage
from ebpfn.gate1 import x_only_sliced
from ebpfn.priors import Dataset


def _bnn_prior():
    return build_prior(PriorConfig(scm_linear_weight=0.0, scm_mlp_weight=0.0, bnn_weight=1.0))


def _fast_cfg(**kw):
    base = dict(n_proj=100, cloud_n_tasks=15, cloud_n_rows=200, k=5, n_boot=200)
    base.update(kw)
    return CoverageConfig(**base)


def test_knn_mean_recall_picks_k_smallest():
    # encode each cloud member's distance in its Y[0]; dist_fn reads it back.
    dists = [3.0, 1.0, 2.0, 5.0, 4.0]
    cloud = [Dataset(X=np.zeros((2, 1)), Y=np.array([v, 0.0])) for v in dists]
    dist_fn = lambda a, b: float(b.Y[0])
    probe = Dataset(X=np.zeros((2, 1)), Y=np.zeros(2))
    assert knn_mean_recall(probe, cloud, dist_fn, k=2) == pytest.approx(1.5)  # mean(1,2)
    assert knn_mean_recall(probe, cloud, dist_fn, k=1) == pytest.approx(1.0)


def test_x_only_sliced_zero_for_identical_positive_for_different():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 3))
    same = Dataset(X=X, Y=np.zeros(300))
    assert x_only_sliced(same, same, n_proj=100, rng=rng) == pytest.approx(0.0, abs=1e-6)
    uniform = Dataset(X=rng.uniform(-2, 2, (300, 3)), Y=np.zeros(300))
    assert x_only_sliced(same, uniform, n_proj=100, rng=rng) > 1e-3


def test_self_null_band_well_formed():
    prior, cfg = _bnn_prior(), _fast_cfg()
    band = prior_self_null(prior, d=4, cfg=cfg, rng=np.random.default_rng(0))
    assert band["band_lo"] <= band["mean"] <= band["band_hi"]
    assert np.isfinite([band["band_lo"], band["mean"], band["band_hi"]]).all()


def test_off_distribution_task_has_higher_coverage_distance():
    # NB: s-OTDD standardizes per task, so a "far" task must differ in *shape*,
    # not just mean/variance -- heavy tails (Student-t) are robustly detected;
    # uniform/bimodal at matched variance are nearly invisible to it.
    prior, cfg = _bnn_prior(), _fast_cfg()
    rng = np.random.default_rng(0)
    in_task = prior.sample_task(400, 4, rng)  # prior-typical (Gaussian features)
    far = Dataset(X=rng.standard_t(2, (400, 4)), Y=rng.standard_t(2, 400))  # heavy-tailed, off-distribution
    cov_in = task_coverage(in_task, prior, 4, cfg, np.random.default_rng(1))["coverage"]
    cov_far = task_coverage(far, prior, 4, cfg, np.random.default_rng(1))["coverage"]
    assert cov_far > cov_in


def test_task_coverage_deterministic():
    prior, cfg = _bnn_prior(), _fast_cfg()
    task = prior.sample_task(300, 3, np.random.default_rng(7))
    a = task_coverage(task, prior, 3, cfg, np.random.default_rng(2))
    b = task_coverage(task, prior, 3, cfg, np.random.default_rng(2))
    assert a == b


def test_config_validation():
    with pytest.raises(ValueError):
        CoverageConfig(lam=3.0)  # not in lam_grid
    with pytest.raises(ValueError):
        CoverageConfig(k=20, cloud_n_tasks=10)  # cloud must have >= k tasks
