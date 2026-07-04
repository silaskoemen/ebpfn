"""Gate-2 descriptor + coverage: the properties the gate's validity rests on.

The descriptor must be (1) affine-invariant (the standardization reversal), (2)
deterministic given a seed, and (3) actually *varying* across structurally
different tasks -- the no-variance failure that killed Gate-1 must not recur. The
coverage quantity must put a structurally-foreign task further from the prior
cloud than a prior-typical one, and `variance_check` must FAIL on in-distribution
tasks and PASS on out-of-distribution ones.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ebpfn.gate1 import PriorConfig, build_prior
from ebpfn.gate2 import (
    DescriptorConfig,
    Gate2Config,
    Gate2CoverageConfig,
    build_cloud,
    conditional_descriptor,
    corpus_coverage,
    descriptor_names,
    variance_check,
)
from ebpfn.priors import Dataset


def _linear_task(n, d, rng, noise=0.2):
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = X @ w + noise * rng.standard_normal(n)
    return Dataset(X=X, Y=y)


def _nonlinear_hetero_task(n, d, rng):
    X = rng.standard_normal((n, d))
    mu = np.sin(2.0 * X[:, 0]) + (X[:, min(1, d - 1)] ** 2)
    scale = 0.2 + 0.6 * (X[:, 0] - X[:, 0].min())  # variance increases monotonically along X[:,0]
    y = mu + scale * rng.standard_normal(n)
    return Dataset(X=X, Y=y)


def test_descriptor_length_matches_names():
    cfg = DescriptorConfig(n_proj=16)
    rng = np.random.default_rng(0)
    desc = conditional_descriptor(_linear_task(120, 4, rng), cfg, np.random.default_rng(1))
    assert desc.shape == (len(descriptor_names(cfg)),)
    assert np.isfinite(desc).all()
    names = descriptor_names(cfg)
    for name in ("y_raw_skew", "y_raw_excess_kurt", "y_raw_mass_abs_gt3"):
        assert name in names
    for block in ("raw", "poly", "rbf_local", "rbf_global"):
        assert f"{block}_cca_mode_1" in names
        assert f"{block}_cca_cka" in names
        assert f"{block}_y_moment_1_r2" in names
        assert f"{block}_y_moment_{cfg.y_moments}_r2" in names


def test_descriptor_deterministic():
    cfg = DescriptorConfig(n_proj=16)
    task = _nonlinear_hetero_task(150, 5, np.random.default_rng(3))
    a = conditional_descriptor(task, cfg, np.random.default_rng(7))
    b = conditional_descriptor(task, cfg, np.random.default_rng(7))
    np.testing.assert_allclose(a, b)


def test_descriptor_affine_invariant():
    """Per-feature affine rescaling of X and an affine map of Y must not move it."""
    cfg = DescriptorConfig(n_proj=24)
    task = _nonlinear_hetero_task(200, 5, np.random.default_rng(4))
    base = conditional_descriptor(task, cfg, np.random.default_rng(11))

    rng = np.random.default_rng(5)
    scale = np.exp(rng.uniform(-1, 1, task.d))
    shift = rng.uniform(-3, 3, task.d)
    tX = Dataset(X=task.X * scale + shift, Y=3.5 * task.Y - 7.0)
    transformed = conditional_descriptor(tX, cfg, np.random.default_rng(11))
    np.testing.assert_allclose(base, transformed, atol=1e-6)


def test_descriptor_varies_across_structure():
    """The Gate-1 killer was zero cross-task variance; here the descriptor must
    separate a linear-homoskedastic task from a nonlinear-heteroskedastic one."""
    cfg = DescriptorConfig(n_proj=48)
    names = descriptor_names(cfg)
    rng = np.random.default_rng(0)
    lin = conditional_descriptor(_linear_task(256, 5, rng), cfg, np.random.default_rng(1))
    nl = conditional_descriptor(_nonlinear_hetero_task(256, 5, rng), cfg, np.random.default_rng(1))
    idx = {nm: i for i, nm in enumerate(names)}
    assert nl[idx["nonlinearity_mean"]] > lin[idx["nonlinearity_mean"]] + 0.02
    # single-coordinate heteroskedasticity dilutes under random projections, so it
    # surfaces in the upper-quantile aggregation, not the mean (a real property).
    assert nl[idx["hetero_q90"]] > lin[idx["hetero_q90"]] + 0.02
    assert np.linalg.norm(nl - lin) > 0.1


def test_raw_y_tail_diagnostics_preserve_marginal_tail_shape():
    cfg = DescriptorConfig(n_proj=16)
    names = descriptor_names(cfg)
    idx = {nm: i for i, nm in enumerate(names)}
    rng = np.random.default_rng(13)
    X = rng.standard_normal((400, 4))
    normal = Dataset(X=X, Y=rng.standard_normal(400))
    heavy = Dataset(X=X, Y=rng.standard_t(df=2.5, size=400))

    d_normal = conditional_descriptor(normal, cfg, np.random.default_rng(2))
    d_heavy = conditional_descriptor(heavy, cfg, np.random.default_rng(2))
    assert d_heavy[idx["y_raw_excess_kurt"]] > d_normal[idx["y_raw_excess_kurt"]] + 1.0
    assert d_heavy[idx["y_raw_mass_abs_gt3"]] > d_normal[idx["y_raw_mass_abs_gt3"]]


def _corpus(tasks):
    return [SimpleNamespace(data=t, d=t.d, n=t.n, source_did=i, target="y") for i, t in enumerate(tasks)]


def test_coverage_foreign_task_is_farther():
    prior = build_prior(PriorConfig(scm_linear_weight=1.0, scm_mlp_weight=0.0, bnn_weight=0.0,
                                    scm_noise_scale=0.15))
    desc_cfg = DescriptorConfig(n_proj=24)
    cov_cfg = Gate2CoverageConfig(cloud_n_tasks=24, cloud_n_rows=200)
    rng = np.random.default_rng(0)
    cloud = build_cloud(prior, 5, desc_cfg, cov_cfg, rng)
    # A single (in, foreign) pair is a coin-flip on two noisy draws (either can land
    # in the prior's own distance tail); the property under test is distributional --
    # foreign tasks sit farther from the cloud than the prior's own samples do.
    d_in = [cloud.distance(conditional_descriptor(prior.sample_task(200, 5, rng), desc_cfg, rng))
            for _ in range(8)]
    d_foreign = [cloud.distance(conditional_descriptor(_nonlinear_hetero_task(200, 5, rng), desc_cfg, rng))
                 for _ in range(8)]
    assert np.median(d_foreign) > np.median(d_in)


def test_variance_check_fails_in_distribution_passes_ood():
    prior = build_prior(PriorConfig(scm_linear_weight=1.0, scm_mlp_weight=0.0, bnn_weight=0.0,
                                    scm_noise_scale=0.15))
    desc_cfg = DescriptorConfig(n_proj=24)
    cov_cfg = Gate2CoverageConfig(cloud_n_tasks=24, cloud_n_rows=200)
    gcfg = Gate2Config()
    rng = np.random.default_rng(1)

    in_corpus = _corpus([prior.sample_task(200, 5, rng) for _ in range(20)])
    rows_in = corpus_coverage(in_corpus, prior, desc_cfg, cov_cfg, rng)
    assert not variance_check(rows_in, gcfg)["passes"]  # prior covers its own samples

    ood_corpus = _corpus([_nonlinear_hetero_task(200, 5, rng) for _ in range(20)])
    rows_ood = corpus_coverage(ood_corpus, prior, desc_cfg, cov_cfg, rng)
    assert variance_check(rows_ood, gcfg)["passes"]  # foreign structure is detected
