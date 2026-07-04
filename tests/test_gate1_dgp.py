"""Gate-1 prior substrate: shapes, determinism, finiteness, and -- the one that
matters -- that each DGP produces a target that is actually learnable from X
(not pure noise). plans/gate1_revised.md §3.1/§6."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from ebpfn.gate1 import BnnDgp, MixturePrior, PriorConfig, ScmDgp, build_prior

ALL_DGPS = [
    ScmDgp(activation="linear"),
    ScmDgp(activation="tanh"),
    ScmDgp(activation="relu"),
    BnnDgp(),
]


@pytest.mark.parametrize("dgp", ALL_DGPS, ids=lambda d: d.name + "_" + getattr(d, "activation", ""))
@pytest.mark.parametrize("d", [1, 3, 8])
def test_shape_and_finite(dgp, d):
    n = 256
    D = dgp.sample(n, d, np.random.default_rng(0))
    assert D.X.shape == (n, d)
    assert D.Y.shape == (n,)
    assert np.isfinite(D.X).all() and np.isfinite(D.Y).all()


@pytest.mark.parametrize("dgp", ALL_DGPS, ids=lambda d: d.name)
def test_determinism(dgp):
    a = dgp.sample(200, 4, np.random.default_rng(7))
    b = dgp.sample(200, 4, np.random.default_rng(7))
    assert np.array_equal(a.X, b.X) and np.array_equal(a.Y, b.Y)
    c = dgp.sample(200, 4, np.random.default_rng(8))
    assert not np.array_equal(a.Y, c.Y)


@pytest.mark.parametrize("dgp", ALL_DGPS, ids=lambda d: d.name + "_" + getattr(d, "activation", ""))
def test_target_is_learnable(dgp):
    """A GBM on a train split must beat the marginal (R^2 > 0) on held-out --
    confirms the DGP makes Y depend on X (the corpus learnability filter's bar)."""
    D = dgp.sample(3000, 5, np.random.default_rng(1))
    ntr = 2000
    model = HistGradientBoostingRegressor(max_iter=150, random_state=0)
    model.fit(D.X[:ntr], D.Y[:ntr])
    r2 = r2_score(D.Y[ntr:], model.predict(D.X[ntr:]))
    assert r2 > 0.1, f"{dgp.name} target barely learnable: R^2={r2:.3f}"


def test_scm_validation():
    with pytest.raises(ValueError):
        ScmDgp(n_hidden=0)
    with pytest.raises(ValueError):
        ScmDgp(activation="sigmoid")
    with pytest.raises(ValueError):
        ScmDgp(edge_prob=1.5)


def test_mixture_uniform_default_and_cloud():
    prior = MixturePrior(dgps=tuple(ALL_DGPS))
    np.testing.assert_allclose(prior._probs(), np.full(len(ALL_DGPS), 1 / len(ALL_DGPS)))
    cloud = prior.sample_cloud(n_tasks=6, n=128, d=3, rng=np.random.default_rng(0))
    assert len(cloud) == 6
    assert all(Dc.X.shape == (128, 3) for Dc in cloud)


def test_mixture_weight_validation():
    with pytest.raises(ValueError):
        MixturePrior(dgps=())
    with pytest.raises(ValueError):
        MixturePrior(dgps=tuple(ALL_DGPS), weights=(1.0, 1.0))  # wrong length
    with pytest.raises(ValueError):
        MixturePrior(dgps=(ScmDgp(),), weights=(0.0,))  # non-positive sum


def test_build_prior_drops_zero_weight_members():
    prior = build_prior(PriorConfig(scm_linear_weight=0.0, scm_mlp_weight=1.0, bnn_weight=2.0))
    assert len(prior.dgps) == 2
    names = {dgp.name for dgp in prior.dgps}
    assert names == {"scm_tanh", "bnn"}
    np.testing.assert_allclose(prior._probs(), np.array([1.0, 2.0]) / 3.0)


def test_build_prior_determinism_through_mixture():
    prior = build_prior(PriorConfig())
    a = prior.sample_task(300, 4, np.random.default_rng(3))
    b = prior.sample_task(300, 4, np.random.default_rng(3))
    assert np.array_equal(a.Y, b.Y)
