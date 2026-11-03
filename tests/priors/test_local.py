"""Local prior: draws stay on the fitted rows' support and carry their dependence."""

import numpy as np
import pytest
from ebpfn.priors.local import LocalPrior


@pytest.fixture(scope="module")
def prior() -> LocalPrior:
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200)
    x = np.column_stack([z, z + 0.3 * rng.standard_normal(200), rng.integers(0, 3, 200), rng.exponential(size=200)])
    return LocalPrior.fit(x, np.sin(2 * x[:, 0]) + 0.1 * rng.standard_normal(200), seed=0)


def test_copula_keeps_marginal_support_and_dependence(prior):
    x = prior.sample_x(2000, np.random.default_rng(1))
    assert set(np.unique(x[:, 2])) <= set(np.unique(prior.x[:, 2]))
    assert x[:, 3].min() >= prior.x[:, 3].min()
    assert x[:, 3].max() <= prior.x[:, 3].max()
    assert np.corrcoef(x[:, 0], x[:, 1])[0, 1] > 0.8


@pytest.mark.parametrize("surrogate", ["gp", "trees"])
def test_draws_track_the_fitted_function_and_vary(prior, surrogate):
    rng = np.random.default_rng(2)
    x, y = prior.sample_task(300, rng, surrogate=surrogate)
    assert x.shape == (300, 4)
    assert np.isfinite(y).all()
    assert np.corrcoef(y, np.sin(2 * x[:, 0]))[0, 1] > 0.7
    # Each task is its own DGP: a second draw at the same x is a different function.
    again = prior.sample_gp(x, rng) if surrogate == "gp" else prior.sample_trees(x, rng)
    assert 0.0 < np.std(y - again) < np.std(y)


def test_rejects_an_unknown_surrogate(prior):
    with pytest.raises(ValueError, match="unknown surrogate"):
        prior.sample_task(10, np.random.default_rng(0), surrogate="knn")
