"""Stick-breaking coordinates for simplex-valued hyperparameters."""

import numpy as np
import pytest
from ebpfn.priors.simplex import sticks_to_weights, weights_to_sticks


@pytest.mark.parametrize("count", [2, 3, 4, 8])
def test_every_box_point_is_a_valid_mixture(count: int) -> None:
    """The property the direct simplex lacks: no infeasible region, so no rejection."""
    rng = np.random.default_rng(0)
    for sticks in rng.uniform(size=(200, count - 1)):
        weights = sticks_to_weights(sticks)
        assert len(weights) == count
        assert all(weight >= 0.0 for weight in weights)
        assert sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize("count", [2, 3, 4, 8])
def test_coordinates_round_trip(count: int) -> None:
    rng = np.random.default_rng(1)
    for sticks in rng.uniform(size=(100, count - 1)):
        weights = sticks_to_weights(sticks)
        assert weights_to_sticks(weights) == pytest.approx(sticks, abs=1e-9)


def test_vertices_and_interior_points_round_trip() -> None:
    """Boundary points are representable, including exhausted-stick vertices."""
    for weights in [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.25, 0.25, 0.5), (0.25, 0.25, 0.375, 0.125)]:
        assert sticks_to_weights(weights_to_sticks(weights)) == pytest.approx(weights, abs=1e-12)


def test_uniform_box_draws_are_uniform_on_the_simplex() -> None:
    """The Beta(1, n-1-k) exponent is what buys this; plain breaks are front-loaded."""
    count = 4
    rng = np.random.default_rng(2)
    draws = np.array([sticks_to_weights(sticks) for sticks in rng.uniform(size=(20_000, count - 1))])
    assert draws.mean(axis=0) == pytest.approx(np.full(count, 1.0 / count), abs=0.01)
    # Dirichlet(1,...,1) variance: (1/n)(1 - 1/n)/(n + 1)
    expected = (1.0 / count) * (1.0 - 1.0 / count) / (count + 1)
    assert draws.var(axis=0) == pytest.approx(np.full(count, expected), rel=0.1)


def test_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="at least two members"):
        sticks_to_weights([])
    with pytest.raises(ValueError, match="outside"):
        sticks_to_weights([1.5])
    with pytest.raises(ValueError, match="sum to one"):
        weights_to_sticks([0.3, 0.3])
    with pytest.raises(ValueError, match="nonnegative"):
        weights_to_sticks([-0.5, 1.5])
