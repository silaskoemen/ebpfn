"""Gaussian scoring for the baseline comparators.

NLL and coverage were already exercised through the diagnostics; CRPS is new and is a
decision metric, so its closed form is pinned against both its analytic value and a
Monte-Carlo estimate of the definition it is supposed to implement.
"""

import math

import numpy as np
import pytest
from benchmarks.baselines import gaussian_coverage, gaussian_crps


def test_crps_at_the_mean_matches_the_analytic_value() -> None:
    expected = 2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)
    assert gaussian_crps(np.zeros(1), np.zeros(1), np.ones(1)) == pytest.approx(expected)


def test_crps_scales_with_the_forecast_spread() -> None:
    """CRPS has the units of y, so scaling the forecast and the error scales it too."""
    unit = gaussian_crps(np.array([1.3]), np.zeros(1), np.ones(1))
    scaled = gaussian_crps(np.array([2.6]), np.zeros(1), np.full(1, 2.0))
    assert scaled == pytest.approx(2.0 * unit)


def test_crps_matches_its_monte_carlo_definition() -> None:
    """CRPS = E|X - y| - E|X - X'|/2 for independent draws from the forecast."""
    rng = np.random.default_rng(0)
    y, mean, std = 1.3, 0.4, 0.8
    x, x_prime = (mean + std * rng.standard_normal(2_000_000) for _ in range(2))
    monte_carlo = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x - x_prime))
    assert gaussian_crps(np.array([y]), np.array([mean]), np.array([std])) == pytest.approx(monte_carlo, abs=2e-3)


def test_a_calibrated_gaussian_covers_at_its_nominal_level() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200_000)
    for level in (0.5, 0.8, 0.9, 0.95):
        empirical = gaussian_coverage(y, np.zeros_like(y), np.ones_like(y), level)
        assert empirical == pytest.approx(level, abs=0.005)


def test_coverage_rejects_a_level_outside_the_unit_interval() -> None:
    y = np.zeros(3)
    for level in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="level must lie"):
            gaussian_coverage(y, y, np.ones(3), level)
