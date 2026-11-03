import numpy as np
import scipy.stats
from benchmarks.studies.prequential import log_bin_mass, target_bins


def test_target_bins_split_distinct_values_at_midpoints() -> None:
    low, high = target_bins(np.array([2.0, 0.0, 1.0, 0.0, 4.0]))
    np.testing.assert_array_equal(low, [1.5, -0.5, 0.5, -0.5, 3.0])
    np.testing.assert_array_equal(high, [3.0, 0.5, 1.5, 0.5, 5.0])


def test_target_bins_tile_the_line_without_gaps() -> None:
    y = np.random.default_rng(0).integers(0, 7, 50).astype(float)
    low, high = target_bins(y)
    edges = np.unique(np.concatenate([low, high]))
    assert len(edges) == len(np.unique(y)) + 1


def test_log_bin_mass_matches_density_on_a_narrow_bin() -> None:
    y, width = np.array([0.3]), 1e-4
    mass = log_bin_mass(scipy.stats.norm.cdf, y - width / 2, y + width / 2)
    np.testing.assert_allclose(mass, scipy.stats.norm.logpdf(y) + np.log(width), atol=1e-6)
