"""The amortized eta posterior: a stable summary, and an estimator that learns what is identifiable."""

import numpy as np
import pytest
from ebpfn.posterior.estimator import N_BINS, fit_posterior, posterior_probs, to_bins
from ebpfn.posterior.summaries import summarize


def test_summary_has_a_fixed_finite_layout_across_shapes() -> None:
    rng = np.random.default_rng(0)
    layouts = []
    for n, p in [(64, 1), (200, 5), (300, 30)]:
        x = rng.standard_normal((n, p))
        summary = summarize(x, x[:, 0] + 0.1 * rng.standard_normal(n))
        assert all(np.isfinite(v) for v in summary.values())
        layouts.append(tuple(summary))
    assert len(set(layouts)) == 1


def test_summary_is_invariant_to_feature_order() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((150, 6))
    y = np.sin(x[:, 0]) + x[:, 1] ** 2
    plain = summarize(x, y)
    permuted = summarize(x[:, rng.permutation(6)], y)
    # Learner-based entries depend on tree randomness over column order; marginal ones must not.
    for key in plain:
        if not key.startswith(("fit.", "res.")):
            assert plain[key] == pytest.approx(permuted[key], abs=1e-9)


def test_summary_tolerates_constant_and_discrete_columns() -> None:
    rng = np.random.default_rng(2)
    x = np.column_stack([np.ones(100), rng.integers(0, 3, 100), rng.standard_normal(100)])
    summary = summarize(x, x[:, 2] + rng.standard_normal(100))
    assert all(np.isfinite(v) for v in summary.values())


def test_bins_cover_the_closed_cube() -> None:
    assert to_bins(np.array([0.0, 0.0999, 0.1, 1.0])).tolist() == [0, 0, 1, N_BINS - 1]


def test_estimator_gains_information_only_where_the_data_carries_it() -> None:
    """Coordinate 0 sets the summary, coordinate 1 is invisible: only the first contracts."""
    rng = np.random.default_rng(3)
    eta = rng.uniform(size=(3000, 2))
    summaries = np.column_stack([eta[:, 0] + 0.02 * rng.standard_normal(3000), rng.standard_normal(3000)])
    model, scaler, _ = fit_posterior(summaries, eta, val_fraction=0.2, seed=0)
    probs = posterior_probs(model, scaler, summaries[:500])
    picked = np.take_along_axis(probs, to_bins(eta[:500])[..., None], -1)[..., 0]
    gain = (np.log(N_BINS) + np.log(picked)).mean(0)
    assert gain[0] > 1.0
    assert abs(gain[1]) < 0.1
