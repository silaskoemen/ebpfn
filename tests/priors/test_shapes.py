import numpy as np
import pytest
from ebpfn.config import ShapeJitterConfig
from ebpfn.data import CharacterizationShape
from ebpfn.priors import sample_training_shape


def _anchor() -> CharacterizationShape:
    return CharacterizationShape(800, 200, 20, 0, "regression")


def test_shared_sampler_is_deterministic_for_baseline_and_tuned():
    anchor, jitter = _anchor(), ShapeJitterConfig()
    first, _ = sample_training_shape(anchor, jitter, np.random.default_rng(11))
    second, _ = sample_training_shape(anchor, jitter, np.random.default_rng(11))
    assert (first.n_probe_fit, first.n_probe_score, first.p_numeric) == (
        second.n_probe_fit,
        second.n_probe_score,
        second.p_numeric,
    )


def test_jitter_factors_are_mean_one():
    anchor, jitter = _anchor(), ShapeJitterConfig()
    rng = np.random.default_rng(0)
    factors = [sample_training_shape(anchor, jitter, rng)[1] for _ in range(5000)]
    assert abs(float(np.mean([f["j_n"] for f in factors])) - 1.0) < 0.03
    assert abs(float(np.mean([f["j_p"] for f in factors])) - 1.0) < 0.03


def test_shapes_are_clamped_to_compute_bounds():
    anchor = CharacterizationShape(4000, 1000, 90, 0, "regression")
    jitter = ShapeJitterConfig(n_max=2048, p_max=64)
    for seed in range(200):
        shape, _ = sample_training_shape(anchor, jitter, np.random.default_rng(seed))
        assert shape.n_probe_fit + shape.n_probe_score <= 2048
        assert shape.p_numeric <= 64
        assert shape.n_probe_fit >= 1
        assert shape.n_probe_score >= 1


def test_categorical_anchor_is_rejected():
    with pytest.raises(ValueError, match="categorical"):
        sample_training_shape(
            CharacterizationShape(100, 50, 4, 2, "regression"), ShapeJitterConfig(), np.random.default_rng(0)
        )
