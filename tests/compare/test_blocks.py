import numpy as np
import pytest
from ebpfn.characterize import characterize_multiresolution
from ebpfn.compare import (
    assert_comparable,
    block_distance,
    budget_weights,
    directed_coverage,
    energy_score,
    group_by_budget_block,
    validity_report,
)
from ebpfn.config import CharacterizationConfig, CompareConfig, HyperPriorConfig
from ebpfn.data import CharacterizationShape, characterization_shape
from ebpfn.priors import build_hyperprior, sample_cloud
from ebpfn.utils import RandomStreams


def _real_and_cloud(n_members: int, seed: int = 0):
    streams = RandomStreams(seed)
    eta = build_hyperprior(HyperPriorConfig())
    config = CharacterizationConfig()
    real_task = sample_cloud(eta, CharacterizationShape(300, 140, 5, 0, "regression"), 1, streams, "real")[0].tuning
    real = characterize_multiresolution(real_task, config)
    cloud_tasks = sample_cloud(eta, characterization_shape(real_task), n_members, streams, "cloud")
    cloud = [characterize_multiresolution(member.tuning, config) for member in cloud_tasks]
    return real, cloud


def test_identical_clouds_have_zero_distance():
    real, _ = _real_and_cloud(4)
    groups = group_by_budget_block(real)
    for block_indices in groups.values():
        distance = block_distance(real.values, real.values, block_indices)
        assert distance.total == pytest.approx(0.0, abs=1e-12)
        assert all(value == pytest.approx(0.0, abs=1e-12) for value in distance.per_block.values())


def test_identical_objectives_are_zero():
    real, _ = _real_and_cloud(6)
    identical = [real] * 6
    assert directed_coverage(real, identical, CompareConfig()).total == pytest.approx(0.0, abs=1e-12)
    assert energy_score(real, identical, CompareConfig()).total == pytest.approx(0.0, abs=1e-12)


def test_group_by_budget_covers_all_coordinates_and_six_blocks():
    real, _ = _real_and_cloud(2)
    groups = group_by_budget_block(real)
    grouped = sum(indices.size for blocks in groups.values() for indices in blocks.values())
    assert grouped == len(real.coordinates)
    blocks = {block for blocks in groups.values() for block in blocks}
    assert blocks == {"observation", "location", "scale_tail", "nonlinear", "interaction", "feature_concentration"}


def test_budget_weights_are_normalized():
    real, _ = _real_and_cloud(2)
    weights = budget_weights(real)
    assert len(weights) >= 2
    assert sum(weights.values()) == pytest.approx(1.0)


def test_validity_report_is_all_valid_for_real_characterization():
    real, _ = _real_and_cloud(2)
    report = validity_report(real)
    assert report.all_valid is True
    assert report.overall_fraction == pytest.approx(1.0)


def test_incomparable_schemas_are_rejected():
    real, _ = _real_and_cloud(2, seed=0)
    other, _ = _real_and_cloud(2, seed=1)
    # A different representation (contrast vs. raw) yields different coordinate
    # names, hence an incomparable schema.
    streams = RandomStreams(3)
    eta = build_hyperprior(HyperPriorConfig())
    mismatched_task = sample_cloud(eta, CharacterizationShape(300, 140, 5, 0, "regression"), 1, streams, "mismatch")[
        0
    ].tuning
    mismatched = characterize_multiresolution(mismatched_task, CharacterizationConfig(representation="contrast"))
    assert_comparable(real, other)  # same shape and representation -> comparable
    with pytest.raises(ValueError, match="not comparable"):
        assert_comparable(real, mismatched)


def test_block_distance_requires_a_populated_block():
    real, _ = _real_and_cloud(2)
    with pytest.raises(ValueError, match="at least one populated block"):
        block_distance(real.values, real.values, {})


def test_distance_is_positive_between_distinct_tasks():
    real, cloud = _real_and_cloud(4)
    groups = group_by_budget_block(real)
    block_indices = next(iter(groups.values()))
    distance = block_distance(real.values, cloud[0].values, block_indices)
    assert distance.total > 0.0
    assert np.isfinite(distance.total)
