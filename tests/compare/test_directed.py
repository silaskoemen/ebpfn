import math

import pytest
from ebpfn.characterize import characterize_multiresolution
from ebpfn.compare import directed_coverage
from ebpfn.config import CharacterizationConfig, CompareConfig, HyperPriorConfig, MapConfig, RowBudgetConfig
from ebpfn.data import CharacterizationShape, characterization_shape
from ebpfn.priors import build_hyperprior, sample_cloud
from ebpfn.utils import RandomStreams


def _real_and_cloud(n_members: int, seed: int = 0):
    streams = RandomStreams(seed)
    eta = build_hyperprior(HyperPriorConfig())
    config = CharacterizationConfig(
        row_budgets=RowBudgetConfig(minimum=32),
        maps=MapConfig(max_products=8, max_conjunctions=8, max_rff=16, rff_distance_rows=64),
    )
    real_task = sample_cloud(eta, CharacterizationShape(64, 32, 5, 0, "regression"), 1, streams, "real")[0].tuning
    real = characterize_multiresolution(real_task, config)
    cloud_tasks = sample_cloud(eta, characterization_shape(real_task), n_members, streams, "cloud")
    cloud = [characterize_multiresolution(member.tuning, config) for member in cloud_tasks]
    return real, cloud


def test_directed_coverage_is_asymmetric():
    real, cloud = _real_and_cloud(6)
    # Coverage of the real task by the cloud vs. of a single cloud member by a
    # cloud that replaces one member with the real task: the directions differ.
    forward = directed_coverage(real, cloud, CompareConfig()).total
    backward = directed_coverage(cloud[0], [real, *cloud[1:]], CompareConfig()).total
    assert forward != pytest.approx(backward)


def test_neighborhood_policy_matches_specification():
    real, cloud = _real_and_cloud(6)
    config = CompareConfig(directed_k_floor=2, directed_k_fraction=0.5)
    result = directed_coverage(real, cloud, config)
    expected_k = max(2, math.ceil(0.5 * len(cloud)))
    for budget, k in result.k_by_budget.items():
        assert k == expected_k
        assert len(result.neighbors_by_budget[budget]) == expected_k


def test_k_is_capped_at_cloud_size():
    real, cloud = _real_and_cloud(3)
    config = CompareConfig(directed_k_floor=5, directed_k_fraction=0.01)
    result = directed_coverage(real, cloud, config)
    assert all(k == len(cloud) for k in result.k_by_budget.values())


def test_per_block_contributions_are_recorded():
    real, cloud = _real_and_cloud(3)
    result = directed_coverage(real, cloud, CompareConfig())
    assert set(result.per_block).issubset(
        {"observation", "location", "scale_tail", "nonlinear", "interaction", "feature_concentration"}
    )
    assert result.total > 0.0


def test_full_validity_is_an_invariant():
    real, cloud = _real_and_cloud(2)
    # 100% validity is an invariant with no bypass: an invalid member is rejected.
    import dataclasses

    invalid = dataclasses.replace(cloud[0], valid=cloud[0].valid & False)
    with pytest.raises(ValueError, match="not fully valid"):
        directed_coverage(real, [invalid, *cloud[1:]], CompareConfig())
