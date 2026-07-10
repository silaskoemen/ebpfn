import numpy as np
import pytest
from ebpfn.characterize import build_row_budget_manifests, characterize
from ebpfn.compare import block_distance, energy_score, group_by_budget_block, sample_energy_pairs
from ebpfn.config import CharacterizationConfig, CompareConfig, HyperPriorConfig
from ebpfn.data import CharacterizationShape, characterization_shape
from ebpfn.priors import build_hyperprior, sample_cloud
from ebpfn.utils import RandomStreams


def _single_budget_real_and_cloud(n_members: int, seed: int = 0):
    # Use the smallest single budget so the block-balanced distance has one budget
    # and brute-forcing the V-statistic is straightforward.
    streams = RandomStreams(seed)
    eta = build_hyperprior(HyperPriorConfig())
    config = CharacterizationConfig()
    real_task = sample_cloud(eta, CharacterizationShape(300, 140, 5, 0, "regression"), 1, streams, "real")[0].tuning
    manifest = build_row_budget_manifests(real_task, config)[0]
    real = characterize(real_task, manifest, config)
    cloud_tasks = sample_cloud(eta, characterization_shape(real_task), n_members, streams, "cloud")
    cloud = [
        characterize(member.tuning, build_row_budget_manifests(member.tuning, config)[0], config)
        for member in cloud_tasks
    ]
    return real, cloud


def _brute_force_energy(real, cloud):
    block_indices = next(iter(group_by_budget_block(real).values()))
    observation = np.mean([block_distance(real.values, m.values, block_indices).total for m in cloud])
    ensemble = np.mean([block_distance(a.values, b.values, block_indices).total for a in cloud for b in cloud])
    return float(observation - 0.5 * ensemble)


def test_v_statistic_matches_brute_force():
    real, cloud = _single_budget_real_and_cloud(8)
    result = energy_score(real, cloud, CompareConfig())
    assert result.total == pytest.approx(_brute_force_energy(real, cloud), rel=1e-9, abs=1e-12)


def test_energy_is_nonnegative():
    real, cloud = _single_budget_real_and_cloud(10, seed=2)
    assert energy_score(real, cloud, CompareConfig()).total >= -1e-9


def test_energy_is_zero_when_members_equal_the_observation():
    real, _ = _single_budget_real_and_cloud(4)
    result = energy_score(real, [real] * 5, CompareConfig())
    assert result.total == pytest.approx(0.0, abs=1e-12)
    assert result.observation_term == pytest.approx(0.0, abs=1e-12)
    assert result.ensemble_term == pytest.approx(0.0, abs=1e-12)


def test_terms_are_stored_separately():
    real, cloud = _single_budget_real_and_cloud(6)
    result = energy_score(real, cloud, CompareConfig())
    assert result.total == pytest.approx(result.observation_term - result.ensemble_term, rel=1e-9)
    assert result.pair_ids is None


def test_common_pair_sample_path_is_deterministic():
    from ebpfn.utils import RandomRole

    real, cloud = _single_budget_real_and_cloud(6)
    rng = RandomStreams(7).generator(RandomRole.SEARCH, "energy-pairs")
    pairs = sample_energy_pairs(len(cloud), 12, rng)
    config = CompareConfig(energy_pair_sample=12)
    first = energy_score(real, cloud, config, pair_ids=pairs)
    second = energy_score(real, cloud, config, pair_ids=pairs)
    assert first.total == pytest.approx(second.total)
    assert first.pair_ids == pairs


def test_pair_sample_requires_ids_when_enabled():
    real, cloud = _single_budget_real_and_cloud(4)
    with pytest.raises(ValueError, match="no common pair ids"):
        energy_score(real, cloud, CompareConfig(energy_pair_sample=8))
