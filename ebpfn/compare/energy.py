"""Objective B: block-balanced energy score.

Using the block-balanced distance ``d`` at each row budget,

    ES(r, S) = mean_i d(r, s_i) - 0.5 * mean_{i,j} d(s_i, s_j)

is evaluated as a deterministic V-statistic including the diagonal pairs, so it
is nonnegative and equals zero exactly when every ensemble member equals the
observation. The observation and ensemble terms are stored separately. Budgets
combine by explicit-weight mean. Exact pairwise evaluation is the default; a
fixed common pair sample (its ids fixed across candidates) is an optional
approximation whose ids become cache state.
"""

from collections.abc import Sequence

import numpy as np

from ebpfn.characterize import TaskCharacterization
from ebpfn.config import CompareConfig

from .blocks import assert_comparable
from .blocks import budget_weights
from .blocks import group_by_budget_block
from .blocks import validity_report
from .contracts import EnergyScoreResult


def sample_energy_pairs(n_members: int, count: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    """Draw a fixed common sample of ``count`` ordered member-index pairs."""
    if n_members < 1 or count < 1:
        raise ValueError("pair sampling needs a nonempty cloud and positive count")
    left = rng.integers(0, n_members, size=count)
    right = rng.integers(0, n_members, size=count)
    return tuple((int(i), int(j)) for i, j in zip(left, right, strict=True))


def _require_validity(real: TaskCharacterization, cloud: Sequence[TaskCharacterization]) -> None:
    # 100% validity is an invariant of an admitted objective vector, not a toggle.
    if not validity_report(real).all_valid:
        raise ValueError("real characterization is not fully valid")
    if any(not validity_report(member).all_valid for member in cloud):
        raise ValueError("a cloud member characterization is not fully valid")


def _block_distance_matrix(
    block_matrices: list[np.ndarray], left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Block-balanced distance between rows of ``left`` and rows of ``right``.

    ``block_matrices`` selects, per block, the coordinate columns; ``left`` and
    ``right`` are row-index arrays. Returns the balanced distance matrix and the
    per-block squared-distance matrices.
    """
    squared_block = []
    for columns in block_matrices:
        diff = columns[left][:, None, :] - columns[right][None, :, :]
        squared_block.append(np.mean(np.square(diff), axis=2))
    balanced = np.sqrt(np.mean(np.stack(squared_block, axis=0), axis=0))
    return balanced, squared_block


def energy_score(
    real: TaskCharacterization,
    cloud: Sequence[TaskCharacterization],
    config: CompareConfig,
    pair_ids: Sequence[tuple[int, int]] | None = None,
) -> EnergyScoreResult:
    if not cloud:
        raise ValueError("energy score requires a nonempty cloud")
    for member in cloud:
        assert_comparable(real, member)
    _require_validity(real, cloud)

    groups = group_by_budget_block(real)
    weights = budget_weights(real)
    n_members = len(cloud)
    cloud_values = np.stack([member.values for member in cloud])

    used_pairs: tuple[tuple[int, int], ...] | None = None
    if config.energy_pair_sample is not None:
        if pair_ids is None:
            raise ValueError("energy_pair_sample is set but no common pair ids were supplied")
        used_pairs = tuple((int(i), int(j)) for i, j in pair_ids)

    total = 0.0
    observation_term = 0.0
    ensemble_term = 0.0
    per_budget: dict[int, float] = {}
    block_accumulator: dict[str, float] = {}
    for budget, block_indices in groups.items():
        blocks = list(block_indices)
        block_columns = [cloud_values[:, block_indices[block]] for block in blocks]
        real_columns = [real.values[block_indices[block]][None, :] for block in blocks]

        # Observation term: distance from the real vector to every member.
        obs_squared = [
            np.mean(np.square(columns - reference), axis=1)
            for columns, reference in zip(block_columns, real_columns, strict=True)
        ]
        obs_balanced = np.sqrt(np.mean(np.stack(obs_squared, axis=0), axis=0))
        obs_mean = float(np.mean(obs_balanced))

        # Ensemble term: pairwise member distances (exact or common pair sample).
        if used_pairs is None:
            index = np.arange(n_members)
            balanced, squared_block = _block_distance_matrix(block_columns, index, index)
            ensemble_mean = float(np.mean(balanced))
            block_ensemble = [float(np.mean(np.sqrt(square))) for square in squared_block]
        else:
            left = np.array([i for i, _ in used_pairs])
            right = np.array([j for _, j in used_pairs])
            squared_block = [np.mean(np.square(columns[left] - columns[right]), axis=1) for columns in block_columns]
            balanced = np.sqrt(np.mean(np.stack(squared_block, axis=0), axis=0))
            ensemble_mean = float(np.mean(balanced))
            block_ensemble = [float(np.mean(np.sqrt(square))) for square in squared_block]

        weight = weights[budget]
        score_budget = obs_mean - 0.5 * ensemble_mean
        total += weight * score_budget
        observation_term += weight * obs_mean
        ensemble_term += weight * 0.5 * ensemble_mean
        per_budget[budget] = score_budget
        for position, block in enumerate(blocks):
            block_obs = float(np.mean(np.sqrt(obs_squared[position])))
            block_score = block_obs - 0.5 * block_ensemble[position]
            block_accumulator[block] = block_accumulator.get(block, 0.0) + weight * block_score

    return EnergyScoreResult(
        total=total,
        per_block=block_accumulator,
        per_budget=per_budget,
        observation_term=observation_term,
        ensemble_term=ensemble_term,
        pair_ids=used_pairs,
    )
