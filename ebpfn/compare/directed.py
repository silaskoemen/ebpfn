"""Objective A: directed k-nearest coverage.

For a matched cloud ``S`` of size ``N`` and neighbourhood ``k(N) = max(k_floor,
ceil(k_fraction * N))``, the directed coverage at row budget ``m`` is the mean
block-balanced distance from the real task to its ``k`` nearest cloud members.
Budgets combine as ``c_multi**2 = sum_m w_m c(m)**2``. This measures support
coverage only: it is intentionally asymmetric (real -> cloud) and has no
synthetic-to-real waste term.
"""

import math
from collections.abc import Sequence

import numpy as np

from ebpfn.characterize import TaskCharacterization
from ebpfn.config import CompareConfig

from .blocks import assert_comparable
from .blocks import block_distance
from .blocks import budget_weights
from .blocks import group_by_budget_block
from .blocks import validity_report
from .contracts import DirectedCoverageResult


def _require_validity(real: TaskCharacterization, cloud: Sequence[TaskCharacterization]) -> None:
    # 100% validity is an invariant of an admitted objective vector, not a toggle.
    if not validity_report(real).all_valid:
        raise ValueError("real characterization is not fully valid")
    if any(not validity_report(member).all_valid for member in cloud):
        raise ValueError("a cloud member characterization is not fully valid")


def directed_coverage(
    real: TaskCharacterization, cloud: Sequence[TaskCharacterization], config: CompareConfig
) -> DirectedCoverageResult:
    if not cloud:
        raise ValueError("directed coverage requires a nonempty cloud")
    for member in cloud:
        assert_comparable(real, member)
    _require_validity(real, cloud)

    groups = group_by_budget_block(real)
    weights = budget_weights(real)
    n_members = len(cloud)
    k = max(config.directed_k_floor, math.ceil(config.directed_k_fraction * n_members))
    k = min(k, n_members)

    total_sq = 0.0
    per_budget: dict[int, float] = {}
    k_by_budget: dict[int, int] = {}
    neighbors_by_budget: dict[int, tuple[int, ...]] = {}
    block_accumulator: dict[str, float] = {}
    for budget, block_indices in groups.items():
        distances = [block_distance(real.values, member.values, block_indices) for member in cloud]
        totals = np.array([distance.total for distance in distances])
        nearest = np.argsort(totals, kind="stable")[:k]
        coverage = float(np.mean(totals[nearest]))
        weight = weights[budget]
        total_sq += weight * coverage**2
        per_budget[budget] = coverage
        k_by_budget[budget] = k
        neighbors_by_budget[budget] = tuple(int(index) for index in nearest)
        for block in block_indices:
            block_mean = float(np.mean([distances[index].per_block[block] for index in nearest]))
            block_accumulator[block] = block_accumulator.get(block, 0.0) + weight * block_mean**2

    return DirectedCoverageResult(
        total=math.sqrt(total_sq),
        per_block={block: math.sqrt(value) for block, value in block_accumulator.items()},
        per_budget=per_budget,
        k_by_budget=k_by_budget,
        neighbors_by_budget=neighbors_by_budget,
    )
