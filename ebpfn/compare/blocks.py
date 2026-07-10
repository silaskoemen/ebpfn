"""Comparable block geometry over characterization vectors.

There is no fitted reference normalization: every admitted coordinate is bounded
to ``[-1, 1]`` and mutually comparable. The distance is block-balanced at each
row budget (root-mean-square within a block, then root-mean-square across the
blocks the budget populates, with uniform block weights). No covariance geometry,
cosine distance, or pairwise coordinate deletion is used.
"""

import numpy as np

from ebpfn.characterize import TaskCharacterization

from .contracts import BlockDistance
from .contracts import ValidityReport


def assert_comparable(real: TaskCharacterization, other: TaskCharacterization) -> None:
    """Require identical coordinate schema (name and order) between two vectors."""
    real_names = tuple(coordinate.name for coordinate in real.coordinates)
    other_names = tuple(coordinate.name for coordinate in other.coordinates)
    if real_names != other_names:
        raise ValueError("characterizations are not comparable: coordinate schemas differ")


def group_by_budget_block(char: TaskCharacterization) -> dict[int, dict[str, np.ndarray]]:
    """Map each row budget to ``block -> coordinate index array`` into ``values``."""
    groups: dict[int, dict[str, list[int]]] = {}
    for index, coordinate in enumerate(char.coordinates):
        budget = coordinate.row_budget
        if budget is None:
            raise ValueError(f"coordinate {coordinate.name!r} has no row budget")
        groups.setdefault(budget, {}).setdefault(coordinate.block, []).append(index)
    return {
        budget: {block: np.asarray(indices, dtype=np.intp) for block, indices in blocks.items()}
        for budget, blocks in groups.items()
    }


def budget_weights(char: TaskCharacterization) -> dict[int, float]:
    """Explicit per-budget weights, renormalized over the budgets present."""
    entries = char.metadata.get("budgets")
    if entries:
        raw = {int(entry["row_budget"]): float(entry["weight"]) for entry in entries}
    else:
        raw = {int(char.metadata["row_budget"]): float(char.metadata["weight"])}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("budget weights must have positive sum")
    return {budget: weight / total for budget, weight in raw.items()}


def block_distance(a_values: np.ndarray, b_values: np.ndarray, block_indices: dict[str, np.ndarray]) -> BlockDistance:
    """Block-balanced distance for one budget's ``block -> indices`` grouping."""
    if not block_indices:
        raise ValueError("block distance requires at least one populated block")
    per_block: dict[str, float] = {}
    for block, indices in block_indices.items():
        diff = a_values[indices] - b_values[indices]
        per_block[block] = float(np.sqrt(np.mean(np.square(diff))))
    total = float(np.sqrt(np.mean(np.square(np.array(list(per_block.values()))))))
    return BlockDistance(total=total, per_block=per_block)


def validity_report(char: TaskCharacterization) -> ValidityReport:
    """Overall and within-block valid fractions (quality-control diagnostics)."""
    groups = group_by_budget_block(char)
    per_block_counts: dict[str, list[int]] = {}
    for blocks in groups.values():
        for block, indices in blocks.items():
            valid = char.valid[indices]
            per_block_counts.setdefault(block, [0, 0])
            per_block_counts[block][0] += int(np.count_nonzero(valid))
            per_block_counts[block][1] += int(valid.size)
    within_block = {block: (valid / total if total else 1.0) for block, (valid, total) in per_block_counts.items()}
    overall = float(np.mean(char.valid)) if char.valid.size else 1.0
    return ValidityReport(
        all_valid=bool(np.all(char.valid)),
        overall_fraction=overall,
        within_block_fraction=within_block,
    )
