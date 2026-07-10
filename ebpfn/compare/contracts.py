"""Result contracts for the block-balanced comparison objectives."""

from dataclasses import dataclass

# The six live characterization blocks. A per-budget block-balanced distance is
# the root-mean-square over whichever of these blocks the budget's coordinates
# populate, with uniform block weights.
BLOCKS: tuple[str, ...] = (
    "observation",
    "location",
    "scale_tail",
    "nonlinear",
    "interaction",
    "feature_concentration",
)


@dataclass(frozen=True)
class BlockDistance:
    """A single block-balanced distance and its per-block decomposition."""

    total: float
    per_block: dict[str, float]


@dataclass(frozen=True)
class ValidityReport:
    """Validity accounting for one characterization vector."""

    all_valid: bool
    overall_fraction: float
    within_block_fraction: dict[str, float]


@dataclass(frozen=True)
class DirectedCoverageResult:
    """Objective A: directed k-nearest coverage of the real task by the cloud."""

    total: float
    per_block: dict[str, float]
    per_budget: dict[int, float]
    k_by_budget: dict[int, int]
    neighbors_by_budget: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class EnergyScoreResult:
    """Objective B: block-balanced energy score V-statistic."""

    total: float
    per_block: dict[str, float]
    per_budget: dict[int, float]
    observation_term: float
    ensemble_term: float
    pair_ids: tuple[tuple[int, int], ...] | None
