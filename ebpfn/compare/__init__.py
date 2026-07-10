"""Block-balanced comparison objectives over characterization vectors."""

from ebpfn.compare.blocks import (
    assert_comparable,
    block_distance,
    budget_weights,
    group_by_budget_block,
    validity_report,
)
from ebpfn.compare.contracts import BLOCKS, BlockDistance, DirectedCoverageResult, EnergyScoreResult, ValidityReport
from ebpfn.compare.directed import directed_coverage
from ebpfn.compare.energy import energy_score, sample_energy_pairs

__all__ = [
    "BLOCKS",
    "BlockDistance",
    "DirectedCoverageResult",
    "EnergyScoreResult",
    "ValidityReport",
    "assert_comparable",
    "block_distance",
    "budget_weights",
    "directed_coverage",
    "energy_score",
    "group_by_budget_block",
    "sample_energy_pairs",
    "validity_report",
]
