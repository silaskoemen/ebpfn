"""Block-balanced comparison objectives over characterization vectors."""

from ebpfn.compare.blocks import assert_comparable
from ebpfn.compare.blocks import block_distance
from ebpfn.compare.blocks import budget_weights
from ebpfn.compare.blocks import group_by_budget_block
from ebpfn.compare.blocks import validity_report
from ebpfn.compare.contracts import BLOCKS
from ebpfn.compare.contracts import BlockDistance
from ebpfn.compare.contracts import DirectedCoverageResult
from ebpfn.compare.contracts import EnergyScoreResult
from ebpfn.compare.contracts import ValidityReport
from ebpfn.compare.directed import directed_coverage
from ebpfn.compare.energy import energy_score
from ebpfn.compare.energy import sample_energy_pairs

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
