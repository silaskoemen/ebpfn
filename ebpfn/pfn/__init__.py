"""PyTorch PFN subsystem: a vendored TabICLv2 backbone with a bar-distribution head,
trained on ebpfn's own prior. Imported only downstream of the simulator-only tuning
stack — ``ebpfn/tune`` and ``ebpfn/compare`` must never import this package (the
candidate-evaluation path stays likelihood-free)."""

from ebpfn.pfn.data import PairedPriorTaskSource, PriorTaskSource, TaskBatch, collate_tasks
from ebpfn.pfn.distribution import BarDistribution, fixed_borders
from ebpfn.pfn.model import EBPFNModel

__all__ = [
    "BarDistribution",
    "EBPFNModel",
    "PairedPriorTaskSource",
    "PriorTaskSource",
    "TaskBatch",
    "collate_tasks",
    "fixed_borders",
]
