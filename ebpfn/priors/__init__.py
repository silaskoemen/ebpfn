"""Synthetic task prior: the vendored TabICLv2 generator under a tunable eta."""

from ebpfn.priors.contracts import GeneratedTask
from ebpfn.priors.nanoprior import (
    ALL_BLOCKS,
    GROUP_ORDER,
    ORIGIN,
    ORIGIN_ETA,
    EtaSpace,
    MechanismEta,
    sample_task,
)

__all__ = [
    "ALL_BLOCKS",
    "GROUP_ORDER",
    "ORIGIN",
    "ORIGIN_ETA",
    "EtaSpace",
    "GeneratedTask",
    "MechanismEta",
    "sample_task",
]
