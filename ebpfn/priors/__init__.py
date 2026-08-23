"""Synthetic task prior: the vendored TabICLv2 generator under a tunable eta."""

from ebpfn.priors.contracts import GeneratedTask
from ebpfn.priors.nanoprior import (
    ALL_GROUPS,
    GROUP_ORDER,
    ORIGIN,
    ORIGIN_ETA,
    SITE_NAMES,
    EtaSpace,
    PriorEta,
    groups_in_block,
    sample_task,
)

__all__ = [
    "ALL_GROUPS",
    "GROUP_ORDER",
    "ORIGIN",
    "ORIGIN_ETA",
    "SITE_NAMES",
    "EtaSpace",
    "GeneratedTask",
    "PriorEta",
    "groups_in_block",
    "sample_task",
]
