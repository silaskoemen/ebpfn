"""Disjoint, reproducible evaluation panels for the three random stages.

Cloud generation always runs through the GENERATION stream; a panel's
``stage``/``token`` identity is threaded into the sampling so search, selection,
and final-audit panels draw disjoint synthetic tasks while reusing common random
numbers across candidates within a panel.
"""

from ebpfn.compare import sample_energy_pairs
from ebpfn.config import TuningConfig
from ebpfn.utils import RandomRole
from ebpfn.utils import RandomStreams

from .contracts import Panel

_STAGE_ROLES: dict[str, RandomRole] = {
    "search": RandomRole.SEARCH,
    "selection": RandomRole.SELECTION,
    "final_audit": RandomRole.FINAL_AUDIT,
}


def stage_role(stage: str) -> RandomRole:
    if stage not in _STAGE_ROLES:
        raise ValueError(f"unknown evaluation stage {stage!r}")
    return _STAGE_ROLES[stage]


def make_panel(stage: str, token: int, config: TuningConfig, streams: RandomStreams) -> Panel:
    """Build one panel, drawing a common energy pair sample when configured."""
    role = stage_role(stage)
    pair_ids = None
    if config.objective == "energy" and config.compare.energy_pair_sample is not None:
        rng = streams.generator(role, "energy-pairs", token)
        pair_ids = sample_energy_pairs(config.cloud.n_members, config.compare.energy_pair_sample, rng)
    return Panel(stage=stage, token=token, energy_pair_ids=pair_ids)


def make_panels(stage: str, count: int, config: TuningConfig, streams: RandomStreams) -> list[Panel]:
    if count < 1:
        raise ValueError("panel count must be at least one")
    return [make_panel(stage, token, config, streams) for token in range(count)]
