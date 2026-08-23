"""Unit-cube coordinate space for the nanoprior hyperprior.

The map is total: every finite vector in the cube decodes to a valid eta, so there is no
feasibility predicate and no repair step. Sampling is not here -- a Sobol ladder, a
population optimizer and Optuna all want different designs over the same box, and the
space only has to name coordinates, encode, decode, and report its schema.

Two kinds of coordinate group share the cube. A :class:`SimplexGroup` carries one role's
mechanism mixture as stick-breaking coordinates; a :class:`ScalarGroup` carries one lifted
sampling site as ``(shift, spread)``. Both are namespaced by the causal block they belong
to, so ``p(y|z)`` coordinates will land beside them without renaming what exists. ``block``
is that causal block throughout -- ``group`` is the unit of activation, which is why these
are groups rather than blocks.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ebpfn.data import content_hash
from ebpfn.priors.nanoprior.eta import ETA_VERSION, ORIGIN_ETA, PriorEta
from ebpfn.priors.nanoprior.mechanisms import GROUP_ORDER, GROUPING_ID, ORIGIN, ROLES, Role
from ebpfn.priors.nanoprior.scalars import SCALARS_VERSION, SITES, CausalBlock, ScalarSite
from ebpfn.priors.simplex import sticks_to_weights, weights_to_sticks

SPACE_VERSION = "nanoprior-eta-space-2"

# The all-multiplicative vertex is the one simplex point with no ``only_cheap`` support,
# so the box stops just short of it. Every other mixture stays reachable, and the excluded
# face is a documented property of the search space rather than a repair inside decode.
_MAX_STICK = 1.0 - 1e-9


@dataclass(frozen=True)
class SimplexGroup:
    """One mixture-weight group, carried as ``len(members) - 1`` stick coordinates."""

    name: str
    block: CausalBlock
    role: Role
    grouping: str
    members: tuple[str, ...]
    origin: tuple[float, ...]
    kind: str = "simplex"

    @property
    def width(self) -> int:
        return len(self.members) - 1

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        # A stick is a share of what earlier sticks left, not one group's mass, so it is
        # named by position rather than by the group it partly controls.
        return tuple(f"{self.name}.stick.{index}" for index in range(self.width))

    def schema_extra(self) -> dict[str, Any]:
        return {"simplex": self.name, "grouping": self.grouping}


@dataclass(frozen=True)
class ScalarGroup:
    """One lifted sampling site, carried as a location and a dispersion coordinate."""

    site: ScalarSite
    kind: str = "scalar"

    @property
    def name(self) -> str:
        return self.site.name

    @property
    def block(self) -> CausalBlock:
        return self.site.block

    @property
    def width(self) -> int:
        return 2

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        return (f"{self.name}.shift", f"{self.name}.spread")

    def schema_extra(self) -> dict[str, Any]:
        return {"site": self.name, "reachable": self.site.reachable}


Group = SimplexGroup | ScalarGroup

SPECS: tuple[Group, ...] = (
    *(SimplexGroup(f"z.{role}", "z", role, GROUPING_ID, GROUP_ORDER, ORIGIN) for role in ROLES),
    *(ScalarGroup(site) for site in SITES),
)

# The default search surface excludes sites that only fire behind the categorical branch,
# which the numeric task contracts never enter. They stay in SPECS so a study can activate
# one explicitly once the contracts carry categoricals.
ALL_GROUPS: tuple[str, ...] = tuple(
    spec.name for spec in SPECS if isinstance(spec, SimplexGroup) or spec.site.reachable
)

_BY_NAME: dict[str, Group] = {spec.name: spec for spec in SPECS}


def groups_in_block(block: CausalBlock, *, reachable_only: bool = True) -> tuple[str, ...]:
    """Every group name in one causal block -- the unit a study activates to search it."""
    names = ALL_GROUPS if reachable_only else tuple(spec.name for spec in SPECS)
    return tuple(name for name in names if _BY_NAME[name].block == block)


class EtaSpace:
    """Encodes an active subset of the hyperprior's groups as a unit-cube vector.

    Inactive groups are held at ``base``, so a study can search one group while pinning
    the rest at the origin or at a previously selected eta.
    """

    def __init__(self, active: Sequence[str] = ALL_GROUPS, base: PriorEta = ORIGIN_ETA) -> None:
        unknown = [name for name in active if name not in _BY_NAME]
        if unknown:
            raise ValueError(f"unknown groups: {unknown}")
        if len(set(active)) != len(active):
            raise ValueError("groups must be unique")
        if not active:
            raise ValueError("at least one group must be active")
        self.active = tuple(active)
        self.base = base
        self._specs = tuple(_BY_NAME[name] for name in self.active)

    @property
    def dimension(self) -> int:
        return sum(spec.width for spec in self._specs)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        return tuple(name for spec in self._specs for name in spec.coordinate_names)

    @property
    def origin_vector(self) -> np.ndarray:
        """Where upstream's own distribution sits. Nowhere near the centre of the cube."""
        return self.encode(ORIGIN_ETA)

    def encode(self, eta: PriorEta) -> np.ndarray:
        values: list[float] = []
        for spec in self._specs:
            if isinstance(spec, SimplexGroup):
                values.extend(weights_to_sticks(eta.weights_for(spec.role)))
            else:
                values.extend(eta.coordinates_for(spec.name))
        return np.array(values, dtype=float)

    def decode(self, vector: np.ndarray) -> PriorEta:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError(f"expected {self.dimension} coordinates, got {vector.shape}")
        if not np.all(np.isfinite(vector)) or vector.min() < 0.0 or vector.max() > 1.0:
            raise ValueError("coordinates must be finite and inside the unit cube")

        weights = {role: self.base.weights_for(role) for role in ROLES}
        scalars = dict(self.base.scalars)
        offset = 0
        for spec in self._specs:
            chunk = vector[offset : offset + spec.width]
            offset += spec.width
            if isinstance(spec, SimplexGroup):
                weights[spec.role] = sticks_to_weights(np.minimum(chunk, _MAX_STICK).tolist())
            else:
                scalars[spec.name] = (float(chunk[0]), float(chunk[1]))
        return PriorEta(**weights, scalars=scalars)

    def schema(self) -> dict[str, Any]:
        """Identity of the search space, for run and checkpoint provenance.

        Kept apart from ``eta.to_dict()``: two studies with different active sets that
        decode to the same distribution describe the same prior and must hash alike.
        """
        coordinates = [
            {"name": name, "block": spec.block, "kind": spec.kind, **spec.schema_extra()}
            for spec in self._specs
            for name in spec.coordinate_names
        ]
        payload = {
            "version": SPACE_VERSION,
            "eta_version": ETA_VERSION,
            "sites": SCALARS_VERSION,
            "active": list(self.active),
            "coordinates": coordinates,
        }
        return {**payload, "hash": content_hash(payload, namespace=SPACE_VERSION)}
