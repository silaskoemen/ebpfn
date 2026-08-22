"""Unit-cube coordinate space for the nanoprior hyperprior.

The map is total: every finite vector in the cube decodes to a valid eta, so there is no
feasibility predicate and no repair step. Sampling is not here — a Sobol ladder, a
population optimizer and Optuna all want different designs over the same box, and the
space only has to name coordinates, encode, decode, and report its schema.

Coordinates are namespaced by the causal block they belong to, so the ``p(y|z)`` and
``g`` blocks land beside the ``z`` block without renaming what already exists.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ebpfn.data import content_hash
from ebpfn.priors.nanoprior.mechanisms import (
    ETA_VERSION,
    GROUP_ORDER,
    GROUPING_ID,
    ORIGIN,
    ORIGIN_ETA,
    ROLES,
    MechanismEta,
    Role,
)
from ebpfn.priors.simplex import sticks_to_weights, weights_to_sticks

SPACE_VERSION = "nanoprior-eta-space-1"

# The all-multiplicative vertex is the one simplex point with no ``only_cheap`` support,
# so the box stops just short of it. Every other mixture stays reachable, and the excluded
# face is a documented property of the search space rather than a repair inside decode.
_MAX_STICK = 1.0 - 1e-9

CausalBlock = Literal["z", "p_y_z", "g"]


@dataclass(frozen=True)
class SimplexBlock:
    """One mixture-weight group, carried as ``len(members) - 1`` stick coordinates."""

    name: str
    block: CausalBlock
    role: Role
    grouping: str
    members: tuple[str, ...]
    origin: tuple[float, ...]

    @property
    def width(self) -> int:
        return len(self.members) - 1

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        # A stick is a share of what earlier sticks left, not one group's mass, so it is
        # named by position rather than by the group it partly controls.
        return tuple(f"{self.name}.stick.{index}" for index in range(self.width))


SPECS: tuple[SimplexBlock, ...] = tuple(
    SimplexBlock(f"z.{role}", "z", role, GROUPING_ID, GROUP_ORDER, ORIGIN) for role in ROLES
)
ALL_BLOCKS: tuple[str, ...] = tuple(spec.name for spec in SPECS)

_BY_NAME = {spec.name: spec for spec in SPECS}


class EtaSpace:
    """Encodes an active subset of the hyperprior's blocks as a unit-cube vector.

    Inactive blocks are held at ``base``, so a study can search one block while pinning
    the rest at the origin or at a previously selected eta.
    """

    def __init__(self, active: Sequence[str] = ALL_BLOCKS, base: MechanismEta = ORIGIN_ETA) -> None:
        unknown = [name for name in active if name not in _BY_NAME]
        if unknown:
            raise ValueError(f"unknown blocks: {unknown}")
        if len(set(active)) != len(active):
            raise ValueError("blocks must be unique")
        if not active:
            raise ValueError("at least one block must be active")
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
        """Where upstream's own mixture sits. Nowhere near the centre of the cube."""
        return self.encode(ORIGIN_ETA)

    def encode(self, eta: MechanismEta) -> np.ndarray:
        sticks = [value for spec in self._specs for value in weights_to_sticks(eta.weights_for(spec.role))]
        return np.array(sticks, dtype=float)

    def decode(self, vector: np.ndarray) -> MechanismEta:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError(f"expected {self.dimension} coordinates, got {vector.shape}")
        if not np.all(np.isfinite(vector)) or vector.min() < 0.0 or vector.max() > 1.0:
            raise ValueError("coordinates must be finite and inside the unit cube")

        weights = {role: self.base.weights_for(role) for role in ROLES}
        offset = 0
        for spec in self._specs:
            sticks = np.minimum(vector[offset : offset + spec.width], _MAX_STICK)
            weights[spec.role] = sticks_to_weights(sticks.tolist())
            offset += spec.width
        return MechanismEta(**weights)

    def schema(self) -> dict[str, Any]:
        """Identity of the search space, for run and checkpoint provenance.

        Kept apart from ``eta.to_dict()``: two studies with different active sets that
        decode to the same mixture describe the same distribution and must hash alike.
        """
        coordinates = [
            {
                "name": name,
                "block": spec.block,
                "kind": "simplex",
                "simplex": spec.name,
                "grouping": spec.grouping,
            }
            for spec in self._specs
            for name in spec.coordinate_names
        ]
        payload = {
            "version": SPACE_VERSION,
            "eta_version": ETA_VERSION,
            "active": list(self.active),
            "coordinates": coordinates,
        }
        return {**payload, "hash": content_hash(payload, namespace=SPACE_VERSION)}
