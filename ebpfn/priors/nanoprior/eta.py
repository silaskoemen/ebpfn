"""The hyperprior eta: what is tunable about the vendored TabICLv2 generator.

Two kinds of coordinate, kept in one object because they are one distribution:

* **Mixture weights**, per role, over the mechanism groups in
  :mod:`ebpfn.priors.nanoprior.mechanisms` -- which function family a node draws.
* **Scalar sites**, per named sampling site in :mod:`ebpfn.priors.nanoprior.scalars` --
  where each family's hardcoded log-uniform interval sits and how wide it is.

:data:`ORIGIN_ETA` is upstream's own distribution and the null arm every tuned variant is
measured against: the released checkpoint was pretrained on exactly it, so at the origin the
base sits at its optimum and any movement is measured from a null that is exact rather than
assumed. Scalars omitted from ``scalars`` are left at upstream's constants, so a partially
lifted eta is meaningful and the empty default is upstream exactly.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ebpfn.priors.nanoprior.mechanisms import GROUPING_ID, ORIGIN, Role, validate_weights
from ebpfn.priors.nanoprior.scalars import BY_NAME, ORIGIN_COORD, SCALARS_VERSION

ETA_VERSION = "nanoprior-eta-3"


def _validate_scalars(scalars: Mapping[str, Sequence[float]]) -> Mapping[str, tuple[float, float]]:
    unknown = sorted(name for name in scalars if name not in BY_NAME)
    if unknown:
        raise ValueError(f"unknown sampling sites: {unknown}")
    checked: dict[str, tuple[float, float]] = {}
    for name in sorted(scalars):
        coordinates = tuple(float(value) for value in scalars[name])
        if len(coordinates) != 2:
            raise ValueError(f"{name} needs (shift, spread) coordinates, got {len(coordinates)}")
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{name} coordinates must lie in the unit interval, got {coordinates}")
        checked[name] = (coordinates[0], coordinates[1])
    return MappingProxyType(checked)


@dataclass(frozen=True)
class PriorEta:
    """Group weights per role plus per-site scalar coordinates -- the tunable nanoprior."""

    mechanism: tuple[float, ...] = ORIGIN
    root: tuple[float, ...] = ORIGIN
    scalars: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mechanism", validate_weights(self.mechanism, "mechanism"))
        object.__setattr__(self, "root", validate_weights(self.root, "root"))
        object.__setattr__(self, "scalars", _validate_scalars(self.scalars))

    def weights_for(self, role: Role) -> tuple[float, ...]:
        return self.mechanism if role == "mechanism" else self.root

    def coordinates_for(self, site: str) -> tuple[float, float]:
        """A site's coordinates, or the origin pair when it is left at upstream's interval."""
        return self.scalars.get(site, (ORIGIN_COORD, ORIGIN_COORD))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": ETA_VERSION,
            "grouping": GROUPING_ID,
            "sites": SCALARS_VERSION,
            "mechanism": list(self.mechanism),
            "root": list(self.root),
            "scalars": {name: list(values) for name, values in self.scalars.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PriorEta":
        if payload.get("version") != ETA_VERSION:
            raise ValueError(f"unsupported eta version {payload.get('version')!r}")
        if payload.get("grouping") != GROUPING_ID:
            raise ValueError(f"weights are grouped as {payload.get('grouping')!r}, not {GROUPING_ID}")
        if payload.get("sites") != SCALARS_VERSION:
            raise ValueError(f"scalars are named under {payload.get('sites')!r}, not {SCALARS_VERSION}")
        return cls(tuple(payload["mechanism"]), tuple(payload["root"]), payload.get("scalars", {}))


# Upstream's own distribution, and the null arm every tuned variant is measured against.
ORIGIN_ETA = PriorEta()
