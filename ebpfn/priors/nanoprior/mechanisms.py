"""Mechanism families of the vendored TabICLv2 generator, grouped for tuning.

ebpfn code, not vendored. Upstream draws each node's function uniformly from eight
families. This module groups them so a mixture costs three free coordinates per role
instead of seven, and draws from that mixture; :mod:`ebpfn.priors.nanoprior.eta` owns the
weights themselves. Two roles are kept separate -- ``mechanism`` (how parents map to a
child) and ``root`` (how latent source features are shaped) -- because feature skew is a
root-marginal problem and near-determinism a mechanism one. At :data:`ORIGIN` the induced
per-family probability is exactly ``1/8``, and exactly ``1/5`` under ``only_cheap``:
upstream in both modes. The grouping is a hypothesis the Jacobian audit tests.

Passing no eta at all leaves upstream's own ``randchoice`` in place, which is bit-identical
rather than merely equal in distribution. That path is a fidelity check against the vendored
generator, not an experimental arm: it consumes a different RNG primitive than weighted
selection does, so it is not seed-identical to the origin. Experiments use the origin eta,
which keeps every arm on one code path. (Lifted *scalar* sites do not have this wrinkle --
see :mod:`ebpfn.priors.nanoprior.scalars`.)
"""

from collections.abc import Sequence
from typing import Literal, Protocol

import numpy as np
from torch import Tensor

Role = Literal["mechanism", "root"]
ROLES: tuple[Role, ...] = ("mechanism", "root")


class Family(Protocol):
    """One of ``prior.py``'s mechanism functions, identified by its ``__name__``."""

    __name__: str

    def __call__(self, x: Tensor, d_out: int, /) -> Tensor: ...


# Group -> the ``prior.py`` functions it covers, keyed by name so this module stays
# importable from ``prior.py`` without a cycle.
GROUPS: dict[str, tuple[str, ...]] = {
    "smooth": ("rand_gp_func", "rand_mlp_func"),
    "polynomial": ("rand_lin_func", "rand_quad_func"),
    "piecewise": ("rand_tree_func", "rand_discretization_func", "rand_em_func"),
    "multiplicative": ("rand_prod_func",),
}
GROUP_ORDER: tuple[str, ...] = ("smooth", "polynomial", "piecewise", "multiplicative")
N_FAMILIES = sum(len(members) for members in GROUPS.values())

# Names the grouping a weight vector is written against. An ungrouped variant would be a
# different identifier, so positional payloads can never be read under the wrong scheme.
GROUPING_ID = "mechanism-groups-4-v1"

# Group mass reproducing upstream's uniform-over-eight draw.
ORIGIN: tuple[float, ...] = tuple(len(GROUPS[group]) / N_FAMILIES for group in GROUP_ORDER)

# ``only_cheap`` excludes every multiplicative family, so all-mass-on-multiplicative has
# empty restricted support. Rejected at construction rather than mid-generation.
CHEAP_GROUPS: tuple[str, ...] = ("smooth", "polynomial", "piecewise")

_GROUP_OF: dict[str, str] = {name: group for group, members in GROUPS.items() for name in members}
_SUM_TOL = 1e-6


def validate_weights(weights: Sequence[float], role: str) -> tuple[float, ...]:
    if len(weights) != len(GROUP_ORDER):
        raise ValueError(f"{role} weights need one entry per group, got {len(weights)}")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"{role} weights must be nonnegative")
    if abs(sum(weights) - 1.0) > _SUM_TOL:
        raise ValueError(f"{role} weights must sum to one, got {sum(weights)}")
    if sum(weights[GROUP_ORDER.index(group)] for group in CHEAP_GROUPS) <= 0.0:
        raise ValueError(f"{role} weights put all mass on multiplicative, which has no only_cheap support")
    return tuple(float(weight) for weight in weights)


def _family_mass(name: str, weights: Sequence[float]) -> float:
    """Per-family probability: the group's mass split evenly across its members."""
    group = _GROUP_OF[name]
    return weights[GROUP_ORDER.index(group)] / len(GROUPS[group])


def select(candidates: Sequence[Family], weights: Sequence[float]) -> Family:
    """Draw one family from ``candidates`` under one role's group ``weights``.

    Takes weights rather than an eta so this module stays free of the eta type, which needs
    the grouping defined here. Restriction (``only_cheap``) masks per-family probabilities
    and renormalizes rather than dropping groups; dropping them would break the exact
    ``1/5``. Draws from the global numpy RNG, which is what ``tasks.seeded_globals`` seeds.
    """
    masses = [_family_mass(func.__name__, weights) for func in candidates]
    total = sum(masses)
    if total <= 0.0:
        raise ValueError(f"eta gives no mass to any of {[func.__name__ for func in candidates]}")

    draw = float(np.random.uniform()) * total
    cumulative = 0.0
    for func, mass in zip(candidates, masses, strict=True):
        cumulative += mass
        if draw < cumulative:
            return func
    return candidates[-1]
