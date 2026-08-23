"""Named sampling sites in the vendored generator, and the map that reparameterizes them.

ebpfn code, not vendored -- see NOTICE. Upstream draws ~20 scalars from hardcoded
log-uniform intervals inline in leaf mechanisms that ``eta`` cannot reach: ``rand_gp_func``
and its siblings take ``(x, d_out)`` and are called positionally through ``rand_func``.
Widening every leaf signature would make the vendor diff unreviewable, so instead each
interval is *named at its call site* and the sampling primitive resolves it here, against a
table installed for the duration of one task by :func:`active_scalars`.

The reparameterization is geometric -- a location shift and a log-halfwidth offset, which
is the "eta = mean, theta = draw around it" idea in log space. Three properties follow:

* **Exact at the origin.** ``2 * 0.5 - 1`` is ``0.0``, ``exp(0.0)`` is ``1.0``, and
  ``low * 1.0 / 1.0`` is ``low`` -- the same bits, so the same ``np.random.uniform`` draw
  yields the same float. A site at the origin is *bit-identical* to an unlifted one. That
  is stronger than the mechanism simplex manages, where ``select`` consumes a different RNG
  primitive than ``randchoice`` and the origin is only distributionally upstream.
* **Total.** ``shift`` is multiplicative, so positivity survives; ``spread`` offsets the
  log-halfwidth by a fraction of its own width, so ``0 < low < high`` holds everywhere in
  the cube. No feasibility predicate and no repair step, matching the stick map.
* **Checked against upstream.** :func:`resolve` asserts the call site still passes the
  constants the registry recorded. Re-vendoring a release that changed one raises here
  instead of silently retuning against a stale origin.

A site with ``low``/``high`` of ``None`` computes that end at the call site, so only the
fixed end is checked. Sites marked ``reachable=False`` sit behind the categorical branch of
``rand_converter``, which our numeric-only task contracts never enter; they are registered
so the table mirrors upstream, but excluded from the default search space. Upstream's
``rand_cat_sizes`` is not registered at all -- the adapter passes column sizes directly, so
that function is never called.
"""

import contextlib
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

SCALARS_VERSION = "nanoprior-scalar-sites-1"

# The coordinate value at which a site reproduces upstream exactly.
ORIGIN_COORD = 0.5
# Location reach: the interval slides by up to this factor either way.
SHIFT_MAX = math.log(3.0)
# Dispersion reach, as a fraction of the site's own log-halfwidth. At 0.9 an interval
# narrows to a tenth of upstream's width or widens to just under twice it.
SPREAD_FRAC = 0.9

CausalBlock = Literal["z", "p_y_z", "g"]


@dataclass(frozen=True)
class ScalarSite:
    """One log-uniform interval in the generator, named for eta to reach."""

    name: str
    block: CausalBlock
    low: float | None
    high: float | None
    reachable: bool = True


SITES: tuple[ScalarSite, ...] = (
    # -- z: the latent computation graph --------------------------------------------
    ScalarSite("z.graph.n_nodes", "z", 2, 33),
    ScalarSite("z.node.extra_width", "z", 1, 32),
    ScalarSite("z.node.output_scale", "z", 0.1, 10.0),
    ScalarSite("z.mlp.hidden_width", "z", 1, 128),
    ScalarSite("z.mlp.depth", "z", 1, 4),
    ScalarSite("z.tree.n_trees", "z", 1, 128),
    ScalarSite("z.discretization.n_centers", "z", 2, None),  # capped by the row count
    ScalarSite("z.discretization.metric", "z", 0.5, 4.0),
    ScalarSite("z.gp.decay", "z", 2.0, 20.0),
    ScalarSite("z.gp.lengthscale", "z", 0.5, 10.0),
    ScalarSite("z.em.n_inducing", "z", 2, None),  # capped by the output width
    ScalarSite("z.em.metric", "z", 1.0, 4.0),
    ScalarSite("z.act.power_relu_exponent", "z", 0.1, 10.0),
    ScalarSite("z.act.power_exponent", "z", 0.1, 10.0),
    ScalarSite("z.act.rescale", "z", 1.0, 10.0),
    ScalarSite("z.matrix.kernel_scale", "z", 0.1, 10.0),
    ScalarSite("z.points.n_centers", "z", 1, 16),
    ScalarSite("z.weights.decay", "z", None, 6.0),  # low falls with the column count
    ScalarSite("z.weights.dispersion", "z", 1e-4, 10.0),
    # -- g: the collapse from latent state to observed columns ------------------------
    ScalarSite("g.column.kumaraswamy_shape", "g", 0.2, 5.0),
    ScalarSite("g.converter.softmax_temperature", "g", 0.1, 10.0, reachable=False),
    ScalarSite("g.converter.neighbor_metric", "g", 0.5, 4.0, reachable=False),
)

BY_NAME: dict[str, ScalarSite] = {site.name: site for site in SITES}
SITE_NAMES: tuple[str, ...] = tuple(BY_NAME)
REACHABLE_SITES: tuple[ScalarSite, ...] = tuple(site for site in SITES if site.reachable)

_ACTIVE: dict[str, tuple[float, float]] = {}


def resolve(name: str, low: float, high: float) -> tuple[float, float]:
    """Reparameterize one site's interval under the active coordinates.

    Returns ``(low, high)`` unchanged -- the identical floats, not merely equal ones -- when
    the site is absent from the active table or sits at its origin coordinates.
    """
    site = BY_NAME.get(name)
    if site is None:
        raise ValueError(f"unknown sampling site {name!r}")
    if site.low is not None and low != site.low:
        raise ValueError(f"{name} is registered at low={site.low!r} but the call site passed {low!r}")
    if site.high is not None and high != site.high:
        raise ValueError(f"{name} is registered at high={site.high!r} but the call site passed {high!r}")

    coordinates = _ACTIVE.get(name)
    if coordinates is None:
        return low, high
    shift_coord, spread_coord = coordinates
    halfwidth = (math.log(high) - math.log(low)) / 2.0
    shift = math.exp((2.0 * shift_coord - 1.0) * SHIFT_MAX)
    spread = math.exp((2.0 * spread_coord - 1.0) * halfwidth * SPREAD_FRAC)
    return low * shift / spread, high * shift * spread


@contextlib.contextmanager
def active_scalars(coordinates: Mapping[str, tuple[float, float]]) -> Iterator[None]:
    """Install the site table for one task, restoring whatever was there before.

    Ambient rather than threaded, deliberately and narrowly: the vendored generator already
    draws from the global numpy RNG, so ``sample_task`` already has to establish ambient
    state for the duration of a draw. This rides that same seam instead of widening twenty
    leaf signatures. It is a sibling of ``seeded_globals`` rather than a part of it, so the
    two kinds of ambient state stay separately named.
    """
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = dict(coordinates)
    try:
        yield
    finally:
        _ACTIVE = previous
