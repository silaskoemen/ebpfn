"""Direct-simplex vectorization of the hyperprior.

The vectorizer maps a configurable active subset of ``eta`` coordinates to and
from a unit-hypercube vector with stable names, finite bounds, and fixed
transforms. Route weights use the direct simplex: three nonreference weights are
free and the reference route (``compositional``) takes one minus their sum;
vectors outside the simplex are rejected rather than normalized. Positive
quantities use log transforms, other bounded scalars affine transforms.
"""

import dataclasses
from typing import Any
from typing import Literal

import numpy as np

from ebpfn.priors.contracts import REFERENCE_ROUTE
from ebpfn.priors.contracts import ROUTE_ORDER
from ebpfn.priors.contracts import HyperPrior

_TOL = 1e-9
_VERSION = "eta-vectorizer-1"

TransformKind = Literal["affine", "log"]

# name -> (transform kind, natural lower bound, natural upper bound)
_COORDINATES: dict[str, tuple[TransformKind, float, float]] = {
    "w_scm": ("affine", 0.0, 1.0),
    "w_bnn": ("affine", 0.0, 1.0),
    "w_tree": ("affine", 0.0, 1.0),
    "corr_strength_mean": ("affine", 0.0, 1.0),
    "log_snr_mean": ("affine", -2.0, 3.0),
    "heteroskedastic_rate": ("affine", 0.0, 1.0),
    "heavy_tail_rate": ("affine", 0.0, 1.0),
    "scm_target_indegree_mean": ("log", 0.5, 8.0),
    "scm_weight_scale": ("log", 0.25, 4.0),
    "bnn_weight_scale": ("log", 0.25, 4.0),
    "compositional_active_fraction_mean": ("affine", 0.05, 1.0),
}

_WEIGHT_COORDINATES = ("w_scm", "w_bnn", "w_tree")

# Provisional active set (~10 coordinates). The frozen active-space decision is a
# Step 5 recovery outcome; `EtaVectorizer` accepts any active subset until then.
DEFAULT_ACTIVE: tuple[str, ...] = (
    "w_scm",
    "w_bnn",
    "w_tree",
    "corr_strength_mean",
    "log_snr_mean",
    "heteroskedastic_rate",
    "heavy_tail_rate",
    "scm_target_indegree_mean",
    "bnn_weight_scale",
    "compositional_active_fraction_mean",
)


def _forward(name: str, value: float) -> float:
    kind, lo, hi = _COORDINATES[name]
    if kind == "log":
        return (np.log(value) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (value - lo) / (hi - lo)


def _inverse(name: str, unit: float) -> float:
    kind, lo, hi = _COORDINATES[name]
    if kind == "log":
        return float(np.exp(np.log(lo) + unit * (np.log(hi) - np.log(lo))))
    return float(lo + unit * (hi - lo))


def _eta_to_natural(eta: HyperPrior) -> dict[str, float]:
    return {
        "w_scm": eta.generator_weights["scm"],
        "w_bnn": eta.generator_weights["bnn"],
        "w_tree": eta.generator_weights["tree"],
        "corr_strength_mean": eta.corr_strength_mean,
        "log_snr_mean": eta.log_snr_mean,
        "heteroskedastic_rate": eta.heteroskedastic_rate,
        "heavy_tail_rate": eta.heavy_tail_rate,
        "scm_target_indegree_mean": eta.scm.target_indegree_mean,
        "scm_weight_scale": eta.scm.weight_scale,
        "bnn_weight_scale": eta.bnn.weight_scale,
        "compositional_active_fraction_mean": eta.compositional.active_fraction_mean,
    }


def _natural_to_eta(base: HyperPrior, natural: dict[str, float]) -> HyperPrior:
    nonreference = {name: natural[f"w_{name}"] for name in ROUTE_ORDER if name != REFERENCE_ROUTE}
    weights = dict(nonreference)
    weights[REFERENCE_ROUTE] = 1.0 - sum(nonreference.values())
    return dataclasses.replace(
        base,
        generator_weights=weights,
        corr_strength_mean=natural["corr_strength_mean"],
        log_snr_mean=natural["log_snr_mean"],
        heteroskedastic_rate=natural["heteroskedastic_rate"],
        heavy_tail_rate=natural["heavy_tail_rate"],
        scm=dataclasses.replace(
            base.scm,
            target_indegree_mean=natural["scm_target_indegree_mean"],
            weight_scale=natural["scm_weight_scale"],
        ),
        bnn=dataclasses.replace(base.bnn, weight_scale=natural["bnn_weight_scale"]),
        compositional=dataclasses.replace(
            base.compositional, active_fraction_mean=natural["compositional_active_fraction_mean"]
        ),
    )


class EtaVectorizer:
    """Encode/decode an active subset of ``eta`` to a feasible unit vector."""

    def __init__(self, base: HyperPrior, active: tuple[str, ...] = DEFAULT_ACTIVE) -> None:
        if not active:
            raise ValueError("at least one active coordinate is required")
        unknown = [name for name in active if name not in _COORDINATES]
        if unknown:
            raise ValueError(f"unknown active coordinates: {unknown}")
        if len(set(active)) != len(active):
            raise ValueError("active coordinates must be unique")
        self.base = base
        self.active = tuple(active)

    @property
    def dimension(self) -> int:
        return len(self.active)

    def encode(self, eta: HyperPrior) -> np.ndarray:
        natural = _eta_to_natural(eta)
        return np.array([_forward(name, natural[name]) for name in self.active], dtype=float)

    def _decode_natural(self, vector: np.ndarray) -> dict[str, float]:
        natural = _eta_to_natural(self.base)
        for name, unit in zip(self.active, vector, strict=True):
            natural[name] = _inverse(name, float(unit))
        return natural

    def decode(self, vector: np.ndarray) -> HyperPrior:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError("vector dimension does not match the active coordinates")
        if not self.is_feasible(vector):
            raise ValueError("vector is infeasible and cannot be decoded")
        return _natural_to_eta(self.base, self._decode_natural(vector))

    def is_feasible(self, vector: np.ndarray) -> bool:
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,):
            return False
        if not np.all(np.isfinite(vector)) or np.any(vector < -_TOL) or np.any(vector > 1.0 + _TOL):
            return False
        natural = self._decode_natural(vector)
        weight_sum = sum(natural[name] for name in _WEIGHT_COORDINATES)
        return bool(weight_sum <= 1.0 + _TOL)

    def sobol(self, m: int, rng: np.random.Generator) -> np.ndarray:
        """Return up to ``m`` feasible unit vectors from a scrambled Sobol design."""
        if m < 1:
            raise ValueError("m must be at least one")
        from scipy.stats import qmc  # lazy: keeps `import ebpfn` free of scipy

        sampler = qmc.Sobol(d=self.dimension, scramble=True, seed=rng)
        draw = int(2 ** np.ceil(np.log2(max(8 * m, 8))))
        points = sampler.random(n=draw)
        feasible = points[np.array([self.is_feasible(point) for point in points])]
        return feasible[:m]

    def schema(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "route_order": list(ROUTE_ORDER),
            "reference_route": REFERENCE_ROUTE,
            "active": list(self.active),
            "coordinates": {
                name: {"transform": _COORDINATES[name][0], "bounds": [_COORDINATES[name][1], _COORDINATES[name][2]]}
                for name in self.active
            },
        }
