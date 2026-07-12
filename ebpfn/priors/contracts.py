"""Frozen contracts for the hierarchical synthetic prior.

`HyperPrior` (`eta`) is the compact object the vectorizer round-trips and the
generator samples from. Per-route hyperpriors hold the means/rates `eta`
controls; task-level `theta` is drawn around them with fixed dispersion. Realized
route, parameters, and diagnostics live on `GeneratedTask.diagnostics`, never on
the `TuningTask` the characterizer sees.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

from ebpfn.data import TuningTask

RouteName: TypeAlias = Literal["scm", "bnn", "tree", "compositional"]

# Order is part of the vectorizer schema; the reference route's weight is derived
# as one minus the three nonreference weights and is never optimized directly.
ROUTE_ORDER: tuple[RouteName, ...] = ("scm", "bnn", "tree", "compositional")
REFERENCE_ROUTE: RouteName = "compositional"

_SIMPLEX_TOL = 1e-9


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ScmHyperPrior:
    """Random-DAG structural causal model; expected indegree is p-stable."""

    target_indegree_mean: float
    n_hidden: int
    max_parents: int
    weight_scale: float
    nonlinear_prob: float

    def __post_init__(self) -> None:
        _require(self.target_indegree_mean > 0.0, "target_indegree_mean must be positive")
        _require(self.n_hidden >= 1, "n_hidden must be at least one")
        _require(self.max_parents >= 1, "max_parents must be at least one")
        _require(self.weight_scale > 0.0, "weight_scale must be positive")
        _require(0.0 <= self.nonlinear_prob <= 1.0, "nonlinear_prob must be a probability")


@dataclass(frozen=True)
class BnnHyperPrior:
    """Random MLP prior over independent Gaussian features; fan-in scaled."""

    n_layers: int
    hidden: int
    weight_scale: float
    nonlinear_prob: float

    def __post_init__(self) -> None:
        _require(self.n_layers >= 1, "n_layers must be at least one")
        _require(self.hidden >= 1, "hidden must be at least one")
        _require(self.weight_scale > 0.0, "weight_scale must be positive")
        _require(0.0 <= self.nonlinear_prob <= 1.0, "nonlinear_prob must be a probability")


@dataclass(frozen=True)
class TreeHyperPrior:
    """Stacked piecewise-constant tree layers (TabICL-style depth/estimators)."""

    n_layers_max: int
    hidden_dim_max: int
    max_depth_lambda: float
    n_estimators_lambda: float

    def __post_init__(self) -> None:
        _require(self.n_layers_max >= 1, "n_layers_max must be at least one")
        _require(self.hidden_dim_max >= 1, "hidden_dim_max must be at least one")
        _require(self.max_depth_lambda > 0.0, "max_depth_lambda must be positive")
        _require(self.n_estimators_lambda > 0.0, "n_estimators_lambda must be positive")


@dataclass(frozen=True)
class CompositionalHyperPrior:
    """Explicit additive linear/threshold/interaction mechanisms."""

    linear_weight: float
    threshold_weight: float
    interaction_weight: float
    active_fraction_mean: float

    def __post_init__(self) -> None:
        weights = (self.linear_weight, self.threshold_weight, self.interaction_weight)
        _require(all(w >= 0.0 for w in weights), "mechanism weights must be nonnegative")
        _require(sum(weights) > 0.0, "at least one mechanism weight must be positive")
        _require(0.0 < self.active_fraction_mean <= 1.0, "active_fraction_mean must be in (0, 1]")


@dataclass(frozen=True)
class HyperPrior:
    """The compact hyperprior ``eta``."""

    generator_weights: dict[str, float]
    corr_strength_mean: float
    log_snr_mean: float
    heteroskedastic_rate: float
    heavy_tail_rate: float
    snr_dispersion: float
    corr_dispersion: float
    scm: ScmHyperPrior
    bnn: BnnHyperPrior
    tree: TreeHyperPrior
    compositional: CompositionalHyperPrior

    def __post_init__(self) -> None:
        if tuple(sorted(self.generator_weights)) != tuple(sorted(ROUTE_ORDER)):
            raise ValueError("generator_weights must be keyed by the four route names")
        weights = np.array([self.generator_weights[name] for name in ROUTE_ORDER], dtype=float)
        _require(bool(np.all(weights >= -_SIMPLEX_TOL)), "generator weights must be nonnegative")
        _require(abs(float(weights.sum()) - 1.0) <= 1e-6, "generator weights must sum to one")
        _require(0.0 <= self.corr_strength_mean <= 1.0, "corr_strength_mean must be a fraction")
        _require(0.0 <= self.heteroskedastic_rate <= 1.0, "heteroskedastic_rate must be a probability")
        _require(0.0 <= self.heavy_tail_rate <= 1.0, "heavy_tail_rate must be a probability")
        _require(self.snr_dispersion > 0.0, "snr_dispersion must be positive")
        _require(self.corr_dispersion > 0.0, "corr_dispersion must be positive")
        object.__setattr__(
            self, "generator_weights", {name: float(self.generator_weights[name]) for name in ROUTE_ORDER}
        )

    def weight_vector(self) -> np.ndarray:
        return np.array([self.generator_weights[name] for name in ROUTE_ORDER], dtype=float)


def hyperprior_to_dict(eta: HyperPrior) -> dict[str, Any]:
    """Serialize an eta for exact artifact and checkpoint handoffs."""
    return asdict(eta)


def hyperprior_from_dict(payload: dict[str, Any]) -> HyperPrior:
    """Reconstruct an eta from its exact serialized representation."""
    return HyperPrior(
        generator_weights={str(name): float(weight) for name, weight in payload["generator_weights"].items()},
        corr_strength_mean=payload["corr_strength_mean"],
        log_snr_mean=payload["log_snr_mean"],
        heteroskedastic_rate=payload["heteroskedastic_rate"],
        heavy_tail_rate=payload["heavy_tail_rate"],
        snr_dispersion=payload["snr_dispersion"],
        corr_dispersion=payload["corr_dispersion"],
        scm=ScmHyperPrior(**payload["scm"]),
        bnn=BnnHyperPrior(**payload["bnn"]),
        tree=TreeHyperPrior(**payload["tree"]),
        compositional=CompositionalHyperPrior(**payload["compositional"]),
    )


@dataclass(frozen=True)
class SharedTheta:
    """Task-level values drawn from ``eta`` and shared across the route mechanism."""

    route: RouteName
    log_snr: float
    corr_strength: float
    heteroskedastic: bool
    heavy_tail: bool


@dataclass(frozen=True)
class RouteRealization:
    """A route's latent features and its noiseless signal, before target noise."""

    x_raw: np.ndarray
    signal: np.ndarray
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.x_raw.ndim != 2:
            raise ValueError("x_raw must be two-dimensional")
        if self.signal.ndim != 1 or self.signal.shape[0] != self.x_raw.shape[0]:
            raise ValueError("signal must align with x_raw rows")
        if not np.isfinite(self.x_raw).all() or not np.isfinite(self.signal).all():
            raise ValueError("route realization must be finite")


@dataclass(frozen=True)
class GeneratedTask:
    """The generator output: a task the characterizer sees plus hidden diagnostics."""

    tuning: TuningTask
    diagnostics: dict[str, Any]
