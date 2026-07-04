"""Data-generating constructions (spec §1).

Construction A (primary): cross-slice conditional swap with exact marginal match.
Construction B (secondary): heteroskedastic real vs homoskedastic decoy.

A *task* is one i.i.d. dataset drawn from a prior with randomized (beta, gamma,
delta, band width). Only the conditional *noise shape* distinguishes real/decoy.

Parameter draws (`draw_params`) are separated from realization (`realize`) so the
calibration setup can build shared-`f` triples (real-train, decoy-train, real-test
with identical conditional mean) — the swap's miscalibration only shows up when
the mean is held fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ebpfn.config import DataConfig, Prior


@dataclass(frozen=True)
class Dataset:
    """A single task. X: (n, d) covariates; Y: (n,) targets."""

    X: np.ndarray
    Y: np.ndarray

    def __post_init__(self) -> None:
        if self.X.ndim != 2 or self.Y.ndim != 1 or self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"shape mismatch: X {self.X.shape}, Y {self.Y.shape}")

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]


@dataclass(frozen=True)
class TaskParams:
    """Per-task randomized knobs shared by a real/decoy pair to fix f and geometry."""

    beta: float
    gamma: float
    delta: float
    w: float  # Construction A band width


def f_mean(X: np.ndarray, beta: float, gamma: float, delta: float) -> np.ndarray:
    """Conditional mean f(x) = beta*x1 + gamma*sin(delta*x1), x1 only (spec §1)."""
    x1 = X[:, 0]
    return beta * x1 + gamma * np.sin(delta * x1)


def draw_params(dc: DataConfig, rng: np.random.Generator) -> TaskParams:
    """Draw the per-task randomized parameters."""
    w = dc.band_width
    if dc.band_width_jitter > 0:
        w += rng.uniform(-dc.band_width_jitter, dc.band_width_jitter)
    return TaskParams(
        beta=rng.uniform(*dc.beta_range),
        gamma=rng.uniform(*dc.gamma_range),
        delta=rng.uniform(*dc.delta_range),
        w=w,
    )


def _band_masks_a(x2: np.ndarray, prior: Prior, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (in_a, in_b) for the two mirror-symmetric A bands.

    'fixed_edge' (registered run): bands at x2 in [s/2, s/2+w] and its mirror; the
    sweep value s is a position in x2, so per-band mass falls as s grows (the s axis
    is confounded with mass).

    'fixed_mass': bands defined by a CDF interval of width m = band_mass at a symmetric
    quantile gap g = sweep_value. Band A spans CDF [0.5+g/2, 0.5+g/2+m] -> x2 in
    [ppf(0.5+g/2), ppf(0.5+g/2+m)]; band B is its exact mirror. Per-band mass is m for
    any g, so the sweep moves feature-separation alone (mass held constant).
    """
    dc = prior.data
    if dc.band_geometry == "fixed_edge":
        s = prior.sweep_value
        in_a = (x2 >= s / 2) & (x2 <= s / 2 + w)
        in_b = (x2 <= -s / 2) & (x2 >= -s / 2 - w)
        return in_a, in_b
    # fixed_mass
    g, m = prior.sweep_value, dc.band_mass
    if not (0.0 <= g <= 1.0 - 2.0 * m):
        raise ValueError(f"fixed_mass needs 0 <= g <= 1-2m; got g={g}, m={m} (max g={1.0 - 2.0 * m})")
    a_lo, a_hi = norm.ppf(0.5 + g / 2), norm.ppf(0.5 + g / 2 + m)
    in_a = (x2 >= a_lo) & (x2 <= a_hi)
    in_b = (x2 >= -a_hi) & (x2 <= -a_lo)  # exact mirror of A across x2 = 0
    return in_a, in_b


def _sigma_construction_a(X: np.ndarray, prior: Prior, w: float) -> np.ndarray:
    """Per-row noise width for the cross-slice swap.

    Two mirror-symmetric bands (see `_band_masks_a`) with equal mass and identical
    f-distribution. Real assigns (A->hi, B->lo); decoy swaps them. Marginal P(Y) is
    invariant under the swap.
    """
    dc = prior.data
    x2 = X[:, 1]
    in_a, in_b = _band_masks_a(x2, prior, w)
    sigma = np.full(X.shape[0], dc.sigma0)
    hi, lo = (dc.sigma_hi, dc.sigma_lo) if prior.role == "real" else (dc.sigma_lo, dc.sigma_hi)
    sigma[in_a] = hi
    sigma[in_b] = lo
    return sigma


def _sigma_construction_b(X: np.ndarray, prior: Prior) -> np.ndarray:
    """Per-row noise width for hetero (real) vs homoskedastic (decoy).

    Real: sigma(x) = sigma0 * exp(kappa * x2). Decoy: constant sigma_bar with
    sigma_bar^2 = E_x[sigma(x)^2] = sigma0^2 * exp(2*kappa^2) for x2 ~ N(0,1),
    so the marginal Var(Y) matches to 2nd order.
    """
    dc = prior.data
    kappa = prior.sweep_value
    x2 = X[:, 1]
    if prior.role == "real":
        return dc.sigma0 * np.exp(kappa * x2)
    sigma_bar = dc.sigma0 * np.exp(kappa**2)  # sqrt(sigma0^2 * exp(2 kappa^2))
    return np.full(X.shape[0], sigma_bar)


def realize(prior: Prior, params: TaskParams, n: int, rng: np.random.Generator) -> Dataset:
    """Materialize one dataset for `prior` at fixed `params` (fresh X and noise)."""
    dc = prior.data
    if dc.d < 2:
        raise ValueError("constructions A and B both use x2; need d >= 2")
    X = rng.standard_normal((n, dc.d))
    mu = f_mean(X, params.beta, params.gamma, params.delta)
    if prior.construction == "A":
        sigma = _sigma_construction_a(X, prior, params.w)
    else:
        sigma = _sigma_construction_b(X, prior)
    Y = mu + sigma * rng.standard_normal(n)
    return Dataset(X=X, Y=Y)


def sample_task(prior: Prior, rng: np.random.Generator, n: int | None = None) -> Dataset:
    """Draw one task from `prior`. n defaults to a per-task draw in [n_min, n_max]."""
    dc = prior.data
    if n is None:
        n = int(rng.integers(dc.n_min, dc.n_max + 1))
    return realize(prior, draw_params(dc, rng), n, rng)


def sample_cloud(prior: Prior, n_tasks: int, rng: np.random.Generator, n: int | None = None) -> list[Dataset]:
    """Draw a cloud of `n_tasks` i.i.d. tasks from a prior."""
    return [sample_task(prior, rng, n=n) for _ in range(n_tasks)]


def pool(datasets: list[Dataset]) -> Dataset:
    """Concatenate datasets into one (for prior-level pooled statistics)."""
    return Dataset(
        X=np.vstack([d.X for d in datasets]),
        Y=np.concatenate([d.Y for d in datasets]),
    )
