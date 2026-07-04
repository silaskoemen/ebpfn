"""DGP family for the EB-PFN prior (plans/gate1_revised.md §3.1).

A DGP samples one *task* -- a single (X, y) dataset -- from a data-generating
process. The family is pluggable: SCM mechanisms (linear / nonlinear) and a BNN
prior are implemented now; a tree-shaped / piecewise-constant `TreeDGP` is the
intended drop-in -- varying the DGP family is the across-generator coverage axis
(H2) and tree DGPs are the most on-target prior for real tabular tasks (real
tables + GBMs are tree-structured), which is the EB coverage hypothesis itself.

Every DGP returns the shared `Dataset`, so the Gate-0 distance / calibration code
is reused unchanged. All randomness flows through an explicit numpy Generator, so
a (DGP, seed) pair fully determines the task (tested).
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from ebpfn.priors import Dataset

_ACTIVATIONS = {
    "linear": lambda x: x,
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0.0, x),
}


def _activation(name: str):
    if name not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {name!r}")
    return _ACTIVATIONS[name]


def _zscore(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-column unit-variance, zero-mean -- keeps deep propagation stable."""
    return (v - v.mean(axis=0)) / (v.std(axis=0) + eps)


class DGP(ABC):
    """A data-generating process: sample(n, d, rng) -> one task with d features."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def sample(self, n: int, d: int, rng: np.random.Generator) -> Dataset:
        """One task: X (n, d) covariates, Y (n,) target."""


@dataclass(frozen=True)
class ScmDgp(DGP):
    """Random structural causal model over a DAG of scalar nodes.

    A DAG is sampled over `d + n_hidden` nodes in topological order; each
    non-root node is activation(sum_parents w*parent) + Gaussian noise, with
    edge weights ~ N(0, weight_scale^2) scaled by 1/sqrt(#parents) to keep
    pre-activations ~ O(1). Every node is z-scored after assembly so depth does
    not blow up scale. `d` nodes are observed as X and the most-downstream node
    is the target Y; Y's direct parents are forced into X so Y genuinely depends
    on the observed features (learnability is then confirmed empirically).

    activation='linear' is the SCM-linear member; 'tanh'/'relu' are SCM-MLP.
    """

    activation: str = "tanh"
    n_hidden: int = 4  # latent nodes beyond the d observed (>=1 so a node is free to be Y)
    edge_prob: float = 0.5  # P(an earlier node is a parent of a later node)
    max_parents: int = 4
    weight_scale: float = 1.0
    noise_scale: float = 0.3

    def __post_init__(self) -> None:
        _activation(self.activation)
        if self.n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1 (need a non-feature node for Y), got {self.n_hidden}")
        if not (0.0 <= self.edge_prob <= 1.0):
            raise ValueError(f"edge_prob must be in [0, 1], got {self.edge_prob}")
        if self.max_parents < 1:
            raise ValueError(f"max_parents must be >= 1, got {self.max_parents}")

    @property
    def name(self) -> str:
        return f"scm_{self.activation}"

    def _parents(self, M: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Parent index arrays per node in topological order; node M-1 (=Y) is
        forced to have >=1 parent so the target is never pure noise."""
        parents: list[np.ndarray] = [np.array([], dtype=int)]  # node 0 is always a root
        for j in range(1, M):
            mask = rng.random(j) < self.edge_prob
            pa = np.flatnonzero(mask)
            if pa.size > self.max_parents:
                pa = rng.choice(pa, size=self.max_parents, replace=False)
            if j == M - 1 and pa.size == 0:
                pa = np.array([int(rng.integers(0, j))])
            parents.append(pa)
        return parents

    def sample(self, n: int, d: int, rng: np.random.Generator) -> Dataset:
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        act = _activation(self.activation)
        M = d + self.n_hidden
        parents = self._parents(M, rng)

        values = np.empty((n, M))
        for j in range(M):
            pa = parents[j]
            if pa.size == 0:
                values[:, j] = rng.standard_normal(n)
                continue
            w = rng.normal(0.0, self.weight_scale, size=pa.size) / np.sqrt(pa.size)
            lin = values[:, pa] @ w
            values[:, j] = act(lin) + self.noise_scale * rng.standard_normal(n)
            values[:, j] = _zscore(values[:, j])

        y_node = M - 1  # most downstream; guaranteed >=1 parent
        # X: Y's parents first (guarantees dependence), then fill from the rest.
        pa_y = parents[y_node]
        rest = np.array([j for j in range(M) if j != y_node and j not in set(pa_y.tolist())], dtype=int)
        ordered = np.concatenate([rng.permutation(pa_y), rng.permutation(rest)])
        x_nodes = ordered[:d]
        return Dataset(X=values[:, x_nodes].copy(), Y=values[:, y_node].copy())


@dataclass(frozen=True)
class BnnDgp(DGP):
    """Bayesian-neural-net prior: X ~ N(0, I_d) pushed through a random MLP.

    Unlike the SCM, features are independent Gaussians (no causal graph among
    them) and the conditional mean is one random network, giving a smoother,
    globally-coupled f(x). Weights use fan-in scaling so activations stay ~O(1).
    """

    n_layers: int = 2
    hidden: int = 16
    activation: str = "tanh"
    weight_scale: float = 1.0
    noise_scale: float = 0.3

    def __post_init__(self) -> None:
        _activation(self.activation)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {self.hidden}")

    @property
    def name(self) -> str:
        return "bnn"

    def sample(self, n: int, d: int, rng: np.random.Generator) -> Dataset:
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        act = _activation(self.activation)
        X = rng.standard_normal((n, d))
        h, fan_in = X, d
        for _ in range(self.n_layers):
            W = rng.standard_normal((fan_in, self.hidden)) * (self.weight_scale / np.sqrt(fan_in))
            h = act(h @ W)
            fan_in = self.hidden
        w_out = rng.standard_normal(fan_in) * (self.weight_scale / np.sqrt(fan_in))
        mu = _zscore(h @ w_out)
        Y = mu + self.noise_scale * rng.standard_normal(n)
        return Dataset(X=X, Y=Y)
