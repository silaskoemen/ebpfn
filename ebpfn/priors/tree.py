"""Tree route: stacked piecewise-constant layers (TabICL ``_tree_scm`` template).

Features are pushed through 1-2 layers of random axis-aligned regression trees;
each layer output is a mean over a small ensemble of shallow trees, z-scored with
added noise. Depth and ensemble size are drawn from bounded exponentials so
complexity stays controlled and independent of `p`. This is the most on-target
route for real tabular data (tables + GBMs are tree-structured).
"""

import numpy as np

from ebpfn.priors.contracts import RouteRealization
from ebpfn.priors.contracts import SharedTheta
from ebpfn.priors.contracts import TreeHyperPrior
from ebpfn.priors.features import sample_features
from ebpfn.priors.features import zscore

_MAX_DEPTH = 4
_MAX_ESTIMATORS = 4


def _random_tree(x: np.ndarray, depth: int, rng: np.random.Generator) -> np.ndarray:
    """One shallow axis-aligned regression tree: piecewise-constant over ``x``."""
    n, p = x.shape
    out = np.empty(n)

    def recurse(idx: np.ndarray, remaining: int) -> None:
        if remaining == 0 or idx.size <= 1:
            out[idx] = rng.standard_normal()
            return
        feature = int(rng.integers(0, p))
        column = x[idx, feature]
        threshold = float(np.median(column))
        left = idx[column <= threshold]
        right = idx[column > threshold]
        if left.size == 0 or right.size == 0:
            out[idx] = rng.standard_normal()
            return
        recurse(left, remaining - 1)
        recurse(right, remaining - 1)

    recurse(np.arange(n), depth)
    return out


def _tree_layer(
    x: np.ndarray, hidden_dim: int, hp: TreeHyperPrior, rng: np.random.Generator
) -> tuple[np.ndarray, list[int], list[int]]:
    depths: list[int] = []
    estimators: list[int] = []
    columns = np.empty((x.shape[0], hidden_dim))
    for j in range(hidden_dim):
        depth = min(_MAX_DEPTH, 2 + int(rng.exponential(1.0 / hp.max_depth_lambda)))
        n_estimators = min(_MAX_ESTIMATORS, 1 + int(rng.exponential(1.0 / hp.n_estimators_lambda)))
        depths.append(depth)
        estimators.append(n_estimators)
        ensemble = np.mean([_random_tree(x, depth, rng) for _ in range(n_estimators)], axis=0)
        columns[:, j] = zscore(ensemble.reshape(-1, 1)).ravel() + 0.1 * rng.standard_normal(x.shape[0])
    return columns, depths, estimators


def realize(hp: TreeHyperPrior, n: int, p: int, shared: SharedTheta, rng: np.random.Generator) -> RouteRealization:
    if p < 1:
        raise ValueError(f"p must be at least one, got {p}")
    x_raw = sample_features(n, p, shared.corr_strength, rng)
    n_layers = int(rng.integers(1, hp.n_layers_max + 1))
    hidden_dim = int(rng.integers(1, hp.hidden_dim_max + 1))
    hidden = x_raw
    all_depths: list[int] = []
    all_estimators: list[int] = []
    for _ in range(n_layers):
        hidden, depths, estimators = _tree_layer(hidden, hidden_dim, hp, rng)
        all_depths.extend(depths)
        all_estimators.extend(estimators)
    signal = hidden @ rng.standard_normal(hidden.shape[1])
    diagnostics = {
        "route": "tree",
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "mean_depth": float(np.mean(all_depths)),
        "mean_estimators": float(np.mean(all_estimators)),
    }
    return RouteRealization(x_raw=x_raw, signal=signal, diagnostics=diagnostics)
