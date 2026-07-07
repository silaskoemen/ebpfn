"""SCM route: a random-DAG structural causal model.

Ported from the Gate-era ``ScmDgp`` with one load-bearing change for
`p`-stability: a node's parent probability is ``target_indegree / #candidates``
so expected indegree stays ~constant as `p` grows (the Gate-era fixed
``edge_prob`` gave indegree growing with node position). Weights are fan-in
scaled and every assembled node is z-scored so depth does not blow up scale.
"""

import numpy as np

from ebpfn.priors.contracts import RouteRealization
from ebpfn.priors.contracts import ScmHyperPrior
from ebpfn.priors.contracts import SharedTheta
from ebpfn.priors.features import activation
from ebpfn.priors.features import zscore


def _parents(m: int, target_indegree: float, max_parents: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Parent index arrays per node in topological order.

    Node ``j`` includes each earlier node with probability
    ``min(1, target_indegree / j)`` so the expected parent count is
    ``min(j, target_indegree)`` regardless of the total node count. The terminal
    node is forced to have at least one parent so the target is never pure noise.
    """
    parents: list[np.ndarray] = [np.array([], dtype=int)]
    for j in range(1, m):
        prob = min(1.0, target_indegree / j)
        candidates = np.flatnonzero(rng.random(j) < prob)
        if candidates.size > max_parents:
            candidates = rng.choice(candidates, size=max_parents, replace=False)
        if j == m - 1 and candidates.size == 0:
            candidates = np.array([int(rng.integers(0, j))])
        parents.append(candidates)
    return parents


def realize(hp: ScmHyperPrior, n: int, p: int, shared: SharedTheta, rng: np.random.Generator) -> RouteRealization:
    if p < 1:
        raise ValueError(f"p must be at least one, got {p}")
    _ = shared
    # p observed nodes + n_hidden latent confounders + one reserved target node.
    m = p + hp.n_hidden + 1
    parents = _parents(m, hp.target_indegree_mean, hp.max_parents, rng)

    values = np.empty((n, m))
    indegrees: list[int] = []
    nonlinear = 0
    for j in range(m):
        pa = parents[j]
        if pa.size == 0:
            values[:, j] = rng.standard_normal(n)
            continue
        indegrees.append(int(pa.size))
        act_name = "tanh" if rng.random() < hp.nonlinear_prob else "linear"
        nonlinear += act_name == "tanh"
        act = activation(act_name)
        weights = rng.normal(0.0, hp.weight_scale, size=pa.size) / np.sqrt(pa.size)
        pre = values[:, pa] @ weights
        values[:, j] = zscore((act(pre) + 0.3 * rng.standard_normal(n)).reshape(-1, 1)).ravel()

    y_node = m - 1
    pa_y = parents[y_node]
    weights_y = rng.normal(0.0, hp.weight_scale, size=pa_y.size) / np.sqrt(pa_y.size)
    act_y = activation("tanh" if rng.random() < hp.nonlinear_prob else "linear")
    signal = act_y(values[:, pa_y] @ weights_y)

    parent_set = {int(node) for node in pa_y.tolist()}
    rest = np.array([j for j in range(m) if j != y_node and j not in parent_set], dtype=int)
    # Parents-first placement: every direct cause of y that fits lands in the
    # observed set, so (when |pa_y| <= p) y is identifiable from X and the terminal
    # SNR knob alone sets the ceiling; the n_hidden nodes only drive feature
    # correlation, never confound y. This is a MODE, not a law -- a "free" mode
    # (features at arbitrary nodes, TabICL-style) would admit genuine unobserved
    # confounding but void SNR calibration. Deferred; see plans/v1/decisions.md D5.
    ordered = np.concatenate([rng.permutation(pa_y), rng.permutation(rest)])
    x_nodes = ordered[:p]
    # Observed features that are direct causes of the target (its realized active set).
    active_indices = [i for i, node in enumerate(x_nodes.tolist()) if node in parent_set]
    diagnostics = {
        "route": "scm",
        "mean_indegree": float(np.mean(indegrees)) if indegrees else 0.0,
        "nonlinear_fraction": float(nonlinear / max(1, m - 1)),
        "n_nodes": int(m),
        "n_hidden": int(hp.n_hidden),
        "active_indices": active_indices,
        "n_active": len(active_indices),
    }
    return RouteRealization(x_raw=values[:, x_nodes].copy(), signal=signal.copy(), diagnostics=diagnostics)
