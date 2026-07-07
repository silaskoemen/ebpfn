"""Compositional route: explicit additive linear/threshold/interaction terms.

An interpretable mechanism whose signal is a weighted sum of a linear term, a
threshold-indicator term, and a pairwise-interaction term over a deliberately
sized active-feature set. Inactive features are pure noise carried in ``x_raw``.
"""

import numpy as np

from ebpfn.priors.contracts import CompositionalHyperPrior
from ebpfn.priors.contracts import RouteRealization
from ebpfn.priors.contracts import SharedTheta
from ebpfn.priors.features import sample_features
from ebpfn.priors.features import zscore

_ACTIVE_DISPERSION = 0.1


def _draw_active(p: int, mean_fraction: float, rng: np.random.Generator) -> int:
    fraction = float(np.clip(rng.normal(mean_fraction, _ACTIVE_DISPERSION), 1.0 / max(p, 1), 1.0))
    return int(np.clip(round(fraction * p), 1, p))


def realize(
    hp: CompositionalHyperPrior, n: int, p: int, shared: SharedTheta, rng: np.random.Generator
) -> RouteRealization:
    if p < 1:
        raise ValueError(f"p must be at least one, got {p}")
    x_raw = sample_features(n, p, shared.corr_strength, rng)
    active_count = _draw_active(p, hp.active_fraction_mean, rng)
    active = rng.choice(p, size=active_count, replace=False)
    x_active = x_raw[:, active]

    weights = np.array([hp.linear_weight, hp.threshold_weight, hp.interaction_weight], dtype=float)
    weights = weights / weights.sum()

    beta = rng.standard_normal(active_count)
    linear = zscore((x_active @ beta).reshape(-1, 1)).ravel()

    thresholds = np.array([float(np.quantile(x_active[:, j], rng.uniform(0.25, 0.75))) for j in range(active_count)])
    threshold_term = zscore(((x_active > thresholds).sum(axis=1)).astype(float).reshape(-1, 1)).ravel()

    # Deliberate interaction burden: up to one distinct unordered pair per active
    # feature, sampled without replacement so there are no self- or duplicate pairs.
    pairs: list[tuple[int, int]] = []
    if active_count >= 2:
        candidates = [(a, b) for a in range(active_count) for b in range(a + 1, active_count)]
        n_pairs = min(active_count, len(candidates))
        chosen = rng.choice(len(candidates), size=n_pairs, replace=False)
        pairs = [candidates[int(index)] for index in chosen]
        products = sum(x_active[:, a] * x_active[:, b] for a, b in pairs)
        interaction = zscore(np.asarray(products).reshape(-1, 1)).ravel()
    else:
        interaction = np.zeros(n)
        weights = np.array([weights[0] + weights[2], weights[1], 0.0])

    signal = weights[0] * linear + weights[1] * threshold_term + weights[2] * interaction
    diagnostics = {
        "route": "compositional",
        "active_count": active_count,
        "active_fraction": float(active_count / p),
        "active_indices": [int(index) for index in active],
        "thresholds": [float(value) for value in thresholds],
        "interaction_pairs": [[int(active[a]), int(active[b])] for a, b in pairs],
        "n_interactions": len(pairs),
        "mechanism_weights": weights.tolist(),
    }
    return RouteRealization(x_raw=x_raw, signal=signal, diagnostics=diagnostics)
