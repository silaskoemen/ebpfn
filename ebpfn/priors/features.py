"""Numeric feature processes and the observation step.

Features combine independent noise with a controlled low-rank latent-factor
structure so cross-feature correlation is a deliberate task-level knob rather than
an accident of `p`. The observation step maps latent `X_raw` to the observed
`X_obs` the characterizer sees; proxy and MAR mechanisms replace noise columns
rather than growing `p`.
"""

from typing import Any

import numpy as np

_ACTIVATIONS = {
    "linear": lambda x: x,
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0.0, x),
}


def activation(name: str):
    if name not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {name!r}")
    return _ACTIVATIONS[name]


def zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Column-wise zero-mean unit-variance; keeps deep propagation stable."""
    return (values - values.mean(axis=0)) / (values.std(axis=0) + eps)


def latent_rank(p: int) -> int:
    """A controlled, slowly growing latent-factor count (spectrum stays bounded)."""
    return int(max(1, min(p, round(np.sqrt(p)))))


def sample_features(n: int, p: int, corr_strength: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance columns with correlation ~ ``corr_strength`` via shared factors.

    ``X = sqrt(corr) * (F @ L) + sqrt(1 - corr) * Z``, where each factor loading
    column is unit-norm so both terms have unit column variance and the induced
    cross-correlation scales with ``corr_strength`` at a fixed latent rank.
    """
    if p < 1:
        raise ValueError(f"p must be at least one, got {p}")
    independent = rng.standard_normal((n, p))
    corr = float(np.clip(corr_strength, 0.0, 1.0))
    if corr <= 0.0:
        return independent
    k = latent_rank(p)
    factors = rng.standard_normal((n, k))
    loadings = rng.standard_normal((k, p))
    loadings /= np.linalg.norm(loadings, axis=0, keepdims=True) + 1e-12
    correlated = factors @ loadings
    return np.sqrt(corr) * correlated + np.sqrt(1.0 - corr) * independent


def apply_observation(x_raw: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map latent features to observed features.

    V1 primary observation is the identity (``X_obs = X_raw``); proxy-column and
    MAR mechanisms slot in here without changing callers. Returns the observed
    matrix, per-column missing rates, and an observation-state diagnostic.
    """
    _ = rng
    x_obs = np.array(x_raw, dtype=float, copy=True)
    missing_rates = np.zeros(x_obs.shape[1], dtype=float)
    state = {"proxy_columns": 0, "mar": False}
    return x_obs, missing_rates, state
