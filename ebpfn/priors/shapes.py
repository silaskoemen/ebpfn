"""Anchored mean-one shape jitter for PFN training.

For one complete real-domain anchor ``(n_anchor, p_anchor)``, draw training
shapes with mean-one lognormal factors so the anchor is the distribution mean.
Baseline and tuned PFNs call this identical sampler; Step 4 objective evaluation
does not jitter (it matches the real shape exactly).
"""

import numpy as np

from ebpfn.config.prior import ShapeJitterConfig
from ebpfn.data import CharacterizationShape


def _mean_one_lognormal(sigma: float, z: float) -> float:
    return float(np.exp(sigma * z - sigma**2 / 2.0))


def sample_training_shape(
    anchor: CharacterizationShape, jitter: ShapeJitterConfig, rng: np.random.Generator
) -> tuple[CharacterizationShape, dict[str, float]]:
    """Jitter an anchor into a training shape, preserving the fit/score ratio."""
    if anchor.p_categorical != 0:
        raise ValueError("primary V1 anchors must have no categorical features")
    n_anchor = anchor.n_probe_fit + anchor.n_probe_score
    p_anchor = anchor.p_numeric

    z_n, z_p = rng.standard_normal(2)
    j_n = _mean_one_lognormal(jitter.sigma_n, float(z_n))
    j_p = _mean_one_lognormal(jitter.sigma_p, float(z_p))

    n_train = int(np.clip(round(n_anchor * j_n), jitter.n_min, jitter.n_max))
    p_train = int(np.clip(round(p_anchor * j_p), jitter.p_min, jitter.p_max))

    fit_fraction = anchor.n_probe_fit / n_anchor
    n_fit = int(np.clip(round(n_train * fit_fraction), 1, n_train - 1))
    n_score = n_train - n_fit

    realized = {
        "j_n": j_n,
        "j_p": j_p,
        "n_train": float(n_fit + n_score),
        "p_train": float(p_train),
        "n_probe_fit": float(n_fit),
        "n_probe_score": float(n_score),
    }
    shape = CharacterizationShape(
        n_probe_fit=n_fit,
        n_probe_score=n_score,
        p_numeric=p_train,
        p_categorical=0,
        task_type=anchor.task_type,
    )
    return shape, realized
