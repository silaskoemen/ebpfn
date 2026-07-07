"""Target realization: SNR recalibration, heteroskedasticity, and heavy tails.

The route supplies a noiseless ``signal``; here it is z-scored to unit variance
and corrupted with noise scaled so the realized signal-to-noise ratio matches the
task-level ``exp(log_snr)`` drawn from ``eta`` (spec: realized signal is
recalibrated to sampled SNR *after* mechanism construction).
"""

import numpy as np

from ebpfn.priors.contracts import SharedTheta
from ebpfn.priors.features import zscore

_HEAVY_TAIL_DF = 3.0


def realize_target(
    signal: np.ndarray, x_raw: np.ndarray, shared: SharedTheta, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the noisy target and SNR diagnostics for one task."""
    unit_signal = zscore(signal.reshape(-1, 1)).ravel()
    snr = float(np.exp(shared.log_snr))
    noise_std = float(np.sqrt(1.0 / snr))

    if shared.heavy_tail:
        base = rng.standard_t(_HEAVY_TAIL_DF, size=unit_signal.shape[0])
        base /= np.sqrt(_HEAVY_TAIL_DF / (_HEAVY_TAIL_DF - 2.0))  # unit variance
    else:
        base = rng.standard_normal(unit_signal.shape[0])

    if shared.heteroskedastic:
        driver = np.abs(x_raw[:, 0]) if x_raw.shape[1] else np.zeros_like(base)
        scale = 0.2 + driver
        scale /= np.sqrt(float(np.mean(scale**2)) + 1e-12)  # keep average variance one
        base = base * scale

    noise = noise_std * base
    target = unit_signal + noise
    realized_snr = float(np.var(unit_signal) / (np.var(noise) + 1e-12))
    diagnostics = {
        "target_snr": snr,
        "realized_snr": realized_snr,
        "noise_std": noise_std,
        "heteroskedastic": float(shared.heteroskedastic),
        "heavy_tail": float(shared.heavy_tail),
    }
    return target, diagnostics
