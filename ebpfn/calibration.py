"""Calibration report under fixed P(X) (spec §3.3).

Head-agnostic: works off any fitted `Predictive` (Gaussian or quantile). Reports
NLL (primary) and CRPS as proper scores, PIT non-uniformity (KS vs Uniform), and
central-interval coverage at the requested levels.
"""

from __future__ import annotations

import numpy as np

from ebpfn.config import CalibConfig
from ebpfn.priors import Dataset
from ebpfn.regressor import ProbModel


def _ks_uniform(u: np.ndarray) -> float:
    """KS distance of samples u to Uniform(0,1)."""
    us = np.sort(u)
    n = us.size
    ecdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(ecdf - us)))


def calibration_report(model: ProbModel, D_test: Dataset, config: CalibConfig | None = None) -> dict:
    """nll, crps, pit_stat, coverage@levels on a test task (spec §6 signature)."""
    config = config or CalibConfig()
    pred = model.predictive(D_test.X)
    y = D_test.Y

    coverage = {}
    for p in config.coverage_levels:
        lo = pred.quantile((1 - p) / 2)
        hi = pred.quantile((1 + p) / 2)
        coverage[p] = float(np.mean((y >= lo) & (y <= hi)))

    return {
        "nll": float(pred.nll(y).mean()),
        "crps": float(pred.crps(y).mean()),
        "pit_stat": _ks_uniform(pred.cdf(y)),
        "coverage": coverage,
    }
