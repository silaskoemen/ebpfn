"""Predictive metrics for the bar-distribution PFN.

NLL, CRPS, point error (MAE/RMSE) and interval coverage all come from the single
predictive density, so they are mutually consistent. Metrics are computed in the
model's standardized target space (each task standardized on its own ``probe_fit``);
point error can be reported on the raw scale via :func:`to_raw` when a scale is supplied.
"""

import dataclasses

import torch
from torch import Tensor

from ebpfn.pfn.distribution import BarDistribution

_DEFAULT_COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)


@dataclasses.dataclass(frozen=True)
class PredictiveMetrics:
    nll: float
    crps: float
    mae: float
    rmse: float
    coverage: dict[str, float]  # nominal central-interval level -> empirical coverage
    n: int


def to_raw(values: Tensor, target_mean: Tensor, target_std: Tensor) -> Tensor:
    """Map standardized predictions/targets back to the original target scale."""
    while target_mean.ndim < values.ndim:
        target_mean = target_mean.unsqueeze(-1)
        target_std = target_std.unsqueeze(-1)
    return values * target_std + target_mean


def compute_metrics(
    distribution: BarDistribution,
    logits: Tensor,
    y_std: Tensor,
    *,
    coverage_levels: tuple[float, ...] = _DEFAULT_COVERAGE_LEVELS,
) -> PredictiveMetrics:
    """Aggregate predictive metrics over every (row) prediction in ``logits``/``y_std``.

    ``logits`` is ``(..., n_bins)`` and ``y_std`` is ``(...)`` in standardized space.
    Interval coverage uses central intervals: for nominal ``p`` the interval is
    ``[icdf((1-p)/2), icdf((1+p)/2)]``.
    """
    flat_logits = logits.reshape(-1, distribution.n_bins)
    flat_y = y_std.reshape(-1)

    nll = distribution.nll(flat_logits, flat_y).mean()
    crps = distribution.crps(flat_logits, flat_y).mean()
    mean = distribution.mean(flat_logits)
    errors = mean - flat_y
    mae = errors.abs().mean()
    rmse = torch.sqrt((errors**2).mean())

    coverage: dict[str, float] = {}
    for level in coverage_levels:
        lower = distribution.icdf(flat_logits, (1.0 - level) / 2.0)
        upper = distribution.icdf(flat_logits, (1.0 + level) / 2.0)
        inside = (flat_y >= lower) & (flat_y <= upper)
        coverage[f"{level:g}"] = float(inside.float().mean())

    return PredictiveMetrics(
        nll=float(nll),
        crps=float(crps),
        mae=float(mae),
        rmse=float(rmse),
        coverage=coverage,
        n=int(flat_y.numel()),
    )
