import math

import pytest
import torch
from ebpfn.pfn.metrics import compute_metrics, compute_row_metrics, to_raw
from ebpfn.pfn.quantile import QuantileDistribution, default_levels


def _standard_normal_quantiles(levels: torch.Tensor, n: int) -> torch.Tensor:
    """Outputs of a head that predicts N(0,1) exactly, one row per observation."""
    return (math.sqrt(2.0) * torch.erfinv(2.0 * levels.double() - 1.0)).float().unsqueeze(0).expand(n, -1)


def test_calibrated_model_is_well_calibrated() -> None:
    distribution = QuantileDistribution(default_levels(999))
    n = 20_000
    logits = _standard_normal_quantiles(distribution.levels, n)
    y = torch.randn(n)
    metrics = compute_metrics(distribution, logits, y)

    # NLL ~ differential entropy of N(0,1); coverage ~ nominal on a matched model.
    assert metrics.nll == pytest.approx(0.5 * math.log(2 * math.pi * math.e), abs=0.02)
    assert metrics.n == n
    assert metrics.rmse == pytest.approx(1.0, abs=0.05)
    for level, empirical in metrics.coverage.items():
        assert empirical == pytest.approx(float(level), abs=0.03)


def test_to_raw_backtransforms() -> None:
    values = torch.tensor([0.0, 1.0, -1.0])
    raw = to_raw(values, target_mean=torch.tensor(5.0), target_std=torch.tensor(2.0))
    assert torch.allclose(raw, torch.tensor([5.0, 7.0, 3.0]))


def test_to_raw_applies_batched_task_statistics_by_row() -> None:
    values = torch.tensor([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]])
    raw = to_raw(
        values,
        target_mean=torch.tensor([5.0, -2.0]),
        target_std=torch.tensor([2.0, 0.5]),
    )
    assert torch.allclose(raw, torch.tensor([[5.0, 7.0, 3.0], [-2.0, -1.5, -2.5]]))


def test_aggregate_metrics_reproduce_from_row_contributions() -> None:
    distribution = QuantileDistribution(default_levels(32))
    logits = _standard_normal_quantiles(distribution.levels, 7)
    y = torch.randn(7)

    rows = compute_row_metrics(distribution, logits, y, crps_grid_size=64)
    aggregate = compute_metrics(distribution, logits, y, crps_grid_size=64)

    assert aggregate.nll == pytest.approx(float(rows.nll.mean()))
    assert aggregate.nll_coarse == pytest.approx(float(rows.nll_coarse.mean()))
    assert aggregate.crps == pytest.approx(float(rows.crps.mean()))
    assert aggregate.mae == pytest.approx(float(rows.absolute_error.mean()))
    assert aggregate.rmse == pytest.approx(float(torch.sqrt(rows.squared_error.mean())))
    assert aggregate.coverage == {
        level: pytest.approx(float(inside.float().mean())) for level, (_, _, inside) in rows.intervals.items()
    }
