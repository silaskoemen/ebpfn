import math

import pytest
import torch
from ebpfn.pfn.distribution import BarDistribution, fixed_borders
from ebpfn.pfn.metrics import compute_metrics, to_raw


def _calibrated_gaussian_logits(borders: torch.Tensor, n: int) -> torch.Tensor:
    def phi(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    mass = (phi(borders[1:]) - phi(borders[:-1])).clamp_min(1e-12)
    return torch.log(mass / mass.sum()).unsqueeze(0).expand(n, -1)


def test_calibrated_model_is_well_calibrated() -> None:
    borders = fixed_borders(300)
    bd = BarDistribution(borders)
    n = 20_000
    logits = _calibrated_gaussian_logits(borders, n)
    y = torch.randn(n)
    metrics = compute_metrics(bd, logits, y)

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
