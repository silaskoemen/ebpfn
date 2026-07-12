import math

import pytest
import torch
from ebpfn.pfn.distribution import BarDistribution, fixed_borders


def _gaussian_logits(borders: torch.Tensor, mu: float, sd: float) -> torch.Tensor:
    def phi(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    mass = (phi((borders[1:] - mu) / sd) - phi((borders[:-1] - mu) / sd)).clamp_min(1e-12)
    return torch.log(mass / mass.sum()).unsqueeze(0)


@pytest.fixture
def standard_borders() -> torch.Tensor:
    return fixed_borders(300)


def test_borders_are_strictly_increasing(standard_borders: torch.Tensor) -> None:
    assert torch.all(standard_borders[1:] > standard_borders[:-1])


def test_fixed_borders_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        fixed_borders(2)
    with pytest.raises(ValueError):
        fixed_borders(4, tail_scale=0.0)


def test_fixed_borders_have_uniform_interior_and_explicit_tail_scale() -> None:
    borders = fixed_borders(512, inner_bound=5.0, tail_scale=1.0)
    assert borders.shape == (513,)
    assert float(borders[1]) == pytest.approx(-5.0)
    assert float(borders[-2]) == pytest.approx(5.0)
    assert float(borders[1] - borders[0]) == pytest.approx(1.0)
    assert float(borders[-1] - borders[-2]) == pytest.approx(1.0)
    assert torch.allclose(
        borders[2:-1] - borders[1:-2],
        torch.full((510,), 10.0 / 510.0),
        atol=5e-7,
        rtol=0.0,
    )


def test_tail_targets_are_not_clamped() -> None:
    distribution = BarDistribution(fixed_borders(32))
    logits = torch.zeros(2, 32)
    nll = distribution.nll(logits, torch.tensor([5.0, 6.0]))
    assert torch.isfinite(nll).all()
    assert float(nll[1] - nll[0]) == pytest.approx(0.5)


def test_construction_rejects_non_monotone_borders() -> None:
    with pytest.raises(ValueError):
        BarDistribution(torch.tensor([0.0, 1.0, 0.5]))


def test_density_integrates_to_one(standard_borders: torch.Tensor) -> None:
    bd = BarDistribution(standard_borders)
    logits = _gaussian_logits(standard_borders, 0.2, 0.9)
    grid = torch.linspace(-10.0, 10.0, 40_000)
    density = torch.exp(bd.log_prob(logits.expand(grid.numel(), -1), grid))
    assert float(torch.trapezoid(density, grid)) == pytest.approx(1.0, abs=2e-3)


def test_cdf_is_monotone(standard_borders: torch.Tensor) -> None:
    bd = BarDistribution(standard_borders)
    logits = _gaussian_logits(standard_borders, 0.0, 1.0)
    ys = torch.linspace(-4.0, 4.0, 400)
    cdf = bd.cdf(logits.expand(ys.numel(), -1), ys)
    assert torch.all(cdf[1:] >= cdf[:-1] - 1e-6)
    assert float(cdf[0]) < 0.01
    assert float(cdf[-1]) > 0.99


def test_icdf_inverts_cdf(standard_borders: torch.Tensor) -> None:
    bd = BarDistribution(standard_borders)
    logits = _gaussian_logits(standard_borders, 0.3, 0.8)
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        y = bd.icdf(logits, q)
        assert float(bd.cdf(logits, y)) == pytest.approx(q, abs=1e-3)


def test_mean_and_quantiles_match_gaussian(standard_borders: torch.Tensor) -> None:
    bd = BarDistribution(standard_borders)
    mu, sd = 0.3, 0.8
    logits = _gaussian_logits(standard_borders, mu, sd)
    assert float(bd.mean(logits)) == pytest.approx(mu, abs=0.02)
    for q in (0.1, 0.5, 0.9):
        expected = mu + sd * math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2 * q - 1)))
        assert float(bd.icdf(logits, q)) == pytest.approx(expected, abs=0.03)


def test_crps_matches_gaussian_closed_form(standard_borders: torch.Tensor) -> None:
    bd = BarDistribution(standard_borders)
    mu, sd = 0.3, 0.8
    logits = _gaussian_logits(standard_borders, mu, sd)
    # CRPS of a Gaussian forecast at its own mean: sd * (2*phi(0) - 1/sqrt(pi)).
    expected = sd * (2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi))
    assert float(bd.crps(logits, torch.tensor([mu]))) == pytest.approx(expected, abs=5e-3)


def test_icdf_rejects_out_of_range_level(standard_borders: torch.Tensor) -> None:
    bd = BarDistribution(standard_borders)
    with pytest.raises(ValueError):
        bd.icdf(_gaussian_logits(standard_borders, 0.0, 1.0), 1.0)
