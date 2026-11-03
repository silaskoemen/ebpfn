"""QuantileDistribution: self-consistency of the reconstructed predictive density."""

import pytest
import torch
from ebpfn.pfn.metrics import compute_metrics
from ebpfn.pfn.quantile import QuantileDistribution, default_levels


def normal_quantiles(levels: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Exact Gaussian quantiles at ``levels`` — a distribution with known moments."""
    return mean + std * torch.special.erfinv(2.0 * levels - 1.0) * (2.0**0.5)


@pytest.fixture
def dist() -> QuantileDistribution:
    return QuantileDistribution(default_levels(999))


def test_default_levels_match_upstream_convention():
    levels = default_levels(999)
    assert levels.numel() == 999
    torch.testing.assert_close(levels[0], torch.tensor(1.0 / 1000.0))
    torch.testing.assert_close(levels[-1], torch.tensor(999.0 / 1000.0))


def test_rejects_degenerate_levels():
    with pytest.raises(ValueError, match="strictly increasing"):
        QuantileDistribution(torch.tensor([0.5, 0.2, 0.9]))
    with pytest.raises(ValueError, match="strictly inside"):
        QuantileDistribution(torch.tensor([0.0, 0.5, 1.0]))


def test_cdf_is_monotone_and_inverts_icdf(dist):
    values = normal_quantiles(dist.levels).expand(4, 999)
    grid = torch.linspace(-4.0, 4.0, 200)
    cdf = torch.stack([dist.cdf(values, torch.full((4,), float(g))) for g in grid])
    assert torch.all(cdf[1:] >= cdf[:-1] - 1e-6)
    assert torch.all((cdf >= 0.0) & (cdf <= 1.0))

    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        recovered = dist.cdf(values, dist.icdf(values, q))
        torch.testing.assert_close(recovered, torch.full((4,), q), atol=2e-3, rtol=0)


def test_quantile_crossing_is_repaired(dist):
    ordered = normal_quantiles(dist.levels)
    shuffled = ordered[torch.randperm(999)].expand(2, 999)
    torch.testing.assert_close(dist.mean(shuffled), dist.mean(ordered.expand(2, 999)))


def test_recovers_gaussian_moments_and_density(dist):
    values = normal_quantiles(dist.levels, mean=1.5, std=2.0).expand(3, 999)
    torch.testing.assert_close(dist.mean(values), torch.full((3,), 1.5), atol=5e-3, rtol=0)

    y = torch.tensor([1.5, 3.5, -0.5])
    expected = (
        -0.5 * ((y - 1.5) / 2.0) ** 2 - torch.log(torch.tensor(2.0)) - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    )
    torch.testing.assert_close(dist.log_prob(values, y), expected, atol=2e-2, rtol=0)


def test_density_integrates_to_one(dist):
    values = normal_quantiles(dist.levels).expand(1, 999)
    grid = torch.linspace(-12.0, 12.0, 4001)
    density = torch.stack([dist.log_prob(values, torch.full((1,), float(g))).exp()[0] for g in grid])
    torch.testing.assert_close(torch.trapezoid(density, grid), torch.tensor(1.0), atol=5e-3, rtol=0)


def test_tails_give_full_support(dist):
    """Support is unbounded: the log-density stays finite arbitrarily far out.

    The CDF saturates to 0 (resp. 1) in float32 long before the density does — and the
    right tail saturates sooner, since ``1 - eps`` rounds to 1 at ~1e-7 while ``eps``
    stays representable to ~1e-45. So full support is asserted in log space, and the
    CDF only just outside the knot range where float32 still resolves it.
    """
    values = normal_quantiles(dist.levels).expand(1, 999)
    for far in (torch.tensor([-40.0]), torch.tensor([40.0])):
        assert torch.isfinite(dist.log_prob(values, far)).all()

    outside = torch.tensor([-4.0])  # just beyond the ±3.09 outermost knots
    assert 0.0 < float(dist.cdf(values, outside)) < float(dist.levels[0])
    assert float(dist.levels[-1]) < float(dist.cdf(values, -outside)) < 1.0


def test_pinball_minimized_by_true_quantiles(dist):
    values = normal_quantiles(dist.levels).expand(64, 999)
    y = torch.randn(64)
    assert float(dist.pinball(values, y).mean()) < float(dist.pinball(values + 1.0, y).mean())


def test_crps_beats_a_shifted_forecast(dist):
    values = normal_quantiles(dist.levels).expand(256, 999)
    y = torch.randn(256)
    sharp = float(dist.crps(values, y).mean())
    assert sharp < float(dist.crps(values + 2.0, y).mean())
    assert sharp < float(dist.crps(normal_quantiles(dist.levels, std=6.0).expand(256, 999), y).mean())


def test_crps_grid_size_is_a_real_knob(dist):
    values = normal_quantiles(dist.levels).expand(8, 999)
    y = torch.randn(8)
    coarse = dist.crps(values, y, grid_size=8)
    fine = dist.crps(values, y, grid_size=1024)
    assert not torch.allclose(coarse, fine)
    torch.testing.assert_close(fine, dist.crps(values, y, grid_size=2048), atol=2e-3, rtol=0)
    with pytest.raises(ValueError, match="at least two"):
        dist.crps(values, y, grid_size=1)


def test_plugs_into_shared_metrics(dist):
    values = normal_quantiles(dist.levels).expand(128, 999)
    y = torch.randn(128)
    metrics = compute_metrics(dist, values, y, crps_grid_size=64)
    assert metrics.n == 128
    assert metrics.rmse > 0.0
    # A well-specified forecast should sit close to nominal coverage.
    assert abs(metrics.coverage["0.9"] - 0.9) < 0.1


def test_icdf_rejects_out_of_range_level(dist):
    values = normal_quantiles(dist.levels).expand(2, 999)
    with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
        dist.icdf(values, 1.0)


def test_coarse_nll_matches_nll_on_a_smooth_grid(dist):
    values = normal_quantiles(dist.levels).expand(512, 999)
    y = torch.randn(512)
    torch.testing.assert_close(dist.coarse_nll(values, y, stride=1), dist.nll(values, y))
    assert float(dist.coarse_nll(values, y, stride=10).mean()) == pytest.approx(
        float(dist.nll(values, y).mean()), abs=0.01
    )


def test_coarse_nll_is_robust_to_knot_jitter(dist):
    """Jitter far below the distribution's scale inflates raw NLL, not the coarse one."""
    smooth = normal_quantiles(dist.levels).expand(512, 999)
    jittered = smooth + 2e-3 * torch.randn(512, 999, generator=torch.Generator().manual_seed(0))
    y = torch.randn(512, generator=torch.Generator().manual_seed(1))
    reference = float(dist.nll(smooth, y).mean())
    assert float(dist.nll(jittered, y).mean()) - reference > 0.1
    assert abs(float(dist.coarse_nll(jittered, y, stride=10).mean()) - reference) < 0.02


def test_coarse_nll_rejects_a_stride_that_leaves_one_knot(dist):
    values = normal_quantiles(dist.levels).expand(2, 999)
    with pytest.raises(ValueError, match="fewer than two"):
        dist.coarse_nll(values, torch.zeros(2), stride=999)
