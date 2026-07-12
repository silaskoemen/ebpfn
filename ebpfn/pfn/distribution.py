"""Bar (Riemann) distribution head for regression PFNs.

A predictive distribution over a scalar target expressed as a categorical over
contiguous bins ("bars") plus half-normal tails on the two outer bins, so the
support is the whole real line. This is the TabPFN-style ``FullSupportBarDistribution``
idea: the backbone emits one logit per bin, ``softmax`` gives per-bin mass, the
interior bins hold a piecewise-uniform density, and the outer bins decay as
half-normals (scale = the outer bin width). NLL, CDF, quantiles, mean, CRPS and
interval coverage all derive analytically from this one object, so the metrics stay
mutually consistent and there is no quantile crossing or missing density (the failure
modes of an independent-quantile regression head).

Conventions: ``logits`` has shape ``(..., n_bins)`` and targets ``y`` shape ``(...)``,
both in the shared standardized target space. All methods are batch-shape agnostic
(operations act on the last dim).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_HALF_LOG_2_OVER_PI = 0.5 * math.log(2.0 / math.pi)
_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def fixed_borders(
    n_bins: int,
    *,
    inner_bound: float = 5.0,
    tail_scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Fixed standardized-target grid with uniform interior and full-support tails.

    Two outer bins begin at ``-inner_bound`` and ``inner_bound`` and use
    ``tail_scale`` as their half-normal scale. The remaining ``n_bins - 2`` bins
    uniformly partition the shared standardized interval between those boundaries.
    """
    if n_bins < 3:
        raise ValueError("n_bins must provide two tail bins and at least one interior bin")
    if inner_bound <= 0.0 or tail_scale <= 0.0:
        raise ValueError("inner_bound and tail_scale must be positive")
    interior = torch.linspace(-inner_bound, inner_bound, n_bins - 1, dtype=dtype)
    return torch.cat(
        [
            interior.new_tensor([-inner_bound - tail_scale]),
            interior,
            interior.new_tensor([inner_bound + tail_scale]),
        ]
    )


class BarDistribution(nn.Module):
    """Full-support bar distribution parameterized by fixed bin ``borders``.

    ``borders`` is a strictly increasing 1-D tensor of length ``n_bins + 1`` and is
    registered as a buffer so it follows the module across devices/dtypes.
    """

    borders: Tensor

    def __init__(self, borders: Tensor) -> None:
        super().__init__()
        borders = torch.as_tensor(borders, dtype=torch.float32)
        if borders.ndim != 1 or borders.numel() < 2:
            raise ValueError("borders must be 1-D with at least two entries")
        if not torch.all(borders[1:] > borders[:-1]):
            raise ValueError("borders must be strictly increasing")
        self.register_buffer("borders", borders)

    @property
    def n_bins(self) -> int:
        return self.borders.numel() - 1

    @property
    def widths(self) -> Tensor:
        return self.borders[1:] - self.borders[:-1]

    # -- internal helpers -------------------------------------------------------

    def _bucket_index(self, y: Tensor) -> Tensor:
        """Index of the bin containing ``y`` (outer bins absorb out-of-range values)."""
        idx = torch.searchsorted(self.borders, y.contiguous(), right=True) - 1
        return idx.clamp(0, self.n_bins - 1)

    def _cdf_from_probs(self, probs: Tensor, cum_below: Tensor, y: Tensor) -> Tensor:
        borders, widths = self.borders, self.widths
        last = self.n_bins - 1
        idx = self._bucket_index(y)
        p_k = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        below_k = cum_below.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

        frac = ((y - borders[idx]) / widths[idx]).clamp(0.0, 1.0)
        interior = below_k + p_k * frac

        left_mass = torch.erfc((borders[1] - y).clamp_min(0.0) / (widths[0] * _SQRT_2))
        left = p_k * left_mass
        right_mass = torch.erf((y - borders[last]).clamp_min(0.0) / (widths[last] * _SQRT_2))
        right = below_k + p_k * right_mass

        out = torch.where(idx == 0, left, interior)
        return torch.where(idx == last, right, out)

    # -- public API -------------------------------------------------------------

    def log_prob(self, logits: Tensor, y: Tensor) -> Tensor:
        """Continuous log-density ``log f(y)`` under the predicted distribution."""
        log_probs = F.log_softmax(logits, dim=-1)
        borders, widths = self.borders, self.widths
        last = self.n_bins - 1
        idx = self._bucket_index(y)
        log_p_k = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

        interior = log_p_k - torch.log(widths[idx])
        d_left = (borders[1] - y).clamp_min(0.0)
        left = log_p_k + _HALF_LOG_2_OVER_PI - torch.log(widths[0]) - d_left**2 / (2.0 * widths[0] ** 2)
        d_right = (y - borders[last]).clamp_min(0.0)
        right = log_p_k + _HALF_LOG_2_OVER_PI - torch.log(widths[last]) - d_right**2 / (2.0 * widths[last] ** 2)

        out = torch.where(idx == 0, left, interior)
        return torch.where(idx == last, right, out)

    def nll(self, logits: Tensor, y: Tensor) -> Tensor:
        """Negative log-density; this is the training loss (a proper scoring rule)."""
        return -self.log_prob(logits, y)

    def cdf(self, logits: Tensor, y: Tensor) -> Tensor:
        """``P(Y <= y)`` — monotone in ``y`` by construction (no quantile crossing)."""
        probs = F.softmax(logits, dim=-1)
        cum_below = torch.cumsum(probs, dim=-1) - probs
        return self._cdf_from_probs(probs, cum_below, y)

    def mean(self, logits: Tensor) -> Tensor:
        """Predictive mean ``E[Y]`` (tails contribute their half-normal means)."""
        probs = F.softmax(logits, dim=-1)
        borders, widths = self.borders, self.widths
        centers = (borders[:-1] + borders[1:]) / 2.0
        centers = centers.clone()
        centers[0] = borders[1] - widths[0] * _SQRT_2_OVER_PI
        centers[-1] = borders[-2] + widths[-1] * _SQRT_2_OVER_PI
        return (probs * centers).sum(dim=-1)

    def icdf(self, logits: Tensor, q: float) -> Tensor:
        """Quantile: the ``y`` with ``P(Y <= y) = q`` for a scalar level ``q`` in (0, 1)."""
        if not 0.0 < q < 1.0:
            raise ValueError("q must lie strictly in (0, 1)")
        probs = F.softmax(logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cum_below = cum - probs
        borders, widths = self.borders, self.widths
        last = self.n_bins - 1
        eps = torch.finfo(probs.dtype).tiny

        # interior: first bin whose cumulative mass reaches q
        k = (cum < q).sum(dim=-1).clamp(max=last)
        p_k = probs.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        below_k = cum_below.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        interior = borders[k] + (q - below_k) / p_k.clamp_min(eps) * widths[k]

        p0 = probs[..., 0]
        left_arg = (1.0 - q / p0.clamp_min(eps)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        left = borders[1] - widths[0] * _SQRT_2 * torch.special.erfinv(left_arg)

        p_last = probs[..., last]
        below_last = cum_below[..., last]
        right_arg = ((q - below_last) / p_last.clamp_min(eps)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        right = borders[last] + widths[last] * _SQRT_2 * torch.special.erfinv(right_arg)

        out = torch.where(q < p0, left, interior)
        return torch.where(q > below_last, right, out)

    def quantiles(self, logits: Tensor, levels: tuple[float, ...]) -> Tensor:
        """Stack of quantiles at the given ``levels``; shape ``(..., len(levels))``."""
        return torch.stack([self.icdf(logits, q) for q in levels], dim=-1)

    def crps(self, logits: Tensor, y: Tensor, *, grid_size: int = 512, tail_pad: float = 6.0) -> Tensor:
        """Continuous ranked probability score ``∫ (F(t) - 1{y<=t})^2 dt``.

        Evaluated by quadrature on a fixed grid that extends ``tail_pad`` outer-bin
        widths beyond the border range so the half-normal tails are captured; the
        analytic CDF makes this a clean, monotone integrand. Eval-time metric only.
        """
        borders, widths = self.borders, self.widths
        lo = borders[0] - tail_pad * widths[0]
        hi = borders[-1] + tail_pad * widths[-1]
        grid = torch.linspace(float(lo), float(hi), grid_size, device=logits.device, dtype=logits.dtype)  # (G,)

        logits_flat = logits.reshape(-1, self.n_bins)  # (B, K)
        y_flat = y.reshape(-1)  # (B,)
        probs = F.softmax(logits_flat, dim=-1)
        cum_below = torch.cumsum(probs, dim=-1) - probs
        n_batch = logits_flat.shape[0]

        probs_g = probs.unsqueeze(1).expand(n_batch, grid_size, self.n_bins)
        cum_below_g = cum_below.unsqueeze(1).expand(n_batch, grid_size, self.n_bins)
        grid_g = grid.unsqueeze(0).expand(n_batch, grid_size)
        cdf = self._cdf_from_probs(probs_g, cum_below_g, grid_g)  # (B, G)

        indicator = (grid.unsqueeze(0) >= y_flat.unsqueeze(1)).to(logits.dtype)  # (B, G)
        crps_flat = torch.trapezoid((cdf - indicator) ** 2, grid, dim=-1)  # (B,)
        return crps_flat.reshape(y.shape)
