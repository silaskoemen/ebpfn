"""Bar (Riemann) distribution head for the regression PFN (plans/gate1_revised.md §3.2).

nanoTabPFN ships a classification head only; a PFN regressor needs a distribution
over a continuous target. We model p(y) as piecewise-uniform over `num_bins`
buckets defined by `borders`, with exponential tails on the two outer regions so
the density has *full support* on R (the target can fall outside the border
range). The model emits one logit per bucket; softmax gives bucket masses.

All math is in torch so the NLL is differentiable through the logits for training;
the same object serves inference (cdf for PIT, icdf for quantiles). Borders are on
the standardized-y scale (the regressor de-standardizes around it).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

_CLAMP = 1e-12


def normal_borders(num_bins: int, eps: float = 1e-3) -> torch.Tensor:
    """num_bins+1 bucket edges at evenly-spaced quantiles of N(0,1): finer near
    0 where standardized targets concentrate, finite ends (eps..1-eps)."""
    qs = np.linspace(eps, 1.0 - eps, num_bins + 1)
    return torch.from_numpy(norm.ppf(qs).astype(np.float32))


class BarDistribution:
    """Piecewise-uniform interior + exponential tails. `borders`: (K+1,) sorted.

    Region index r in 0..K-1 for a value y:
      r=0       : left tail,  y < borders[1]            -> exp scale borders[1]-borders[0]
      r (1..K-2): interior,   borders[r] <= y < borders[r+1] (uniform)
      r=K-1     : right tail, y >= borders[K-1]          -> exp scale borders[K]-borders[K-1]
    """

    def __init__(self, borders: torch.Tensor):
        if borders.ndim != 1 or borders.numel() < 4:
            raise ValueError("borders must be 1-D with >=4 edges (>=3 buckets)")
        self.borders = borders
        self.num_bins = borders.numel() - 1
        self.widths = borders[1:] - borders[:-1]  # (K,)
        self.s_left = float(self.widths[0])
        self.s_right = float(self.widths[-1])
        self._inner_edges = borders[1:-1]  # borders[1..K-1], length K-1

    def to(self, device) -> BarDistribution:
        b = self.borders.to(device)
        return BarDistribution(b)

    def _region(self, y: torch.Tensor) -> torch.Tensor:
        """Region index per element of y (MPS-safe, no bucketize)."""
        return (y.unsqueeze(-1) >= self._inner_edges).sum(dim=-1).clamp_(0, self.num_bins - 1)

    def nll(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """-log p(y) per element. logits: (..., K); y: (...). Differentiable in logits."""
        log_p = torch.log_softmax(logits, dim=-1)
        r = self._region(y)
        log_pr = log_p.gather(-1, r.unsqueeze(-1)).squeeze(-1)
        b = self.borders
        left = -np.log(self.s_left) + (y - b[1]) / self.s_left
        right = -np.log(self.s_right) - (y - b[-2]) / self.s_right
        w_r = self.widths[r]
        interior = -torch.log(w_r)
        extra = torch.where(r == 0, left, torch.where(r == self.num_bins - 1, right, interior))
        return -(log_pr + extra)

    def cdf(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """P(Y <= y) per element -> PIT values."""
        p = torch.softmax(logits, dim=-1).clamp_min(_CLAMP)
        cum_before = torch.cumsum(p, dim=-1) - p  # exclusive prefix sum
        r = self._region(y)
        b = self.borders
        p_r = p.gather(-1, r.unsqueeze(-1)).squeeze(-1)
        cb_r = cum_before.gather(-1, r.unsqueeze(-1)).squeeze(-1)
        p0 = p[..., 0]
        left = p0 * torch.exp((y - b[1]) / self.s_left)
        right = cb_r + p_r * (1.0 - torch.exp(-(y - b[-2]) / self.s_right))
        w_r = self.widths[r]
        interior = cb_r + p_r * (y - b[r]) / w_r
        out = torch.where(r == 0, left, torch.where(r == self.num_bins - 1, right, interior))
        return out.clamp(0.0, 1.0)

    def icdf(self, logits: torch.Tensor, q: float) -> torch.Tensor:
        """Quantile at level q (scalar) for each row of logits. logits: (..., K)."""
        p = torch.softmax(logits, dim=-1).clamp_min(_CLAMP)
        cum = torch.cumsum(p, dim=-1)  # inclusive
        cum_before = cum - p
        b = self.borders
        K = self.num_bins
        p0, pK = p[..., 0], p[..., K - 1]
        # region of q: left tail if q<=p0; right tail if q>=cum_before[K-1]; else interior bucket
        cb_last = cum_before[..., K - 1]
        # interior bucket index: largest r with cum_before[r] <= q (clamped to 1..K-2)
        r = (q >= cum).sum(dim=-1).clamp(1, K - 2)
        p_r = p.gather(-1, r.unsqueeze(-1)).squeeze(-1)
        cb_r = cum_before.gather(-1, r.unsqueeze(-1)).squeeze(-1)
        w_r = self.widths[r]
        interior = b[r] + (q - cb_r) / p_r * w_r
        left = b[1] + self.s_left * torch.log(torch.clamp(torch.as_tensor(q, device=p.device) / p0, min=_CLAMP))
        frac = 1.0 - (q - cb_last) / pK.clamp_min(_CLAMP)
        right = b[-2] - self.s_right * torch.log(frac.clamp_min(_CLAMP))
        in_left = q <= p0
        in_right = q >= cb_last
        return torch.where(in_left, left, torch.where(in_right, right, interior))
