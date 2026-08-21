"""Quantile predictive distribution for the released TabICLv2 regression head.

That head emits ``n_quantiles`` values at fixed levels rather than bar logits, so the
predictive density is reconstructed rather than read off directly: quantile crossing
is removed by sorting, the interior is the piecewise-linear quantile function through
the knots, and beyond the outermost knots exponential tails give full support. This
mirrors upstream's defaults (``tail_type="exp"``, ``crossing_method="sort"``, linear
spline interior) so numbers stay comparable to the released model's own reporting;
the heavier GPD tail option and monotone-spline interior are deliberately not ported.

The API matches :class:`ebpfn.pfn.distribution.BarDistribution` — every method takes
the raw head output as its first argument — so both heads feed
:mod:`ebpfn.pfn.metrics` unchanged.
"""

import torch
from torch import Tensor, nn

_MIN_BETA, _MAX_BETA = 0.01, 100.0
_MIN_GAP = 1e-6
_TOL = 1e-6
_TAIL_KNOTS = 20


def default_levels(n_quantiles: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Upstream's level grid: ``linspace(0, 1, n + 2)[1:-1]``, i.e. interior only."""
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least two")
    return torch.linspace(0.0, 1.0, n_quantiles + 2, dtype=dtype)[1:-1]


class QuantileDistribution(nn.Module):
    """Predictive distribution defined by quantiles at fixed ``levels``.

    ``levels`` is a strictly increasing 1-D tensor in (0, 1), registered as a buffer so
    it follows the module across devices.
    """

    levels: Tensor

    def __init__(self, levels: Tensor) -> None:
        super().__init__()
        levels = torch.as_tensor(levels, dtype=torch.float32)
        if levels.ndim != 1 or levels.numel() < 2:
            raise ValueError("levels must be 1-D with at least two entries")
        if not torch.all(levels[1:] > levels[:-1]):
            raise ValueError("levels must be strictly increasing")
        if float(levels[0]) <= 0.0 or float(levels[-1]) >= 1.0:
            raise ValueError("levels must lie strictly inside (0, 1)")
        self.register_buffer("levels", levels)

    @property
    def n_outputs(self) -> int:
        return self.levels.numel()

    # -- internal helpers -------------------------------------------------------

    def _flat(self, values: Tensor) -> Tensor:
        """Sort away quantile crossing and flatten the batch dims."""
        return torch.sort(values.reshape(-1, self.n_outputs), dim=-1).values

    def _tails(self, sorted_q: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Exponential tail coefficients ``(a_left, b_left, a_right, b_right)``.

        Fit by regressing the outermost knots on ``log(alpha)`` (left) and
        ``log(1 - alpha)`` (right), then anchored at the outer knot so the quantile
        function stays continuous there.
        """
        levels = self.levels
        k = min(_TAIL_KNOTS, self.n_outputs // 4) or 1

        log_left = torch.log(levels[:k].clamp_min(_TOL))
        centered = log_left - log_left.mean()
        q_left = sorted_q[:, :k]
        cov = ((q_left - q_left.mean(dim=-1, keepdim=True)) * centered).mean(dim=-1)
        beta_left = (cov / centered.square().mean().clamp_min(_TOL)).abs().clamp(_MIN_BETA, _MAX_BETA)

        log_right = torch.log((1.0 - levels[-k:]).clamp_min(_TOL))
        centered = log_right - log_right.mean()
        q_right = sorted_q[:, -k:]
        cov = ((q_right - q_right.mean(dim=-1, keepdim=True)) * centered).mean(dim=-1)
        beta_right = (cov / centered.square().mean().clamp_min(_TOL)).abs().clamp(_MIN_BETA, _MAX_BETA)

        a_left = beta_left
        b_left = sorted_q[:, 0] - a_left * torch.log(levels[0].clamp_min(_TOL))
        a_right = -beta_right
        b_right = sorted_q[:, -1] - a_right * torch.log((1.0 - levels[-1]).clamp_min(_TOL))
        return a_left, b_left, a_right, b_right

    def _interior_gaps(self, sorted_q: Tensor) -> tuple[Tensor, Tensor]:
        """Knot-to-knot spacing in value and level, floored away from zero."""
        return (sorted_q[:, 1:] - sorted_q[:, :-1]).clamp_min(_MIN_GAP), self.levels[1:] - self.levels[:-1]

    # -- public API -------------------------------------------------------------

    def cdf(self, values: Tensor, y: Tensor) -> Tensor:
        """``P(Y <= y)`` — monotone by construction (quantiles are sorted first)."""
        sorted_q = self._flat(values)
        flat_y = y.reshape(-1, 1)
        a_left, b_left, a_right, b_right = self._tails(sorted_q)
        dq, dalpha = self._interior_gaps(sorted_q)

        idx = (torch.searchsorted(sorted_q, flat_y.contiguous(), right=True) - 1).clamp(0, self.n_outputs - 2)
        gathered_q = sorted_q.gather(-1, idx).squeeze(-1)
        gathered_dq = dq.gather(-1, idx).squeeze(-1)
        base = self.levels[idx.squeeze(-1)]
        step = dalpha[idx.squeeze(-1)]
        flat_y = flat_y.squeeze(-1)
        interior = base + step * ((flat_y - gathered_q) / gathered_dq).clamp(0.0, 1.0)

        left = torch.exp(((flat_y - b_left) / a_left).clamp(max=0.0)).clamp(0.0, float(self.levels[0]))
        right = 1.0 - torch.exp(((flat_y - b_right) / a_right).clamp(max=0.0))
        right = right.clamp(float(self.levels[-1]), 1.0)

        out = torch.where(flat_y < sorted_q[:, 0], left, interior)
        return torch.where(flat_y > sorted_q[:, -1], right, out).reshape(y.shape)

    def log_prob(self, values: Tensor, y: Tensor) -> Tensor:
        """Continuous log-density ``log f(y)``."""
        sorted_q = self._flat(values)
        flat_y = y.reshape(-1, 1)
        a_left, b_left, a_right, b_right = self._tails(sorted_q)
        dq, dalpha = self._interior_gaps(sorted_q)

        idx = (torch.searchsorted(sorted_q, flat_y.contiguous(), right=True) - 1).clamp(0, self.n_outputs - 2)
        density = (dalpha[idx.squeeze(-1)] / dq.gather(-1, idx).squeeze(-1)).log()
        flat_y = flat_y.squeeze(-1)

        left = ((flat_y - b_left) / a_left).clamp(max=0.0) - a_left.log()
        right = ((flat_y - b_right) / a_right).clamp(max=0.0) - (-a_right).log()

        out = torch.where(flat_y < sorted_q[:, 0], left, density)
        return torch.where(flat_y > sorted_q[:, -1], right, out).reshape(y.shape)

    def nll(self, values: Tensor, y: Tensor) -> Tensor:
        """Negative log-density. Reported, not trained on — see :meth:`pinball`."""
        return -self.log_prob(values, y)

    def icdf(self, values: Tensor, q: float) -> Tensor:
        """Quantile at a scalar level ``q`` in (0, 1)."""
        if not 0.0 < q < 1.0:
            raise ValueError("q must lie strictly in (0, 1)")
        sorted_q = self._flat(values)
        a_left, b_left, a_right, b_right = self._tails(sorted_q)
        levels = self.levels

        if q < float(levels[0]):
            out = a_left * torch.log(torch.tensor(q, device=sorted_q.device)) + b_left
        elif q > float(levels[-1]):
            out = a_right * torch.log(torch.tensor(1.0 - q, device=sorted_q.device)) + b_right
        else:
            upper = int(torch.searchsorted(levels, torch.tensor(q, device=levels.device), right=True).clamp(1))
            lower = upper - 1
            weight = (q - float(levels[lower])) / float(levels[upper] - levels[lower])
            out = torch.lerp(sorted_q[:, lower], sorted_q[:, upper], weight)
        return out.reshape(values.shape[:-1])

    def quantiles(self, values: Tensor, levels: tuple[float, ...]) -> Tensor:
        """Stack of quantiles at the given ``levels``; shape ``(..., len(levels))``."""
        return torch.stack([self.icdf(values, q) for q in levels], dim=-1)

    def mean(self, values: Tensor) -> Tensor:
        """Predictive mean ``E[Y] = ∫ Q(alpha) d(alpha)``, tails integrated analytically."""
        sorted_q = self._flat(values)
        a_left, b_left, a_right, b_right = self._tails(sorted_q)
        levels = self.levels

        interior = torch.trapezoid(sorted_q, levels, dim=-1)
        first, last = float(levels[0]), float(1.0 - levels[-1])
        left = a_left * (first * torch.log(torch.tensor(first, device=sorted_q.device)) - first) + b_left * first
        right = a_right * (last * torch.log(torch.tensor(last, device=sorted_q.device)) - last) + b_right * last
        return (interior + left + right).reshape(values.shape[:-1])

    def _icdf_grid(self, sorted_q: Tensor, alphas: Tensor) -> Tensor:
        """Quantiles at many levels at once: ``(n_rows, len(alphas))``."""
        a_left, b_left, a_right, b_right = self._tails(sorted_q)
        levels = self.levels
        upper = torch.searchsorted(levels, alphas.contiguous(), right=True).clamp(1, self.n_outputs - 1)
        lower = upper - 1
        weight = (alphas - levels[lower]) / (levels[upper] - levels[lower])
        interior = torch.lerp(sorted_q[:, lower], sorted_q[:, upper], weight)

        left = a_left[:, None] * torch.log(alphas).clamp(min=-30.0) + b_left[:, None]
        right = a_right[:, None] * torch.log1p(-alphas).clamp(min=-30.0) + b_right[:, None]
        out = torch.where(alphas < levels[0], left, interior)
        return torch.where(alphas > levels[-1], right, out)

    def training_loss(self, values: Tensor, y: Tensor) -> Tensor:
        """The loss this head is fit by — pinball, not NLL.

        NLL here goes through a reconstructed density (sorted knots, linear interior,
        fitted exponential tails) and is far more fragile than the quantiles themselves.
        """
        return self.pinball(values, y)

    def pinball(self, values: Tensor, y: Tensor) -> Tensor:
        """Mean pinball loss over the level grid — the head's native training loss."""
        sorted_q = self._flat(values)
        error = y.reshape(-1, 1) - sorted_q
        levels = self.levels
        return torch.maximum(levels * error, (levels - 1.0) * error).mean(dim=-1).reshape(y.shape)

    def crps(self, values: Tensor, y: Tensor, *, grid_size: int = 512) -> Tensor:
        """Continuous ranked probability score via ``CRPS = 2 ∫ pinball(alpha) d(alpha)``.

        ``grid_size`` sets the quadrature resolution in *level* space (the bar head's
        equivalent knob integrates over target space instead). The grid spans the open
        interval, so the exponential tails contribute rather than being truncated away.
        """
        if grid_size < 2:
            raise ValueError("grid_size must be at least two")
        sorted_q = self._flat(values)
        alphas = torch.linspace(0.0, 1.0, grid_size + 2, device=sorted_q.device, dtype=sorted_q.dtype)[1:-1]
        predicted = self._icdf_grid(sorted_q, alphas)
        error = y.reshape(-1, 1) - predicted
        integrand = torch.maximum(alphas * error, (alphas - 1.0) * error)
        return (2.0 * torch.trapezoid(integrand, alphas, dim=-1)).reshape(y.shape)
