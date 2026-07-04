"""Trained-PFN inference as a ProbModel/Predictive (plans/gate1_revised.md §3.2/§5).

A trained PFN does regression *in context*: `fit` just stores the task's train
split (no gradients); `predictive` runs one forward pass and reads off a bar
distribution per test row. Wrapping it in Gate-0's `ProbModel`/`Predictive`
interface lets `calibration_report` (NLL/CRPS/PIT/coverage) be reused verbatim.

The PFN works on standardized y (matching the bar borders); the Predictive
de-standardizes so all scores are on the task's original y scale (NLL picks up
the +log(std) Jacobian; PIT is transform-invariant).
"""
from __future__ import annotations

import numpy as np
import torch

from ebpfn.gate1.pfn.bar import BarDistribution
from ebpfn.gate1.pfn.model import PFNTransformer
from ebpfn.priors import Dataset
from ebpfn.regressor import Predictive, ProbModel

_CRPS_LEVELS = np.linspace(0.02, 0.98, 49)


class PFNPredictive(Predictive):
    """Bar distribution per test row, de-standardized to original y scale."""

    def __init__(self, logits: torch.Tensor, bar: BarDistribution, y_mean: float, y_std: float):
        self._logits = logits  # (n_test, K), cpu
        self._bar = bar
        self._mean = y_mean
        self._std = y_std

    def _z(self, y: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((y - self._mean) / self._std).astype(np.float32))

    def nll(self, y: np.ndarray) -> np.ndarray:
        z_nll = self._bar.nll(self._logits, self._z(y)).numpy()
        return z_nll + np.log(self._std)  # change-of-variables to original scale

    def cdf(self, y: np.ndarray) -> np.ndarray:
        return self._bar.cdf(self._logits, self._z(y)).numpy()

    def quantile(self, level: float) -> np.ndarray:
        return self._mean + self._std * self._bar.icdf(self._logits, float(level)).numpy()

    def crps(self, y: np.ndarray) -> np.ndarray:
        # CRPS = 2 * integral_0^1 pinball_alpha, grid-averaged over quantile levels.
        Q = np.column_stack([self.quantile(a) for a in _CRPS_LEVELS])  # (n, L)
        diff = y[:, None] - Q
        pin = np.maximum(_CRPS_LEVELS[None, :] * diff, (_CRPS_LEVELS[None, :] - 1.0) * diff)
        return 2.0 * pin.mean(axis=1)


class PFNRegressor(ProbModel):
    """In-context regression with a trained PFN. Not trained per task."""

    def __init__(self, model: PFNTransformer, bar: BarDistribution, device, eps: float = 1e-8):
        self.model = model
        self.bar = bar.to("cpu")
        self.device = device
        self.eps = eps
        self._Xtr: np.ndarray | None = None
        self._ytr_std: np.ndarray | None = None
        self._mean = 0.0
        self._std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PFNRegressor":
        self._Xtr = np.asarray(X, dtype=np.float32)
        self._mean = float(y.mean())
        self._std = float(y.std()) + self.eps
        self._ytr_std = ((y - self._mean) / self._std).astype(np.float32)
        return self

    @torch.no_grad()
    def predictive(self, X: np.ndarray) -> PFNPredictive:
        assert self._Xtr is not None, "fit before predictive"
        n_train = self._Xtr.shape[0]
        X = np.asarray(X, dtype=np.float32)
        x = torch.from_numpy(np.concatenate([self._Xtr, X])[None]).to(self.device)
        y = torch.from_numpy(self._ytr_std[None]).to(self.device)
        self.model.eval()
        logits = self.model((x, y), train_test_split_index=n_train).squeeze(0).cpu()
        return PFNPredictive(logits, self.bar, self._mean, self._std)
