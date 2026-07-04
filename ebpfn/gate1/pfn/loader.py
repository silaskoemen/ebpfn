"""Live prior batches for PFN training (plans/gate1_revised.md §3.2).

Replaces nanoTabPFN's HDF5 prior-dump loader: each step samples one task geometry
(n rows, d features, train/test split) and draws `batch_size` i.i.d. tasks of that
shape from the MixturePrior, so tensors are rectangular. The target is standardized
per task by its train-portion mean/std (the bar-distribution borders live on the
standardized scale); features are left raw (the model normalizes them internally).
"""

from __future__ import annotations

import numpy as np
import torch

from ebpfn.gate1.config import PFNConfig
from ebpfn.gate1.prior import MixturePrior


def _standardize_y(y: np.ndarray, split: int, eps: float = 1e-8) -> np.ndarray:
    mu = y[:split].mean()
    sd = y[:split].std() + eps
    return (y - mu) / sd


class PriorBatchLoader:
    """Iterable of `steps` training batches drawn from `prior`."""

    def __init__(self, prior: MixturePrior, cfg: PFNConfig, device, rng: np.random.Generator):
        self.prior = prior
        self.cfg = cfg
        self.device = device
        self.rng = rng

    def _draw_geometry(self) -> tuple[int, int, int]:
        c = self.cfg
        n = int(self.rng.integers(c.n_rows_min, c.n_rows_max + 1))
        d = int(self.rng.integers(c.d_min, c.d_max + 1))
        frac = self.rng.uniform(c.train_frac_min, c.train_frac_max)
        split = int(np.clip(round(frac * n), 1, n - 1))
        return n, d, split

    def __len__(self) -> int:
        return self.cfg.steps

    def __iter__(self):
        for _ in range(self.cfg.steps):
            n, d, split = self._draw_geometry()
            xs, ys = [], []
            for _ in range(self.cfg.batch_size):
                D = self.prior.sample_task(n, d, self.rng)
                xs.append(D.X.astype(np.float32))
                ys.append(_standardize_y(D.Y, split).astype(np.float32))
            x = torch.from_numpy(np.stack(xs)).to(self.device)  # (B, n, d)
            y = torch.from_numpy(np.stack(ys)).to(self.device)  # (B, n)
            yield {"x": x, "y": y, "train_test_split_index": split}
