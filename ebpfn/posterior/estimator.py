"""Amortized posterior q(eta | D) over the unit cube, one binned marginal per coordinate.

Trained by neural posterior estimation on ``(eta, summary(D))`` pairs with ``eta``
drawn uniformly from the cube, so the network's output *is* the posterior under the
uniform proposal and any other hyperprior psi follows by reweighting:
``p_psi(D) / p_U(D) = E_q[psi(eta)]``. Marginals are factorized, which is the
approximation this estimator makes and the first thing to replace if coordinates prove
jointly identified only.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

N_BINS = 10
HIDDEN = 256
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 256
MAX_EPOCHS = 300
PATIENCE = 20
# Clip on robust-standardized inputs, so a heavy-tailed summary cannot dominate a layer.
INPUT_CLIP = 8.0


@dataclass(frozen=True)
class InputScaler:
    """Median/IQR standardization, fitted on the simulation set and frozen."""

    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, summaries: np.ndarray) -> "InputScaler":
        q25, center, q75 = np.quantile(summaries, (0.25, 0.5, 0.75), axis=0)
        return cls(center, np.where(q75 - q25 > 1e-8, q75 - q25, 1.0))

    def __call__(self, summaries: np.ndarray) -> np.ndarray:
        return np.clip((summaries - self.center) / self.scale, -INPUT_CLIP, INPUT_CLIP)


class BinnedPosterior(nn.Module):
    """Summary vector -> per-coordinate logits over ``N_BINS`` equal-width cube bins."""

    def __init__(self, n_inputs: int, n_coordinates: int) -> None:
        super().__init__()
        self.n_coordinates = n_coordinates
        self.net = nn.Sequential(
            nn.Linear(n_inputs, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, n_coordinates * N_BINS),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).view(-1, self.n_coordinates, N_BINS)


def to_bins(eta: np.ndarray) -> np.ndarray:
    return np.minimum((eta * N_BINS).astype(np.int64), N_BINS - 1)


def _loss(model: BinnedPosterior, inputs: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    logits = model(inputs)
    return nn.functional.cross_entropy(logits.reshape(-1, N_BINS), bins.reshape(-1))


def fit_posterior(
    summaries: np.ndarray, eta: np.ndarray, *, val_fraction: float, seed: int
) -> tuple[BinnedPosterior, InputScaler, dict[str, float]]:
    """Train on a random split, early-stopping on held-out cross-entropy."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = rng.permutation(len(eta))
    n_val = int(len(eta) * val_fraction)
    val, train = order[:n_val], order[n_val:]
    scaler = InputScaler.fit(summaries[train])
    inputs = torch.as_tensor(scaler(summaries), dtype=torch.float32)
    bins = torch.as_tensor(to_bins(eta))

    model = BinnedPosterior(summaries.shape[1], eta.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_state, stale, epoch = float("inf"), None, 0, 0
    for epoch in range(MAX_EPOCHS):  # noqa: B007
        model.train()
        shuffled = torch.as_tensor(rng.permutation(train))
        for start in range(0, len(shuffled), BATCH):
            index = shuffled[start : start + BATCH]
            optimizer.zero_grad()
            _loss(model, inputs[index], bins[index]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(_loss(model, inputs[val], bins[val]))
        if val_loss < best - 1e-4:
            best, stale = val_loss, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("posterior never improved on its validation split")
    model.load_state_dict(best_state)
    model.eval()
    return (
        model,
        scaler,
        {"val_ce": best, "epochs": float(epoch + 1), "n_train": float(len(train)), "n_val": float(n_val)},
    )


@torch.no_grad()
def posterior_probs(model: BinnedPosterior, scaler: InputScaler, summaries: np.ndarray) -> np.ndarray:
    """``(n_datasets, n_coordinates, N_BINS)`` bin probabilities under the uniform proposal."""
    inputs = torch.as_tensor(scaler(np.atleast_2d(summaries)), dtype=torch.float32)
    return torch.softmax(model(inputs), dim=-1).numpy()
