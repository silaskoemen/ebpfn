"""Mixture-over-DGPs prior for EB-PFN (plans/gate1_revised.md §2/§3.1).

`MixturePrior` samples each task by first drawing a DGP from the family by
weight, then sampling a task from it. This is the *single* source that feeds both
the PFN's training batches and the s-OTDD coverage clouds -- the exact prior<->
model pairing H1 requires (§2). The caller supplies (n, d) per task, so coverage
clouds can be d-matched to each real task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ebpfn.gate1.config import PriorConfig
from ebpfn.gate1.dgp import DGP
from ebpfn.gate1.dgp import BnnDgp
from ebpfn.gate1.dgp import ScmDgp
from ebpfn.priors import Dataset


@dataclass(frozen=True)
class MixturePrior:
    """A weighted mixture of DGPs. weights default to uniform."""

    dgps: tuple[DGP, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.dgps:
            raise ValueError("MixturePrior needs at least one DGP")
        if self.weights is not None:
            if len(self.weights) != len(self.dgps):
                raise ValueError(f"weights len {len(self.weights)} != dgps len {len(self.dgps)}")
            if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError(f"weights must be non-negative with positive sum, got {self.weights}")

    def _probs(self) -> np.ndarray:
        if self.weights is None:
            return np.full(len(self.dgps), 1.0 / len(self.dgps))
        w = np.asarray(self.weights, dtype=float)
        return w / w.sum()

    def sample_task(self, n: int, d: int, rng: np.random.Generator) -> Dataset:
        """Draw one task: pick a DGP by weight, then sample (n, d) from it."""
        i = int(rng.choice(len(self.dgps), p=self._probs()))
        return self.dgps[i].sample(n, d, rng)

    def sample_cloud(self, n_tasks: int, n: int, d: int, rng: np.random.Generator) -> list[Dataset]:
        """A cloud of `n_tasks` i.i.d. tasks, all sized (n, d)."""
        return [self.sample_task(n, d, rng) for _ in range(n_tasks)]


def build_prior(cfg: PriorConfig) -> MixturePrior:
    """Assemble the mixture from a PriorConfig (members with weight 0 dropped)."""
    members: list[tuple[DGP, float]] = []
    if cfg.scm_linear_weight > 0:
        members.append(
            (
                ScmDgp(
                    activation="linear",
                    n_hidden=cfg.scm_n_hidden,
                    edge_prob=cfg.scm_edge_prob,
                    max_parents=cfg.scm_max_parents,
                    weight_scale=cfg.scm_weight_scale,
                    noise_scale=cfg.scm_noise_scale,
                ),
                cfg.scm_linear_weight,
            )
        )
    if cfg.scm_mlp_weight > 0:
        members.append(
            (
                ScmDgp(
                    activation=cfg.scm_mlp_activation,
                    n_hidden=cfg.scm_n_hidden,
                    edge_prob=cfg.scm_edge_prob,
                    max_parents=cfg.scm_max_parents,
                    weight_scale=cfg.scm_weight_scale,
                    noise_scale=cfg.scm_noise_scale,
                ),
                cfg.scm_mlp_weight,
            )
        )
    if cfg.bnn_weight > 0:
        members.append(
            (
                BnnDgp(
                    n_layers=cfg.bnn_n_layers,
                    hidden=cfg.bnn_hidden,
                    activation=cfg.bnn_activation,
                    weight_scale=cfg.bnn_weight_scale,
                    noise_scale=cfg.bnn_noise_scale,
                ),
                cfg.bnn_weight,
            )
        )
    dgps = tuple(m[0] for m in members)
    weights = tuple(m[1] for m in members)
    return MixturePrior(dgps=dgps, weights=weights)
