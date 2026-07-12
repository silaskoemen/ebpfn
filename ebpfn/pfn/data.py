"""Task-to-tensor adapter and the pluggable task source the PFN trains against.

The PFN never sees ebpfn's config directly; it consumes a :class:`PriorTaskSource`
built from a ``HyperPrior`` (``eta``). Today that ``eta`` comes from the baseline
prior config; a tuned ``eta`` plugs into the *same* source interface with no change to
the training loop. The adapter maps a :class:`TuningTask` (``probe_fit`` = context/train,
``probe_score`` = query/test) to the ``forward(x, y)`` convention of the vendored
backbone: ``x`` of shape ``(B, n_rows, n_cols)`` with train rows first, ``y`` of shape
``(B, n_train)``.

Targets are standardized per task on ``probe_fit`` (the backbone standardizes features
internally but not targets); the fit statistics travel on the batch so predictions can
be mapped back to the original target scale.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ebpfn.data.types import CharacterizationShape, TaskPartition, TuningTask
from ebpfn.priors import HyperPrior, hyperprior_from_dict, sample_task
from ebpfn.utils import RandomStreams

_TARGET_STD_FLOOR = 1e-6


def _partition_features(partition: TaskPartition) -> np.ndarray:
    return partition.X.to_numpy().astype(np.float32, copy=False)


@dataclasses.dataclass(frozen=True)
class TaskBatch:
    """A shape-homogeneous batch of tasks as tensors, ready for the backbone.

    ``x`` stacks train rows (``[:n_train]``) then test rows; ``y_train_std`` /
    ``y_test_std`` are standardized per task; ``target_mean`` / ``target_std`` are the
    per-task ``probe_fit`` statistics used to standardize, so a prediction in
    standardized space maps back as ``value * target_std + target_mean``.
    """

    x: Tensor  # (B, n_rows, n_cols) float32
    y_train_std: Tensor  # (B, n_train) float32
    y_test_std: Tensor  # (B, n_test) float32
    target_mean: Tensor  # (B,) float32
    target_std: Tensor  # (B,) float32
    n_train: int

    @property
    def batch_size(self) -> int:
        return self.x.shape[0]

    @property
    def n_test(self) -> int:
        return self.x.shape[1] - self.n_train

    def to(self, device: torch.device | str) -> "TaskBatch":
        return TaskBatch(
            x=self.x.to(device),
            y_train_std=self.y_train_std.to(device),
            y_test_std=self.y_test_std.to(device),
            target_mean=self.target_mean.to(device),
            target_std=self.target_std.to(device),
            n_train=self.n_train,
        )


def collate_tasks(tasks: list[TuningTask]) -> TaskBatch:
    """Stack shape-homogeneous ``TuningTask``s into a :class:`TaskBatch`.

    All tasks must share ``n_probe_fit``, ``n_probe_score`` and feature count (the
    backbone requires one ``(n_rows, n_cols)`` per batch). Raises otherwise.
    """
    if not tasks:
        raise ValueError("cannot collate an empty task list")
    shapes = {(t.probe_fit.X.height, t.probe_score.X.height, t.probe_fit.X.width) for t in tasks}
    if len(shapes) != 1:
        raise ValueError(f"tasks in a batch must share (n_train, n_test, n_features); saw {shapes}")
    n_train = tasks[0].probe_fit.X.height

    xs, y_train, y_test, means, stds = [], [], [], [], []
    for task in tasks:
        x = np.concatenate([_partition_features(task.probe_fit), _partition_features(task.probe_score)], axis=0)
        y_fit = task.probe_fit.y.astype(np.float32, copy=False)
        y_score = task.probe_score.y.astype(np.float32, copy=False)
        mean = float(y_fit.mean())
        std = max(float(y_fit.std()), _TARGET_STD_FLOOR)
        xs.append(x)
        y_train.append((y_fit - mean) / std)
        y_test.append((y_score - mean) / std)
        means.append(mean)
        stds.append(std)

    return TaskBatch(
        x=torch.from_numpy(np.stack(xs)),
        y_train_std=torch.from_numpy(np.stack(y_train)),
        y_test_std=torch.from_numpy(np.stack(y_test)),
        target_mean=torch.tensor(means, dtype=torch.float32),
        target_std=torch.tensor(stds, dtype=torch.float32),
        n_train=n_train,
    )


class PriorTaskSource:
    """Yields tasks sampled from a fixed ``eta`` — the seam the PFN trains against.

    Construct from the baseline prior now, or from a tuned ``eta`` later; the training
    loop only ever calls :meth:`sample_batch` / :meth:`tensor_batch`. Determinism comes
    from the ``(*identity)`` tokens threaded into ``RandomStreams`` (order-independent),
    exactly as the tuning studies seed task generation.
    """

    def __init__(self, eta: HyperPrior, streams: RandomStreams) -> None:
        self.eta = eta
        self.streams = streams

    @classmethod
    def from_eta_file(cls, path: Path, streams: RandomStreams) -> "PriorTaskSource":
        """Load an exact tuned eta artifact as a PFN training source."""
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError("eta artifact must contain a JSON object")
        return cls(hyperprior_from_dict(payload), streams)

    def sample_batch(self, batch_size: int, shape: CharacterizationShape, *identity: str | int) -> list[TuningTask]:
        return [sample_task(self.eta, shape, self.streams, *identity, member).tuning for member in range(batch_size)]

    def tensor_batch(self, batch_size: int, shape: CharacterizationShape, *identity: str | int) -> TaskBatch:
        return collate_tasks(self.sample_batch(batch_size, shape, *identity))
