"""Task-to-tensor adapter and the pluggable task source the PFN trains against.

The PFN never sees ebpfn's config directly; it consumes a :class:`TaskSource` sampling
from a fixed ``eta``. A tuned ``eta`` plugs into the *same* interface with no change to
the training loop. The adapter maps a :class:`TuningTask` (``probe_fit`` = context/train,
``probe_score`` = query/test) to the ``forward(x, y)`` convention of the vendored
backbone: ``x`` of shape ``(B, n_rows, n_cols)`` with train rows first, ``y`` of shape
``(B, n_train)``.

Targets are standardized per task on ``probe_fit`` (the backbone standardizes features
internally but not targets); the fit statistics travel on the batch so predictions can
be mapped back to the original target scale.
"""

import dataclasses
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

from ebpfn.data.types import CharacterizationShape, TaskPartition, TuningTask
from ebpfn.priors.nanoprior import ORIGIN_ETA, PriorEta
from ebpfn.priors.nanoprior import sample_task as nanoprior_sample_task
from ebpfn.utils import RandomStreams

# Smallest target spread, relative to the split's own offset, that float32 resolves.
# Below it there is no scale to standardize by and raw units are used instead.
_MIN_TARGET_SPREAD = 16 * float(np.finfo(np.float32).eps)


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
        spread = float(y_fit.std())
        # A constant context carries no scale, so centre it and leave it in raw units.
        # Dividing by a small floor instead would inflate the query targets by 1/floor.
        std = spread if spread > _MIN_TARGET_SPREAD * float(np.abs(y_fit).max()) else 1.0
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


class TaskSource(Protocol):
    """What the training loop needs from wherever tasks come from.

    ``eta_payload`` is the JSON-able identity of the generating prior — a nanoprior
    ``PriorEta``, or ``None`` for a prior with no tunable coordinates. Checkpoints
    record it and resume compares it.
    """

    streams: RandomStreams

    @property
    def stream_provenance(self) -> dict[str, str | int]: ...

    @property
    def eta_payload(self) -> dict[str, Any] | None: ...

    def sample_batch(self, batch_size: int, shape: CharacterizationShape, *identity: str | int) -> list[TuningTask]: ...

    def tensor_batch(self, batch_size: int, shape: CharacterizationShape, *identity: str | int) -> TaskBatch: ...


class NanopriorTaskSource:
    """Task source drawing from the vendored TabICLv2 prior under a mechanism ``eta``.

    Determinism comes from the ``(*identity)`` tokens threaded into ``RandomStreams``
    (order-independent). ``eta`` is deliberately absent from the seed, so two mixtures at
    one identity start from the same draws and diverge only once their mechanism
    selections differ — variance reduction without a pairing protocol. The default is the origin eta:
    upstream's own mixture, and the null arm every tuned variant is measured against. Every arm
    including the null therefore takes the same code path, so arms separated by a small
    eta step differ only in mixture. ``max_attempts`` bounds the learnability rejection
    loop, which corner etas can drive long.
    """

    def __init__(
        self,
        streams: RandomStreams,
        *,
        eta: PriorEta = ORIGIN_ETA,
        max_attempts: int | None = None,
    ) -> None:
        self.streams = streams
        self.eta = eta
        self.max_attempts = max_attempts

    @property
    def stream_provenance(self) -> dict[str, str | int]:
        return {"version": "nanoprior-task-source-3", "base_seed": self.streams.base_seed}

    @property
    def eta_payload(self) -> dict[str, Any] | None:
        return self.eta.to_dict()

    def sample_batch(self, batch_size: int, shape: CharacterizationShape, *identity: str | int) -> list[TuningTask]:
        return [
            nanoprior_sample_task(
                shape, self.streams, *identity, member, eta=self.eta, max_attempts=self.max_attempts
            ).tuning
            for member in range(batch_size)
        ]

    def tensor_batch(self, batch_size: int, shape: CharacterizationShape, *identity: str | int) -> TaskBatch:
        return collate_tasks(self.sample_batch(batch_size, shape, *identity))
