"""Adapt the vendored TabICLv2 prior to ebpfn's task contracts.

ebpfn code, not vendored — see NOTICE. Two mismatches have to be bridged:

* **Shape.** Upstream returns a dict of extracted columns; ebpfn needs a
  :class:`TuningTask` split into ``probe_fit``/``probe_score`` at an exact
  :class:`CharacterizationShape`. One numeric x-column is requested per feature, so
  ``len(x_cat_sizes) == p`` gives exactly ``p`` columns.
* **Randomness.** Upstream draws from the *global* ``numpy.random`` and ``torch`` RNGs
  rather than an injected generator, which would break ebpfn's reproducibility contract.
  :func:`seeded_globals` derives one seed per task from a ``RandomStreams`` draw, seeds
  both globals, and restores the previous global state on exit, so generating a task
  neither depends on nor perturbs surrounding random state.

The upstream target is a raw extracted column with arbitrary scale; that is fine
because ``collate_tasks`` standardizes targets per task on ``probe_fit``.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl
import torch

from ebpfn.data import CharacterizationShape, FeatureSchema, TaskPartition, TuningTask, content_hash
from ebpfn.priors.contracts import GeneratedTask
from ebpfn.priors.nanoprior.prior import rand_dataset_filtered
from ebpfn.utils import RandomRole, RandomStreams

_SEED_BOUND = 2**31 - 1


@contextlib.contextmanager
def seeded_globals(seed: int) -> Iterator[None]:
    """Seed the global numpy/torch RNGs, then restore whatever was there before."""
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def sample_task(shape: CharacterizationShape, streams: RandomStreams, *identity: str | int) -> GeneratedTask:
    """Draw one regression task from the vendored TabICLv2 prior at an exact shape.

    Upstream rejection-samples until an ExtraTrees out-of-bag test finds signal
    (p < 0.05), so the returned task is never pure noise — the learnability filter
    ebpfn's own generator lacks.
    """
    if shape.p_categorical != 0:
        raise ValueError("nanoprior tasks are numeric-only until the task contracts carry categoricals")
    if shape.task_type != "regression":
        raise NotImplementedError(f"nanoprior adapter generates regression only, got {shape.task_type!r}")

    n = shape.n_probe_fit + shape.n_probe_score
    p = shape.p_numeric
    rng = streams.generator(RandomRole.GENERATION, "nanoprior", n, p, *identity)
    seed = int(rng.integers(0, _SEED_BOUND))

    with seeded_globals(seed):
        # y_cat_sizes=[0] selects the numeric (regression) target branch upstream.
        columns = rand_dataset_filtered(x_cat_sizes=[0] * p, y_cat_sizes=[0], n_samples=n)

    features = torch.cat([tensor for name, tensor in columns.items() if name.startswith("x")], dim=-1)
    x_obs = features.numpy().astype(float)
    target = columns["y_0"].squeeze(-1).numpy().astype(float)
    if x_obs.shape != (n, p):
        raise ValueError(f"nanoprior returned features of shape {x_obs.shape}, expected {(n, p)}")
    if not np.isfinite(x_obs).all() or not np.isfinite(target).all():
        raise ValueError("nanoprior returned a non-finite task")

    names = tuple(f"x{index}" for index in range(p))
    schema = FeatureSchema(names, ("numeric",) * p)
    n_fit = shape.n_probe_fit
    row_ids = np.arange(n)
    task_key = content_hash(seed, n, p, identity, namespace="nanoprior-task-1")
    tuning = TuningTask(
        task_id=f"nanoprior-{task_key[:16]}",
        source_id="synthetic",
        task_type="regression",
        outer_split_id="synthetic-outer",
        characterization_split_id=f"synthetic-{task_key[:16]}",
        probe_fit=TaskPartition(pl.DataFrame(x_obs[:n_fit], schema=names), target[:n_fit], row_ids[:n_fit]),
        probe_score=TaskPartition(pl.DataFrame(x_obs[n_fit:], schema=names), target[n_fit:], row_ids[n_fit:]),
        schema=schema,
        preprocessing_id="synthetic-raw",
        probe_fit_missing_rates=(0.0,) * p,
    )
    diagnostics: dict[str, Any] = {
        "route": "nanoprior",
        "seed": seed,
        "shape": {"n": n, "p": p, "n_probe_fit": n_fit, "n_probe_score": shape.n_probe_score},
    }
    return GeneratedTask(tuning=tuning, diagnostics=diagnostics)
