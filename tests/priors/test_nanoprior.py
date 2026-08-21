"""The adapter around the vendored TabICLv2 prior.

Scope is deliberately the *seam*: shape fidelity, determinism across the global-RNG
bridge, and input validation. Upstream's generator behaviour is upstream's to test.
"""

import numpy as np
import pytest
import torch
from ebpfn.data import CharacterizationShape
from ebpfn.pfn.data import NanopriorTaskSource, collate_tasks
from ebpfn.priors.nanoprior import MechanismEta, sample_task
from ebpfn.utils import RandomStreams


def _shape(p: int = 6, n_fit: int = 90, n_score: int = 30) -> CharacterizationShape:
    return CharacterizationShape(n_fit, n_score, p, 0, "regression")


def test_task_matches_requested_shape() -> None:
    task = sample_task(_shape(p=7, n_fit=80, n_score=40), RandomStreams(0), "t").tuning

    assert task.probe_fit.X.shape == (80, 7)
    assert task.probe_score.X.shape == (40, 7)
    assert task.probe_fit.y.shape == (80,)
    assert tuple(task.schema.names) == tuple(f"x{i}" for i in range(7))
    assert np.isfinite(task.probe_fit.X.to_numpy()).all()


def test_identity_determines_the_task() -> None:
    """Upstream draws from global RNGs, so reproducibility is the adapter's to guarantee."""
    first = sample_task(_shape(), RandomStreams(0), "repeat").tuning
    second = sample_task(_shape(), RandomStreams(0), "repeat").tuning
    other = sample_task(_shape(), RandomStreams(0), "different").tuning

    np.testing.assert_array_equal(first.probe_fit.X.to_numpy(), second.probe_fit.X.to_numpy())
    np.testing.assert_array_equal(first.probe_fit.y, second.probe_fit.y)
    assert first.task_id == second.task_id
    assert first.task_id != other.task_id
    assert not np.array_equal(first.probe_fit.y, other.probe_fit.y)


def test_generation_leaves_global_rng_untouched() -> None:
    """The bug the adapter exists to prevent: upstream seeding leaking into callers."""
    np.random.seed(1234)
    torch.manual_seed(1234)
    numpy_before = np.random.get_state()[1].copy()
    torch_before = torch.random.get_rng_state().clone()

    sample_task(_shape(), RandomStreams(0), "isolation")

    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_rejects_shapes_the_adapter_cannot_honour() -> None:
    with pytest.raises(ValueError, match="numeric-only"):
        sample_task(CharacterizationShape(50, 10, 3, 2, "regression"), RandomStreams(0), "cat")
    with pytest.raises(NotImplementedError, match="regression only"):
        sample_task(CharacterizationShape(50, 10, 3, 0, "classification"), RandomStreams(0), "clf")


def test_source_batches_collate_for_the_backbone() -> None:
    source = NanopriorTaskSource(RandomStreams(0))
    tasks = source.sample_batch(3, _shape(), "batch")
    assert len({task.task_id for task in tasks}) == 3, "batch members must be distinct draws"

    batch = collate_tasks(tasks)
    assert batch.x.shape == (3, 120, 6)
    assert batch.n_train == 90
    assert batch.n_test == 30
    assert torch.isfinite(batch.x).all()
    assert torch.isfinite(batch.y_train_std).all()
    torch.testing.assert_close(batch.y_train_std.mean(dim=-1), torch.zeros(3), atol=1e-5, rtol=0)


def test_eta_changes_the_task_and_its_identity() -> None:
    """A tuned mixture must be a different prior, and must not collide in a cache."""
    baseline = sample_task(_shape(), RandomStreams(0), "t").tuning
    smooth = MechanismEta(mechanism=(1.0, 0.0, 0.0, 0.0))
    tuned = sample_task(_shape(), RandomStreams(0), "t", eta=smooth).tuning

    assert tuned.task_id != baseline.task_id
    assert not np.array_equal(tuned.probe_fit.y, baseline.probe_fit.y)
    assert sample_task(_shape(), RandomStreams(0), "t", eta=smooth).tuning.task_id == tuned.task_id


def test_eta_is_recorded_on_the_generated_task() -> None:
    generated = sample_task(_shape(), RandomStreams(0), "t", eta=MechanismEta())
    assert generated.diagnostics["eta"] == MechanismEta().to_dict()
    assert sample_task(_shape(), RandomStreams(0), "t").diagnostics["eta"] is None


def test_exhausted_learnability_retries_raise_rather_than_hang() -> None:
    """Corner etas can make most draws unlearnable; an unbounded loop looks like a hang."""
    with pytest.raises(RuntimeError, match="mostly unlearnable"):
        sample_task(_shape(), RandomStreams(0), "t", max_attempts=0)


def test_task_source_carries_its_eta() -> None:
    shape = _shape()
    streams = RandomStreams(0)
    baseline = NanopriorTaskSource(streams).sample_batch(2, shape, "batch")
    tuned = NanopriorTaskSource(streams, eta=MechanismEta(root=(1.0, 0.0, 0.0, 0.0))).sample_batch(2, shape, "batch")

    assert [task.task_id for task in baseline] != [task.task_id for task in tuned]
    assert collate_tasks(tuned).x.shape == collate_tasks(baseline).x.shape
