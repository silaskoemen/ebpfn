import numpy as np
import polars as pl
import pytest
import torch
from ebpfn.data.types import CharacterizationShape, FeatureSchema, TaskPartition, TuningTask
from ebpfn.pfn.data import NanopriorTaskSource, collate_tasks
from ebpfn.priors import PriorEta
from ebpfn.utils import RandomStreams


@pytest.fixture
def source() -> NanopriorTaskSource:
    return NanopriorTaskSource(RandomStreams(0))


def test_tensor_batch_shapes_and_standardization(source: NanopriorTaskSource) -> None:
    shape = CharacterizationShape(64, 32, 6, 0, "regression")
    batch = source.tensor_batch(4, shape, "test")
    assert batch.x.shape == (4, 96, 6)
    assert batch.y_train_std.shape == (4, 64)
    assert batch.y_test_std.shape == (4, 32)
    assert batch.n_train == 64
    assert batch.n_test == 32
    assert batch.x.dtype == torch.float32
    # standardized on probe_fit: each row's train targets are ~zero-mean, unit-std
    assert float(batch.y_train_std.mean().abs()) < 0.05
    assert float(batch.y_train_std.std()) == pytest.approx(1.0, abs=0.1)


def test_backtransform_statistics_present(source: NanopriorTaskSource) -> None:
    shape = CharacterizationShape(32, 16, 4, 0, "regression")
    batch = source.tensor_batch(3, shape, "test")
    assert batch.target_mean.shape == (3,)
    assert batch.target_std.shape == (3,)
    assert torch.all(batch.target_std > 0)


def test_collate_rejects_heterogeneous_shapes(source: NanopriorTaskSource) -> None:
    a = source.sample_batch(1, CharacterizationShape(32, 16, 4, 0, "regression"), "a")
    b = source.sample_batch(1, CharacterizationShape(40, 16, 4, 0, "regression"), "b")
    with pytest.raises(ValueError, match="share"):
        collate_tasks(a + b)


def test_collate_rejects_empty() -> None:
    with pytest.raises(ValueError):
        collate_tasks([])


def test_batch_to_device_cpu(source: NanopriorTaskSource) -> None:
    batch = source.tensor_batch(2, CharacterizationShape(32, 16, 4, 0, "regression"), "test").to("cpu")
    assert batch.x.device.type == "cpu"


def test_determinism_via_identity(source: NanopriorTaskSource) -> None:
    shape = CharacterizationShape(32, 16, 4, 0, "regression")
    first = source.tensor_batch(2, shape, "same")
    second = source.tensor_batch(2, shape, "same")
    assert torch.equal(first.x, second.x)


def test_default_prior_target_tails_stay_bounded(source: NanopriorTaskSource) -> None:
    """Track how far the origin prior's standardized targets reach.

    The quantile head extrapolates its tails rather than binning them, so this is a drift
    monitor rather than a hard contract: an eta whose targets reach much further than the
    origin's changes what the tail extrapolation is being asked to do.
    """
    inner_bound = 5.0
    shape = CharacterizationShape(512, 128, 100, 0, "regression")
    batch = source.tensor_batch(32, shape, "tail-guard")
    y = torch.cat([batch.y_train_std.reshape(-1), batch.y_test_std.reshape(-1)])
    beyond = float((y.abs() > inner_bound).float().mean())
    # observed 0.17-0.26% across seeds 0-2 (max |y| ~21): the nanoprior's collapse-induced
    # targets are markedly heavier-tailed than a noise-on-signal prior, so the bound is 1%
    # -- ~4x headroom, still tight enough to fire on a further shift. Deterministic under
    # the fixture's fixed seed.
    assert beyond < 1e-2, f"{beyond:.4%} of standardized targets fall beyond +-{inner_bound}"


def test_sources_at_one_identity_share_their_seed() -> None:
    """Two mixtures at the same identity are seeded alike, so a comparison between them
    is not confounded by which tasks each happened to draw."""
    shape = CharacterizationShape(32, 16, 4, 0, "regression")
    streams = RandomStreams(7)
    baseline = NanopriorTaskSource(streams)
    tuned = NanopriorTaskSource(streams, eta=PriorEta(root=(1.0, 0.0, 0.0, 0.0)))

    assert baseline.eta_payload != tuned.eta_payload
    assert (
        baseline.stream_provenance
        == tuned.stream_provenance
        == {
            "version": "nanoprior-task-source-3",
            "base_seed": 7,
        }
    )
    repeated = NanopriorTaskSource(streams).sample_batch(1, shape, "train", 0)[0]
    assert repeated.task_id == baseline.sample_batch(1, shape, "train", 0)[0].task_id


def test_constant_context_standardizes_in_raw_units() -> None:
    """A constant context carries no scale, so the query targets must stay in raw units.

    Dividing by a small floor instead would scale them by 1/floor and spike the loss.
    """
    n_fit, n_score, p = 8, 4, 2
    names = tuple(f"x{i}" for i in range(p))
    features = np.arange((n_fit + n_score) * p, dtype=float).reshape(-1, p)
    y_fit = np.full(n_fit, 3.0)
    y_score = np.array([3.0, 4.0, 2.0, 3.5])
    row_ids = np.arange(n_fit + n_score)
    task = TuningTask(
        task_id="constant-context",
        source_id="synthetic",
        task_type="regression",
        outer_split_id="synthetic-outer",
        characterization_split_id="synthetic-split",
        probe_fit=TaskPartition(pl.DataFrame(features[:n_fit], schema=names), y_fit, row_ids[:n_fit]),
        probe_score=TaskPartition(pl.DataFrame(features[n_fit:], schema=names), y_score, row_ids[n_fit:]),
        schema=FeatureSchema(names, ("numeric",) * p),
        preprocessing_id="synthetic-raw",
        probe_fit_missing_rates=(0.0,) * p,
    )

    batch = collate_tasks([task])
    assert float(batch.target_std[0]) == 1.0
    assert float(batch.target_mean[0]) == 3.0
    torch.testing.assert_close(batch.y_train_std[0], torch.zeros(n_fit))
    torch.testing.assert_close(batch.y_test_std[0], torch.tensor([0.0, 1.0, -1.0, 0.5]))
