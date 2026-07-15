import json
from pathlib import Path

import pytest
import torch
from ebpfn.config.pfn import PfnArchConfig
from ebpfn.config.prior import HyperPriorConfig
from ebpfn.data.types import CharacterizationShape
from ebpfn.pfn.data import PairedPriorTaskSource, PriorTaskSource, collate_tasks
from ebpfn.priors import build_hyperprior, hyperprior_to_dict
from ebpfn.utils import RandomStreams


@pytest.fixture
def source() -> PriorTaskSource:
    return PriorTaskSource(build_hyperprior(HyperPriorConfig()), RandomStreams(0))


def test_tensor_batch_shapes_and_standardization(source: PriorTaskSource) -> None:
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


def test_backtransform_statistics_present(source: PriorTaskSource) -> None:
    shape = CharacterizationShape(32, 16, 4, 0, "regression")
    batch = source.tensor_batch(3, shape, "test")
    assert batch.target_mean.shape == (3,)
    assert batch.target_std.shape == (3,)
    assert torch.all(batch.target_std > 0)


def test_collate_rejects_heterogeneous_shapes(source: PriorTaskSource) -> None:
    a = source.sample_batch(1, CharacterizationShape(32, 16, 4, 0, "regression"), "a")
    b = source.sample_batch(1, CharacterizationShape(40, 16, 4, 0, "regression"), "b")
    with pytest.raises(ValueError, match="share"):
        collate_tasks(a + b)


def test_collate_rejects_empty() -> None:
    with pytest.raises(ValueError):
        collate_tasks([])


def test_batch_to_device_cpu(source: PriorTaskSource) -> None:
    batch = source.tensor_batch(2, CharacterizationShape(32, 16, 4, 0, "regression"), "test").to("cpu")
    assert batch.x.device.type == "cpu"


def test_determinism_via_identity(source: PriorTaskSource) -> None:
    shape = CharacterizationShape(32, 16, 4, 0, "regression")
    first = source.tensor_batch(2, shape, "same")
    second = source.tensor_batch(2, shape, "same")
    assert torch.equal(first.x, second.x)


def test_default_prior_stays_within_target_grid(source: PriorTaskSource) -> None:
    """Guard the prior<->grid contract: the default prior's standardized targets must
    almost all fall inside the fixed bar-distribution interior. If a future prior (or a
    tuned eta) develops heavier standardized tails, mass spills past ``inner_bound`` into
    the half-normal tail bins, silently degrading calibration -- this test flags that.
    """
    inner_bound = PfnArchConfig().target_inner_bound
    shape = CharacterizationShape(512, 128, 100, 0, "regression")
    batch = source.tensor_batch(32, shape, "tail-guard")
    y = torch.cat([batch.y_train_std.reshape(-1), batch.y_test_std.reshape(-1)])
    beyond = float((y.abs() > inner_bound).float().mean())
    # observed ~0.03% on the current prior; 0.5% leaves ~15x headroom yet fires on a
    # materially heavier-tailed prior. Deterministic under the fixture's fixed seed.
    assert beyond < 5e-3, f"{beyond:.4%} of standardized targets fall beyond +-{inner_bound}"


def test_source_loads_exact_tuned_eta_artifact(tmp_path: Path) -> None:
    eta = build_hyperprior(HyperPriorConfig(log_snr_mean=1.25))
    path = tmp_path / "eta_best.json"
    path.write_text(json.dumps(hyperprior_to_dict(eta)))

    source = PriorTaskSource.from_eta_file(path, RandomStreams(7))
    assert source.eta == eta
    assert source.streams.base_seed == 7


def test_paired_sources_share_draws_but_keep_eta_specific_task_identity() -> None:
    baseline = build_hyperprior(HyperPriorConfig(log_snr_mean=1.0))
    perturbed = build_hyperprior(HyperPriorConfig(log_snr_mean=2.0))
    streams = RandomStreams(7)
    shape = CharacterizationShape(32, 16, 4, 0, "regression")
    first_source = PairedPriorTaskSource(baseline, streams, pairing_id="pilot-1")
    second_source = PairedPriorTaskSource(perturbed, streams, pairing_id="pilot-1")

    first = first_source.sample_batch(3, shape, "train", 0)
    second = second_source.sample_batch(3, shape, "train", 0)

    assert all(a.probe_fit.X.equals(b.probe_fit.X) for a, b in zip(first, second, strict=True))
    assert all(a.probe_score.X.equals(b.probe_score.X) for a, b in zip(first, second, strict=True))
    assert all(a.task_id != b.task_id for a, b in zip(first, second, strict=True))
    assert first_source.stream_provenance == {
        "version": "paired-prior-task-source-1",
        "base_seed": 7,
        "common_random_numbers": True,
        "pairing_id": "pilot-1",
    }


def test_pairing_id_separates_paired_experiments() -> None:
    eta = build_hyperprior(HyperPriorConfig())
    streams = RandomStreams(7)
    shape = CharacterizationShape(32, 16, 4, 0, "regression")

    first = PairedPriorTaskSource(eta, streams, pairing_id="pilot-1").sample_batch(1, shape, "train", 0)[0]
    second = PairedPriorTaskSource(eta, streams, pairing_id="pilot-2").sample_batch(1, shape, "train", 0)[0]

    assert first.task_id != second.task_id
    assert not first.probe_fit.X.equals(second.probe_fit.X)
