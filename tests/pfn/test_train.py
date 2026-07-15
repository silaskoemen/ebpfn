import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from ebpfn.config.pfn import PfnArchConfig, PfnTrainConfig
from ebpfn.config.prior import HyperPriorConfig
from ebpfn.data.types import CharacterizationShape
from ebpfn.pfn.data import PairedPriorTaskSource, PriorTaskSource
from ebpfn.pfn.train import (
    MPS_MEMORY_FRACTION,
    build_source,
    configure_device_memory,
    load_checkpoint,
    release_device_cache,
    select_device,
    train_pfn,
)
from ebpfn.priors import build_hyperprior
from ebpfn.utils import RandomStreams


def _configs(steps: int) -> tuple[PfnArchConfig, PfnTrainConfig]:
    arch = PfnArchConfig(
        n_bins=32,
        embed_dim=16,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        n_cls_rows=8,
    )
    train = PfnTrainConfig(
        seed=0,
        steps=steps,
        tasks_per_step=4,
        lr=2e-3,
        warmup_steps=min(5, steps),
        checkpoint_interval=1000,
        anchor_probe_fit=32,
        anchor_probe_score=16,
        anchor_features=4,
        device="cpu",
    )
    return arch, train


def test_select_device_explicit() -> None:
    assert select_device("cpu").type == "cpu"


def test_partial_training_horizon_can_end_during_warmup() -> None:
    assert PfnTrainConfig(steps=1, warmup_steps=100).warmup_steps == 100


def test_mps_memory_guard_and_cache_release(monkeypatch) -> None:
    fractions: list[float] = []
    cache_clears = 0

    def set_fraction(value: float) -> None:
        fractions.append(value)

    def empty_cache() -> None:
        nonlocal cache_clears
        cache_clears += 1

    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", set_fraction)
    monkeypatch.setattr(torch.mps, "empty_cache", empty_cache)

    device = torch.device("mps")
    configure_device_memory(device)
    release_device_cache(device)

    assert fractions == [MPS_MEMORY_FRACTION]
    assert cache_clears == 1


def test_training_reduces_loss() -> None:
    arch, train = _configs(steps=60)
    _, result = train_pfn(arch, train, log_every=0)
    losses = np.asarray(result.losses)
    assert losses[-10:].mean() < losses[:10].mean()
    assert result.steps == 60


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    arch, train = _configs(steps=3)
    model, result = train_pfn(arch, train, checkpoint_dir=tmp_path, log_every=0)
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()

    batch = build_source(train, RandomStreams(train.seed)).tensor_batch(
        2, CharacterizationShape(32, 16, 4, 0, "regression"), "eval"
    )
    reloaded, checkpoint = load_checkpoint(result.checkpoint_path)
    assert checkpoint["step"] == 3
    expected = model.predict_logits(batch.x, batch.y_train_std)
    actual = reloaded.predict_logits(batch.x, batch.y_train_std)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_checkpoints_are_step_specific_and_training_resumes_exactly(tmp_path: Path) -> None:
    arch, full_train = _configs(steps=4)
    full_train = full_train.model_copy(update={"checkpoint_interval": 2, "warmup_steps": 0})
    _, uninterrupted_result = train_pfn(arch, full_train, log_every=0)

    first_train = full_train.model_copy(update={"steps": 2})
    source = PairedPriorTaskSource(
        build_hyperprior(first_train.prior),
        RandomStreams(first_train.seed),
        pairing_id="resume-test",
    )
    _, first_result = train_pfn(arch, first_train, source=source, checkpoint_dir=tmp_path, log_every=0)
    assert first_result.checkpoint_path == tmp_path / "checkpoint_step_00000002.pt"

    resumed, resumed_result = train_pfn(
        arch,
        full_train,
        source=source,
        checkpoint_dir=tmp_path,
        resume_from=first_result.checkpoint_path,
        log_every=0,
    )

    paired_uninterrupted, paired_result = train_pfn(arch, full_train, source=source, log_every=0)
    assert uninterrupted_result.steps == resumed_result.steps == paired_result.steps == 4
    assert resumed_result.losses == paired_result.losses
    assert all(
        torch.equal(resumed.state_dict()[name], value) for name, value in paired_uninterrupted.state_dict().items()
    )
    assert {path.name for path in resumed_result.checkpoint_paths} == {
        "checkpoint_step_00000002.pt",
        "checkpoint_step_00000004.pt",
    }


def test_identical_initialization_produces_identical_parameters_after_one_step() -> None:
    arch, train = _configs(steps=1)
    first, first_result = train_pfn(arch, train, log_every=0)
    _ = torch.rand(100)
    second, second_result = train_pfn(arch, train, log_every=0)

    assert first_result.losses == second_result.losses
    assert all(torch.equal(first.state_dict()[name], value) for name, value in second.state_dict().items())


def test_gradient_accumulation_draws_each_microbatch_and_keeps_optimizer_step_count(monkeypatch) -> None:
    arch, train = _configs(steps=2)
    train = train.model_copy(update={"tasks_per_step": 1, "gradient_accumulation_steps": 3})
    batch_calls = 0
    optimizer_steps = 0
    original_batch = PriorTaskSource.tensor_batch
    original_step = torch.optim.AdamW.step

    def counting_batch(self, *args, **kwargs):
        nonlocal batch_calls
        batch_calls += 1
        return original_batch(self, *args, **kwargs)

    def counting_step(self, closure=None):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(self, closure)

    monkeypatch.setattr(PriorTaskSource, "tensor_batch", counting_batch)
    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)

    train_pfn(arch, train, log_every=0)

    assert batch_calls == 6
    assert optimizer_steps == 2


def test_custom_source_seed_must_match_training_seed() -> None:
    arch, train = _configs(steps=1)
    source = PriorTaskSource(build_hyperprior(HyperPriorConfig()), RandomStreams(train.seed + 1))

    with pytest.raises(ValueError, match="source seed"):
        train_pfn(arch, train, source=source, log_every=0)


def test_baseline_and_tuned_sources_use_identical_fixed_borders() -> None:
    arch, train = _configs(steps=1)
    baseline = PriorTaskSource(build_hyperprior(HyperPriorConfig()), RandomStreams(train.seed))
    tuned = PriorTaskSource(
        build_hyperprior(HyperPriorConfig(log_snr_mean=1.25, heavy_tail_rate=0.8)),
        RandomStreams(train.seed),
    )

    baseline_model, _ = train_pfn(arch, train, source=baseline, log_every=0)
    tuned_model, _ = train_pfn(arch, train, source=tuned, log_every=0)
    assert torch.equal(baseline_model.distribution.borders, tuned_model.distribution.borders)


def test_checkpoint_records_exact_tuned_source_and_training_config(tmp_path: Path) -> None:
    arch, train = _configs(steps=1)
    eta = build_hyperprior(HyperPriorConfig(log_snr_mean=1.25))
    source = PriorTaskSource(eta, RandomStreams(train.seed))

    _, result = train_pfn(arch, train, source=source, checkpoint_dir=tmp_path, log_every=0)
    assert result.checkpoint_path is not None
    _, checkpoint = load_checkpoint(result.checkpoint_path)

    assert checkpoint["train"] == train.model_dump(mode="json")
    assert checkpoint["source_eta"] == dataclasses.asdict(eta)
    assert checkpoint["source_seed"] == train.seed
    assert checkpoint["source_stream"] == source.stream_provenance
