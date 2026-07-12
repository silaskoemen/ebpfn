import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from ebpfn.config.pfn import PfnArchConfig, PfnTrainConfig
from ebpfn.config.prior import HyperPriorConfig
from ebpfn.data.types import CharacterizationShape
from ebpfn.pfn.data import PriorTaskSource
from ebpfn.pfn.train import build_source, load_checkpoint, select_device, train_pfn
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


def test_identical_initialization_produces_identical_parameters_after_one_step() -> None:
    arch, train = _configs(steps=1)
    first, first_result = train_pfn(arch, train, log_every=0)
    _ = torch.rand(100)
    second, second_result = train_pfn(arch, train, log_every=0)

    assert first_result.losses == second_result.losses
    assert all(torch.equal(first.state_dict()[name], value) for name, value in second.state_dict().items())


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
