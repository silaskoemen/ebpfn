"""Prior-fitted-network training loop.

Each step samples a fresh batch of synthetic tasks at one jittered shape from a
:class:`PriorTaskSource` and minimizes the mean bar-distribution NLL of the query
targets. The loop is decoupled from where tasks come from: pass any task source (the
baseline prior today, a tuned prior later) and it trains identically.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import torch
from loguru import logger
from torch import Tensor

from ebpfn.config.pfn import PfnArchConfig, PfnTrainConfig
from ebpfn.data.types import CharacterizationShape
from ebpfn.pfn.data import PriorTaskSource
from ebpfn.pfn.distribution import fixed_borders
from ebpfn.pfn.model import EBPFNModel
from ebpfn.priors import build_hyperprior, hyperprior_to_dict
from ebpfn.priors.shapes import sample_training_shape
from ebpfn.utils import RandomRole, RandomStreams

MPS_MEMORY_FRACTION = 0.6


def select_device(preference: str) -> torch.device:
    """Resolve ``"auto"`` to the best available accelerator (cuda > mps > cpu)."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_device_memory(device: torch.device) -> None:
    """Apply allocator limits that keep accelerator use from starving the host."""
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)


def release_device_cache(device: torch.device) -> None:
    """Release inactive allocations between differently shaped optimizer steps."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def build_model(arch: PfnArchConfig, borders: Tensor) -> EBPFNModel:
    """Construct an :class:`EBPFNModel` from an architecture config and fitted borders."""
    if borders.numel() - 1 != arch.n_bins:
        raise ValueError(f"borders imply {borders.numel() - 1} bins but arch.n_bins is {arch.n_bins}")
    return EBPFNModel(
        borders,
        embed_dim=arch.embed_dim,
        col_num_blocks=arch.col_num_blocks,
        row_num_blocks=arch.row_num_blocks,
        icl_num_blocks=arch.icl_num_blocks,
        col_nhead=arch.col_nhead,
        row_nhead=arch.row_nhead,
        icl_nhead=arch.icl_nhead,
        feature_group_size=arch.feature_group_size,
        n_cls_cols=arch.n_cls_cols,
        n_cls_rows=arch.n_cls_rows,
    )


def anchor_shape(train: PfnTrainConfig) -> CharacterizationShape:
    return CharacterizationShape(
        train.anchor_probe_fit, train.anchor_probe_score, train.anchor_features, 0, "regression"
    )


def build_source(train: PfnTrainConfig, streams: RandomStreams) -> PriorTaskSource:
    return PriorTaskSource(build_hyperprior(train.prior), streams)


def _warmup_scale(step: int, warmup_steps: int) -> float:
    return 1.0 if warmup_steps == 0 else min(1.0, (step + 1) / warmup_steps)


@dataclasses.dataclass
class TrainResult:
    losses: list[float]
    steps: int
    device: str
    checkpoint_path: Path | None
    checkpoint_paths: tuple[Path, ...]


def save_checkpoint(
    directory: Path,
    model: EBPFNModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    arch: PfnArchConfig,
    train: PfnTrainConfig,
    source: PriorTaskSource,
    losses: list[float],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"checkpoint_step_{step:08d}.pt"
    torch.save(
        {
            "checkpoint_version": "pfn-training-checkpoint-2",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "losses": list(losses),
            "borders": model.distribution.borders.detach().cpu(),
            "arch": arch.model_dump(mode="json"),
            "train": train.model_dump(mode="json"),
            "source_eta": hyperprior_to_dict(source.eta),
            "source_seed": source.streams.base_seed,
            "source_stream": source.stream_provenance,
        },
        path,
    )
    return path


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> tuple[EBPFNModel, dict]:
    """Rebuild a model from a checkpoint and return it with the raw checkpoint dict."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    arch = PfnArchConfig.model_validate(checkpoint["arch"])
    model = build_model(arch, checkpoint["borders"].to(map_location))
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def _validate_resume_checkpoint(
    checkpoint: dict,
    arch: PfnArchConfig,
    train: PfnTrainConfig,
    source: PriorTaskSource,
) -> None:
    if checkpoint.get("checkpoint_version") != "pfn-training-checkpoint-2":
        raise ValueError("resume checkpoint has an unsupported version")
    if checkpoint["arch"] != arch.model_dump(mode="json"):
        raise ValueError("resume checkpoint architecture does not match the requested architecture")
    stored_train = dict(checkpoint["train"])
    requested_train = train.model_dump(mode="json")
    stored_steps = int(stored_train.pop("steps"))
    requested_steps = int(requested_train.pop("steps"))
    if stored_train != requested_train:
        raise ValueError("resume checkpoint training config does not match the requested training config")
    completed_steps = int(checkpoint["step"])
    if completed_steps > stored_steps or completed_steps > requested_steps:
        raise ValueError("resume checkpoint step lies beyond its stored or requested training horizon")
    if checkpoint["source_eta"] != hyperprior_to_dict(source.eta):
        raise ValueError("resume checkpoint eta does not match the requested task source")
    if checkpoint["source_stream"] != source.stream_provenance:
        raise ValueError("resume checkpoint stream contract does not match the requested task source")


def train_pfn(
    arch: PfnArchConfig,
    train: PfnTrainConfig,
    *,
    source: PriorTaskSource | None = None,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
    init_weights_from: Path | None = None,
    log_every: int = 50,
    on_step: Callable[[int, float], None] | None = None,
) -> tuple[EBPFNModel, TrainResult]:
    """Train a PFN against ``source`` (defaults to the config's baseline prior).

    ``on_step(step, loss)`` is an optional hook (e.g. an experiment-tracking logger).
    Checkpoints are written every ``train.checkpoint_interval`` steps when
    ``checkpoint_dir`` is set, plus once at the end.

    ``resume_from`` continues an interrupted run and requires an exact eta/config
    match. ``init_weights_from`` is different: it seeds a **fine-tune** from another
    checkpoint's weights under a possibly *different* ``source`` eta -- only the
    architecture must match; optimizer state and step counter start fresh. This is
    the V2 bounded-fine-tune-from-base mechanism (``PLAN.md``).
    """
    if resume_from is not None and init_weights_from is not None:
        raise ValueError("pass at most one of resume_from / init_weights_from")
    device = select_device(train.device)
    configure_device_memory(device)
    streams = RandomStreams(train.seed)
    if source is None:
        source = build_source(train, streams)
    elif source.streams.base_seed != train.seed:
        raise ValueError(f"source seed {source.streams.base_seed} does not match configured PFN seed {train.seed}")
    anchor = anchor_shape(train)
    borders = fixed_borders(
        arch.n_bins,
        inner_bound=arch.target_inner_bound,
        tail_scale=arch.target_tail_scale,
    )
    torch.manual_seed(train.seed)
    model = build_model(arch, borders).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train.lr, weight_decay=train.weight_decay)
    start_step = 0
    losses: list[float] = []
    checkpoint_paths: list[Path] = []
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        _validate_resume_checkpoint(checkpoint, arch, train, source)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        losses = [float(value) for value in checkpoint["losses"]]
        if len(losses) != start_step:
            raise ValueError("resume checkpoint loss history does not align with its completed step")
        checkpoint_paths.append(resume_from)
    elif init_weights_from is not None:
        checkpoint = torch.load(init_weights_from, map_location=device, weights_only=False)
        if checkpoint["arch"] != arch.model_dump(mode="json"):
            raise ValueError("init-weights checkpoint architecture does not match the requested architecture")
        model.load_state_dict(checkpoint["model"])  # weights only: fresh optimizer, start_step=0, new eta allowed

    logger.info(
        f"🧠 pfn train | {train.steps} steps | device={device} | n_bins={arch.n_bins} | anchor={anchor} | "
        f"microbatch={train.tasks_per_step} x accumulation={train.gradient_accumulation_steps}"
    )
    model.train()
    for step in range(start_step, train.steps):
        shape_rng = streams.generator(RandomRole.PFN_TRAINING, "shape", step)
        shape, _ = sample_training_shape(anchor, train.jitter, shape_rng)
        scaled_lr = train.lr * _warmup_scale(step, train.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = scaled_lr

        optimizer.zero_grad(set_to_none=True)
        microbatch_losses: list[float] = []
        for accumulation_step in range(train.gradient_accumulation_steps):
            batch = source.tensor_batch(
                train.tasks_per_step,
                shape,
                "train",
                step,
                "accumulation",
                accumulation_step,
            ).to(device)
            loss = model.loss(batch)
            (loss / train.gradient_accumulation_steps).backward()
            microbatch_losses.append(float(loss.detach()))
            del batch, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), train.grad_clip)
        optimizer.step()
        if device.type == "mps":
            release_device_cache(device)

        loss_value = float(sum(microbatch_losses) / len(microbatch_losses))
        losses.append(loss_value)
        if on_step is not None:
            on_step(step, loss_value)
        if log_every and (step % log_every == 0 or step == train.steps - 1):
            logger.info(f"  step {step + 1}/{train.steps} | loss={loss_value:.4f} | lr={scaled_lr:.2e}")
        if checkpoint_dir is not None and (step + 1) % train.checkpoint_interval == 0:
            checkpoint_paths.append(
                save_checkpoint(checkpoint_dir, model, optimizer, step + 1, arch, train, source, losses)
            )

    checkpoint_path = None
    if checkpoint_dir is not None:
        expected = checkpoint_dir / f"checkpoint_step_{train.steps:08d}.pt"
        if not checkpoint_paths or checkpoint_paths[-1] != expected:
            checkpoint_paths.append(
                save_checkpoint(checkpoint_dir, model, optimizer, train.steps, arch, train, source, losses)
            )
        checkpoint_path = expected
    return model, TrainResult(
        losses=losses,
        steps=train.steps,
        device=str(device),
        checkpoint_path=checkpoint_path,
        checkpoint_paths=tuple(checkpoint_paths),
    )
