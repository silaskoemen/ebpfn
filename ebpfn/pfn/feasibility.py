"""Profile PFN training/inference cost across context sizes.

Answers whether the implemented architecture can train and infer over the intended
row/feature regime: for each ``(rows, features)`` cell it times a train step and an
inference pass and records peak memory, and it reports the realized jittered-shape
distribution so the training regime around the anchor is explicit. The backbone is
context-length agnostic, so one model is profiled at every cell.
"""

import resource
import time
from typing import Any

import numpy as np
import torch

from ebpfn.config.pfn import PfnArchConfig, PfnStudyModeConfig, PfnTrainConfig
from ebpfn.data.types import CharacterizationShape
from ebpfn.pfn.data import PriorTaskSource, TaskBatch
from ebpfn.pfn.distribution import fixed_borders
from ebpfn.pfn.model import EBPFNModel
from ebpfn.pfn.train import (
    anchor_shape,
    build_model,
    build_source,
    configure_device_memory,
    release_device_cache,
    select_device,
)
from ebpfn.priors.shapes import sample_training_shape
from ebpfn.utils import RandomRole, RandomStreams


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _is_out_of_memory(error: RuntimeError) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _process_peak_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if rss > 1e7 else rss / 1e3


def _peak_memory(device: torch.device, cpu_baseline_mb: float) -> tuple[float | None, str]:
    """Return the best available cell memory measurement and its semantics."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6, "cuda_peak_allocated"
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e6, "mps_driver_allocated_after_cell"
    return max(0.0, _process_peak_memory_mb() - cpu_baseline_mb), "cpu_process_peak_rss_growth"


def _time_ms(fn: Any, device: torch.device, reps: int) -> float:
    fn()  # warmup (allocations, lazy init)
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    _synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / reps


def _train_step(
    model: EBPFNModel,
    batch: TaskBatch,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = model.loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()


def profile_cell(
    model: EBPFNModel,
    batch: TaskBatch,
    device: torch.device,
    reps: int,
    *,
    cpu_baseline_mb: float,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    train_ms = _time_ms(lambda: _train_step(model, batch, optimizer, grad_clip), device, reps)
    infer_ms = _time_ms(lambda: model.predict_logits(batch.x, batch.y_train_std), device, reps)
    peak_memory_mb, memory_metric = _peak_memory(device, cpu_baseline_mb)
    return {
        "status": "ok",
        "rows": batch.x.shape[1],
        "features": batch.x.shape[2],
        "n_train": batch.n_train,
        "n_test": batch.n_test,
        "tasks": batch.batch_size,
        "train_ms": train_ms,
        "infer_ms": infer_ms,
        "peak_memory_mb": peak_memory_mb,
        "memory_metric": memory_metric,
    }


def realized_shape_report(
    anchor: CharacterizationShape, train: PfnTrainConfig, streams: RandomStreams, *, n: int = 256
) -> dict[str, Any]:
    """Quantiles of the jittered training shapes drawn around the anchor."""
    n_train, p_train = [], []
    for step in range(n):
        rng = streams.generator(RandomRole.PFN_TRAINING, "profile-shape", step)
        shape, _ = sample_training_shape(anchor, train.jitter, rng)
        n_train.append(shape.n_probe_fit + shape.n_probe_score)
        p_train.append(shape.p_numeric)

    def quantiles(values: list[int]) -> dict[str, float]:
        arr = np.asarray(values)
        return {q: float(np.quantile(arr, float(q))) for q in ("0.1", "0.5", "0.9")}

    return {"n_rows": quantiles(n_train), "n_features": quantiles(p_train)}


def profile(
    arch: PfnArchConfig,
    train: PfnTrainConfig,
    mode: PfnStudyModeConfig,
    *,
    source: PriorTaskSource | None = None,
    reps: int = 3,
) -> dict[str, Any]:
    """Profile ``arch`` over the ``mode`` feasibility grid and return a report dict."""
    device = select_device(train.device)
    configure_device_memory(device)
    streams = RandomStreams(train.seed)
    if source is None:
        source = build_source(train, streams)
    anchor = anchor_shape(train)
    borders = fixed_borders(
        arch.n_bins,
        inner_bound=arch.target_inner_bound,
        tail_scale=arch.target_tail_scale,
    )
    torch.manual_seed(train.seed)
    reference_model = build_model(arch, borders)
    n_parameters = int(sum(parameter.numel() for parameter in reference_model.parameters()))
    del reference_model

    cells: list[dict[str, Any]] = []
    for rows in mode.profile_rows:
        for features in mode.profile_features:
            n_fit = max(2, round(rows * 0.8))
            n_score = max(1, rows - n_fit)
            shape = CharacterizationShape(n_fit, n_score, features, 0, "regression")
            model: EBPFNModel | None = None
            batch: TaskBatch | None = None
            cpu_baseline_mb = _process_peak_memory_mb()
            try:
                batch = source.tensor_batch(mode.profile_tasks, shape, "profile", rows, features).to(device)
                torch.manual_seed(train.seed)
                model = build_model(arch, borders).to(device)
                cell = profile_cell(
                    model,
                    batch,
                    device,
                    reps,
                    cpu_baseline_mb=cpu_baseline_mb,
                    lr=train.lr,
                    weight_decay=train.weight_decay,
                    grad_clip=train.grad_clip,
                )
                cell["projected_train_hours"] = (
                    cell["train_ms"] * train.gradient_accumulation_steps * train.steps / 3_600_000.0
                )
                cells.append(cell)
            except RuntimeError as error:
                if not _is_out_of_memory(error):
                    raise
                cells.append(
                    {
                        "status": "oom",
                        "rows": rows,
                        "features": features,
                        "n_train": n_fit,
                        "n_test": n_score,
                        "tasks": mode.profile_tasks,
                        "train_ms": None,
                        "infer_ms": None,
                        "peak_memory_mb": None,
                        "memory_metric": None,
                        "projected_train_hours": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                del model, batch
                release_device_cache(device)

    anchor_rows = anchor.n_probe_fit + anchor.n_probe_score
    in_regime = (
        train.jitter.n_min <= anchor_rows <= train.jitter.n_max
        and train.jitter.p_min <= anchor.p_numeric <= train.jitter.p_max
        and anchor_rows <= arch.max_context
        and train.jitter.n_max <= arch.max_context
    )
    all_cells_feasible = all(cell["status"] == "ok" for cell in cells)
    return {
        "device": str(device),
        "n_parameters": n_parameters,
        "cells": cells,
        "realized_shapes": realized_shape_report(anchor, train, streams),
        "anchor": {"rows": anchor_rows, "features": anchor.p_numeric},
        "max_context": arch.max_context,
        "in_regime": bool(in_regime),
        "all_cells_feasible": all_cells_feasible,
        "profile_tasks_match_training": mode.profile_tasks == train.tasks_per_step,
        "gradient_accumulation_steps": train.gradient_accumulation_steps,
        "effective_tasks_per_step": train.tasks_per_step * train.gradient_accumulation_steps,
    }
