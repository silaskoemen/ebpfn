"""Small-context gate: is the base's small-context lag an approximation gap that training closes?

The prequential study found Kumo and TabPFN beat the released base by ~0.3-0.4 nats per
row at 8-32 context rows *on samples from the base's own prior*, so the lag is
approximation, not prior mass. TabICLv2 pretrained on long contexts; if it simply never
learned tiny ones, continued training on the same nanoprior closes it.

Training on small contexts alone does (8-128 rows, lr 1e-5, 250 steps: +0.22 nats per
row at 8-32 on real data) but forgets large ones (-0.06 at 128-512). The arms therefore
train on a context mixture, all fine-tuned from the released base on the origin nanoprior:

* **mixed** -- context rows log-uniform in ``MIXED_FIT``, at two learning rates;
* **large** -- the same steps at ``LARGE_FIT`` contexts: the "any more training" null;
* **distill** -- the mixed contexts, but above ``DISTILL_ABOVE`` context rows the target is
  the frozen base's quantiles (L1), not the data.

Both mixed and large lose at 128+ rows, on fresh draws from the very distribution they
train on, pinball included: at a converged optimum fresh-Adam steps are pure noise. The
distillation gradient is exactly zero while the model agrees with the base, so it keeps
the base where the base is already optimal and learns only where it is not.

``train`` saves each arm's weights at ``SAVE_STEPS``, per ``--seed`` (the task stream and
context shapes; the seed spread of one arm is the noise floor a tuned prior must clear);
``evaluate`` scores every saved
checkpoint prequentially on the datasets ``prequential.py`` scores at the same ``--rows``,
so its rows join one to one with the reference models' rows. Real datasets use the first
``REAL_ORDERINGS`` orderings only.

Run from project root (wrap in caffeinate):
    pixi run python -m benchmarks.studies.small_context train [--arms mixed_1e-5 ...] [--seed 1]
    pixi run python -m benchmarks.studies.small_context evaluate [--rows 2048] [--seed 1]
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.base import released_base_checkpoint
from benchmarks.data.regression_corpus import ROWS
from benchmarks.studies.prequential import PRIOR_DATASETS, PRIOR_ORDERINGS, load_datasets, model_scorer, prequential
from ebpfn.data import CharacterizationShape
from ebpfn.pfn.data import NanopriorTaskSource, TaskBatch, collate_tasks
from ebpfn.pfn.model import EBPFNModel
from ebpfn.pfn.train import (
    configure_device_memory,
    load_checkpoint,
    release_device_cache,
    select_device,
    trainable_parameters,
)
from ebpfn.priors.nanoprior.eta import ORIGIN_ETA, PriorEta
from ebpfn.utils import RandomStreams

OUT_DIR = Path("benchmarks/results/small_context")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
SEED = 1  # default training seed; the prequential controls are drawn at seed 0
STEPS = 1000
SAVE_STEPS = (250, 1000)
TASKS_PER_STEP = 16
MICROBATCH_CELLS = 25_000  # (context + query) rows x features, summed over a microbatch's tasks: fits MPS
MICROBATCH_ROWS = 2_500  # rows summed over a microbatch's tasks: row tokens and ICL attention, not cells, dominate
TASK_CELLS = 60_000  # features are cut so one task fits MPS at the longest contexts
N_QUERY = 32
FEATURES = (2, 100)  # log-uniform feature count per step
MIXED_FIT = (8, 1024)
LARGE_FIT = (256, 1024)
DISTILL_ABOVE = 128  # context rows past which the base and Kumo already agree on prior samples
WARMUP_STEPS = 50
GRAD_CLIP = 1.0
MAX_ATTEMPTS = 50
REAL_ORDERINGS = 2
ARMS = {  # name -> (context range, lr, distill above this many context rows)
    "mixed_1e-5": (MIXED_FIT, 1e-5, None),
    "mixed_3e-5": (MIXED_FIT, 3e-5, None),
    "large_1e-5": (LARGE_FIT, 1e-5, None),
    "distill_1e-5": (MIXED_FIT, 1e-5, DISTILL_ABOVE),
}


def log_uniform_int(rng: np.random.Generator, low: int, high: int) -> int:
    return round(np.exp(rng.uniform(np.log(low), np.log(high))))


def run_name(name: str, seed: int) -> str:
    return f"{name}_seed{seed}"


def checkpoint_path(name: str, seed: int, step: int) -> Path:
    return CHECKPOINT_DIR / f"{run_name(name, seed)}_step{step}.pt"


def training_shape(rng: np.random.Generator, fit: tuple[int, int]) -> CharacterizationShape:
    n_fit = log_uniform_int(rng, *fit)
    p = min(log_uniform_int(rng, *FEATURES), TASK_CELLS // (n_fit + N_QUERY))
    return CharacterizationShape(n_fit, N_QUERY, p, 0, "regression")


def distill_loss(model: EBPFNModel, teacher: EBPFNModel, batch: TaskBatch) -> torch.Tensor:
    """Mean L1 between the model's and the frozen teacher's quantiles on the query rows."""
    with torch.no_grad():
        target = teacher(batch.x, batch.y_train_std)
    return (model(batch.x, batch.y_train_std) - target).abs().mean()


def fine_tune(
    label: str,
    fit: tuple[int, int],
    lr: float,
    distill_above: int | None,
    seed: int,
    device: torch.device,
    checkpoints: Mapping[int, Path],
    eta: PriorEta = ORIGIN_ETA,
) -> list[float]:
    """Fine-tune the released base on the nanoprior at ``eta``, saving weights at each of ``checkpoints``' steps.

    The task stream and context shapes depend only on ``seed``, and nanoprior seeds tasks
    from identity rather than ``eta``, so runs at one seed share random numbers across etas.
    """
    model, _ = load_checkpoint(released_base_checkpoint(), map_location=device)
    model.to(device).train()
    teacher = None
    if distill_above is not None:
        teacher, _ = load_checkpoint(released_base_checkpoint(), map_location=device)
        teacher.to(device).eval().requires_grad_(False)
    params = trainable_parameters(model, "all")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    source = NanopriorTaskSource(RandomStreams(seed), eta=eta, max_attempts=MAX_ATTEMPTS)
    losses = []
    for step in range(STEPS):
        shape = training_shape(np.random.default_rng([seed, step]), fit)  # shared across arms with one range
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / WARMUP_STEPS)
        tasks = source.sample_batch(TASKS_PER_STEP, shape, "small_context", step)
        rows = shape.n_probe_fit + shape.n_probe_score
        micro = int(
            np.clip(min(MICROBATCH_CELLS // (rows * shape.p_numeric), MICROBATCH_ROWS // rows), 1, TASKS_PER_STEP)
        )
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for start in range(0, TASKS_PER_STEP, micro):
            chunk = tasks[start : start + micro]
            batch = collate_tasks(chunk).to(device)
            if teacher is not None and distill_above is not None and shape.n_probe_fit > distill_above:
                loss = distill_loss(model, teacher, batch)
            else:
                loss = model.loss(batch)
            (loss * len(chunk) / TASKS_PER_STEP).backward()
            total += float(loss.detach()) * len(chunk) / TASKS_PER_STEP
            del loss
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()
        losses.append(total)
        if device.type == "mps":
            release_device_cache(device)
        if step % 50 == 0:
            print(f"  {label} step {step} | median loss {np.median(losses[-50:]):.4f} | {shape}", flush=True)
        if step + 1 in checkpoints:
            torch.save(model.state_dict(), checkpoints[step + 1])
            print(f"  {label} saved step {step + 1}", flush=True)
    return losses


def train_arm(name: str, seed: int, device: torch.device) -> None:
    fit, lr, distill_above = ARMS[name]
    checkpoints = {step: checkpoint_path(name, seed, step) for step in SAVE_STEPS}
    losses = fine_tune(run_name(name, seed), fit, lr, distill_above, seed, device, checkpoints)
    (OUT_DIR / f"{run_name(name, seed)}_losses.json").write_text(json.dumps(losses))


def evaluate(model: EBPFNModel, device: torch.device, datasets: list, out: Path) -> None:
    """Prequential rows for every dataset, written the way ``prequential.py`` writes them."""
    out.mkdir(parents=True, exist_ok=True)
    score = model_scorer(model, device)
    for source, name, x, y in datasets:
        path = out / f"{source}__{name.replace('/', '_')}.parquet"
        if path.exists():
            continue
        y_std = (y - y.mean()) / (y.std() or 1.0)
        orderings = REAL_ORDERINGS if source == "real" else PRIOR_ORDERINGS
        frame = prequential(score, x, y_std, orderings, f"{source}/{name}")
        frame.with_columns(source=pl.lit(source), dataset=pl.lit(name), model=pl.lit(out.name)).write_parquet(path)
        release_device_cache(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--rows", type=int, default=ROWS, help="evaluate: rows per corpus dataset and control")
    parser.add_argument("--seed", type=int, default=SEED, help="training seed: task stream and context shapes")
    args = parser.parse_args()

    device = select_device("auto")
    configure_device_memory(device)
    if args.command == "train":
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        for name in args.arms:
            if not all(checkpoint_path(name, args.seed, step).exists() for step in SAVE_STEPS):
                train_arm(name, args.seed, device)
        return

    datasets = load_datasets(PRIOR_DATASETS, args.rows)
    model, _ = load_checkpoint(released_base_checkpoint(), map_location=device)
    model.to(device)
    for name in args.arms:
        for step in SAVE_STEPS:
            path = checkpoint_path(name, args.seed, step)
            if path.exists():
                model.load_state_dict(torch.load(path, map_location=device))
                evaluate(model, device, datasets, OUT_DIR / f"rows{args.rows}" / path.stem)
                print(f"  {path.stem} evaluated at {args.rows} rows", flush=True)


if __name__ == "__main__":
    main()
