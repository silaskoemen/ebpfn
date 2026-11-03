"""Multi-step gate: does an alignment-selected pool of origin tasks help after training?

The one-step gate showed single-task updates at a converged base are curvature noise.
What survives, if anything, is the *average*: a pool whose mean preconditioned update
aligns with the target gradient. Per target, ``N_CANDIDATES`` origin tasks at the
target's shape are scored by first-order alignment ``-g_T . u_s`` and three kinds of
pool are trained on for ``STEPS`` warm-Adam steps from the released base:

* **own** -- the top ``POOL_SIZE`` by this target's score;
* **cross** -- the top ``POOL_SIZE`` by another target's score: alignment that is not
  about this target, so whatever it buys is generic;
* **random** -- ``N_RANDOM`` uniform pools, the P2-style null band for this protocol.

Two fixes over the one-step gate. ``g_T`` is averaged over ``N_SPLITS`` resampled
context/query splits of the selection rows, so it no longer carries one split's noise.
And the target's query rows are halved once into SEL (scoring may see them) and TEST
(only ever evaluated, always against the original context).

Run from project root (wrap in caffeinate):
    pixi run python -m benchmarks.studies.pool_training [--lr 3e-5] [--steps 100]
"""

import argparse
import dataclasses
import json
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.base import released_base_checkpoint
from benchmarks.corpus import CHAR_DIR, iter_real_tasks
from benchmarks.studies.gradient_alignment import flat_grad, query_subset
from ebpfn.data import CharacterizationShape, TaskPartition, TuningTask
from ebpfn.pfn.data import NanopriorTaskSource, TaskBatch, collate_tasks
from ebpfn.pfn.metrics import compute_row_metrics
from ebpfn.pfn.model import EBPFNModel
from ebpfn.pfn.train import (
    configure_device_memory,
    load_checkpoint,
    release_device_cache,
    select_device,
    trainable_parameters,
)
from ebpfn.utils import RandomStreams

OUT_DIR = Path("benchmarks/results/pool_training")
SEED = 0
SCOPE = "icl"
N_PRECONDITION = 32
N_SPLITS = 16
N_CANDIDATES = 512
POOL_SIZE = 128
N_RANDOM = 3
TASKS_PER_STEP = 8
EVAL_STEPS = (0, 10, 25, 50, 100, 200)
N_PRIOR_EVAL = 8
BETA1, BETA2 = 0.9, 0.999
ADAM_EPS = 1e-8
WARM_STEP = 10_000  # Adam step count the warm state claims: bias corrections ~= 1
GRAD_CLIP = 1.0
MAX_ATTEMPTS = 50


@dataclasses.dataclass(frozen=True)
class Target:
    name: str
    shape: CharacterizationShape
    task: TuningTask
    sel: np.ndarray  # query-row indices scoring may see
    test: np.ndarray  # query-row indices only ever evaluated


def load_targets() -> list[Target]:
    """Repeat 0 of every corpus dataset, query rows halved into SEL and TEST."""
    targets = []
    for name, repeat, task in iter_real_tasks(CHAR_DIR):
        if repeat != 0:
            continue
        n_test = task.probe_score.X.height
        order = np.random.default_rng(SEED).permutation(n_test)
        shape = CharacterizationShape(task.probe_fit.X.height, n_test, task.probe_fit.X.width, 0, "regression")
        targets.append(Target(name, shape, task, order[: n_test // 2], order[n_test // 2 :]))
    return targets


def _rows(partition: TaskPartition, index: np.ndarray) -> TaskPartition:
    return TaskPartition(partition.X[index], partition.y[index], partition.row_ids[index])


def resplits(target: Target, count: int) -> list[TaskBatch]:
    """Fresh context/query splits of the context and SEL rows, keeping the context size."""
    fit, sel = target.task.probe_fit, _rows(target.task.probe_score, target.sel)
    pool = TaskPartition(
        pl.concat([fit.X, sel.X]), np.concatenate([fit.y, sel.y]), np.concatenate([fit.row_ids, sel.row_ids])
    )
    rng = np.random.default_rng([SEED, zlib.crc32(target.name.encode())])
    batches = []
    for _ in range(count):
        order = rng.permutation(pool.X.height)
        split = dataclasses.replace(
            target.task, probe_fit=_rows(pool, order[: fit.X.height]), probe_score=_rows(pool, order[fit.X.height :])
        )
        batches.append(collate_tasks([split]))
    return batches


def target_gradient(
    model: EBPFNModel, params: list[torch.nn.Parameter], device: torch.device, target: Target
) -> torch.Tensor:
    return torch.stack([flat_grad(model, params, batch.to(device)) for batch in resplits(target, N_SPLITS)]).mean(0)


@torch.no_grad()
def evaluate(model: EBPFNModel, device: torch.device, test: TaskBatch, sel: TaskBatch, prior: list[TaskBatch]) -> dict:
    batch = test.to(device)
    rows = compute_row_metrics(model.distribution, model(batch.x, batch.y_train_std)[0], batch.y_test_std[0])
    return {
        "test_nll": float(rows.nll.mean()),
        "test_nll_coarse": float(rows.nll_coarse.mean()),
        "test_crps": float(rows.crps.mean()),
        "test_nll_rows": rows.nll.cpu().tolist(),
        "test_nll_coarse_rows": rows.nll_coarse.cpu().tolist(),
        "test_crps_rows": rows.crps.cpu().tolist(),
        "sel_loss": float(model.loss(sel.to(device))),
        "prior_loss": float(np.mean([float(model.loss(b.to(device))) for b in prior])),
    }


def train_pool(
    model: EBPFNModel,
    params: list[torch.nn.Parameter],
    device: torch.device,
    initial: list[torch.Tensor],
    second: torch.Tensor,
    pool: list[TaskBatch],
    seed: int,
    lr: float,
    steps: int,
    probe,
) -> list[dict]:
    """Warm-Adam steps on batches drawn from ``pool``; ``probe()`` at every eval step."""
    with torch.no_grad():
        for p, value in zip(params, initial, strict=True):
            p.copy_(value)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=0.0)
    offset = 0
    for p in params:
        state = optimizer.state[p]
        state["step"] = torch.tensor(float(WARM_STEP))
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = second[offset : offset + p.numel()].view_as(p).clone()
        offset += p.numel()
    rng = np.random.default_rng(seed)
    records = [{"step": 0} | probe()]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for index in rng.integers(len(pool), size=TASKS_PER_STEP):
            (model.loss(pool[index].to(device)) / TASKS_PER_STEP).backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()
        if device.type == "mps":
            release_device_cache(device)
        if step in EVAL_STEPS:
            records.append({"step": step} | probe())
    return records


def run_target(
    model: EBPFNModel,
    params: list[torch.nn.Parameter],
    device: torch.device,
    target: Target,
    cross: str,
    reference: dict[str, torch.Tensor],
    source: NanopriorTaskSource,
    lr: float,
    steps: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Training curves per arm, and every candidate's score under every target."""
    initial = [p.detach().clone() for p in params]
    second = torch.zeros_like(reference[target.name])
    for task in source.sample_batch(N_PRECONDITION, target.shape, "precondition", target.name):
        second += flat_grad(model, params, collate_tasks([task]).to(device)) ** 2 / N_PRECONDITION

    names = list(reference)
    stacked = torch.stack([reference[name] for name in names])
    candidates = [
        collate_tasks([task]) for task in source.sample_batch(N_CANDIDATES, target.shape, "candidate", target.name)
    ]
    scores = []
    for batch in candidates:
        g = flat_grad(model, params, batch.to(device))
        update = g / (torch.sqrt(BETA2 * second + (1 - BETA2) * g**2) + ADAM_EPS)
        scores.append((-(stacked @ update)).cpu().numpy())  # predicted change in loss per unit lr
    score_frame = pl.DataFrame(np.array(scores), schema=names).with_row_index("candidate")
    model.zero_grad(set_to_none=True)

    batch = collate_tasks([target.task])
    test, sel = query_subset(batch, target.test), query_subset(batch, target.sel)
    prior = [
        collate_tasks([task]) for task in source.sample_batch(N_PRIOR_EVAL, target.shape, "prior-eval", target.name)
    ]

    def probe() -> dict:
        return evaluate(model, device, test, sel, prior)

    arms = {
        "own": np.argsort(score_frame[target.name].to_numpy())[:POOL_SIZE],
        "cross": np.argsort(score_frame[cross].to_numpy())[:POOL_SIZE],
    }
    pick = np.random.default_rng([SEED, zlib.crc32(target.name.encode()), 1])
    arms |= {f"random{r}": pick.choice(N_CANDIDATES, POOL_SIZE, replace=False) for r in range(N_RANDOM)}
    rows = []
    for arm, index in arms.items():
        pool = [candidates[i] for i in index]
        own_score = float(score_frame[target.name].to_numpy()[index].mean())
        records = train_pool(model, params, device, initial, second, pool, SEED, lr, steps, probe)
        rows += [{"target": target.name, "arm": arm, "pool_own_score": own_score} | record for record in records]
        print(
            f"  {target.name} {arm}: test_nll {records[0]['test_nll']:+.4f} -> {records[-1]['test_nll']:+.4f}",
            flush=True,
        )
    with torch.no_grad():
        for p, value in zip(params, initial, strict=True):
            p.copy_(value)
    return pl.DataFrame(rows), score_frame.with_columns(pl.lit(target.name).alias("target"))


def summarize(curves: pl.DataFrame) -> pl.DataFrame:
    """Per target and step: each arm's change from the base against the random band."""
    base = (
        curves.filter(pl.col("step") == 0)
        .group_by("target")
        .agg(pl.col("test_nll", "test_crps", "prior_loss").first().name.suffix("_base"))
    )
    frame = curves.join(base, on="target").with_columns(
        d_nll=pl.col("test_nll") - pl.col("test_nll_base"),
        d_crps=pl.col("test_crps") - pl.col("test_crps_base"),
        d_prior=pl.col("prior_loss") - pl.col("prior_loss_base"),
    )
    random = (
        frame.filter(pl.col("arm").str.starts_with("random"))
        .group_by("target", "step")
        .agg(
            rand_mean_nll=pl.col("d_nll").mean(),
            rand_min_nll=pl.col("d_nll").min(),
            rand_max_nll=pl.col("d_nll").max(),
            rand_mean_crps=pl.col("d_crps").mean(),
            rand_sd_crps=pl.col("d_crps").std(),
            rand_sd_nll=pl.col("d_nll").std(),
        )
    )
    arms = frame.filter(~pl.col("arm").str.starts_with("random")).pivot(
        on="arm", index=["target", "step"], values=["d_nll", "d_crps", "d_prior", "pool_own_score"]
    )
    return arms.join(random, on=["target", "step"]).sort("target", "step")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps not in EVAL_STEPS:
        raise ValueError(f"--steps must be one of the eval steps {EVAL_STEPS}")

    out = OUT_DIR / f"lr_{args.lr:g}_steps_{args.steps}"
    out.mkdir(parents=True, exist_ok=True)
    device = select_device("auto")
    configure_device_memory(device)
    model, _ = load_checkpoint(released_base_checkpoint(), map_location=device)
    model.to(device).eval()
    params = trainable_parameters(model, SCOPE)
    print(
        f"device={device} | scope={SCOPE} ({sum(p.numel() for p in params):,} params) | lr={args.lr:g} | {args.steps} steps"
    )

    targets = load_targets()
    reference = {t.name: target_gradient(model, params, device, t) for t in targets}
    source = NanopriorTaskSource(RandomStreams(SEED), max_attempts=MAX_ATTEMPTS)
    curves, scores = [], []
    for position, target in enumerate(targets):
        curve_path, score_path = out / f"{target.name}.parquet", out / f"{target.name}.scores.parquet"
        if not curve_path.exists():
            cross = targets[(position + 1) % len(targets)].name
            curve, score = run_target(model, params, device, target, cross, reference, source, args.lr, args.steps)
            curve.with_columns(pl.lit(cross).alias("cross")).write_parquet(curve_path)
            score.write_parquet(score_path)
            print(f"{target.name} done", flush=True)
        curves.append(pl.read_parquet(curve_path))
        scores.append(pl.read_parquet(score_path))

    summary = summarize(pl.concat(curves))
    summary.write_csv(out / "summary.csv")
    (out / "config.json").write_text(
        json.dumps(
            {
                "lr": args.lr,
                "steps": args.steps,
                "scope": SCOPE,
                "seed": SEED,
                "n_splits": N_SPLITS,
                "n_candidates": N_CANDIDATES,
                "pool_size": POOL_SIZE,
                "n_random": N_RANDOM,
                "tasks_per_step": TASKS_PER_STEP,
                "n_precondition": N_PRECONDITION,
            },
            indent=2,
        )
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=250):
        print(summary.filter(pl.col("step") == args.steps))


if __name__ == "__main__":
    main()
