"""Local-prior gate: does a target-shaped prior beat fine-tuning on the target's rows?

η cannot be read off one dataset, but the task-level latent φ can, so the prior to
adapt on is a cloud of DGPs fitted to the target (:class:`ebpfn.priors.local.LocalPrior`).
Per target and context size ``n``, the target's data is ``n`` rows drawn from its
characterization context; its query rows are TEST, never trained or fitted on. Every arm
continues the released base for ``STEPS`` steps at each lr in ``LRS``:

* **finetune** -- resampled context/query splits of the ``n`` rows (pure fine-tuning);
* **local** -- fresh tasks from the local prior fitted to the ``n`` rows;
* **local_origin** -- half local, half origin-prior tasks per step;
* **cross_local** -- the local prior of *another* target: separates "this target's
  structure helps" from "smooth surrogate tasks help";
* **origin** -- origin-prior tasks at the target's shape, the P2-style null.

All arms are scored on TEST with the ``n`` rows as context, on CRPS and ``nll_coarse``
(raw NLL is reported but not decisive, see ``metrics.COARSE_NLL_STRIDE``). A same-data
GP is the outside reference.

Run from project root (long; wrap in caffeinate):
    pixi run python -m benchmarks.studies.local_prior [--ns 64 128 256]
"""

import argparse
import dataclasses
import json
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.base import released_base_checkpoint
from benchmarks.baselines import gaussian_crps, gaussian_nll, gp_fit_predict
from benchmarks.corpus import CHAR_DIR, iter_real_tasks
from ebpfn.data import CharacterizationShape
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
from ebpfn.priors.local import LocalPrior
from ebpfn.utils import RandomStreams

OUT_DIR = Path("benchmarks/results/local_prior")
SEED = 0
SCOPE = "icl"
NS = (64, 128, 256)
N_QUERY = 64  # query rows of every synthetic task
FINETUNE_QUERY_FRACTION = 0.25
POOL_SIZE = 512
TASKS_PER_STEP = 8
STEPS = 200
EVAL_STEPS = (25, 50, 100, 200)
LRS = (1e-5, 1e-4)
WARMUP_STEPS = 10
GRAD_CLIP = 1.0
MAX_ATTEMPTS = 50
ARMS = ("finetune", "local", "local_origin", "cross_local", "origin")


@dataclasses.dataclass(frozen=True)
class Target:
    name: str
    x: np.ndarray  # characterization context rows the n rows are drawn from
    y: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def rows(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        index = np.random.default_rng([SEED, zlib.crc32(self.name.encode()), n]).permutation(len(self.y))[:n]
        return self.x[index], self.y[index]


def load_targets() -> list[Target]:
    targets = []
    for name, repeat, task in iter_real_tasks(CHAR_DIR):
        if repeat == 0:
            fit, score = task.probe_fit, task.probe_score
            targets.append(
                Target(
                    name,
                    fit.X.to_numpy().astype(np.float64),
                    fit.y.astype(np.float64),
                    score.X.to_numpy().astype(np.float64),
                    score.y.astype(np.float64),
                )
            )
    return targets


def standardized_batch(x: np.ndarray, y: np.ndarray, n_train: int) -> TaskBatch:
    """``(B, rows, p)`` / ``(B, rows)`` arrays as a batch, standardized on each context like ``collate_tasks``."""
    mean = y[:, :n_train].mean(1, keepdims=True)
    std = y[:, :n_train].std(1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    z = ((y - mean) / std).astype(np.float32)
    return TaskBatch(
        x=torch.from_numpy(x.astype(np.float32)),
        y_train_std=torch.from_numpy(z[:, :n_train]),
        y_test_std=torch.from_numpy(z[:, n_train:]),
        target_mean=torch.from_numpy(mean[:, 0].astype(np.float32)),
        target_std=torch.from_numpy(std[:, 0].astype(np.float32)),
        n_train=n_train,
    )


def take(batch: TaskBatch, index: np.ndarray) -> TaskBatch:
    rows = torch.as_tensor(index)
    return dataclasses.replace(
        batch,
        x=batch.x[rows],
        y_train_std=batch.y_train_std[rows],
        y_test_std=batch.y_test_std[rows],
        target_mean=batch.target_mean[rows],
        target_std=batch.target_std[rows],
    )


def concat(a: TaskBatch, b: TaskBatch) -> TaskBatch:
    return dataclasses.replace(
        a,
        x=torch.cat([a.x, b.x]),
        y_train_std=torch.cat([a.y_train_std, b.y_train_std]),
        y_test_std=torch.cat([a.y_test_std, b.y_test_std]),
        target_mean=torch.cat([a.target_mean, b.target_mean]),
        target_std=torch.cat([a.target_std, b.target_std]),
    )


def local_pool(target: Target, n: int) -> TaskBatch:
    """``POOL_SIZE`` local-prior tasks of ``n + N_QUERY`` rows, half per surrogate."""
    x, y = target.rows(n)
    prior = LocalPrior.fit(x, y, seed=SEED)
    rng = np.random.default_rng([SEED, zlib.crc32(target.name.encode()), n, 1])
    draws = [prior.sample_task(n + N_QUERY, rng, surrogate="gp" if i % 2 == 0 else "trees") for i in range(POOL_SIZE)]
    return standardized_batch(np.stack([d[0] for d in draws]), np.stack([d[1] for d in draws]), n)


def origin_pool(target: Target, n: int, source: NanopriorTaskSource) -> TaskBatch:
    shape = CharacterizationShape(n, N_QUERY, target.x.shape[1], 0, "regression")
    return collate_tasks(source.sample_batch(POOL_SIZE, shape, "local-prior-origin", target.name, n))


def finetune_draw(target: Target, n: int) -> Callable[[np.random.Generator], TaskBatch]:
    x, y = target.rows(n)
    n_query = max(1, int(n * FINETUNE_QUERY_FRACTION))

    def draw(rng: np.random.Generator) -> TaskBatch:
        order = np.stack([rng.permutation(n) for _ in range(TASKS_PER_STEP)])
        return standardized_batch(x[order], y[order], n - n_query)

    return draw


def pool_draw(*pools: TaskBatch) -> Callable[[np.random.Generator], TaskBatch]:
    """Equal shares of each pool per step."""
    share = TASKS_PER_STEP // len(pools)

    def draw(rng: np.random.Generator) -> TaskBatch:
        parts = [take(pool, rng.integers(pool.batch_size, size=share)) for pool in pools]
        batch = parts[0]
        for part in parts[1:]:
            batch = concat(batch, part)
        return batch

    return draw


@torch.no_grad()
def probe(model: EBPFNModel, device: torch.device, test: TaskBatch) -> dict:
    model.eval()
    batch = test.to(device)
    rows = compute_row_metrics(model.distribution, model(batch.x, batch.y_train_std)[0], batch.y_test_std[0])
    return {
        "crps": float(rows.crps.mean()),
        "nll": float(rows.nll.mean()),
        "nll_coarse": float(rows.nll_coarse.mean()),
        "crps_rows": rows.crps.cpu().tolist(),
        "nll_coarse_rows": rows.nll_coarse.cpu().tolist(),
    }


def train_arm(
    model: EBPFNModel,
    params: list[torch.nn.Parameter],
    device: torch.device,
    initial: list[torch.Tensor],
    draw: Callable[[np.random.Generator], TaskBatch],
    lr: float,
    test: TaskBatch,
) -> list[dict]:
    with torch.no_grad():
        for p, value in zip(params, initial, strict=True):
            p.copy_(value)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    records = []
    for step in range(1, STEPS + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, step / WARMUP_STEPS)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model.loss(draw(rng).to(device)).backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()
        if device.type == "mps":
            release_device_cache(device)
        if step in EVAL_STEPS:
            records.append({"step": step} | probe(model, device, test))
    return records


def gp_reference(target: Target, n: int) -> dict:
    x, y = target.rows(n)
    mu, sd = float(y.mean()), float(y.std()) or 1.0
    mean, std = gp_fit_predict(x, (y - mu) / sd, target.x_test)
    y_test = (target.y_test - mu) / sd
    return {"crps": gaussian_crps(y_test, mean, std), "nll": gaussian_nll(y_test, mean, std)}


def run_target(
    model: EBPFNModel,
    params: list[torch.nn.Parameter],
    device: torch.device,
    target: Target,
    n: int,
    pools: dict[str, TaskBatch],
    cross: str,
    source: NanopriorTaskSource,
) -> pl.DataFrame:
    initial = [p.detach().clone() for p in params]
    x, y = target.rows(n)
    test = standardized_batch(np.concatenate([x, target.x_test])[None], np.concatenate([y, target.y_test])[None], n)
    origin = origin_pool(target, n, source)
    draws = {
        "finetune": finetune_draw(target, n),
        "local": pool_draw(pools[target.name]),
        "local_origin": pool_draw(pools[target.name], origin),
        "cross_local": pool_draw(pools[cross]),
        "origin": pool_draw(origin),
    }
    identity = {"target": target.name, "n": n, "cross": cross}
    gp = gp_reference(target, n)
    rows = [identity | {"arm": "base", "lr": 0.0, "step": 0} | probe(model, device, test)]
    rows.append(
        identity | {"arm": "gp", "lr": 0.0, "step": 0, "crps": gp["crps"], "nll": gp["nll"], "nll_coarse": gp["nll"]}
    )
    for arm in ARMS:
        for lr in LRS:
            records = train_arm(model, params, device, initial, draws[arm], lr, test)
            rows += [identity | {"arm": arm, "lr": lr} | record for record in records]
            print(
                f"  n={n} {target.name} {arm} lr={lr:g}: crps {rows[0]['crps']:.4f} -> {records[-1]['crps']:.4f}",
                flush=True,
            )
    with torch.no_grad():
        for p, value in zip(params, initial, strict=True):
            p.copy_(value)
    return pl.DataFrame(rows)


def summarize(frame: pl.DataFrame) -> pl.DataFrame:
    """Per n, arm, lr, step: mean change from the base over targets, and how many it helps."""
    base = frame.filter(pl.col("arm") == "base").select("target", "n", crps_base="crps", nll_coarse_base="nll_coarse")
    delta = (
        frame.filter(pl.col("arm") != "base")
        .join(base, on=["target", "n"])
        .with_columns(
            d_crps=(pl.col("crps") - pl.col("crps_base")) / pl.col("crps_base"),
            d_nll_coarse=pl.col("nll_coarse") - pl.col("nll_coarse_base"),
        )
    )
    return (
        delta.group_by("n", "arm", "lr", "step")
        .agg(
            rel_crps_mean=pl.col("d_crps").mean(),
            rel_crps_median=pl.col("d_crps").median(),
            crps_helped=(pl.col("d_crps") < 0).sum(),
            d_nll_coarse_mean=pl.col("d_nll_coarse").mean(),
            nll_coarse_helped=(pl.col("d_nll_coarse") < 0).sum(),
        )
        .sort("n", "arm", "lr", "step")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ns", type=int, nargs="+", default=list(NS))
    args = parser.parse_args()

    device = select_device("auto")
    configure_device_memory(device)
    model, _ = load_checkpoint(released_base_checkpoint(), map_location=device)
    model.to(device)
    params = trainable_parameters(model, SCOPE)
    print(
        f"device={device} | scope={SCOPE} ({sum(p.numel() for p in params):,} params) | ns={args.ns} | lrs={LRS} | {STEPS} steps"
    )

    all_targets = load_targets()
    source = NanopriorTaskSource(RandomStreams(SEED), max_attempts=MAX_ATTEMPTS)
    frames = []
    for n in args.ns:
        targets = [t for t in all_targets if len(t.y) >= n]
        if skipped := [f"{t.name} ({len(t.y)} rows)" for t in all_targets if len(t.y) < n]:
            print(f"n={n} skipping {', '.join(skipped)}", flush=True)
        names = [t.name for t in targets]
        out = OUT_DIR / f"n_{n}"
        out.mkdir(parents=True, exist_ok=True)
        pending = [t for t in targets if not (out / f"{t.name}.parquet").exists()]
        # Every target's pool is needed: each one is some other target's cross arm.
        pools = {t.name: local_pool(t, n) for t in targets} if pending else {}
        for target in pending:
            cross = names[(names.index(target.name) + 1) % len(names)]
            run_target(model, params, device, target, n, pools, cross, source).write_parquet(
                out / f"{target.name}.parquet"
            )
            print(f"n={n} {target.name} done", flush=True)
        frames += [pl.read_parquet(out / f"{t.name}.parquet") for t in targets]

    frame = pl.concat(frames, how="diagonal_relaxed")
    summary = summarize(frame)
    summary.write_csv(OUT_DIR / "summary.csv")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "ns": args.ns,
                "scope": SCOPE,
                "seed": SEED,
                "n_query": N_QUERY,
                "pool_size": POOL_SIZE,
                "tasks_per_step": TASKS_PER_STEP,
                "steps": STEPS,
                "eval_steps": list(EVAL_STEPS),
                "lrs": list(LRS),
                "warmup_steps": WARMUP_STEPS,
                "arms": list(ARMS),
            },
            indent=2,
        )
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=250):
        print(summary.filter(pl.col("step").is_in([0, 100, STEPS])))


if __name__ == "__main__":
    main()
