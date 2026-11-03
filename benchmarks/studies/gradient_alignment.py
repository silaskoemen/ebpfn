"""P5 gate: does gradient alignment predict what one real step does to a target?

For each corpus target, origin-prior candidate tasks at the target's shape get one
preconditioned update ``u_s`` each. Two training-free scores predict the change in target
loss on half its query rows (A):

* **first order** -- ``-lr * g_A . u_s``, the classic alignment score;
* **trial step** -- the step actually taken and A re-measured: one forward, exact for one
  step, so it carries the curvature the first-order score drops.

Both are judged against the change on the held-out half V; a score that tracks A but not
V is fitting target noise. Nulls guard the claim that a score is *target-specific*:

* **generic** -- alignment with the mean origin gradient (first order), or the trial
  change on held-out origin tasks (*disruption*). At a converged base most steps hurt, so
  a score can rank V purely by how disruptive a step is; ``partial_trial_v`` is the trial
  score's rank correlation with V after both are residualized on disruption.
* **cross-target** -- another dataset's ``g_A`` or trial change on its A. Same reading.

The preconditioner is a warm Adam second moment estimated from origin tasks at the
target's shape, since mid-training that is what turns a gradient into a step. Loss is
the head's native pinball loss, a CRPS proxy; NLL is not differentiated here.

Run from project root (wrap in caffeinate):
    pixi run python -m benchmarks.studies.gradient_alignment [--candidates 128] [--scope icl]
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.base import released_base_checkpoint
from benchmarks.corpus import CHAR_DIR, iter_real_tasks
from ebpfn.data import CharacterizationShape, TuningTask
from ebpfn.pfn.data import NanopriorTaskSource, TaskBatch, collate_tasks
from ebpfn.pfn.model import EBPFNModel
from ebpfn.pfn.train import (
    configure_device_memory,
    load_checkpoint,
    release_device_cache,
    select_device,
    trainable_parameters,
)
from ebpfn.utils import RandomStreams
from scipy import stats

OUT_DIR = Path("benchmarks/results/gradient_alignment")
SEED = 0
N_PRECONDITION = 32
LRS = (3e-6, 3e-5, 3e-4)
N_PRIOR_EVAL = 8
BETA2 = 0.999
ADAM_EPS = 1e-8
TOP_FRACTION = 0.1
MAX_ATTEMPTS = 50


@dataclasses.dataclass(frozen=True)
class Target:
    name: str
    shape: CharacterizationShape
    align: TaskBatch
    held: TaskBatch


def query_subset(batch: TaskBatch, index: np.ndarray) -> TaskBatch:
    rows = torch.as_tensor(index)
    return dataclasses.replace(
        batch,
        x=torch.cat([batch.x[:, : batch.n_train], batch.x[:, batch.n_train + rows]], dim=1),
        y_test_std=batch.y_test_std[:, rows],
    )


def load_targets() -> list[Target]:
    """Repeat 0 of every corpus dataset, query rows split into disjoint A and V halves."""
    targets = []
    for name, repeat, task in iter_real_tasks(CHAR_DIR):
        if repeat != 0:
            continue
        batch = collate_tasks([task])
        order = np.random.default_rng(SEED).permutation(batch.n_test)
        half = batch.n_test // 2
        shape = CharacterizationShape(
            task.probe_fit.X.height, task.probe_score.X.height, task.probe_fit.X.width, 0, "regression"
        )
        targets.append(Target(name, shape, query_subset(batch, order[:half]), query_subset(batch, order[half:])))
    return targets


def flat_grad(model: EBPFNModel, params: list[torch.nn.Parameter], batch: TaskBatch) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    model.loss(batch).backward()
    grads = [p.grad for p in params]
    if any(grad is None for grad in grads):
        raise RuntimeError("a trainable parameter received no gradient")
    return torch.cat([grad.flatten() for grad in grads if grad is not None])


@torch.no_grad()
def _loss(model: EBPFNModel, batch: TaskBatch) -> float:
    return float(model.loss(batch))


@torch.no_grad()
def _shift(params: list[torch.nn.Parameter], delta: torch.Tensor) -> None:
    offset = 0
    for p in params:
        p.add_(delta[offset : offset + p.numel()].view_as(p))
        offset += p.numel()


def _tasks(source: NanopriorTaskSource, count: int, shape: CharacterizationShape, *identity: str) -> list[TuningTask]:
    return source.sample_batch(count, shape, *identity)


def run_target(
    model: EBPFNModel,
    params: list[torch.nn.Parameter],
    device: torch.device,
    target: Target,
    targets: list[Target],
    reference: torch.Tensor,
    source: NanopriorTaskSource,
    n_candidates: int,
) -> pl.DataFrame:
    """One row per candidate; ``reference`` stacks every target's ``g_A`` in ``targets`` order."""
    # Warm second moment and mean gradient of the origin prior at this target's shape.
    second = torch.zeros_like(reference[0])
    generic = torch.zeros_like(second)
    for task in _tasks(source, N_PRECONDITION, target.shape, "precondition", target.name):
        g = flat_grad(model, params, collate_tasks([task]).to(device))
        second += g**2 / N_PRECONDITION
        generic += g / N_PRECONDITION
    names = [t.name for t in targets]

    aligns = {t.name: t.align.to(device) for t in targets}
    held = target.held.to(device)
    # One task per forward: a stacked batch at the widest corpus shape does not fit beside the gradients.
    prior = [
        collate_tasks([t]).to(device) for t in _tasks(source, N_PRIOR_EVAL, target.shape, "prior-eval", target.name)
    ]

    def prior_loss() -> float:
        return float(np.mean([_loss(model, batch) for batch in prior]))

    base_a = {name: _loss(model, batch) for name, batch in aligns.items()}
    base_v, base_prior = _loss(model, held), prior_loss()
    rows = []
    for index, task in enumerate(_tasks(source, n_candidates, target.shape, "candidate", target.name)):
        g = flat_grad(model, params, collate_tasks([task]).to(device))
        update = g / (torch.sqrt(BETA2 * second + (1 - BETA2) * g**2) + ADAM_EPS)
        dots = torch.cat([reference @ update, (generic @ update)[None]]).cpu().numpy()
        row = {"target": target.name, "candidate": index, "grad_norm": float(g.norm())}
        row |= {f"dot.{name}": float(-d) for name, d in zip([*names, "generic"], dots, strict=True)}
        for lr in LRS:
            _shift(params, -lr * update)
            row |= {f"delta_a.{name}.{lr}": _loss(model, batch) - base_a[name] for name, batch in aligns.items()}
            row[f"delta_v.{lr}"] = _loss(model, held) - base_v
            row[f"delta_prior.{lr}"] = prior_loss() - base_prior
            _shift(params, lr * update)
        rows.append(row)
        if device.type == "mps":
            release_device_cache(device)
    return pl.DataFrame(rows)


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.spearmanr(a, b).statistic)


def _partial_rho(score: np.ndarray, outcome: np.ndarray, control: np.ndarray) -> float:
    """Rank correlation of ``score`` and ``outcome`` after regressing ``control`` out of both."""
    ranks = [stats.rankdata(v) for v in (score, outcome, control)]
    design = np.column_stack([np.ones(len(score)), ranks[2]])
    residual = [r - design @ np.linalg.lstsq(design, r, rcond=None)[0] for r in ranks[:2]]
    return float(np.corrcoef(*residual)[0, 1])


def evaluate(frame: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Per target and lr: every score against the actual change on held-out V."""
    rows = []
    for target in names:
        part = frame.filter(pl.col("target") == target)
        k = max(1, int(len(part) * TOP_FRACTION))
        for lr in LRS:
            delta_v = part[f"delta_v.{lr}"].to_numpy()
            scores = {
                "first": lr * part[f"dot.{target}"].to_numpy(),
                "first_generic": lr * part["dot.generic"].to_numpy(),
                "trial": part[f"delta_a.{target}.{lr}"].to_numpy(),
                "disruption": part[f"delta_prior.{lr}"].to_numpy(),
            }
            others = [o for o in names if o != target]
            row = {
                "target": target,
                "lr": lr,
                "frac_helpful_v": float(np.mean(delta_v < 0)),
                "mean_delta_v": float(delta_v.mean()),
                "first_vs_trial_a": _rho(scores["first"], scores["trial"]),
            }
            row |= {f"rho_{name}_v": _rho(score, delta_v) for name, score in scores.items()}
            row["rho_first_cross_v"] = float(np.mean([_rho(part[f"dot.{o}"].to_numpy(), delta_v) for o in others]))
            row["rho_trial_cross_v"] = float(
                np.mean([_rho(part[f"delta_a.{o}.{lr}"].to_numpy(), delta_v) for o in others])
            )
            row["partial_trial_v"] = _partial_rho(scores["trial"], delta_v, scores["disruption"])
            row["partial_trial_cross_v"] = float(
                np.mean(
                    [_partial_rho(part[f"delta_a.{o}.{lr}"].to_numpy(), delta_v, scores["disruption"]) for o in others]
                )
            )
            row |= {f"top_{name}_v": float(delta_v[np.argsort(score)[:k]].mean()) for name, score in scores.items()}
            row["top_oracle_v"] = float(np.sort(delta_v)[:k].mean())
            rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--scope", choices=("head", "icl", "all"), default="icl")
    args = parser.parse_args()

    out = OUT_DIR / f"scope_{args.scope}"
    out.mkdir(parents=True, exist_ok=True)
    device = select_device("auto")
    configure_device_memory(device)
    model, _ = load_checkpoint(released_base_checkpoint(), map_location=device)
    model.to(device).eval()
    params = trainable_parameters(model, args.scope)
    print(
        f"device={device} | scope={args.scope} ({sum(p.numel() for p in params):,} params) | {args.candidates} candidates"
    )

    targets = load_targets()
    reference = torch.stack([flat_grad(model, params, t.align.to(device)) for t in targets])
    source = NanopriorTaskSource(RandomStreams(SEED), max_attempts=MAX_ATTEMPTS)
    frames = []
    for target in targets:
        path = out / f"{target.name}.parquet"
        if not path.exists():
            run_target(model, params, device, target, targets, reference, source, args.candidates).write_parquet(path)
            print(f"{target.name} done", flush=True)
        frames.append(pl.read_parquet(path))

    summary = evaluate(pl.concat(frames), [t.name for t in targets])
    summary.write_csv(out / "summary.csv")
    (out / "config.json").write_text(
        json.dumps(
            {
                "candidates": args.candidates,
                "scope": args.scope,
                "lrs": list(LRS),
                "n_precondition": N_PRECONDITION,
                "n_prior_eval": N_PRIOR_EVAL,
                "seed": SEED,
            },
            indent=2,
        )
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=250):
        print(summary)


if __name__ == "__main__":
    main()
