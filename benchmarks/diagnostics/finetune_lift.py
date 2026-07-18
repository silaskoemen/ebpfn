"""E3 sufficiency / lift probe (V2 -- see ``PLAN.md``).

The go/no-go for the whole thesis: E1+E2 showed the prior can be *steered* toward
the energy targets via log_snr. E3 asks the sufficiency question -- does training
on that steered prior actually PRODUCE A BETTER MODEL on those targets, or is
z-proximity not the same as good training data?

Procedure: fine-tune a COPY of the frozen baseline for N steps under a prior that
differs from the base by the single knob log_snr_mean (set to the E2/finer-sweep
optimum; everything else = base eta -- a clean single-knob test, not joint
optimization). Then evaluate base vs fine-tuned on the energy targets, with an
inside dataset (kin8nm) as a selectivity control. Standardized NLL + RMSE.

Run from project root (long; wrap in caffeinate):
    caffeinate -i pixi run python -m benchmarks.diagnostics.finetune_lift [--steps 400] [--log-snr 4.2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from benchmarks.diagnostics.controllability import override_knob
from benchmarks.diagnostics.reachability import CHAR_DIR, DEFAULT_BASE, iter_real_tasks
from ebpfn.config import PfnArchConfig, PfnTrainConfig
from ebpfn.pfn.data import PriorTaskSource, collate_tasks
from ebpfn.pfn.train import load_checkpoint, select_device, train_pfn
from ebpfn.priors import hyperprior_from_dict, hyperprior_to_dict
from ebpfn.utils import RandomStreams

OUT_DIR = Path("benchmarks/results/diagnostics/finetune_lift")
OPTIMUM_JSON = Path("benchmarks/results/diagnostics/controllability/reach_vs_snr_optimum.json")
EVAL_TARGETS = ["energy_heating", "energy_cooling", "kin8nm"]


@torch.no_grad()
def evaluate(model, device, datasets: list[str]) -> dict[str, dict[str, float]]:
    """Mean standardized NLL and RMSE per dataset over all its repeats."""
    tasks_by: dict[str, list] = {}
    for name, _repeat, task in iter_real_tasks(CHAR_DIR):
        if name in datasets:
            tasks_by.setdefault(name, []).append(task)
    out: dict[str, dict[str, float]] = {}
    for name in datasets:
        nlls, rmses = [], []
        for task in tasks_by[name]:
            batch = collate_tasks([task]).to(device)
            logits = model.predict_logits(batch.x, batch.y_train_std)[0]
            y = batch.y_test_std[0]
            nlls.append(float(model.distribution.nll(logits, y).mean()))
            rmses.append(float(torch.sqrt(torch.mean((model.distribution.mean(logits) - y) ** 2))))
        out[name] = {"nll": float(np.mean(nlls)), "rmse": float(np.mean(rmses))}
    return out


def run(checkpoint: Path, steps: int, log_snr: float | None, batch_size_note: str = "") -> None:
    device = select_device("auto")
    base_model, ck = load_checkpoint(checkpoint, map_location=device)
    base_model.to(device).eval()
    arch = PfnArchConfig.model_validate(ck["arch"])
    train = PfnTrainConfig.model_validate(ck["train"]).model_copy(update={"steps": steps})

    if log_snr is None:
        log_snr = float(json.loads(OPTIMUM_JSON.read_text())["optimal_log_snr_mean"])
    base_eta = hyperprior_to_dict(hyperprior_from_dict(ck["source_eta"]))
    print(
        f"base step={ck['step']} | log_snr base={base_eta['log_snr_mean']} -> tuned={log_snr} | fine-tune {steps} steps\n"
    )
    eta_tuned = hyperprior_from_dict(override_knob(base_eta, "log_snr_mean", log_snr))
    source = PriorTaskSource(eta_tuned, RandomStreams(train.seed))

    # Resumable: continue from the latest fine-tune checkpoint if present (kills on
    # lid-close sleep are frequent for long mps runs), else seed from base weights.
    ckpt_dir = OUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(ckpt_dir.glob("checkpoint_step_*.pt"))
    latest = existing[-1] if existing else None
    latest_step = int(latest.stem.rsplit("_", 1)[-1]) if latest else 0
    final_loss: float | None = None
    final_ckpt = str(latest) if latest else None
    if latest is not None and latest_step >= steps:
        print(f"fine-tune already complete at step {latest_step}; loading for eval")
        tuned_model, _ = load_checkpoint(latest, map_location=device)
        tuned_model.to(device).eval()
    else:
        print(f"{'resuming from step ' + str(latest_step) if latest else 'init from base weights'}")
        tuned_model, result = train_pfn(
            arch,
            train,
            source=source,
            checkpoint_dir=ckpt_dir,
            resume_from=latest,
            init_weights_from=None if latest else checkpoint,
            log_every=max(1, steps // 10),
        )
        tuned_model.to(device).eval()
        final_loss = result.losses[-1]
        final_ckpt = str(result.checkpoint_path)

    print("\nevaluating base vs fine-tuned...")
    base_metrics = evaluate(base_model, device, EVAL_TARGETS)
    tuned_metrics = evaluate(tuned_model, device, EVAL_TARGETS)

    rows = []
    print(f"\n{'dataset':<16}{'metric':<6}{'base':>9}{'tuned':>9}{'lift %':>9}")
    for name in EVAL_TARGETS:
        for metric in ("nll", "rmse"):
            b, t = base_metrics[name][metric], tuned_metrics[name][metric]
            lift = 100.0 * (b - t) / abs(b)  # positive = improvement (lower is better)
            role = "control" if name == "kin8nm" else "target"
            rows.append({"dataset": name, "role": role, "metric": metric, "base": b, "tuned": t, "lift_pct": lift})
            flag = "  better" if lift > 1 else ("  WORSE" if lift < -1 else "")
            print(f"{name:<16}{metric:<6}{b:>9.3f}{t:>9.3f}{lift:>+8.1f}%{flag}")

    pl.DataFrame(rows).write_parquet(OUT_DIR / "lift.parquet")
    (OUT_DIR / "lift_summary.json").write_text(
        json.dumps(
            {
                "log_snr_tuned": log_snr,
                "steps": steps,
                "final_train_loss": final_loss,
                "checkpoint": final_ckpt,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nartifacts -> {OUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument(
        "--log-snr", type=float, default=None, help="tuned log_snr_mean (default: read finer-sweep optimum)"
    )
    args = ap.parse_args()
    run(args.checkpoint, args.steps, args.log_snr)


if __name__ == "__main__":
    main()
