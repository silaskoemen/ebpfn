"""Gate-0 sweep driver (spec §4/§5).

Runs the pre-registered sweep for one construction, writes per-seed raw metrics +
config + git SHA to benchmarks/results/<run>/, and emits the 3-panel figure.

    pixi run python benchmarks/scripts/run_sweep.py --construction A
    pixi run python benchmarks/scripts/run_sweep.py --construction A --quick

`--quick` is a fast pilot (few seeds/tasks) to validate the pipeline -- NOT a
pre-registered run. Fix the grid and thresholds before the real run (spec §5).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import polars as pl
from ebpfn.config import ExperimentConfig
from ebpfn.config import SweepConfig
from ebpfn.experiment import run_null
from ebpfn.experiment import run_sweep
from ebpfn.experiment import suggest_thresholds
from ebpfn.experiment import summarize
from ebpfn.plotting import make_sweep_figure
from ebpfn.results import save_run

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    base = ExperimentConfig()
    values = tuple(float(v) for v in args.values.split(",")) if args.values else base.sweep.values
    sweep = SweepConfig(
        construction=args.construction,
        values=values,
        n_seeds=args.seeds,
        n_tasks_per_prior=args.tasks,
        cloud_n_rows=base.sweep.cloud_n_rows,
        n_calib_tasks=args.calib_tasks,
        calib_n_train=base.sweep.calib_n_train,
        calib_n_test=base.sweep.calib_n_test,
    )
    return dataclasses.replace(base, sweep=sweep, seed=args.seed)


def quick_config(args: argparse.Namespace) -> ExperimentConfig:
    """Small, fast pilot to validate the pipeline end to end."""
    cfg = build_config(args)
    sweep = dataclasses.replace(
        cfg.sweep,
        values=(0.1, 0.5, 2.0) if not args.values else cfg.sweep.values,
        n_seeds=3,
        n_tasks_per_prior=12,
        cloud_n_rows=400,
        n_calib_tasks=1,
        calib_n_train=1500,
        calib_n_test=1500,
    )
    distance = dataclasses.replace(cfg.distance, n_proj=100)
    mmd = dataclasses.replace(cfg.mmd, n_cells_grid=(8, 16))
    model = dataclasses.replace(cfg.model, catboost_iterations=200)
    return dataclasses.replace(cfg, sweep=sweep, distance=distance, mmd=mmd, model=model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--construction", choices=["A", "B"], default="A")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tasks", type=int, default=50, help="tasks per prior cloud")
    ap.add_argument("--calib-tasks", type=int, default=3, dest="calib_tasks")
    ap.add_argument("--values", type=str, default="", help="comma-separated sweep grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="fast pilot, not pre-registered")
    ap.add_argument("--null", action="store_true", help="real-vs-real' null run to set T_cond/T_cal")
    ap.add_argument("--name", type=str, default="", help="run directory name")
    args = ap.parse_args()

    cfg = quick_config(args) if args.quick else build_config(args)
    suffix = "_null" if args.null else ("_quick" if args.quick else "")
    tag = args.name or f"{cfg.sweep.construction}{suffix}"
    out_dir = RESULTS_ROOT / tag

    print(
        f"[run] construction={cfg.sweep.construction} seeds={cfg.sweep.n_seeds} "
        f"tasks/prior={cfg.sweep.n_tasks_per_prior} values={cfg.sweep.values} "
        f"mode={'null' if args.null else 'sweep'}"
    )

    if args.null:
        frames = run_null(cfg)
        save_run(out_dir, cfg, frames)
        thresholds = suggest_thresholds(frames, cfg)
        (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
        print("[null] pre-registration thresholds (99th pct of real-vs-real' null):")
        print(json.dumps(thresholds, indent=2))
        print(f"[null] wrote null frames + thresholds to {out_dir}")
        return

    frames = run_sweep(cfg)
    save_run(out_dir, cfg, frames)

    fig = make_sweep_figure(frames, cfg)
    fig_path = out_dir / f"figure_{cfg.sweep.construction}.png"
    fig.savefig(fig_path, dpi=130)

    summary = summarize(frames, cfg)
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=4):
        print(summary)
    print(f"[run] wrote frames + config + figure to {out_dir}")
    print(f"[run] figure: {fig_path}")


if __name__ == "__main__":
    main()
