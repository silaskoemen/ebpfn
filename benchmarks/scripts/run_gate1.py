"""Gate-1 (revised) end-to-end driver (plans/gate1_revised.md §6).

Builds the SCM+BNN prior, trains our own PFN on it, loads the TabArena corpus,
measures per-task prior-coverage and the trained PFN's calibration, then runs the
n,d-partial correlation test that decides H1. Writes tables + configs + the
scatter figure to benchmarks/results/<run>/.

    pixi run python benchmarks/scripts/run_gate1.py --quick
    pixi run python benchmarks/scripts/run_gate1.py --name gate1_full --steps 4000

`--quick` is a fast pilot (small PFN, few datasets) to validate the chain -- NOT
a pre-registered run. Fix configs/thresholds before the confirmatory run (§4).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl

from ebpfn.gate1 import (
    CorpusConfig,
    CoverageConfig,
    DownstreamConfig,
    GateConfig,
    PFNConfig,
    PriorConfig,
    build_prior,
    corpus_calibration,
    corpus_coverage,
    corpus_null,
    gate1_test,
)
from ebpfn.gate1.corpus import load_corpus
from ebpfn.gate1.pfn import train_pfn
from ebpfn.gate1.plotting import make_gate_figure
from ebpfn.results import _git_sha

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def build_configs(args: argparse.Namespace) -> dict:
    if args.quick:
        return {
            "prior": PriorConfig(),
            "pfn": PFNConfig(steps=args.steps or 300, embedding_size=64, num_layers=2,
                             mlp_hidden_size=128, seed=args.seed),
            "corpus": CorpusConfig(max_datasets=4, max_tasks_per_dataset=3, n_max=1200, seed=args.seed),
            "coverage": CoverageConfig(cloud_n_tasks=20, cloud_n_rows=300, n_proj=100, n_boot=500),
            "downstream": DownstreamConfig(in_context_cap=200, test_cap=300, seed=args.seed),
            "gate": GateConfig(n_boot=1000),
        }
    # Pre-registered confirmatory config (§4). d_max is matched between the PFN's
    # training regime and the corpus so out-of-regime d-extrapolation is not
    # confounded with coverage; higher-d tasks are variance-trimmed to d_max.
    d_max = 16
    return {
        "prior": PriorConfig(),
        "pfn": PFNConfig(steps=args.steps or 4000, d_max=d_max, seed=args.seed),
        "corpus": CorpusConfig(max_datasets=args.max_datasets, d_max=d_max, seed=args.seed),
        "coverage": CoverageConfig(),
        "downstream": DownstreamConfig(seed=args.seed),
        "gate": GateConfig(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fast pilot, not pre-registered")
    ap.add_argument("--steps", type=int, default=0, help="PFN training steps (0 = config default)")
    ap.add_argument("--max-datasets", type=int, default=51, dest="max_datasets")  # full TabArena-v0.1
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", type=str, default="")
    args = ap.parse_args()

    cfg = build_configs(args)
    out_dir = RESULTS_ROOT / (args.name or ("gate1_quick" if args.quick else "gate1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    prior = build_prior(cfg["prior"])  # one prior feeds both training and coverage clouds (§2)

    print(f"[gate1] training PFN ({cfg['pfn'].steps} steps)...")
    reg = train_pfn(prior, cfg["pfn"], log_every=max(1, cfg["pfn"].steps // 5))

    print("[gate1] loading TabArena corpus...")
    corpus = load_corpus(cfg["corpus"], rng, verbose=True)
    print(f"[gate1] {len(corpus)} real tasks")

    print("[gate1] measuring coverage + calibration...")
    cov_rows = corpus_coverage(corpus, prior, cfg["coverage"], rng)
    null = corpus_null(corpus, prior, cfg["coverage"], rng)
    cal_rows = corpus_calibration(reg, corpus, cfg["downstream"], rng=rng)

    result = gate1_test(cov_rows, cal_rows, cfg["gate"])

    pl.DataFrame(cov_rows).write_parquet(out_dir / "coverage.parquet")
    pl.DataFrame(cal_rows).write_parquet(out_dir / "calibration.parquet")
    (out_dir / "null.json").write_text(json.dumps(null, indent=2))
    (out_dir / "gate.json").write_text(json.dumps(result, indent=2))
    (out_dir / "config.json").write_text(json.dumps({k: dataclasses.asdict(v) for k, v in cfg.items()}, indent=2, default=str))
    (out_dir / "meta.json").write_text(json.dumps({"git_sha": _git_sha(), "n_tasks": len(corpus)}, indent=2))

    make_gate_figure(cov_rows, cal_rows, result).savefig(out_dir / "gate1_scatter.png", dpi=130)

    print("[gate1] result:")
    print(json.dumps(result, indent=2))
    print(f"[gate1] wrote tables + figure to {out_dir}")


if __name__ == "__main__":
    main()
