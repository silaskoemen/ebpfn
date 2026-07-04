"""Gate-2 end-to-end driver (plans/gate2.md).

For each prior in the ladder: train a PFN on it (exact prior<->model pairing, the
same guarantee as Gate-1), measure per-task descriptor coverage to that prior's
cloud and per-task in-context calibration on the TabArena corpus. Then run the
two pre-committed parts -- the variance go/no-go and the across-prior
fixed-effects ablation -- and print one decisive verdict.

    pixi run python benchmarks/scripts/run_gate2.py --quick          # fast chain check
    pixi run python benchmarks/scripts/run_gate2.py --name gate2_full --steps 4000

`--quick` is a pilot (tiny PFN, few datasets) to validate the chain end to end --
NOT pre-registered. Thresholds/configs are frozen in ebpfn/gate2/config.py before
the confirmatory run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
from ebpfn.gate1 import CorpusConfig
from ebpfn.gate1 import DownstreamConfig
from ebpfn.gate1 import PFNConfig
from ebpfn.gate1 import build_prior
from ebpfn.gate1 import corpus_calibration
from ebpfn.gate1.corpus import load_corpus
from ebpfn.gate1.pfn import train_pfn
from ebpfn.gate2 import DescriptorConfig
from ebpfn.gate2 import Gate2Config
from ebpfn.gate2 import Gate2CoverageConfig
from ebpfn.gate2 import ablation_test
from ebpfn.gate2 import corpus_coverage
from ebpfn.gate2 import format_report
from ebpfn.gate2 import gate2_verdict
from ebpfn.gate2 import prior_ladder
from ebpfn.gate2 import variance_check
from ebpfn.results import _git_sha

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def build_configs(args: argparse.Namespace) -> dict:
    if args.quick:
        return {
            "pfn": PFNConfig(
                steps=args.steps or 300, embedding_size=64, num_layers=2, mlp_hidden_size=128, d_max=8, seed=args.seed
            ),
            "corpus": CorpusConfig(max_datasets=4, max_tasks_per_dataset=3, n_max=1200, d_max=8, seed=args.seed),
            "descriptor": DescriptorConfig(n_proj=24, n0=200, seed=args.seed),
            "coverage": Gate2CoverageConfig(cloud_n_tasks=16, cloud_n_rows=200, seed=args.seed),
            "downstream": DownstreamConfig(in_context_cap=200, test_cap=300, seed=args.seed),
            "gate": Gate2Config(n_boot=1000, seed=args.seed),
        }
    d_max = 16  # matched between PFN training regime and corpus (same as Gate-1 confirm)
    return {
        "pfn": PFNConfig(steps=args.steps or 4000, d_max=d_max, seed=args.seed),
        "corpus": CorpusConfig(max_datasets=args.max_datasets, d_max=d_max, seed=args.seed),
        "descriptor": DescriptorConfig(seed=args.seed),
        "coverage": Gate2CoverageConfig(seed=args.seed),
        "downstream": DownstreamConfig(seed=args.seed),
        "gate": Gate2Config(seed=args.seed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fast pilot, not pre-registered")
    ap.add_argument("--steps", type=int, default=0, help="PFN steps per prior (0 = config default)")
    ap.add_argument("--max-datasets", type=int, default=51, dest="max_datasets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", type=str, default="")
    args = ap.parse_args()

    cfg = build_configs(args)
    out_dir = RESULTS_ROOT / (args.name or ("gate2_quick" if args.quick else "gate2"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[gate2] loading TabArena corpus...")
    corpus = load_corpus(cfg["corpus"], rng, verbose=True)
    print(f"[gate2] {len(corpus)} real tasks")

    ladder = prior_ladder()
    all_rows: list[dict] = []
    for name, pcfg in ladder.items():
        print(f"\n[gate2] === prior '{name}' ===")
        prior = build_prior(pcfg)  # same prior feeds PFN training and coverage clouds
        reg = train_pfn(prior, cfg["pfn"], log_every=max(1, cfg["pfn"].steps // 5))
        print(f"[gate2] '{name}': measuring coverage + calibration...")
        cov_rows = corpus_coverage(corpus, prior, cfg["descriptor"], cfg["coverage"], rng)
        cal_rows = corpus_calibration(reg, corpus, cfg["downstream"], rng=rng)
        cal_by = {(r["source_did"], r["target"]): r for r in cal_rows}
        for r in cov_rows:
            cal = cal_by.get((r["source_did"], r["target"]))
            if cal is None:
                continue
            all_rows.append({"prior": name, **r, "nll": cal["nll"], "crps": cal["crps"], "pit_stat": cal["pit_stat"]})

    priors = list(ladder)
    # Part A uses the reference (balanced) prior's coverage rows.
    ref = "balanced" if "balanced" in priors else priors[0]
    ref_rows = [r for r in all_rows if r["prior"] == ref]
    variance = variance_check(ref_rows, cfg["gate"])
    ablation = ablation_test(all_rows, priors, cfg["gate"])
    result = gate2_verdict(variance, ablation)

    pl.DataFrame(all_rows).write_parquet(out_dir / "rows.parquet")
    (out_dir / "gate2.json").write_text(json.dumps(result, indent=2))
    (out_dir / "config.json").write_text(
        json.dumps({k: dataclasses.asdict(v) for k, v in cfg.items()}, indent=2, default=str)
    )
    (out_dir / "meta.json").write_text(
        json.dumps({"git_sha": _git_sha(), "n_tasks": len(corpus), "ref_prior": ref, "priors": priors}, indent=2)
    )

    print("\n" + format_report(result))
    print(f"\n[gate2] wrote tables + verdict to {out_dir}")


if __name__ == "__main__":
    main()
