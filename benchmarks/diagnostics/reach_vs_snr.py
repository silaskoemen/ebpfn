"""Finer log_snr sweep (V2, follows E2 -- see ``PLAN.md``).

E2 showed log_snr steers the cloud toward the high-SNR energy targets. This sweeps
a fine log_snr grid and reports the E1 reachability *ratio* (self-whitened, own-
cloud spread) per target -- so we can see whether log_snr brings energy fully
inside (ratio<=1.5) or only to the edge, and pick the optimal value for the E3
fine-tune. Optimum = the log_snr minimizing the mean energy-target ratio.

Run from project root:
    pixi run python -m benchmarks.diagnostics.reach_vs_snr [--n-prior 150]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from benchmarks.diagnostics.controllability import override_knob
from benchmarks.diagnostics.reachability import (
    CHAR_DIR,
    DEFAULT_BASE,
    iter_real_tasks,
    pooled_embed,
    prior_cloud_z,
    whitened_knn,
)
from ebpfn.data import characterization_shape
from ebpfn.pfn.data import collate_tasks
from ebpfn.pfn.train import load_checkpoint, select_device
from ebpfn.priors import hyperprior_from_dict, hyperprior_to_dict

OUT_DIR = Path("benchmarks/results/diagnostics/controllability")
LOG_SNR_GRID = [2.7, 3.2, 3.7, 4.0, 4.2, 4.5, 4.8, 5.2, 5.7, 6.5]
ENERGY_TARGETS = ["energy_heating", "energy_cooling"]
DEFAULT_TARGETS = [*ENERGY_TARGETS, "naval", "kin8nm"]


def reach_ratio(model, device, eta, shape, real_z, n_prior, batch_size, k, tag) -> float:
    cloud = prior_cloud_z(model, device, eta, shape, n_prior, batch_size, tag)
    mean, std = cloud.mean(0), cloud.std(0) + 1e-6
    intra = float(np.median(whitened_knn(cloud, cloud, mean, std, k, drop_self=True)))
    d_real = float(np.median(whitened_knn(real_z, cloud, mean, std, k, drop_self=False)))
    return d_real / intra


def run(checkpoint: Path, n_prior: int, k: int, batch_size: int, targets: list[str]) -> pl.DataFrame:
    device = select_device("auto")
    model, ck = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    base_eta_dict = hyperprior_to_dict(hyperprior_from_dict(ck["source_eta"]))
    print(f"base step={ck['step']} log_snr_base={base_eta_dict.get('log_snr_mean')} | grid={LOG_SNR_GRID}\n")

    real_by_name: dict[str, list] = {}
    for name, _repeat, task in iter_real_tasks(CHAR_DIR):
        if name in targets:
            real_by_name.setdefault(name, []).append(task)

    records: list[dict] = []
    for name in targets:
        tasks = real_by_name[name]
        shape = characterization_shape(tasks[0])
        real_z = np.stack([pooled_embed(model, collate_tasks([t]), device)[0] for t in tasks])
        line = []
        for v in LOG_SNR_GRID:
            eta_v = hyperprior_from_dict(override_knob(base_eta_dict, "log_snr_mean", v))
            ratio = reach_ratio(model, device, eta_v, shape, real_z, n_prior, batch_size, k, f"{name}-{v}")
            records.append({"target": name, "log_snr_mean": v, "ratio": ratio})
            line.append(ratio)
        print(f"{name:<16} ratio by log_snr: " + " ".join(f"{r:.2f}" for r in line))

    df = pl.DataFrame(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_DIR / "reach_vs_snr.parquet")

    # optimum = log_snr minimizing the mean ratio across the energy targets
    energy = (
        df.filter(pl.col("target").is_in(ENERGY_TARGETS))
        .group_by("log_snr_mean")
        .agg(pl.col("ratio").mean().alias("mean_ratio"))
        .sort("log_snr_mean")
    )
    best = energy.sort("mean_ratio").row(0, named=True)
    print("\nenergy mean-ratio by log_snr:")
    for r in energy.iter_rows(named=True):
        star = "  <== optimal" if abs(r["log_snr_mean"] - best["log_snr_mean"]) < 1e-9 else ""
        inside = " inside" if r["mean_ratio"] <= 1.5 else ""
        print(f"  log_snr={r['log_snr_mean']:<4} mean_ratio={r['mean_ratio']:.2f}{inside}{star}")
    (OUT_DIR / "reach_vs_snr_optimum.json").write_text(
        json.dumps({"optimal_log_snr_mean": best["log_snr_mean"], "energy_mean_ratio": best["mean_ratio"]}, indent=2)
    )
    print(f"\noptimal log_snr_mean = {best['log_snr_mean']} (energy mean ratio {best['mean_ratio']:.2f})")
    print(f"artifacts -> {OUT_DIR}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--n-prior", type=int, default=150)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    args = ap.parse_args()
    run(args.checkpoint, args.n_prior, args.k, args.batch_size, args.targets)


if __name__ == "__main__":
    main()
