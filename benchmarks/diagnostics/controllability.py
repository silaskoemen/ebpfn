"""E2 controllability diagnostic (V2, see ``PLAN.md``).

Question: do the survivor knobs move the prior ``z``-cloud in embedding space, and
in particular *toward* the real targets E1 found off-manifold (energy_heating,
energy_cooling)? A knob is a usable optimizer lever only if sweeping it moves the
cloud-to-target distance monotonically.

For each (target, knob, value): override that one knob on the base eta, draw a
prior cloud at the target's exact shape, embed with the frozen base, and measure
cloud->target distance two ways -- an **energy distance** (whole-cloud vs the
target's repeat points) and a **kNN distance** (local). Both are computed in a
frame whitened by the *base* cloud so values are comparable across the sweep.

Run from project root:
    pixi run python -m benchmarks.diagnostics.controllability [--n-prior 100]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
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
from scipy.spatial.distance import cdist

OUT_DIR = Path("benchmarks/results/diagnostics/controllability")

# Knob grids centred on the perturbed-eta base (log_snr_mean=2.7, snr_dispersion=0.5,
# corr_strength_mean=0.6, heteroskedastic_rate=0.2). Energy targets are high-SNR, so
# the log_snr hypothesis predicts distance falling as log_snr rises.
KNOB_GRIDS: dict[str, list[float]] = {
    "log_snr_mean": [1.2, 1.95, 2.7, 3.45, 4.2, 5.0],
    "snr_dispersion": [0.1, 0.3, 0.5, 0.7, 0.9],
    "corr_strength_mean": [0.0, 0.3, 0.6, 0.9],
    "heteroskedastic_rate": [0.0, 0.2, 0.4, 0.6],
}
DEFAULT_TARGETS = ["energy_heating", "energy_cooling", "naval", "kin8nm"]


def override_knob(eta_dict: dict, knob: str, value: float) -> dict:
    d = json.loads(json.dumps(eta_dict))  # deep copy
    d[knob] = value
    return d


def energy_distance(cloud: np.ndarray, real: np.ndarray) -> float:
    """Energy distance between the cloud point set and the real point set."""
    d_cr = cdist(cloud, real).mean()
    d_cc = cdist(cloud, cloud).mean()  # includes zero diagonal; bias cancels in comparisons
    d_rr = cdist(real, real).mean()
    return float(2.0 * d_cr - d_cc - d_rr)


def run(checkpoint: Path, n_prior: int, k: int, batch_size: int, targets: list[str]) -> pl.DataFrame:
    device = select_device("auto")
    model, ck = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    base_eta_dict = hyperprior_to_dict(hyperprior_from_dict(ck["source_eta"]))
    print(f"base step={ck['step']} | device={device} | n_prior={n_prior} | targets={targets}\n")

    real_by_name: dict[str, list] = {}
    for name, _repeat, task in iter_real_tasks(CHAR_DIR):
        if name in targets:
            real_by_name.setdefault(name, []).append(task)

    records: list[dict] = []
    for name in targets:
        tasks = real_by_name[name]
        shape = characterization_shape(tasks[0])
        real_z = np.stack([pooled_embed(model, collate_tasks([t]), device)[0] for t in tasks])
        # fixed whitening frame from the base-eta cloud at this shape
        base_cloud = prior_cloud_z(
            model, device, hyperprior_from_dict(base_eta_dict), shape, n_prior, batch_size, f"{name}-base"
        )
        mean, std = base_cloud.mean(0), base_cloud.std(0) + 1e-6
        real_w = (real_z - mean) / std
        for knob, grid in KNOB_GRIDS.items():
            for value in grid:
                eta_v = hyperprior_from_dict(override_knob(base_eta_dict, knob, value))
                try:
                    cloud = prior_cloud_z(model, device, eta_v, shape, n_prior, batch_size, f"{name}-{knob}-{value}")
                except Exception as error:
                    print(f"  {name} {knob}={value} FAIL: {type(error).__name__}: {error}")
                    continue
                cloud_w = (cloud - mean) / std
                edist = energy_distance(cloud_w, real_w)
                knn = float(np.median(whitened_knn(real_z, cloud, mean, std, k, drop_self=False)))
                is_base = abs(value - base_eta_dict.get(knob, np.nan)) < 1e-9
                records.append(
                    {"target": name, "knob": knob, "value": value, "is_base": is_base, "energy": edist, "knn": knn}
                )
        print(f"{name}: swept {len(KNOB_GRIDS)} knobs")

    df = pl.DataFrame(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_DIR / "controllability.parquet")

    # slope report: does distance move monotonically across each knob, and toward the target?
    print(f"\n{'target':<15}{'knob':<22}{'energy lo->hi':>16}{'knn lo->hi':>14}  direction")
    for target in targets:
        for knob in KNOB_GRIDS:
            sub = df.filter((pl.col("target") == target) & (pl.col("knob") == knob)).sort("value")
            if sub.height < 2:
                continue
            e = sub["energy"].to_numpy()
            n = sub["knn"].to_numpy()
            # Spearman-sign: does energy fall as value rises? (toward target = falling)
            rank_corr = np.corrcoef(np.argsort(np.argsort(sub["value"].to_numpy())), np.argsort(np.argsort(e)))[0, 1]
            arrow = "TOWARD" if e[-1] < e[0] else "away"
            mono = "mono" if abs(rank_corr) > 0.9 else ""
            print(f"{target:<15}{knob:<22}{e[0]:>7.2f}->{e[-1]:<7.2f}{n[0]:>6.2f}->{n[-1]:<6.2f}  {arrow} {mono}")

    print(f"\nartifacts -> {OUT_DIR}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--n-prior", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    args = ap.parse_args()
    run(args.checkpoint, args.n_prior, args.k, args.batch_size, args.targets)


if __name__ == "__main__":
    main()
