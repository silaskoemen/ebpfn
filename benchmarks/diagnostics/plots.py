"""Plots for the V2 diagnostics (see ``PLAN.md``), rendered from the persisted
artifacts of ``reachability.py`` and ``controllability.py`` -- no model needed.

    pixi run python -m benchmarks.diagnostics.plots [--which reachability|controllability|all]

E1: per-dataset small multiples (local PCA, honest per-shape) + a pooled-shape
overview. E2: cloud->target energy distance vs knob, normalized to the base
setting (<1 toward target, >1 away).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REACH_DIR = Path("benchmarks/results/diagnostics/reachability")
CTRL_DIR = Path("benchmarks/results/diagnostics/controllability")

CLOUD = "#86b6ef"  # light blue
CRIMSON = "#d1183c"
STRONG = ["#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834", "#0d6b6b", "#7a4a1e"]
KNOBS = ["log_snr_mean", "snr_dispersion", "corr_strength_mean", "heteroskedastic_rate"]
KNOB_BASE = {"log_snr_mean": 2.7, "snr_dispersion": 0.5, "corr_strength_mean": 0.6, "heteroskedastic_rate": 0.2}
TARGET_COLOR = {"energy_heating": "#e34948", "energy_cooling": "#eb6834", "naval": "#4a3aa7", "kin8nm": "#1baf7a"}


def _pca(fit: np.ndarray, n: int):
    mean = fit.mean(0)
    _, _, vt = np.linalg.svd(fit - mean, full_matrices=False)
    comp = vt[:n]
    return lambda x: (x - mean) @ comp.T


def plot_reachability() -> None:
    z = np.load(REACH_DIR / "embeddings.npz", allow_pickle=True)
    prior_z, real_z = z["prior_z"], z["real_z"]
    names = [str(s) for s in z["names"]]
    bounds = np.cumsum([0, *[int(s) for s in z["prior_sizes"]]])
    clouds = {names[i]: prior_z[bounds[i] : bounds[i + 1]] for i in range(len(names))}
    ratios = {
        r["dataset"]: r["ratio"] for r in pl.read_parquet(REACH_DIR / "reachability.parquet").iter_rows(named=True)
    }

    # small multiples (local PCA per dataset)
    order = sorted(names, key=lambda d: ratios.get(d, 0.0))
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, name in zip(axes.flat, order, strict=True):
        cloud = clouds[name]
        real = real_z[names.index(name)][None, :]
        proj = _pca(np.vstack([cloud, real]), 2)
        pc, pr = proj(cloud), proj(real)
        ax.scatter(pc[:, 0], pc[:, 1], s=14, c=CLOUD, alpha=0.5, edgecolors="none", label="prior cloud")
        ax.scatter(pr[:, 0], pr[:, 1], s=170, c=CRIMSON, edgecolors="white", linewidths=2, zorder=3, label="real")
        ratio = float(ratios[name])
        verdict = "inside" if ratio <= 1.5 else "far" if ratio > 3 else "edge"
        ax.set_title(f"{name}   ratio={ratio:.2f} ({verdict})", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    axes.flat[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("E1 per-dataset (local PCA): real point vs its own-shape prior cloud", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(REACH_DIR / "e1_small_multiples.png", dpi=130)
    plt.close(fig)

    # pooled overview
    proj = _pca(np.vstack([prior_z, real_z]), 2)
    pc, pr = proj(prior_z), proj(real_z)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(pc[:, 0], pc[:, 1], s=12, c=CLOUD, alpha=0.5, edgecolors="none", label="prior cloud (pooled)")
    for i, name in enumerate(names):
        ax.scatter(pr[i, 0], pr[i, 1], s=200, c=STRONG[i % len(STRONG)], edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(
            name, (pr[i, 0], pr[i, 1]), fontsize=10, fontweight="bold", xytext=(7, 5), textcoords="offset points"
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9)
    ax.set_title("E1 overview: prior cloud vs real datasets (pooled across shapes — gestalt only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(REACH_DIR / "e1_overview.png", dpi=130)
    plt.close(fig)
    print(f"saved -> {REACH_DIR}/e1_small_multiples.png, e1_overview.png")


def plot_controllability() -> None:
    df = pl.read_parquet(CTRL_DIR / "controllability.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, knob in zip(axes.flat, KNOBS, strict=True):
        for target, color in TARGET_COLOR.items():
            sub = df.filter((pl.col("target") == target) & (pl.col("knob") == knob)).sort("value")
            if sub.height < 2:
                continue
            v, e = sub["value"].to_numpy(), sub["energy"].to_numpy()
            e0 = e[np.argmin(np.abs(v - KNOB_BASE[knob]))]
            ax.plot(v, e / e0, "-o", color=color, lw=2, ms=6, label=target)
        ax.axhline(1.0, color="#999", lw=1, ls="--", zorder=0)
        ax.axvline(KNOB_BASE[knob], color="#ccc", lw=1, ls=":", zorder=0)
        ax.set_title(knob, fontsize=12)
        ax.set_xlabel("knob value")
        ax.set_ylabel("energy dist / base (1.0)")
    axes.flat[0].legend(fontsize=9, framealpha=0.9)
    fig.suptitle(
        "E2 controllability: cloud->target energy distance vs knob (<1 toward, >1 away). log_snr steers selectively.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(CTRL_DIR / "e2_sweeps.png", dpi=130)
    plt.close(fig)
    print(f"saved -> {CTRL_DIR}/e2_sweeps.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["reachability", "controllability", "all"], default="all")
    args = ap.parse_args()
    if args.which in ("reachability", "all"):
        plot_reachability()
    if args.which in ("controllability", "all"):
        plot_controllability()


if __name__ == "__main__":
    main()
