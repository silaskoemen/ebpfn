"""Exploratory Gate-0 region-mapping pilot (NOT pre-registered).

Purpose (see plans/spec.md §0 and the de-confound investigation):
  - Construction A uses *fixed-mass* quantile bands so the separation axis g moves
    feature-separation alone (band mass held constant) -- de-confounding the old s axis.
  - We do not hunt for a single passing point; we MAP the Gate-0-passing region and
    report its *width* across (g, sigma-contrast). A fat region = robust pass; a
    knife-edge that needs extreme sigma-contrast is barely better than a fail.
  - Construction B (heteroskedastic vs homoskedastic) is co-primary and geometry-free:
    masking, if any, comes from the genuine OT-averaging mechanism. A robust B pass is
    stronger evidence than any tuned A pass. If de-confounded A only passes on a
    knife-edge AND B never passes, that is a real finding -- report it, don't tune it away.

A cell "passes Gate-0" iff (spec §0): decoy s-OTDD recall is INSIDE the real-real' null
band AND cond-MMD(mean) > T_cond AND NLL gap > T_cal with seed-paired Wilcoxon p < 0.05,
where T_cond/T_cal are the 99th-pct real-real' null thresholds for that cell.

    pixi run python benchmarks/scripts/pilot_region.py            # A and B
    pixi run python benchmarks/scripts/pilot_region.py --quick    # fast smoke
    pixi run python benchmarks/scripts/pilot_region.py --only A
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from ebpfn.config import ExperimentConfig
from ebpfn.config import SweepConfig
from ebpfn.experiment import run_null
from ebpfn.experiment import run_sweep
from ebpfn.experiment import suggest_thresholds
from ebpfn.experiment import summarize
from ebpfn.results import save_run
from scipy.stats import spearmanr

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

# sigma-contrast: matched geometric mean (=1, the baseline sigma0), ratio = sigma_hi/sigma_lo
SIGMA_RATIOS = (4.0, 8.0, 16.0)
A_G_GRID = (0.05, 0.15, 0.3, 0.45, 0.6)  # separation (prob-gap); needs g <= 1 - 2*band_mass
B_KAPPA_GRID = (0.25, 0.5, 1.0, 1.5, 2.0)
WILCOXON_ALPHA = 0.05


def _sigma_pair(ratio: float) -> tuple[float, float]:
    """sigma_hi, sigma_lo with geometric mean 1.0 and sigma_hi/sigma_lo == ratio."""
    r = float(np.sqrt(ratio))
    return r, 1.0 / r


def _base_sweep(args: argparse.Namespace) -> dict:
    if args.quick:
        return dict(
            n_seeds=3, n_tasks_per_prior=12, cloud_n_rows=400, n_calib_tasks=1, calib_n_train=1500, calib_n_test=1500
        )
    return dict(
        n_seeds=args.seeds,
        n_tasks_per_prior=args.tasks,
        cloud_n_rows=600,
        n_calib_tasks=1,
        calib_n_train=2000,
        calib_n_test=2000,
    )


def _quick_trims(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if not args.quick:
        return cfg
    distance = dataclasses.replace(cfg.distance, n_proj=100)
    mmd = dataclasses.replace(cfg.mmd, n_cells_grid=(8, 16))
    model = dataclasses.replace(cfg.model, catboost_iterations=200)
    return dataclasses.replace(cfg, distance=distance, mmd=mmd, model=model)


def _run_cell(cfg: ExperimentConfig) -> pl.DataFrame:
    """One (construction x geometry x sigma) config swept over its values.

    Returns the per-value summary with the null thresholds merged in and a `passes`
    column applying the spec-§0 gate rule.
    """
    null = run_null(cfg)
    thr = suggest_thresholds(null, cfg, q=99.0)
    summary = summarize(run_sweep(cfg), cfg)
    return summary.with_columns(
        pl.lit(thr["T_cond_mean"]).alias("T_cond"),
        pl.lit(thr["T_cal_nll"]).alias("T_cal"),
    ).with_columns(
        (
            pl.col("otdd_covered")
            & (pl.col("cond_mmd_mean") > pl.col("T_cond"))
            & (pl.col("nll_gap") > pl.col("T_cal"))
            & (pl.col("nll_gap_wilcoxon_p") < WILCOXON_ALPHA)
        ).alias("passes")
    )


def _region_summary(df: pl.DataFrame, axis: str, group: str | None) -> pl.DataFrame:
    """Per-group passing-region width along `axis` + Spearman(cond_mmd, nll_gap)."""
    rows = []
    groups = [None] if group is None else df[group].unique().sort().to_list()
    for gval in groups:
        sub = df if group is None else df.filter(pl.col(group) == gval)
        sub = sub.sort(axis)
        passed = sub.filter(pl.col("passes"))
        rho = (
            spearmanr(sub["cond_mmd_mean"].to_numpy(), sub["nll_gap"].to_numpy()).correlation
            if sub.height >= 3
            else float("nan")
        )
        row = {
            "n_pass": passed.height,
            "frac_pass": passed.height / sub.height if sub.height else 0.0,
            f"{axis}_min_pass": float(passed[axis].min()) if passed.height else None,
            f"{axis}_max_pass": float(passed[axis].max()) if passed.height else None,
            "spearman_cond_cal": float(rho) if rho is not None else float("nan"),
        }
        if group is not None:
            row[group] = gval
        rows.append(row)
    return pl.DataFrame(rows)


def _heatmap(df: pl.DataFrame, row_key: str, col_key: str, out_path: Path, title: str) -> None:
    rows = df[row_key].unique().sort().to_list()
    cols = df[col_key].unique().sort().to_list()
    M = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            cell = df.filter((pl.col(row_key) == r) & (pl.col(col_key) == c))
            if cell.height:
                M[i, j] = 1.0 if bool(cell["passes"][0]) else 0.0
    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 2, 1.0 * len(rows) + 2))
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [f"{c:g}" for c in cols])
    ax.set_yticks(range(len(rows)), [f"{r:g}" for r in rows])
    ax.set_xlabel(col_key)
    ax.set_ylabel(row_key)
    ax.set_title(title)
    for i in range(len(rows)):
        for j in range(len(cols)):
            txt = "pass" if M[i, j] == 1.0 else ("fail" if M[i, j] == 0.0 else "-")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_construction_a(base: ExperimentConfig, args: argparse.Namespace) -> None:
    sweep_kw = _base_sweep(args)
    g_grid = tuple(float(v) for v in args.values.split(",")) if args.values else A_G_GRID
    cells = []
    for ratio in SIGMA_RATIOS:
        hi, lo = _sigma_pair(ratio)
        data = dataclasses.replace(
            base.data, band_geometry="fixed_mass", band_mass=args.band_mass, sigma_hi=hi, sigma_lo=lo
        )
        sweep = SweepConfig(construction="A", values=g_grid, **sweep_kw)
        cfg = _quick_trims(dataclasses.replace(base, data=data, sweep=sweep, seed=args.seed), args)
        print(f"[A] sigma_ratio={ratio:g} (hi={hi:.3f} lo={lo:.3f}) band_mass={args.band_mass} g={g_grid}")
        summ = (
            _run_cell(cfg)
            .with_columns(pl.lit(ratio).alias("sigma_ratio"), pl.lit(args.band_mass).alias("band_mass"))
            .rename({"value": "g"})
        )
        cells.append(summ)

    df = pl.concat(cells)
    region = _region_summary(df, axis="g", group="sigma_ratio")
    out = RESULTS_ROOT / "pilot_A"
    save_run(out, base, {"cells": df, "region": region})
    _heatmap(
        df,
        "sigma_ratio",
        "g",
        out / "pass_heatmap_A.png",
        f"Construction A (fixed_mass m={args.band_mass}): Gate-0 pass region",
    )
    _report("A", df, region, out)


def run_construction_b(base: ExperimentConfig, args: argparse.Namespace) -> None:
    sweep_kw = _base_sweep(args)
    k_grid = tuple(float(v) for v in args.values.split(",")) if args.values else B_KAPPA_GRID
    sweep = SweepConfig(construction="B", values=k_grid, **sweep_kw)
    cfg = _quick_trims(dataclasses.replace(base, sweep=sweep, seed=args.seed), args)
    print(f"[B] geometry-free, kappa={k_grid}")
    df = _run_cell(cfg).rename({"value": "kappa"})
    region = _region_summary(df, axis="kappa", group=None)
    out = RESULTS_ROOT / "pilot_B"
    save_run(out, base, {"cells": df, "region": region})
    _report("B", df, region, out)


def _report(tag: str, df: pl.DataFrame, region: pl.DataFrame, out: Path) -> None:
    (out / "region.json").write_text(json.dumps(region.to_dicts(), indent=2, default=str))
    with pl.Config(tbl_rows=-1, tbl_width_chars=220, float_precision=4):
        print(f"\n=== Construction {tag}: per-cell Gate-0 inputs ===")
        print(df)
        print(f"\n=== Construction {tag}: passing-region width ===")
        print(region)
    n_pass = int(df["passes"].sum())
    print(f"[{tag}] {n_pass}/{df.height} cells pass Gate-0. Frames + region.json -> {out}")
    if n_pass == 0:
        print(f"[{tag}] NO passing cell -- candidate real finding (OTDD robust to this mismatch), not a bug.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["A", "B"], default=None, help="run a single construction")
    ap.add_argument("--quick", action="store_true", help="fast smoke (few seeds/tasks); not confirmatory")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=20, help="tasks per prior cloud")
    ap.add_argument("--band-mass", type=float, default=0.15, dest="band_mass")
    ap.add_argument("--values", type=str, default="", help="override sweep grid (g for A, kappa for B)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = ExperimentConfig()
    if args.only in (None, "A"):
        run_construction_a(base, args)
    if args.only in (None, "B"):
        run_construction_b(base, args)


if __name__ == "__main__":
    main()
