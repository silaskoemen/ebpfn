"""The primary output figure (spec §4): one figure per construction, x = sweep
variable, three stacked panels with seed CIs:

1. s-OTDD decoy recall with the real-real' null band shaded;
2. conditional-coverage MMD (mean and max);
3. calibration gap (NLL and CRPS), with the zero line marked.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from ebpfn.config import ExperimentConfig

_Z = 1.96  # 95% normal CI


def _agg(df: pl.DataFrame, ycol: str) -> pl.DataFrame:
    """Per-value mean and 95% CI half-width across seeds, sorted by value."""
    return (
        df.group_by("value")
        .agg(
            pl.col(ycol).mean().alias("mean"),
            (_Z * pl.col(ycol).std() / pl.col(ycol).count().sqrt()).alias("ci"),
        )
        .sort("value")
    )


def _line(ax, df, ycol, label, **kw):
    a = _agg(df, ycol)
    x = a["value"].to_numpy()
    m = a["mean"].to_numpy()
    ci = a["ci"].fill_null(0.0).to_numpy()
    ax.plot(x, m, marker="o", label=label, **kw)
    ax.fill_between(x, m - ci, m + ci, alpha=0.2)


def make_sweep_figure(frames: dict[str, pl.DataFrame], cfg: ExperimentConfig) -> plt.Figure:
    sweep_name = "s" if cfg.sweep.construction == "A" else "kappa"
    lam, k = cfg.distance.lam, cfg.mmd.n_cells

    sotdd = frames["sotdd"].filter(pl.col("lam") == lam)
    mmd = frames["mmd"].filter(pl.col("n_cells") == k)
    calib = frames["calib"]

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # Panel 1: s-OTDD recall + null band.
    ax = axes[0]
    _line(ax, sotdd, "decoy_recall", "decoy recall", color="C3")
    nb = sotdd.group_by("value").agg(
        pl.col("null_lo").mean().alias("lo"), pl.col("null_hi").mean().alias("hi")
    ).sort("value")
    ax.fill_between(nb["value"].to_numpy(), nb["lo"].to_numpy(), nb["hi"].to_numpy(),
                    color="C0", alpha=0.25, label="real-real' null band")
    ax.set_ylabel("s-OTDD recall")
    ax.set_title(f"Construction {cfg.sweep.construction}: gate-0 sweep (lambda={lam}, K={k})")
    ax.legend(fontsize=8)

    # Panel 2: conditional-coverage MMD (mean and max).
    ax = axes[1]
    _line(ax, mmd, "mmd_mean", "cond-MMD mean", color="C2")
    _line(ax, mmd, "mmd_max", "cond-MMD max", color="C4")
    ax.set_ylabel("conditional MMD^2")
    ax.legend(fontsize=8)

    # Panel 3: calibration gap (NLL and CRPS).
    ax = axes[2]
    _line(ax, calib, "nll_gap", "NLL gap (primary)", color="C1")
    _line(ax, calib, "crps_gap", "CRPS gap", color="C5")
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("calibration gap (decoy - real)")
    ax.set_xlabel(f"sweep variable {sweep_name}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
