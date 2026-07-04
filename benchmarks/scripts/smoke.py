"""End-to-end smoke run for one Construction-A sweep point (not a full sweep).

Builds real/decoy/real' clouds, computes s-OTDD decoy recall vs the real-real'
null band, the conditional-coverage MMD, and the calibration gap on a fresh real
test task. Run: `pixi run python benchmarks/scripts/smoke.py`.
"""

from __future__ import annotations

import warnings

import numpy as np

# LightGBM fits on ndarrays; sklearn's predict-time feature-name check is noise here.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from ebpfn.calibration import calibration_report
from ebpfn.config import DataConfig
from ebpfn.config import DistanceConfig
from ebpfn.config import MMDConfig
from ebpfn.config import ModelConfig
from ebpfn.config import Prior
from ebpfn.distance import cloud_recall
from ebpfn.distance import inside_band
from ebpfn.distance import make_sotdd_fn
from ebpfn.distance import null_band
from ebpfn.mmd import CellPartition
from ebpfn.mmd import aggregate
from ebpfn.mmd import per_cell_mmd
from ebpfn.priors import sample_cloud
from ebpfn.priors import sample_task
from ebpfn.regressor import train_prob_regressor


def main() -> None:
    rng = np.random.default_rng(0)
    dc = DataConfig()
    dist_cfg = DistanceConfig()
    mmd_cfg = MMDConfig()
    s = 0.25  # small s: the regime where the thesis should live

    real = Prior("A", "real", s, dc)
    decoy = Prior("A", "decoy", s, dc)

    n_tasks = 20  # small for the smoke run
    print(f"[smoke] Construction A, s={s}, {n_tasks} tasks/cloud")
    real_probe = sample_cloud(real, n_tasks, rng, n=600)
    real_ref = sample_cloud(real, n_tasks, rng, n=600)
    decoy_cloud = sample_cloud(decoy, n_tasks, rng, n=600)

    dist_fn = make_sotdd_fn(dist_cfg.lam, dist_cfg.n_proj, rng, p=dist_cfg.p)

    band = null_band(real_probe, real_ref, dist_fn, rng, alpha=dist_cfg.null_alpha)
    decoy_recalls = cloud_recall(real_probe, decoy_cloud, dist_fn)
    decoy_mean = float(decoy_recalls.mean())
    covered = inside_band(decoy_mean, band)
    print(f"[s-OTDD] null band (real-real')  = [{band['band_lo']:.4f}, {band['band_hi']:.4f}], mean={band['mean']:.4f}")
    print(f"[s-OTDD] decoy recall mean       = {decoy_mean:.4f}  -> inside band: {covered}")

    # Conditional-coverage meter on one paired real/decoy task.
    Dr = sample_task(real, rng, n=2000)
    Dd = sample_task(decoy, rng, n=2000)
    cells = CellPartition(mmd_cfg.n_cells, mmd_cfg.method, rng).fit(np.vstack([Dr.X, Dd.X]))
    cell_mmd = per_cell_mmd(Dr, Dd, cells, min_per_cell=mmd_cfg.min_per_cell, rng=rng)
    agg = aggregate(cell_mmd)
    print(f"[cond-MMD] cells used={agg['n_cells']}  mean={agg['mean']:.4f}  max={agg['max']:.4f}")

    # Calibration gap on a fresh real test task under natural P(X), for both heads.
    D_real_train = sample_task(real, rng, n=2000)
    D_decoy_train = sample_task(decoy, rng, n=2000)
    D_test = sample_task(real, rng, n=2000)
    for kind in ("catboost_gauss", "qgbm"):
        cfg = ModelConfig(kind=kind)
        rep_real = calibration_report(train_prob_regressor(D_real_train, kind, cfg), D_test)
        rep_decoy = calibration_report(train_prob_regressor(D_decoy_train, kind, cfg), D_test)
        print(
            f"[calib:{kind:14s}] gap (decoy-real)  "
            f"nll={rep_decoy['nll'] - rep_real['nll']:+.4f}  "
            f"crps={rep_decoy['crps'] - rep_real['crps']:+.4f}  "
            f"(real nll={rep_real['nll']:.3f}, decoy nll={rep_decoy['nll']:.3f})"
        )


if __name__ == "__main__":
    main()
