"""Sweep orchestration (spec §4/§5).

Per sweep point (construction x value x seed) we compute three metric families,
each on the data regime that makes it a fair, clean measurement:

- s-OTDD recall + real-real' null band: clouds of INDEPENDENT tasks (prior-level
  coverage), swept over lambda. Same prior-level data OTDD would see.
- conditional-coverage MMD: the same clouds pooled (prior-level P(Y|cell)), swept
  over K. A fair head-to-head with OTDD on identical data.
- calibration gap: shared-f triples (real-train, decoy-train, real-test with one
  fixed conditional mean), so only the noise-shape miscalibration moves; averaged
  over several triples per point.

Returns three tidy (long) polars frames so the lambda and K sub-sweeps stay
separable. Seeds give the CIs; the calibration gap is seed-paired (spec §5).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ebpfn.calibration import calibration_report
from ebpfn.config import ExperimentConfig, Prior
from ebpfn.distance import cloud_recall, inside_band, make_sotdd_fn, null_band
from ebpfn.mmd import CellPartition, aggregate, per_cell_mmd
from ebpfn.priors import draw_params, pool, realize, sample_cloud
from ebpfn.regressor import train_prob_regressor


def _point_rng(base_seed: int, value_idx: int, seed_idx: int) -> np.random.Generator:
    """Deterministic, independent rng per sweep point."""
    return np.random.default_rng([base_seed, value_idx, seed_idx])


def sotdd_rows(real_probe, real_ref, decoy_cloud, value, seed_idx, cfg, rng) -> list[dict]:
    """s-OTDD decoy recall vs real-real' null band, swept over lambda."""
    dc = cfg.distance
    rows = []
    for lam in dc.lam_grid:
        dist_fn = make_sotdd_fn(lam, dc.n_proj, rng, p=dc.p)
        band = null_band(real_probe, real_ref, dist_fn, rng, alpha=dc.null_alpha)
        decoy_mean = float(cloud_recall(real_probe, decoy_cloud, dist_fn).mean())
        rows.append({
            "value": value, "seed": seed_idx, "lam": lam,
            "decoy_recall": decoy_mean,
            "null_lo": band["band_lo"], "null_hi": band["band_hi"], "null_mean": band["mean"],
            "inside_band": inside_band(decoy_mean, band),
        })
    return rows


def mmd_rows(real_pool, decoy_pool, value, seed_idx, cfg, rng) -> list[dict]:
    """Conditional-coverage MMD on pooled prior-level data, swept over K."""
    mc = cfg.mmd
    rows = []
    Xcat = np.vstack([real_pool.X, decoy_pool.X])
    for k in mc.n_cells_grid:
        cells = CellPartition(k, mc.method, rng).fit(Xcat)
        cell_mmd = per_cell_mmd(
            real_pool, decoy_pool, cells,
            bandwidth=mc.bandwidth, min_per_cell=mc.min_per_cell, max_per_cell=mc.max_per_cell, rng=rng,
        )
        agg = aggregate(cell_mmd)
        rows.append({
            "value": value, "seed": seed_idx, "n_cells": k,
            "mmd_mean": agg["mean"], "mmd_max": agg["max"], "cells_used": agg["n_cells"],
        })
    return rows


def calib_gap_row(prior_ref: Prior, prior_alt: Prior, value, seed_idx, cfg, rng) -> dict:
    """Calibration gap (alt - ref) on shared-f triples, averaged (spec §3.3).

    The reference (ref) and alternative (alt) models are trained on shared-f draws
    from their priors; both are evaluated on a fresh ref-prior test task. For the
    real run, ref=real and alt=decoy. For the null run, ref=alt=real (two
    independent real draws) so the gap reflects only estimation noise.
    """
    sc, mc, kind = cfg.sweep, cfg.model, cfg.model.kind
    nll_gaps, crps_gaps, nll_ref_l, nll_alt_l = [], [], [], []
    for _ in range(sc.n_calib_tasks):
        params = draw_params(cfg.data, rng)  # one shared conditional mean
        D_ref_tr = realize(prior_ref, params, sc.calib_n_train, rng)
        D_alt_tr = realize(prior_alt, params, sc.calib_n_train, rng)
        D_test = realize(prior_ref, params, sc.calib_n_test, rng)
        rep_ref = calibration_report(train_prob_regressor(D_ref_tr, kind, mc), D_test, cfg.calib)
        rep_alt = calibration_report(train_prob_regressor(D_alt_tr, kind, mc), D_test, cfg.calib)
        nll_gaps.append(rep_alt["nll"] - rep_ref["nll"])
        crps_gaps.append(rep_alt["crps"] - rep_ref["crps"])
        nll_ref_l.append(rep_ref["nll"])
        nll_alt_l.append(rep_alt["nll"])
    return {
        "value": value, "seed": seed_idx, "head": kind,
        "nll_gap": float(np.mean(nll_gaps)), "crps_gap": float(np.mean(crps_gaps)),
        "nll_ref": float(np.mean(nll_ref_l)), "nll_alt": float(np.mean(nll_alt_l)),
    }


def summarize(frames: dict[str, pl.DataFrame], cfg: ExperimentConfig) -> pl.DataFrame:
    """Per-value decision inputs (spec §0/§4) at primary lambda/K: whether decoy
    recall sits inside the null band, the conditional score, and a seed-paired
    Wilcoxon on the NLL gap (one-sided gap > 0)."""
    from scipy.stats import wilcoxon

    lam, k = cfg.distance.lam, cfg.mmd.n_cells
    sotdd = frames["sotdd"].filter(pl.col("lam") == lam)
    mmd = frames["mmd"].filter(pl.col("n_cells") == k)
    calib = frames["calib"]

    rows = []
    for value in cfg.sweep.values:
        s = sotdd.filter(pl.col("value") == value)
        m = mmd.filter(pl.col("value") == value)
        c = calib.filter(pl.col("value") == value)
        gaps = c["nll_gap"].to_numpy()
        # one-sided Wilcoxon signed-rank that the seed-paired gap exceeds 0
        try:
            p = float(wilcoxon(gaps, alternative="greater").pvalue) if (gaps != 0).any() else 1.0
        except ValueError:
            p = 1.0
        rows.append({
            "value": value,
            "decoy_recall": float(s["decoy_recall"].mean()),
            "null_lo": float(s["null_lo"].mean()),
            "null_hi": float(s["null_hi"].mean()),
            "otdd_covered": bool(s["inside_band"].mean() >= 0.5),
            "cond_mmd_mean": float(m["mmd_mean"].mean()),
            "cond_mmd_max": float(m["mmd_max"].mean()),
            "nll_gap": float(gaps.mean()),
            "nll_gap_wilcoxon_p": p,
        })
    return pl.DataFrame(rows).sort("value")


def run_sweep(cfg: ExperimentConfig) -> dict[str, pl.DataFrame]:
    """Run the full sweep; return tidy frames keyed 'sotdd', 'mmd', 'calib'."""
    sc = cfg.sweep
    sotdd, mmd, calib = [], [], []
    for vi, value in enumerate(sc.values):
        real = Prior(sc.construction, "real", value, cfg.data)
        decoy = Prior(sc.construction, "decoy", value, cfg.data)
        for si in range(sc.n_seeds):
            rng = _point_rng(cfg.seed, vi, si)
            real_probe = sample_cloud(real, sc.n_tasks_per_prior, rng, n=sc.cloud_n_rows)
            real_ref = sample_cloud(real, sc.n_tasks_per_prior, rng, n=sc.cloud_n_rows)
            decoy_cloud = sample_cloud(decoy, sc.n_tasks_per_prior, rng, n=sc.cloud_n_rows)

            sotdd += sotdd_rows(real_probe, real_ref, decoy_cloud, value, si, cfg, rng)
            mmd += mmd_rows(pool(real_probe + real_ref), pool(decoy_cloud), value, si, cfg, rng)
            calib.append(calib_gap_row(real, decoy, value, si, cfg, rng))

    return {
        "sotdd": pl.DataFrame(sotdd),
        "mmd": pl.DataFrame(mmd),
        "calib": pl.DataFrame(calib),
    }


def run_null(cfg: ExperimentConfig) -> dict[str, pl.DataFrame]:
    """Real-vs-real' null run for threshold pre-registration (spec §4).

    Computes the conditional-MMD and calibration-gap statistics with no decoy, so
    their upper percentiles give T_cond and T_cal before the real run.
    """
    sc = cfg.sweep
    mmd, calib = [], []
    for vi, value in enumerate(sc.values):
        real = Prior(sc.construction, "real", value, cfg.data)
        for si in range(sc.n_seeds):
            rng = _point_rng(cfg.seed, vi, si)
            real_a = sample_cloud(real, sc.n_tasks_per_prior, rng, n=sc.cloud_n_rows)
            real_b = sample_cloud(real, sc.n_tasks_per_prior, rng, n=sc.cloud_n_rows)
            mmd += mmd_rows(pool(real_a), pool(real_b), value, si, cfg, rng)
            calib.append(calib_gap_row(real, real, value, si, cfg, rng))
    return {"mmd_null": pl.DataFrame(mmd), "calib_null": pl.DataFrame(calib)}


def suggest_thresholds(null_frames: dict[str, pl.DataFrame], cfg: ExperimentConfig, q: float = 99.0) -> dict:
    """T_cond / T_cal as the qth percentile of the real-vs-real' null statistics."""
    k = cfg.mmd.n_cells
    mmd_null = null_frames["mmd_null"].filter(pl.col("n_cells") == k)
    calib_null = null_frames["calib_null"]
    return {
        "q": q,
        "primary_n_cells": k,
        "T_cond_mean": float(np.percentile(mmd_null["mmd_mean"].to_numpy(), q)),
        "T_cond_max": float(np.percentile(mmd_null["mmd_max"].to_numpy(), q)),
        "T_cal_nll": float(np.percentile(calib_null["nll_gap"].to_numpy(), q)),
        "T_cal_crps": float(np.percentile(calib_null["crps_gap"].to_numpy(), q)),
    }
