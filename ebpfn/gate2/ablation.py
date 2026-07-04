"""The across-prior ablation -- Gate-2's PRIMARY test (plans/gate2.md §3).

The cleanest design (promoted from a Gate-1 fallback to the spine here): change
the prior in a controlled way and ask whether the *change* in a task's coverage
predicts the *change* in its calibration. Differencing within task removes
task-intrinsic difficulty (the confound that, together with zero predictor
variance, made the Gate-1 cross-sectional correlation uninterpretable), and the
single directional coefficient leaves almost no room for forking-paths search.

We stack the K priors, keep tasks measured under all of them, demean coverage and
calibration within task across priors, and take the fixed-effects slope /
correlation of the demeaned pairs. CI is a cluster bootstrap over tasks. The
predicted sign is positive: a prior under which a task is *worse* covered (larger
descriptor distance) should give that task *worse* calibration (larger NLL).

A secondary cross-sectional test (per-prior partial Spearman of coverage vs
calibration, controlling n and d) is reported for direct comparison with the
Gate-1 number (0.083) -- but it is explicitly NOT the headline.
"""

from __future__ import annotations

import numpy as np

from ebpfn.gate1.gate import partial_spearman
from ebpfn.gate2.config import Gate2Config


def _pivot(rows: list[dict], priors: list[str], metric: str) -> dict:
    """tasks x priors matrices for coverage and calib over tasks seen in all priors."""
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault((r["source_did"], r["target"]), {})[r["prior"]] = r
    keys = [k for k, d in by_key.items() if all(p in d for p in priors)]
    if len(keys) < 4:
        raise ValueError(f"need >= 4 tasks present in all {len(priors)} priors, got {len(keys)}")
    cov = np.array([[by_key[k][p]["coverage"] for p in priors] for k in keys])
    cal = np.array([[by_key[k][p][metric] for p in priors] for k in keys])
    n = np.array([by_key[k][priors[0]]["n"] for k in keys], dtype=float)
    d = np.array([by_key[k][priors[0]]["d"] for k in keys], dtype=float)
    return {"keys": keys, "cov": cov, "cal": cal, "n": n, "d": d}


def _fe_slope_corr(cov: np.ndarray, cal: np.ndarray) -> tuple[float, float]:
    """Fixed-effects (within-task) slope and correlation of demeaned pairs."""
    c = cov - cov.mean(axis=1, keepdims=True)
    y = cal - cal.mean(axis=1, keepdims=True)
    cf, yf = c.ravel(), y.ravel()
    denom = float(np.sum(cf * cf))
    slope = 0.0 if denom < 1e-12 else float(np.sum(cf * yf) / denom)
    corr = 0.0 if cf.std() == 0 or yf.std() == 0 else float(np.corrcoef(cf, yf)[0, 1])
    return slope, corr


def ablation_test(rows: list[dict], priors: list[str], cfg: Gate2Config | None = None) -> dict:
    """Run the across-prior fixed-effects ablation. Returns the decision inputs."""
    cfg = cfg or Gate2Config()
    piv = _pivot(rows, priors, cfg.calib_metric)
    cov, cal = piv["cov"], piv["cal"]
    keys = piv["keys"]
    m = len(keys)

    coverage_spread = float(np.mean(cov.std(axis=1)))  # within-task variation across priors
    slope, corr = _fe_slope_corr(cov, cal)

    rng = np.random.default_rng(cfg.seed)
    boot = np.empty(cfg.n_boot)
    for b in range(cfg.n_boot):
        idx = rng.integers(0, m, m)  # cluster bootstrap: resample whole tasks
        boot[b] = _fe_slope_corr(cov[idx], cal[idx])[1]
    lo, hi = np.percentile(boot, [100 * cfg.alpha / 2, 100 * (1 - cfg.alpha / 2)])

    has_variation = coverage_spread >= cfg.min_coverage_spread
    enough_tasks = m >= cfg.min_ablation_tasks
    thr = cfg.ablation_effect_threshold
    # Three-way Part B decision (one-sided positive hypothesis):
    #   link    : CI lower bound clears the bar       -> a usable positive effect is shown
    #   no_link : CI upper bound below the bar        -> a usable positive effect is RULED OUT
    #   else    : CI straddles the bar (underpowered) -> can't tell; report inconclusive
    # Both decisive calls also require enough tasks; a wide/unstable CI on few tasks
    # falls through to inconclusive rather than masquerading as a true null.
    passes = bool(has_variation and enough_tasks and lo > thr)  # the "link" call
    ruled_out_effect = bool(has_variation and enough_tasks and hi < thr)  # the "no link" call

    # secondary: per-prior cross-sectional partial Spearman (compare to Gate-1 0.083)
    cross = {}
    for j, p in enumerate(priors):
        cross[p] = float(partial_spearman(cov[:, j], cal[:, j], [piv["n"], piv["d"]]))

    return {
        "metric": cfg.calib_metric,
        "n_tasks": m,
        "priors": priors,
        "coverage_spread": coverage_spread,
        "has_coverage_variation": bool(has_variation),
        "fe_slope": slope,
        "fe_corr": corr,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_half_width": float((hi - lo) / 2),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "effect_threshold": cfg.ablation_effect_threshold,
        "enough_tasks": enough_tasks,
        "min_ablation_tasks": cfg.min_ablation_tasks,
        "ruled_out_effect": ruled_out_effect,
        "passes": passes,
        "cross_sectional_partial_spearman": cross,
        "gate1_reference": 0.083,  # the joint s-OTDD number this replaces
    }
