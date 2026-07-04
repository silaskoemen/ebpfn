"""The Gate-1 H1 test (plans/gate1_revised.md §1/§3.5/§4).

One number decides the gate: the rank correlation between per-task prior-coverage
distance and per-task calibration error, **partialled on task size n and
dimensionality d** -- the most likely false-pass source (§4). We report the raw
Spearman, the n,d-partial Spearman with a task-level bootstrap CI, whether it
clears the pre-registered threshold and excludes 0, and the §4 secondary: the
incremental rank-R^2 of the joint (conditional-aware) coverage over the X-only
marginal coverage (does conditional structure add predictive value?).

Partial Spearman = Pearson correlation of the rank residuals after regressing
rank(coverage) and rank(calib) on a **flexible basis** of the controls (linear +
squares + pairwise interactions of standardized n, d). The flexible basis matters:
a confound that is non-linear or interaction-driven in n,d would survive a
linear-in-ranks control and manufacture a false pass (§4) -- the planted-confound
test guards exactly this.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ebpfn.gate1.config import GateConfig


def _control_design(controls: list[np.ndarray], n: int) -> np.ndarray:
    """Intercept + standardized linear controls, plus squares + pairwise
    interactions once there are enough tasks to afford them.

    The flexible (square/interaction) terms are what defeat a non-linear n,d
    confound (§4), but they cost degrees of freedom: with few tasks they leave
    too little residual dof and the partial correlation degenerates. We add them
    only when n >= 6 * (#flexible columns) so the control stays well-posed.
    """
    std = [(c - c.mean()) / (c.std() + 1e-12) for c in controls]
    cols = [np.ones(n)] + std
    k = len(std)
    n_flex = k + k * (k - 1) // 2  # squares + pairwise interactions
    if n >= 6 * (k + 1 + n_flex):
        cols += [s * s for s in std]
        for i in range(k):
            for j in range(i + 1, k):
                cols.append(std[i] * std[j])
    return np.column_stack(cols)


def _residualize(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    """Spearman of x,y after partialling out a flexible basis of `controls`."""
    rx, ry = rankdata(x), rankdata(y)
    if not controls:
        return _pearson(rx, ry)
    design = _control_design(controls, rx.size)
    return _pearson(_residualize(rx, design), _residualize(ry, design))


def _join(coverage_rows: list[dict], calib_rows: list[dict]) -> dict[str, np.ndarray]:
    """Join coverage and calibration tables by (source_did, target)."""
    cal_by_key = {(r["source_did"], r["target"]): r for r in calib_rows}
    cov, xcov, n, d, keys = [], [], [], [], []
    for r in coverage_rows:
        key = (r["source_did"], r["target"])
        if key not in cal_by_key:
            continue
        cov.append(r["coverage"])
        xcov.append(r["x_only_coverage"])
        n.append(r["n"])
        d.append(r["d"])
        keys.append(key)
    cal = cal_by_key
    return {
        "coverage": np.array(cov),
        "x_only_coverage": np.array(xcov),
        "n": np.array(n, dtype=float),
        "d": np.array(d, dtype=float),
        "keys": keys,
        "_cal": cal,
    }


def gate1_test(coverage_rows: list[dict], calib_rows: list[dict], cfg: GateConfig | None = None) -> dict:
    """Run the H1 test. Returns the decision inputs (one dict)."""
    cfg = cfg or GateConfig()
    j = _join(coverage_rows, calib_rows)
    cov, xcov, n, d, keys = j["coverage"], j["x_only_coverage"], j["n"], j["d"], j["keys"]
    calib = np.array([j["_cal"][k][cfg.calib_metric] for k in keys])
    if cov.size < 4:
        raise ValueError(f"need >= 4 joined tasks for the gate test, got {cov.size}")

    controls = [n, d]
    raw = partial_spearman(cov, calib, [])
    partial = partial_spearman(cov, calib, controls)

    rng = np.random.default_rng(0)
    boot = np.empty(cfg.n_boot)
    m = cov.size
    for b in range(cfg.n_boot):
        idx = rng.integers(0, m, m)
        boot[b] = partial_spearman(cov[idx], calib[idx], [n[idx], d[idx]])
    lo, hi = np.percentile(boot, [100 * cfg.alpha / 2, 100 * (1 - cfg.alpha / 2)])

    return {
        "metric": cfg.calib_metric,
        "n_tasks": int(m),
        "spearman_raw": raw,
        "partial_spearman": partial,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "threshold": cfg.effect_threshold,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "passes": bool(lo > cfg.effect_threshold),  # H1: higher coverage distance -> higher calib error
        "x_only_partial_spearman": partial_spearman(xcov, calib, controls),
        # §4 secondary: does joint (conditional-aware) coverage add beyond n,d AND the X-only marginal?
        "joint_over_xonly_partial_spearman": partial_spearman(cov, calib, [n, d, xcov]),
    }
