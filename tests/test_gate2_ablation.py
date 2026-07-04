"""Gate-2 ablation + verdict: the primary test must recover a planted within-task
coverage->calibration link and stay null when there is none, and the verdict
must route correctly through the three pre-committed branches.
"""
from __future__ import annotations

import numpy as np

from ebpfn.gate2 import Gate2Config, ablation_test, gate2_verdict
from ebpfn.gate2.report import DEAD, INCONCLUSIVE, INCONCLUSIVE_POWER, LINK, NO_LINK


def _rows(M, priors, rng, slope, noise):
    """M tasks x K priors. coverage varies within task across priors; calib is
    task_baseline + slope*coverage + noise (slope>0 = the planted link)."""
    rows = []
    for i in range(M):
        base = rng.normal(0, 2.0)  # task-intrinsic difficulty (the confound)
        for p in priors:
            cov = rng.uniform(0.5, 4.0)  # within-task coverage variation across priors
            nll = base + slope * cov + noise * rng.standard_normal()
            rows.append({"prior": p, "source_did": i, "target": "y", "n": 200, "d": 5,
                         "coverage": cov, "nll": nll, "crps": nll, "pit_stat": 0.1})
    return rows


def test_ablation_recovers_planted_link():
    priors = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    rows = _rows(60, priors, rng, slope=1.0, noise=0.5)
    res = ablation_test(rows, priors, Gate2Config(n_boot=500))
    assert res["has_coverage_variation"]
    assert res["fe_corr"] > 0.5
    assert res["ci_lo"] > res["effect_threshold"]
    assert res["passes"]


def test_ablation_null_when_no_link():
    priors = ["a", "b", "c"]
    rng = np.random.default_rng(1)
    rows = _rows(60, priors, rng, slope=0.0, noise=1.0)
    res = ablation_test(rows, priors, Gate2Config(n_boot=500))
    assert not res["passes"]
    assert abs(res["fe_corr"]) < 0.2


def test_ablation_differences_out_task_difficulty():
    """A huge task-intrinsic spread with zero true slope must NOT manufacture a
    link -- the within-task demeaning is exactly what removes it."""
    priors = ["a", "b", "c"]
    rng = np.random.default_rng(2)
    rows = []
    for i in range(80):
        base = rng.normal(0, 50.0)  # enormous between-task difficulty
        for p in priors:
            cov = rng.uniform(0.5, 4.0)
            rows.append({"prior": p, "source_did": i, "target": "y", "n": 200, "d": 5,
                         "coverage": cov, "nll": base + rng.standard_normal(),
                         "crps": 0.0, "pit_stat": 0.0})
    res = ablation_test(rows, priors, Gate2Config(n_boot=500))
    assert not res["passes"]


def test_verdict_routing():
    good_var = {"passes": True, "frac_outside": 0.3, "min_frac_outside": 0.15,
                "median_ratio": 1.5, "min_median_ratio": 1.25}
    bad_var = {**good_var, "passes": False}

    # a well-powered LINK: CI lower bound clears the bar
    link = {"has_coverage_variation": True, "enough_tasks": True, "passes": True,
            "ruled_out_effect": False, "coverage_spread": 1.0, "fe_corr": 0.4,
            "ci_lo": 0.2, "ci_hi": 0.6, "ci_half_width": 0.2, "n_tasks": 60,
            "min_ablation_tasks": 12, "effect_threshold": 0.15}
    # a true NULL: whole CI sits below the bar (effect ruled out)
    no_link = {**link, "passes": False, "ruled_out_effect": True, "fe_corr": 0.0,
               "ci_lo": -0.08, "ci_hi": 0.08, "ci_half_width": 0.08}
    # underpowered: CI straddles the bar -- neither link nor null
    wide = {**link, "passes": False, "ruled_out_effect": False, "fe_corr": 0.05,
            "ci_lo": -0.5, "ci_hi": 0.7, "ci_half_width": 0.6}
    # too few tasks: refuse a decisive call
    few = {**link, "passes": False, "ruled_out_effect": False, "enough_tasks": False, "n_tasks": 7}
    flat = {**link, "has_coverage_variation": False, "passes": False, "coverage_spread": 1e-9}

    assert gate2_verdict(bad_var, link)["verdict"] == DEAD
    assert gate2_verdict(good_var, flat)["verdict"] == INCONCLUSIVE
    assert gate2_verdict(good_var, link)["verdict"] == LINK
    assert gate2_verdict(good_var, no_link)["verdict"] == NO_LINK
    assert gate2_verdict(good_var, wide)["verdict"] == INCONCLUSIVE_POWER
    assert gate2_verdict(good_var, few)["verdict"] == INCONCLUSIVE_POWER
