"""The Gate-1 test statistics, with the false-pass guard front and centre:
when calibration is driven entirely by n,d and coverage only correlates with
calibration *through* n,d, the n,d-partial correlation must null out (CI includes
0) even though the raw correlation is spuriously non-zero. And a genuine
coverage->calibration signal must still be detected. §4."""

from __future__ import annotations

import numpy as np
import pytest
from ebpfn.gate1.config import GateConfig
from ebpfn.gate1.gate import gate1_test
from ebpfn.gate1.gate import partial_spearman
from scipy.stats import spearmanr


def _rows(n, d, coverage, x_only, calib):
    """Build coverage/calib row dicts with unique keys per task."""
    cov_rows, cal_rows = [], []
    for i in range(len(n)):
        key = {"source_did": i, "target": "t"}
        cov_rows.append(
            {**key, "coverage": float(coverage[i]), "x_only_coverage": float(x_only[i]), "n": int(n[i]), "d": int(d[i])}
        )
        cal_rows.append({**key, "n": int(n[i]), "d": int(d[i]), "nll": float(calib[i])})
    return cov_rows, cal_rows


def test_partial_spearman_no_controls_matches_scipy():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=50), rng.normal(size=50)
    assert partial_spearman(x, y, []) == pytest.approx(spearmanr(x, y).statistic, abs=1e-9)


def test_partial_spearman_perfect_monotone():
    x = np.linspace(0, 3, 40)
    assert partial_spearman(x, np.exp(x), []) == pytest.approx(1.0, abs=1e-9)


def test_planted_confound_nulls_out():
    """coverage and calib share an n,d-driven 'hardness' but have no direct link.
    Both keep substantial variation independent of n,d (the identifiable regime --
    near-perfect collinearity with n,d is non-identifiable by construction, and
    correctly yields a wide CI rather than a clean zero)."""
    rng = np.random.default_rng(1)
    M = 150
    n = rng.integers(100, 2000, M)
    d = rng.integers(2, 50, M)
    hardness = -0.01 * n + 0.05 * d
    hardness = (hardness - hardness.mean()) / hardness.std()  # unit-scale shared component
    coverage = hardness + rng.normal(0, 1, M)  # correlates with calib ONLY through n,d
    calib = hardness + rng.normal(0, 1, M)
    x_only = hardness + rng.normal(0, 1, M)
    cov_rows, cal_rows = _rows(n, d, coverage, x_only, calib)
    res = gate1_test(cov_rows, cal_rows, GateConfig(n_boot=400, effect_threshold=0.2))

    assert abs(res["spearman_raw"]) > 0.3  # spuriously correlated through n,d
    assert abs(res["partial_spearman"]) < 0.15  # vanishes once n,d partialled out
    assert not res["ci_excludes_zero"]
    assert not res["passes"]


def test_genuine_signal_detected():
    """calib is driven by a coverage signal that is independent of n,d."""
    rng = np.random.default_rng(2)
    M = 150
    n = rng.integers(100, 2000, M)
    d = rng.integers(2, 50, M)
    hardness = -0.01 * n + 0.05 * d
    signal = rng.normal(0, 1, M)
    coverage = signal + 0.3 * hardness + rng.normal(0, 0.1, M)
    calib = hardness + 1.0 * signal + rng.normal(0, 0.2, M)
    x_only = rng.normal(0, 1, M)
    cov_rows, cal_rows = _rows(n, d, coverage, x_only, calib)
    res = gate1_test(cov_rows, cal_rows, GateConfig(n_boot=400, effect_threshold=0.2))

    assert res["partial_spearman"] > 0.2
    assert res["ci_lo"] > 0
    assert res["ci_excludes_zero"]
    assert res["passes"]


def test_joint_over_xonly_credits_conditional_signal():
    """When calib is driven by joint coverage (not the X-only marginal), the
    §4 secondary -- joint coverage partialled on n,d AND x_only -- is positive."""
    rng = np.random.default_rng(3)
    M = 200
    n = rng.integers(100, 2000, M)
    d = rng.integers(2, 50, M)
    joint = rng.normal(size=M)
    x_only = rng.normal(size=M)
    calib = 1.0 * joint + rng.normal(0, 0.2, M)  # driven by joint, not x_only
    cov_rows, cal_rows = _rows(n, d, joint, x_only, calib)
    res = gate1_test(cov_rows, cal_rows, GateConfig(n_boot=200))
    assert res["joint_over_xonly_partial_spearman"] > 0.3


def test_requires_minimum_tasks():
    cov_rows, cal_rows = _rows(np.arange(3) + 100, np.arange(3) + 2, np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        gate1_test(cov_rows, cal_rows)
