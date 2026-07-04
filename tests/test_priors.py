"""The falsifiable heart of Construction A: the cross-slice swap keeps the
marginal P(Y) (essentially) invariant while changing Y|X inside the bands."""

import numpy as np
from ebpfn.config import DataConfig
from ebpfn.config import Prior
from ebpfn.priors import sample_task


def _bands(D, s, w):
    x2 = D.X[:, 1]
    in_a = (x2 >= s / 2) & (x2 <= s / 2 + w)
    in_b = (x2 <= -s / 2) & (x2 >= -s / 2 - w)
    return in_a, in_b


def test_marginal_py_invariant_under_swap():
    # The swap preserves P(Y) only for a FIXED task parameterization (same
    # beta, gamma, delta, w). Seed real/decoy generators identically so X, the
    # conditional mean, and the noise draws match; only the role-dependent sigma
    # assignment differs. Then bands A and B (mirror images, equal mass, identical
    # f-distribution) make the pooled Y distributionally equal.
    dc = DataConfig(band_width=0.5)
    s = 0.5
    real = Prior("A", "real", s, dc)
    decoy = Prior("A", "decoy", s, dc)

    from scipy.stats import ks_2samp

    stats = []
    for seed in range(8):
        Dr = sample_task(real, np.random.default_rng(seed), n=20000)
        Dd = sample_task(decoy, np.random.default_rng(seed), n=20000)
        stats.append(ks_2samp(Dr.Y, Dd.Y).statistic)
    # Should sit at the equal-distributions null level (KS ~ 0.01-0.02 for n=20000).
    assert np.mean(stats) < 0.02, f"marginal P(Y) drifted under swap: KS={np.mean(stats):.4f}"


def test_conditional_differs_inside_bands():
    # Inside band A, real uses sigma_hi and decoy sigma_lo -> Var(Y|A) differs.
    dc = DataConfig(band_width=0.5, sigma_hi=2.0, sigma_lo=0.5)
    s, w = 0.5, 0.5
    rng = np.random.default_rng(0)
    Dr = sample_task(Prior("A", "real", s, dc), rng, n=50000)
    Dd = sample_task(Prior("A", "decoy", s, dc), rng, n=50000)
    in_a_r, _ = _bands(Dr, s, w)
    in_a_d, _ = _bands(Dd, s, w)
    var_real_a = Dr.Y[in_a_r].var()
    var_decoy_a = Dd.Y[in_a_d].var()
    assert var_real_a > 2.0 * var_decoy_a, (var_real_a, var_decoy_a)


def test_mean_depends_on_x1_only():
    dc = DataConfig(sigma0=0.0, sigma_hi=0.0, sigma_lo=0.0)
    rng = np.random.default_rng(1)
    D = sample_task(Prior("A", "real", 0.5, dc), rng, n=5000)
    # With zero noise, Y is exactly f(x1); regressing residual on x2 should be flat.
    # Sort by x2 and check Y has no trend with x2 beyond what x1 explains: corr(Y, x2) ~ 0.
    assert abs(np.corrcoef(D.Y, D.X[:, 1])[0, 1]) < 0.05
