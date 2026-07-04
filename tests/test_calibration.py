"""Both probabilistic heads should show a positive NLL calibration gap: a model
trained on the decoy prior is worse-calibrated on real test data than one trained
on the real prior, even though the conditional means match by construction."""

import numpy as np
import pytest
from ebpfn.calibration import calibration_report
from ebpfn.config import DataConfig
from ebpfn.config import ModelConfig
from ebpfn.config import Prior
from ebpfn.priors import sample_task
from ebpfn.regressor import train_prob_regressor


@pytest.mark.parametrize("kind", ["catboost_gauss", "qgbm"])
def test_nll_gap_positive(kind):
    dc = DataConfig(sigma_hi=2.0, sigma_lo=0.5)
    s = 0.25
    real = Prior("A", "real", s, dc)
    decoy = Prior("A", "decoy", s, dc)
    rng = np.random.default_rng(0)

    cfg = ModelConfig(kind=kind, catboost_iterations=150, n_estimators=150)
    m_real = train_prob_regressor(sample_task(real, rng, n=3000), kind, cfg)
    m_decoy = train_prob_regressor(sample_task(decoy, rng, n=3000), kind, cfg)
    D_test = sample_task(real, rng, n=3000)

    gap = calibration_report(m_decoy, D_test)["nll"] - calibration_report(m_real, D_test)["nll"]
    assert gap > 0, f"{kind}: expected positive NLL gap, got {gap:.4f}"


def test_gaussian_predictive_closed_forms():
    # A well-specified Gaussian fit on homoskedastic data should be ~calibrated:
    # PIT close to uniform and central coverage near nominal.
    from ebpfn.regressor import GaussianPredictive

    rng = np.random.default_rng(0)
    y = rng.standard_normal(20000)
    pred = GaussianPredictive(mean=np.zeros_like(y), sigma=np.ones_like(y))
    pit = pred.cdf(y)
    assert abs(pit.mean() - 0.5) < 0.02
    cov80 = np.mean((y >= pred.quantile(0.1)) & (y <= pred.quantile(0.9)))
    assert abs(cov80 - 0.8) < 0.02
