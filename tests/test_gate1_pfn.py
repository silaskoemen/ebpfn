"""PFN regressor: bar-distribution correctness (the delicate piece) + an
end-to-end train/predict plumbing smoke. plans/gate1_revised.md §3.2."""

from __future__ import annotations

import numpy as np
import torch
from ebpfn.calibration import calibration_report
from ebpfn.gate1 import PriorConfig
from ebpfn.gate1 import build_prior
from ebpfn.gate1.config import PFNConfig
from ebpfn.gate1.pfn import BarDistribution
from ebpfn.gate1.pfn import build_model
from ebpfn.gate1.pfn import normal_borders
from ebpfn.gate1.pfn import train_pfn
from ebpfn.priors import Dataset


def _bar_and_logits(rows=6, seed=0):
    torch.manual_seed(seed)
    bar = BarDistribution(normal_borders(48))
    return bar, torch.randn(rows, bar.num_bins)


def test_bar_cdf_endpoints_and_monotone():
    bar, logits = _bar_and_logits()
    ys = torch.linspace(-8, 8, 300)
    cdf = bar.cdf(logits[0:1].repeat(300, 1), ys)
    assert cdf[0] < 1e-3 and cdf[-1] > 1 - 1e-3
    assert bool((cdf[1:] >= cdf[:-1] - 1e-6).all())


def test_bar_pdf_integrates_to_one():
    bar, logits = _bar_and_logits()
    yg = torch.linspace(-15, 15, 6001)
    pdf = torch.exp(-bar.nll(logits[0:1].repeat(6001, 1), yg))
    assert abs(torch.trapz(pdf, yg).item() - 1.0) < 0.02


def test_bar_icdf_inverts_cdf():
    bar, logits = _bar_and_logits()
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        back = bar.cdf(logits, bar.icdf(logits, q))
        assert torch.allclose(back, torch.full_like(back, q), atol=1e-3)


def test_bar_nll_matches_cdf_derivative():
    bar, logits = _bar_and_logits()
    y0, h = torch.tensor([0.3]), 1e-3
    num = (bar.cdf(logits[0:1], y0 + h) - bar.cdf(logits[0:1], y0 - h)) / (2 * h)
    ana = torch.exp(-bar.nll(logits[0:1], y0))
    assert abs(num.item() - ana.item()) < 1e-3


def test_model_forward_shape():
    cfg = PFNConfig(num_bins=32, embedding_size=32, num_attention_heads=2, mlp_hidden_size=64, num_layers=1)
    model = build_model(cfg)
    B, n, d, split = 2, 20, 4, 12
    x = torch.randn(B, n, d)
    y = torch.randn(B, split)
    out = model((x, y), train_test_split_index=split)
    assert out.shape == (B, n - split, cfg.num_bins)


def test_train_and_predict_smoke():
    """Tiny train -> in-context predict -> calibration_report returns finite,
    in-range scores. Plumbing only (too few steps to assert calibration)."""
    prior = build_prior(PriorConfig())
    cfg = PFNConfig(
        steps=20,
        batch_size=4,
        num_bins=32,
        embedding_size=32,
        num_attention_heads=2,
        mlp_hidden_size=64,
        num_layers=1,
        n_rows_min=64,
        n_rows_max=96,
        device="cpu",
        seed=0,
    )
    reg = train_pfn(prior, cfg, log_every=0)

    D = prior.sample_task(160, 4, np.random.default_rng(1))
    reg.fit(D.X[:100], D.Y[:100])
    pred = reg.predictive(D.X[100:])
    y = D.Y[100:]
    assert np.isfinite(pred.nll(y)).all()
    assert np.isfinite(pred.crps(y)).all()
    pit = pred.cdf(y)
    assert (pit >= 0).all() and (pit <= 1).all()

    rep = calibration_report(reg, Dataset(X=D.X[100:], Y=y))
    assert np.isfinite(rep["nll"]) and np.isfinite(rep["crps"])
    assert all(0.0 <= rep["coverage"][p] <= 1.0 for p in rep["coverage"])
