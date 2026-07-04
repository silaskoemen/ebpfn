"""PFN training loop (plans/gate1_revised.md §3.2/§6.2).

Adapted from nanoTabPFN's `train` (Apache-2.0): same shape, but the loss is the
bar-distribution NLL (not cross-entropy) and batches come from our live
MixturePrior loader. Plain AdamW (no schedulefree dependency). The trained model
is paired *by construction* with the prior it saw -- the exact pairing H1 needs.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ebpfn.gate1.config import PFNConfig
from ebpfn.gate1.pfn.bar import BarDistribution
from ebpfn.gate1.pfn.bar import normal_borders
from ebpfn.gate1.pfn.loader import PriorBatchLoader
from ebpfn.gate1.pfn.model import PFNTransformer
from ebpfn.gate1.pfn.regressor import PFNRegressor
from ebpfn.gate1.prior import MixturePrior


def resolve_device(name: str) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: PFNConfig) -> PFNTransformer:
    return PFNTransformer(
        embedding_size=cfg.embedding_size,
        num_attention_heads=cfg.num_attention_heads,
        mlp_hidden_size=cfg.mlp_hidden_size,
        num_layers=cfg.num_layers,
        num_outputs=cfg.num_bins,
    )


def train_pfn(prior: MixturePrior, cfg: PFNConfig, log_every: int = 100) -> PFNRegressor:
    """Train a regression PFN on `prior`; return an inference-ready PFNRegressor."""
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = build_model(cfg).to(device)
    bar = BarDistribution(normal_borders(cfg.num_bins, cfg.border_eps)).to(device)
    loader = PriorBatchLoader(prior, cfg, device, rng)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    t0 = time.time()
    for step, batch in enumerate(loader):
        split = batch["train_test_split_index"]
        x, y = batch["x"], batch["y"]
        logits = model((x, y[:, :split]), train_test_split_index=split)  # (B, n_test, K)
        targets = y[:, split:]  # (B, n_test), standardized
        loss = bar.nll(logits.reshape(-1, cfg.num_bins), targets.reshape(-1)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad()
        if log_every and (step % log_every == log_every - 1 or step == 0):
            print(f"step {step + 1:5d}/{cfg.steps} | nll {loss.item():7.4f} | {time.time() - t0:6.1f}s")

    return PFNRegressor(model, bar, device)
