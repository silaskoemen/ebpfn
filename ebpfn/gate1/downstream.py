"""Per-task downstream calibration of the trained PFN (plans/gate1_revised.md §3.5).

Each real task is split train/test; the PFN takes the train split in context
(capped to its context regime -- no gradient steps) and the test split is scored
with Gate-0's `calibration_report` (NLL/CRPS/PIT/interval-coverage). The output
table is joined against `coverage.py`'s table in the gate test.
"""
from __future__ import annotations

import numpy as np

from ebpfn.calibration import calibration_report
from ebpfn.config import CalibConfig
from ebpfn.gate1.config import DownstreamConfig
from ebpfn.gate1.corpus import RealTask
from ebpfn.gate1.pfn.regressor import PFNRegressor
from ebpfn.priors import Dataset


def task_calibration(
    reg: PFNRegressor, task: Dataset, cfg: DownstreamConfig, calib_cfg: CalibConfig, rng: np.random.Generator
) -> dict:
    """Fit-in-context on the train split, score calibration on the test split."""
    perm = rng.permutation(task.n)
    n_train = int(round(cfg.train_frac * task.n))
    tr, te = perm[:n_train], perm[n_train:]
    if tr.size > cfg.in_context_cap:
        tr = rng.choice(tr, size=cfg.in_context_cap, replace=False)
    if te.size > cfg.test_cap:
        te = rng.choice(te, size=cfg.test_cap, replace=False)
    reg.fit(task.X[tr], task.Y[tr])
    rep = calibration_report(reg, Dataset(X=task.X[te], Y=task.Y[te]), calib_cfg)
    return {
        "nll": rep["nll"], "crps": rep["crps"], "pit_stat": rep["pit_stat"],
        **{f"cov@{p}": rep["coverage"][p] for p in rep["coverage"]},
    }


def corpus_calibration(
    reg: PFNRegressor, corpus: list[RealTask], cfg: DownstreamConfig,
    calib_cfg: CalibConfig | None = None, rng: np.random.Generator | None = None,
) -> list[dict]:
    """Per-task calibration row for the whole corpus (keyed for the join)."""
    calib_cfg = calib_cfg or CalibConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    rows = []
    for t in corpus:
        cal = task_calibration(reg, t.data, cfg, calib_cfg, rng)
        rows.append({"source_did": t.source_did, "target": t.target, "n": t.n, "d": t.d, **cal})
    return rows
