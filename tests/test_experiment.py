"""Harness smoke: run_sweep / run_null produce the expected tidy schema, the
lambda and K sub-sweeps are present, and threshold suggestion is well-formed.
Tiny config -- correctness of the schema, not the science."""

import pytest
from ebpfn.config import DistanceConfig
from ebpfn.config import ExperimentConfig
from ebpfn.config import MMDConfig
from ebpfn.config import ModelConfig
from ebpfn.config import SweepConfig
from ebpfn.experiment import run_null
from ebpfn.experiment import run_sweep
from ebpfn.experiment import suggest_thresholds
from ebpfn.experiment import summarize


def _tiny_cfg(construction="A"):
    return ExperimentConfig(
        sweep=SweepConfig(
            construction=construction,
            values=(0.25, 1.0),
            n_seeds=2,
            n_tasks_per_prior=4,
            cloud_n_rows=300,
            n_calib_tasks=1,
            calib_n_train=800,
            calib_n_test=800,
        ),
        distance=DistanceConfig(n_proj=40),
        mmd=MMDConfig(n_cells=8, n_cells_grid=(4, 8)),
        model=ModelConfig(catboost_iterations=60),
    )


def test_run_sweep_schema():
    cfg = _tiny_cfg()
    frames = run_sweep(cfg)
    assert set(frames) == {"sotdd", "mmd", "calib"}
    # lambda sub-sweep present in sotdd, K sub-sweep present in mmd.
    assert sorted(frames["sotdd"]["lam"].unique().to_list()) == list(cfg.distance.lam_grid)
    assert sorted(frames["mmd"]["n_cells"].unique().to_list()) == list(cfg.mmd.n_cells_grid)
    # one calib row per (value, seed).
    assert frames["calib"].height == len(cfg.sweep.values) * cfg.sweep.n_seeds
    for col in ("nll_gap", "crps_gap"):
        assert col in frames["calib"].columns


def test_summarize_schema():
    cfg = _tiny_cfg()
    summary = summarize(run_sweep(cfg), cfg)
    assert summary.height == len(cfg.sweep.values)
    for col in ("otdd_covered", "cond_mmd_mean", "nll_gap", "nll_gap_wilcoxon_p"):
        assert col in summary.columns


def test_run_null_and_thresholds():
    cfg = _tiny_cfg()
    null_frames = run_null(cfg)
    assert set(null_frames) == {"mmd_null", "calib_null"}
    th = suggest_thresholds(null_frames, cfg)
    for key in ("T_cond_mean", "T_cond_max", "T_cal_nll", "T_cal_crps"):
        assert key in th and th[key] == pytest.approx(th[key])  # finite, not nan
