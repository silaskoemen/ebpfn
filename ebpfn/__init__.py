"""ebpfn -- Gate-0: conditional coverage beyond joint dataset distance.

Library surface matching the spec §6 signatures. The experiment harness lives
under benchmarks/.
"""
from ebpfn.calibration import calibration_report
from ebpfn.config import (
    CalibConfig,
    DataConfig,
    DistanceConfig,
    ExperimentConfig,
    MMDConfig,
    ModelConfig,
    Prior,
    SweepConfig,
)
from ebpfn.experiment import run_null, run_sweep, suggest_thresholds, summarize
from ebpfn.plotting import make_sweep_figure
from ebpfn.results import save_run
from ebpfn.distance import (
    cloud_recall,
    exact_otdd,
    inside_band,
    make_sotdd_fn,
    null_band,
    recall_to_cloud,
    s_otdd,
    standardize_per_task,
)
from ebpfn.mmd import CellPartition, aggregate, per_cell_mmd
from ebpfn.priors import Dataset, f_mean, sample_cloud, sample_task
from ebpfn.regressor import (
    GaussianCatBoost,
    ProbModel,
    QuantileGBM,
    train_prob_regressor,
)

__all__ = [
    "CalibConfig",
    "CellPartition",
    "DataConfig",
    "Dataset",
    "DistanceConfig",
    "ExperimentConfig",
    "GaussianCatBoost",
    "MMDConfig",
    "ModelConfig",
    "Prior",
    "ProbModel",
    "QuantileGBM",
    "SweepConfig",
    "aggregate",
    "calibration_report",
    "cloud_recall",
    "exact_otdd",
    "f_mean",
    "inside_band",
    "make_sotdd_fn",
    "make_sweep_figure",
    "null_band",
    "per_cell_mmd",
    "recall_to_cloud",
    "run_null",
    "run_sweep",
    "s_otdd",
    "sample_cloud",
    "sample_task",
    "save_run",
    "standardize_per_task",
    "suggest_thresholds",
    "summarize",
    "train_prob_regressor",
]
