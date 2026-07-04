"""ebpfn -- Gate-0: conditional coverage beyond joint dataset distance.

Library surface matching the spec §6 signatures. The experiment harness lives
under benchmarks/.
"""

from ebpfn.calibration import calibration_report
from ebpfn.config import CalibConfig
from ebpfn.config import DataConfig
from ebpfn.config import DistanceConfig
from ebpfn.config import ExperimentConfig
from ebpfn.config import MMDConfig
from ebpfn.config import ModelConfig
from ebpfn.config import Prior
from ebpfn.config import SweepConfig
from ebpfn.distance import cloud_recall
from ebpfn.distance import exact_otdd
from ebpfn.distance import inside_band
from ebpfn.distance import make_sotdd_fn
from ebpfn.distance import null_band
from ebpfn.distance import recall_to_cloud
from ebpfn.distance import s_otdd
from ebpfn.distance import standardize_per_task
from ebpfn.experiment import run_null
from ebpfn.experiment import run_sweep
from ebpfn.experiment import suggest_thresholds
from ebpfn.experiment import summarize
from ebpfn.mmd import CellPartition
from ebpfn.mmd import aggregate
from ebpfn.mmd import per_cell_mmd
from ebpfn.plotting import make_sweep_figure
from ebpfn.priors import Dataset
from ebpfn.priors import f_mean
from ebpfn.priors import sample_cloud
from ebpfn.priors import sample_task
from ebpfn.regressor import GaussianCatBoost
from ebpfn.regressor import ProbModel
from ebpfn.regressor import QuantileGBM
from ebpfn.regressor import train_prob_regressor
from ebpfn.results import save_run

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
