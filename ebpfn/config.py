"""Explicit configuration for Gate-0 (spec §5: every knob in a Config; no magic
numbers at call sites)."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np

_DEFAULT_QUANTILES: tuple[float, ...] = tuple(round(float(q), 4) for q in np.linspace(0.05, 0.95, 19))


@dataclass(frozen=True)
class DataConfig:
    """Shared data-generating knobs for both priors (spec §1).

    X ~ N(0, I_d); conditional mean f(x) = beta*x1 + gamma*sin(delta*x1) depends
    on x1 only. Per-task (beta, gamma, delta) and band width are randomized within
    the ranges below; the *noise shape* is what distinguishes real from decoy.
    """

    d: int = 2  # 2 primary, 5 for the robustness check
    # conditional-mean coefficients, drawn per task
    beta_range: tuple[float, float] = (0.5, 1.5)
    gamma_range: tuple[float, float] = (0.5, 1.5)
    delta_range: tuple[float, float] = (0.5, 2.0)
    # noise widths (Construction A regions, and Construction B baseline)
    sigma0: float = 1.0  # baseline width outside the bands
    sigma_hi: float = 2.0
    sigma_lo: float = 0.5
    # Construction A band geometry (mirror-symmetric about x2 = 0 to keep P(Y) exactly invariant)
    # band_geometry: 'fixed_edge' = bands at x2 in +/-[s/2, s/2+w] (registered run; band mass
    # falls as s grows -> confounds the s axis). 'fixed_mass' = bands defined by a CDF interval
    # of width `band_mass` at a symmetric quantile gap g (=sweep_value); per-band mass is
    # constant in g, so the sweep moves feature-separation alone.
    band_geometry: str = "fixed_edge"
    band_width: float = 0.5  # w (fixed_edge only)
    band_width_jitter: float = 0.0  # per-task uniform jitter on w; keep >=0 (fixed_edge only)
    band_mass: float = 0.15  # m: per-band probability mass (fixed_mass only)
    # rows per task, drawn per task
    n_min: int = 500
    n_max: int = 2000

    def __post_init__(self) -> None:
        if self.band_geometry not in ("fixed_edge", "fixed_mass"):
            raise ValueError(f"band_geometry must be 'fixed_edge' or 'fixed_mass', got {self.band_geometry!r}")
        if self.band_geometry == "fixed_mass" and not (0.0 < self.band_mass < 0.5):
            raise ValueError(f"band_mass must be in (0, 0.5), got {self.band_mass}")


@dataclass(frozen=True)
class Prior:
    """A task prior: a construction + role + a fixed point on the sweep axis.

    construction: 'A' (cross-slice conditional swap) or 'B' (hetero vs homoskedastic).
    role:         'real' or 'decoy'.
    sweep_value:  s (band separation in x2) for A; kappa (hetero strength) for B.
    """

    construction: str
    role: str
    sweep_value: float
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        if self.construction not in ("A", "B"):
            raise ValueError(f"construction must be 'A' or 'B', got {self.construction!r}")
        if self.role not in ("real", "decoy"):
            raise ValueError(f"role must be 'real' or 'decoy', got {self.role!r}")


@dataclass(frozen=True)
class DistanceConfig:
    """s-OTDD / OTDD knobs (spec §3.1). lambda has no silent default."""

    lam: float = 1.0  # primary feature/label weight in the ground cost (headline)
    lam_grid: tuple[float, ...] = (0.5, 1.0, 2.0)  # sensitivity sweep (spec §3.1/§8)
    p: int = 2  # order of the Wasserstein distance
    n_proj: int = 200  # sliced-Wasserstein projections
    standardize: bool = True  # per-task unit-variance X and Y before any distance
    null_alpha: float = 0.05  # 95% null band

    def __post_init__(self) -> None:
        if self.lam not in self.lam_grid:
            raise ValueError(f"primary lam={self.lam} must be in lam_grid={self.lam_grid}")


@dataclass(frozen=True)
class MMDConfig:
    """Conditional-coverage meter knobs (spec §3.2)."""

    n_cells: int = 16  # primary K (headline)
    n_cells_grid: tuple[int, ...] = (8, 16, 32)  # granularity sweep (spec §3.2/§8)
    method: str = "kmeans"  # label-agnostic X-only partition
    bandwidth: str = "median"  # RBF bandwidth rule on 1-D Y
    min_per_cell: int = 20  # cells below this are skipped
    max_per_cell: int = 500  # subsample cap for the quadratic MMD estimator

    def __post_init__(self) -> None:
        if self.n_cells not in self.n_cells_grid:
            raise ValueError(f"primary n_cells={self.n_cells} must be in n_cells_grid={self.n_cells_grid}")


@dataclass(frozen=True)
class ModelConfig:
    """Probabilistic regressor knobs (spec §3.3).

    kind: 'catboost_gauss' (CatBoost RMSEWithUncertainty -> exact Gaussian NLL,
    the primary head) or 'qgbm' (lightgbm quantile grid -> second opinion).
    """

    kind: str = "catboost_gauss"
    # qgbm (lightgbm quantile grid)
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    # catboost_gauss
    catboost_iterations: int = 500
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 6


@dataclass(frozen=True)
class CalibConfig:
    """Calibration report knobs (spec §3.3)."""

    coverage_levels: tuple[float, ...] = (0.5, 0.8, 0.9)


@dataclass(frozen=True)
class SweepConfig:
    """The pre-registered sweep grid (spec §4/§5). Fix before the first non-pilot run."""

    construction: str = "A"
    values: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)  # s (A) or kappa (B)
    n_seeds: int = 10
    # s-OTDD clouds (independent tasks; prior-level coverage / null band)
    n_tasks_per_prior: int = 50
    cloud_n_rows: int = 600
    # calibration: shared-f triples per sweep point, averaged
    n_calib_tasks: int = 3
    calib_n_train: int = 2000
    calib_n_test: int = 2000


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    mmd: MMDConfig = field(default_factory=MMDConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    seed: int = 0
