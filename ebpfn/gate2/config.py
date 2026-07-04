"""Gate-2 configuration (plans/gate2.md).

Gate-2 re-specifies the coverage target after Gate-1's joint s-OTDD distance came
out non-discriminating (151-task partial Spearman = 0.083, CI included 0). The
new target is a **conditional-structure descriptor** -- "how a learner sees the
problem" -- and the primary test is an **across-prior fixed-effects ablation**
that differences out task-intrinsic difficulty. Everything here is frozen before
any calibration number is looked at (pre-registration discipline, §4 of the
Gate-1 plan carried forward).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ebpfn.gate1.config import PriorConfig


@dataclass(frozen=True)
class DescriptorConfig:
    """The conditional-structure descriptor (FROZEN before touching calibration).

    For a task, X is z-scored per feature and Y is rank-Gaussian transformed
    (rank -> inverse-normal CDF) -- both affine nuisances removed, and the Y map
    is invariant to *any* monotone reparametrization, not just affine (the Gate-1
    standardization reversal, tightened: it also kills the heavy-tail Y^3 blow-up
    the plain z-score left in). We draw `n_proj` random unit directions u, project
    t = u^T X, and per direction extract affine-invariant features of Y | t. These
    are aggregated across projections by mean + quantiles into a fixed-length,
    dimension-adaptive vector. Two **multivariate** features (effective dimension
    and interaction gain) are appended -- the structure 1D projections cannot see
    (the explicit pushback that the projection-marginal profile is incomplete).
    A second block then estimates operator modes: regularized CCA/CKA between
    raw, polynomial, local-RBF and global-RBF feature maps of X and Hermite moment
    functions of rank-Gaussianized Y.

    Cross-task comparability: every task is measured at a matched row budget `n0`.
    Tasks with n > n0 are evaluated on `n_sub` independent subsamples of size n0
    and the descriptor vectors are averaged; this equalizes the estimator's
    bias/variance across tasks of different n -- the n-confound that flat-lined
    Gate-1, removed at the estimator level rather than just capped.
    """

    n_proj: int = 64
    n_bins: int = 10  # quantile bins for the conditional-mean curve along t
    n0: int = 256  # matched row budget; every task is measured at min(n, n0) rows
    n_sub: int = 4  # matched-n subsamples to average when n > n0 (variance reduction)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)  # cross-projection aggregation (plus mean)
    gbm_max_iter: int = 60  # multivariate interaction/eff-dim probe
    gbm_val_frac: float = 0.3  # held-out split for the GBM/additive R^2 (no in-sample inflation)
    # Operator-spectrum block: regularized CCA/CKA between feature maps of X and
    # Hermite moment functions of rank-Gaussianized Y. This is the cheap
    # finite-feature version of the conditional-operator spectrum discussed in
    # plans/gate2.md: raw + polynomial + local/global RBF feature blocks of X,
    # and the first `y_moments` orthonormal moment modes of Y. Per-moment ridge
    # multiple-correlations are also kept so CKA does not hide mean/scale/tail
    # mismatches inside one aggregate score. Raw-standardized marginal Y tail
    # diagnostics are always included separately before this block.
    y_moments: int = 6
    cca_modes: int = 3
    cca_ridge: float = 1e-3
    n_rff: int = 32
    n_poly: int = 16
    rbf_local_scale: float = 0.5
    rbf_global_scale: float = 2.0
    min_rows: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_proj < 1:
            raise ValueError(f"n_proj must be >= 1, got {self.n_proj}")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        if self.n0 < 8:
            raise ValueError(f"n0 must be >= 8 for a stable descriptor, got {self.n0}")
        if self.n_sub < 1:
            raise ValueError(f"n_sub must be >= 1, got {self.n_sub}")
        if not (0.0 < self.gbm_val_frac < 1.0):
            raise ValueError("need 0 < gbm_val_frac < 1")
        if self.y_moments < 1:
            raise ValueError("y_moments must be >= 1")
        if self.y_moments > 12:
            raise ValueError("y_moments > 12 is numerically unstable for this finite-sample descriptor")
        if self.cca_modes < 1:
            raise ValueError("cca_modes must be >= 1")
        if self.cca_ridge <= 0.0:
            raise ValueError("cca_ridge must be > 0")
        if self.n_rff < 1:
            raise ValueError("n_rff must be >= 1")
        if self.n_poly < 1:
            raise ValueError("n_poly must be >= 1")
        if self.rbf_local_scale <= 0.0 or self.rbf_global_scale <= 0.0:
            raise ValueError("RBF bandwidth scales must be > 0")
        if self.min_rows < 8:
            raise ValueError("min_rows must be >= 8 for stable conditional features")


@dataclass(frozen=True)
class Gate2CoverageConfig:
    """Mahalanobis coverage of a real task vs the prior's descriptor cloud.

    The prior cloud is sampled d-matched to each real task (descriptors are
    dimension-adaptive but conditional structure still depends on d). Coverage is
    the (shrunk-covariance) Mahalanobis distance of the real descriptor to the
    cloud. An independent prior-probe distance distribution is the null band.
    """

    cloud_n_tasks: int = 48  # prior tasks per d in the descriptor cloud
    cloud_n_rows: int = 256
    outside_quantile: float = 0.95  # a real task is "outside" beyond this null quantile
    seed: int = 0

    def __post_init__(self) -> None:
        if self.cloud_n_tasks < 8:
            raise ValueError("cloud_n_tasks must be >= 8 to estimate a covariance")
        if not (0.5 < self.outside_quantile < 1.0):
            raise ValueError("outside_quantile must be in (0.5, 1)")


@dataclass(frozen=True)
class Gate2Config:
    """The decision thresholds for both pre-committed parts (FROZEN).

    Part A (variance go/no-go): does descriptor coverage discriminate real tasks
    from the prior at all? If not, coverage-gating is dead regardless of any
    calibration link -- the exact failure that flat-lined Gate-1, checked here
    *before* calibration. Part B (ablation): the within-task across-prior slope of
    calibration on coverage, cluster-bootstrapped over tasks.
    """

    calib_metric: str = "nll"
    # Part A -- variance go/no-go (pre-registered)
    min_frac_outside: float = 0.15  # >= this fraction of real tasks beyond the null band
    min_median_ratio: float = 1.25  # median(real dist) / null-median must exceed this
    # Part B -- ablation (pre-registered)
    ablation_effect_threshold: float = 0.15  # the FE-correlation bar: lo>thr=link, hi<thr=null
    min_coverage_spread: float = 1e-3  # within-task coverage must actually vary across priors
    min_ablation_tasks: int = 12  # below this, refuse a decisive Part B call (underpowered)
    n_boot: int = 2000
    alpha: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.calib_metric not in ("nll", "crps", "pit_stat"):
            raise ValueError(f"calib_metric must be nll/crps/pit_stat, got {self.calib_metric!r}")


def prior_ladder() -> dict[str, PriorConfig]:
    """K priors spanning a coverage range over real tabular conditional structure.

    Real tabular targets are typically nonlinear, heteroskedastic and
    interaction-rich, so a smooth additive prior should *under*-cover them and a
    richer one should cover better -- the gradient the across-prior ablation
    needs. Ordered narrow -> rich along a single richness axis (linear-only +
    low-noise up to deep relu-MLP + BNN + higher-noise).

    Why six rungs, not three: the ablation estimates a *within-task* slope across
    these priors, so each task contributes (K-1) points after demeaning -- K=3
    gave only 2, the dominant reason the pilot CI was [-0.55, +0.70]. Six rungs
    roughly triple the within-task leverage. The names are stable keys used in the
    result tables; the two anchors (linear_narrow, rich_nonlinear) are unchanged.

    Rung *spacing* here is by generative richness, not by measured coverage. Before
    the confirmatory run, validate that these produce a roughly even coverage
    gradient by measuring each rung's descriptor distance to the real corpus --
    which is calibration-free, so re-spacing on it does NOT break pre-registration
    (never re-space on calibration).
    """
    return {
        "linear_narrow": PriorConfig(  # pure linear, low noise -- under-covers real structure
            scm_linear_weight=1.0, scm_mlp_weight=0.0, bnn_weight=0.0,
            scm_noise_scale=0.15,
        ),
        "linear": PriorConfig(  # pure linear, default noise
            scm_linear_weight=1.0, scm_mlp_weight=0.0, bnn_weight=0.0,
            scm_noise_scale=0.3,
        ),
        "mild_nonlinear": PriorConfig(  # mostly linear with a touch of smooth nonlinearity
            scm_linear_weight=1.0, scm_mlp_weight=0.5, bnn_weight=0.25,
            scm_mlp_activation="tanh",
        ),
        "balanced": PriorConfig(),  # the Gate-1 default mixture (linear + MLP + BNN)
        "nonlinear": PriorConfig(  # no linear member; fully nonlinear, deeper SCM
            scm_linear_weight=0.0, scm_mlp_weight=1.0, bnn_weight=1.0,
            scm_mlp_activation="tanh", scm_n_hidden=6,
        ),
        "rich_nonlinear": PriorConfig(  # deep relu-MLP + larger BNN + higher noise
            scm_linear_weight=0.0, scm_mlp_weight=1.0, bnn_weight=1.5,
            scm_mlp_activation="relu", scm_n_hidden=6, scm_noise_scale=0.5,
            bnn_n_layers=3, bnn_hidden=24, bnn_noise_scale=0.5,
        ),
    }
