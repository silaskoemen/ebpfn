"""Gate-1 (revised) configuration -- explicit knobs, no magic numbers at call
sites (same discipline as Gate-0's config.py)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorConfig:
    """The mixture-over-DGPs prior (plans/gate1_revised.md §3.1).

    First cut: SCM-linear, SCM-MLP, and BNN, mixed by the three weights below
    (a zero weight drops that member). DGP knobs are exposed so the across-
    generator sweep (H2) can vary them; the tree DGP slots in later.
    """

    # mixture weights (relative; normalized; a 0 drops the member)
    scm_linear_weight: float = 1.0
    scm_mlp_weight: float = 1.0
    bnn_weight: float = 1.0
    # shared SCM knobs
    scm_n_hidden: int = 4
    scm_edge_prob: float = 0.5
    scm_max_parents: int = 4
    scm_weight_scale: float = 1.0
    scm_noise_scale: float = 0.3
    scm_mlp_activation: str = "tanh"  # SCM-MLP nonlinearity (SCM-linear uses 'linear')
    # BNN knobs
    bnn_n_layers: int = 2
    bnn_hidden: int = 16
    bnn_activation: str = "tanh"
    bnn_weight_scale: float = 1.0
    bnn_noise_scale: float = 0.3

    def __post_init__(self) -> None:
        weights = (self.scm_linear_weight, self.scm_mlp_weight, self.bnn_weight)
        if any(w < 0 for w in weights):
            raise ValueError(f"mixture weights must be >= 0, got {weights}")
        if sum(weights) <= 0:
            raise ValueError("at least one mixture weight must be positive")


@dataclass(frozen=True)
class DownstreamConfig:
    """Per-task in-context calibration of the trained PFN (§3.5/§5).

    Each real task is split train/test; the train split is the PFN's in-context
    set (capped to its context regime), the test split is scored with
    NLL/CRPS/PIT/interval-coverage. No gradient steps.
    """

    train_frac: float = 0.5
    in_context_cap: int = 256  # cap in-context train rows to the PFN's regime
    test_cap: int = 500
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("need 0 < train_frac < 1")
        if self.in_context_cap < 1 or self.test_cap < 1:
            raise ValueError("caps must be >= 1")


@dataclass(frozen=True)
class GateConfig:
    """The H1 partial-correlation test (§1/§5)."""

    calib_metric: str = "nll"  # calibration error to correlate against coverage
    effect_threshold: float = 0.2  # pre-registered min |partial Spearman| to call a pass
    n_boot: int = 2000
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.calib_metric not in ("nll", "crps", "pit_stat"):
            raise ValueError(f"calib_metric must be nll/crps/pit_stat, got {self.calib_metric!r}")


@dataclass(frozen=True)
class CoverageConfig:
    """Per-task prior-coverage measurement (§3.4).

    Coverage = k-NN-mean s-OTDD distance from a real task to a prior cloud
    sampled at the task's own d (d-matched, so the joint z=[X,sqrt(lam)Y]
    dimensions align). Task and cloud are subsampled to a common row count so the
    sliced distance compares equal-size point clouds (the task's true n is kept
    separately as the confound covariate). The X-only sliced distance is the
    P(X) over-matching diagnostic (§4). Null floor = prior self-recall per d.
    """

    lam: float = 1.0
    lam_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    p: int = 2
    n_proj: int = 200
    standardize: bool = True
    k: int = 5  # k-NN-mean recall
    cloud_n_tasks: int = 50
    cloud_n_rows: int = 600
    null_alpha: float = 0.05
    n_boot: int = 1000

    def __post_init__(self) -> None:
        if self.lam not in self.lam_grid:
            raise ValueError(f"primary lam={self.lam} must be in lam_grid={self.lam_grid}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.cloud_n_tasks < self.k:
            raise ValueError(f"cloud_n_tasks ({self.cloud_n_tasks}) must be >= k ({self.k})")


@dataclass(frozen=True)
class CorpusConfig:
    """Real-task corpus by column rotation over TabArena (§3.3).

    Each source table is rotated: every continuous column becomes a regression
    target in turn. Learnability keeps only targets a GBM predicts above its
    marginal; redundancy drops near-deterministic targets. Tasks are clamped to
    the PFN's (n, d) regime; n, d, schema are recorded for the n,d control.
    """

    suite_id: int = 457  # TabArena-v0.1 OpenML task-suite
    cache_dir: str = "data/raw/openml"
    n_min: int = 200
    n_max: int = 2000
    d_max: int = 50
    target_min_unique: int = 20  # a target needs >= this many distinct values to count as continuous
    learnability_min: float = 0.05  # held-out R^2 must beat the marginal by this
    redundancy_max: float = 0.999  # drop near-deterministic targets above this R^2
    max_tasks_per_dataset: int = 4  # cap rotations per table so no table dominates
    max_datasets: int = 10  # cap source tables (raise for the full corpus)
    test_frac: float = 0.3  # split for the learnability/redundancy probe
    learnability_max_iter: int = 150
    seed: int = 0

    def __post_init__(self) -> None:
        if not (2 <= self.n_min <= self.n_max):
            raise ValueError("need 2 <= n_min <= n_max")
        if self.d_max < 1:
            raise ValueError("d_max must be >= 1")
        if not (0.0 <= self.learnability_min <= self.redundancy_max <= 1.0):
            raise ValueError("need 0 <= learnability_min <= redundancy_max <= 1")


@dataclass(frozen=True)
class PFNConfig:
    """Regression PFN: architecture, bar-distribution head, and training (§3.2).

    Nano-scale by default so one training is hours on MPS (no CUDA here) and the
    across-generator sweep stays tractable. Per-step task geometry (n, d, split)
    is drawn from the ranges below; the model is d-agnostic (per-feature
    embedding + attention between features), so d may vary across steps.
    """

    # architecture
    embedding_size: int = 96
    num_attention_heads: int = 4
    mlp_hidden_size: int = 192
    num_layers: int = 3
    # bar-distribution head
    num_bins: int = 64
    border_eps: float = 1e-3
    # training
    steps: int = 2000
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # per-step task geometry (drawn each step)
    n_rows_min: int = 128
    n_rows_max: int = 256
    d_min: int = 1
    d_max: int = 8
    train_frac_min: float = 0.4  # single_eval_pos = train_frac * n_rows
    train_frac_max: float = 0.8
    device: str = ""  # "" -> auto (cuda > mps > cpu)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_bins < 3:
            raise ValueError(f"num_bins must be >= 3, got {self.num_bins}")
        if not (0.0 < self.train_frac_min <= self.train_frac_max < 1.0):
            raise ValueError("need 0 < train_frac_min <= train_frac_max < 1")
        if not (1 <= self.d_min <= self.d_max):
            raise ValueError("need 1 <= d_min <= d_max")
        if not (2 <= self.n_rows_min <= self.n_rows_max):
            raise ValueError("need 2 <= n_rows_min <= n_rows_max")
