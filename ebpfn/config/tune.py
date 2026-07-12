"""Strict configuration for the simulator-only objective evaluator and search.

These configs are the sole run state hashed into tuning cache keys. They never
carry PFN settings: candidate evaluation is likelihood-free by construction.
"""

from typing import Literal

from pydantic import field_validator, model_validator

from ebpfn.config.base import StrictConfigModel
from ebpfn.config.characterization import CharacterizationConfig
from ebpfn.config.prior import HyperPriorConfig

# Provisional active hyperprior coordinates. Kept in sync with
# ``ebpfn.priors.vectorize.DEFAULT_ACTIVE`` (a test asserts equality); duplicated
# here to keep the config package a dependency leaf below ``ebpfn.priors``.
DEFAULT_ACTIVE_COORDINATES: tuple[str, ...] = (
    "w_scm",
    "w_bnn",
    "w_tree",
    "corr_strength_mean",
    "log_snr_mean",
    "heteroskedastic_rate",
    "heavy_tail_rate",
    "scm_target_indegree_mean",
    "bnn_weight_scale",
    "compositional_active_fraction_mean",
)


class CompareConfig(StrictConfigModel):
    """Block-balanced distance geometry and objective term policy.

    100% coordinate validity is an invariant, not a toggle: the objectives always
    reject a cloud member (or real vector) that is not fully valid. The ``qc_*``
    fractions are reported diagnostics only and never admit a partial vector.
    """

    version: str = "compare-1"
    block_weights: Literal["uniform"] = "uniform"
    directed_k_floor: int = 5
    directed_k_fraction: float = 0.01
    energy_include_diagonal: Literal[True] = True
    energy_pair_sample: int | None = None  # None = exact pairwise V-statistic
    qc_within_block: float = 0.75  # diagnostic only; never admits a partial vector
    qc_overall: float = 0.90

    @model_validator(mode="after")
    def validate_values(self) -> "CompareConfig":
        if not self.version:
            raise ValueError("compare version must be nonempty")
        if self.directed_k_floor < 1:
            raise ValueError("directed_k_floor must be at least one")
        if not 0.0 < self.directed_k_fraction <= 1.0:
            raise ValueError("directed_k_fraction must be in (0, 1]")
        if self.energy_pair_sample is not None and self.energy_pair_sample < 1:
            raise ValueError("energy_pair_sample must be positive when set")
        for fraction in (self.qc_within_block, self.qc_overall):
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("quality-control fractions must be in [0, 1]")
        return self


class CloudConfig(StrictConfigModel):
    """Matched synthetic cloud drawn per real task at each evaluation.

    ``on_failure`` is the D1 hook for synthetic generation/characterization
    failures: ``raise`` propagates (strict, the default) while ``exclude`` drops
    the failed member and counts it. Members are never silently resampled.
    """

    n_members: int = 64
    on_failure: Literal["raise", "exclude"] = "raise"

    @model_validator(mode="after")
    def validate_values(self) -> "CloudConfig":
        if self.n_members < 2:
            raise ValueError("cloud must have at least two members")
        return self


class SearchConfig(StrictConfigModel):
    """Multifidelity Sobol screen, population optimizer, and selection rerank."""

    sobol_candidates: int = 64
    retain_strong: int = 8
    retain_diverse: int = 4
    # Enabled only after the Step 4 pilots show evaluator signal above noise.
    optimizer: Literal["differential_evolution", "none"] = "none"
    de_maxiter: int = 20
    de_popsize: int = 15
    de_fidelity: Literal["min", "full"] = "min"
    selection_panel_size: int = 3
    single_task_regularization: Literal["none", "prior_distance", "closest_to_baseline"] = "none"  # D3 hook
    prior_distance_penalty: float | None = None
    competitive_tolerance: float | None = None

    @model_validator(mode="after")
    def validate_values(self) -> "SearchConfig":
        if self.sobol_candidates < 1 or self.retain_strong < 1 or self.retain_diverse < 0:
            raise ValueError("candidate retention counts must be positive (diverse may be zero)")
        if self.de_maxiter < 1 or self.de_popsize < 1:
            raise ValueError("differential-evolution schedule values must be positive")
        if self.selection_panel_size < 1:
            raise ValueError("selection panel size must be at least one")
        if self.single_task_regularization == "prior_distance":
            if self.prior_distance_penalty is None or self.prior_distance_penalty <= 0.0:
                raise ValueError("prior_distance regularization requires a positive penalty")
        if self.single_task_regularization == "closest_to_baseline":
            if self.competitive_tolerance is None or self.competitive_tolerance <= 0.0:
                raise ValueError("closest_to_baseline regularization requires a positive competitive tolerance")
        return self


class CacheConfig(StrictConfigModel):
    """Run-local content-addressed evaluation cache."""

    enabled: bool = True
    root: str = ".cache/tuning"
    cache_version: str = "tuning-cache-3"

    @model_validator(mode="after")
    def validate_values(self) -> "CacheConfig":
        if not self.root or not self.cache_version:
            raise ValueError("cache root and version must be nonempty")
        return self


class TuningConfig(StrictConfigModel):
    """Complete resolved run state for a simulator-only tuning evaluation."""

    seed: int = 0
    objective: Literal["directed", "energy"] = "energy"
    active: tuple[str, ...] = DEFAULT_ACTIVE_COORDINATES
    characterization: CharacterizationConfig = CharacterizationConfig()
    prior: HyperPriorConfig = HyperPriorConfig()
    compare: CompareConfig = CompareConfig()
    cloud: CloudConfig = CloudConfig()
    search: SearchConfig = SearchConfig()
    cache: CacheConfig = CacheConfig()

    @field_validator("active", mode="before")
    @classmethod
    def freeze_active(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "TuningConfig":
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not self.active:
            raise ValueError("at least one active coordinate is required")
        if len(set(self.active)) != len(self.active):
            raise ValueError("active coordinates must be unique")
        if (
            self.objective == "energy"
            and self.compare.energy_pair_sample is not None
            and self.cloud.on_failure == "exclude"
        ):
            # Excluding members shifts indices, which invalidates a fixed common
            # energy pair sample keyed on member index.
            raise ValueError("energy_pair_sample is incompatible with cloud on_failure='exclude'")
        return self


class TuningStudyModeConfig(StrictConfigModel):
    """Scale and artifact location for the Step 4 recovery matrix."""

    name: Literal["fast", "audit", "sweep"]
    output_dir: str
    repeats: int
    n_probe_fit: int
    n_probe_score: int
    n_features: int
    n_real_tasks: int
    cloud_sizes: tuple[int, ...]
    regularization_policies: tuple[Literal["none", "prior_distance", "closest_to_baseline"], ...]

    @field_validator("cloud_sizes", mode="before")
    @classmethod
    def freeze_cloud_sizes(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("regularization_policies", mode="before")
    @classmethod
    def freeze_regularization_policies(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "TuningStudyModeConfig":
        if not self.output_dir or self.repeats < 1 or self.n_real_tasks < 1:
            raise ValueError("study output and repeat/task counts must be positive")
        if min(self.n_probe_fit, self.n_probe_score) < 2 or self.n_features < 2:
            raise ValueError("study partitions and feature count must be at least two")
        if not self.cloud_sizes or any(size < 2 for size in self.cloud_sizes):
            raise ValueError("cloud-size grid must contain values of at least two")
        if tuple(sorted(set(self.cloud_sizes))) != self.cloud_sizes:
            raise ValueError("cloud-size grid must be unique and increasing")
        if not self.regularization_policies or len(set(self.regularization_policies)) != len(
            self.regularization_policies
        ):
            raise ValueError("regularization policies must be nonempty and unique")
        if self.regularization_policies != ("none",) and self.n_real_tasks != 1:
            raise ValueError("single-task regularization comparisons require one real task")
        return self


class TuningStudyConfig(StrictConfigModel):
    """Step 4 planted/null recovery and search-protocol study."""

    mode: TuningStudyModeConfig
    tuning: TuningConfig = TuningConfig()
    # Each planted scenario shifts one active knob by planted_unit_shift in vectorized [0,1] space,
    # so perturbations (and parameter_error) are comparable across knobs. log_snr_mean is kept as one
    # knob for a direct comparison against the structural levers. Every entry must be in tuning.active.
    planted_knobs: tuple[str, ...] = (
        "log_snr_mean",
        "heteroskedastic_rate",
        "compositional_active_fraction_mean",
        "scm_target_indegree_mean",
        "corr_strength_mean",
    )
    planted_unit_shift: float = 0.25
    prior_distance_penalty: float = 0.08
    competitive_tolerance: float = 0.02
    decision_owner: str
    decision_date: str
    multiresolution_decision: str = "pending"
    synthetic_failure_decision: Literal["pending", "raise", "exclude"] = "pending"
    single_task_regularization_decision: Literal["pending", "none", "prior_distance", "closest_to_baseline"] = "pending"

    @field_validator("planted_knobs", mode="before")
    @classmethod
    def freeze_planted_knobs(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "TuningStudyConfig":
        if (
            not 0.0 < self.planted_unit_shift < 1.0
            or self.prior_distance_penalty <= 0.0
            or self.competitive_tolerance <= 0.0
        ):
            raise ValueError("planted_unit_shift must be in (0, 1) and regularization thresholds positive")
        if not self.planted_knobs or len(set(self.planted_knobs)) != len(self.planted_knobs):
            raise ValueError("planted_knobs must be nonempty and unique")
        unknown = [knob for knob in self.planted_knobs if knob not in self.tuning.active]
        if unknown:
            raise ValueError(f"planted_knobs must be a subset of tuning.active; unknown: {unknown}")
        if not self.decision_owner or not self.decision_date or not self.multiresolution_decision:
            raise ValueError("decision metadata must be nonempty")
        if (
            self.synthetic_failure_decision != "pending"
            and self.synthetic_failure_decision != self.tuning.cloud.on_failure
        ):
            raise ValueError("synthetic-failure decision must match the resulting tuning config")
        if (
            self.single_task_regularization_decision != "pending"
            and self.single_task_regularization_decision != self.tuning.search.single_task_regularization
        ):
            raise ValueError("regularization decision must match the resulting tuning config")
        return self
