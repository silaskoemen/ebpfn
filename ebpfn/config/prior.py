"""Strict configuration for the hierarchical synthetic prior and its p-audit."""

from typing import Literal

from pydantic import field_validator
from pydantic import model_validator

from ebpfn.config.base import StrictConfigModel


class ScmRouteConfig(StrictConfigModel):
    target_indegree_mean: float = 2.0
    n_hidden: int = 4
    max_parents: int = 4
    weight_scale: float = 1.0
    nonlinear_prob: float = 0.5

    @model_validator(mode="after")
    def validate_values(self) -> "ScmRouteConfig":
        if self.target_indegree_mean <= 0.0 or self.weight_scale <= 0.0:
            raise ValueError("scm indegree and weight scale must be positive")
        if self.n_hidden < 1 or self.max_parents < 1:
            raise ValueError("scm hidden and max parents must be at least one")
        if not 0.0 <= self.nonlinear_prob <= 1.0:
            raise ValueError("scm nonlinear_prob must be a probability")
        return self


class BnnRouteConfig(StrictConfigModel):
    n_layers: int = 2
    hidden: int = 16
    weight_scale: float = 1.0
    nonlinear_prob: float = 0.75

    @model_validator(mode="after")
    def validate_values(self) -> "BnnRouteConfig":
        if self.n_layers < 1 or self.hidden < 1:
            raise ValueError("bnn layers and hidden width must be at least one")
        if self.weight_scale <= 0.0:
            raise ValueError("bnn weight scale must be positive")
        if not 0.0 <= self.nonlinear_prob <= 1.0:
            raise ValueError("bnn nonlinear_prob must be a probability")
        return self


class TreeRouteConfig(StrictConfigModel):
    n_layers_max: int = 2
    hidden_dim_max: int = 8
    max_depth_lambda: float = 1.0
    n_estimators_lambda: float = 1.0

    @model_validator(mode="after")
    def validate_values(self) -> "TreeRouteConfig":
        if self.n_layers_max < 1 or self.hidden_dim_max < 1:
            raise ValueError("tree layers and hidden dim must be at least one")
        if self.max_depth_lambda <= 0.0 or self.n_estimators_lambda <= 0.0:
            raise ValueError("tree exponential rates must be positive")
        return self


class CompositionalRouteConfig(StrictConfigModel):
    linear_weight: float = 1.0
    threshold_weight: float = 1.0
    interaction_weight: float = 1.0
    active_fraction_mean: float = 0.5

    @model_validator(mode="after")
    def validate_values(self) -> "CompositionalRouteConfig":
        weights = (self.linear_weight, self.threshold_weight, self.interaction_weight)
        if any(w < 0.0 for w in weights) or sum(weights) <= 0.0:
            raise ValueError("compositional mechanism weights must be nonnegative with positive sum")
        if not 0.0 < self.active_fraction_mean <= 1.0:
            raise ValueError("compositional active_fraction_mean must be in (0, 1]")
        return self


class HyperPriorConfig(StrictConfigModel):
    generator_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    corr_strength_mean: float = 0.3
    log_snr_mean: float = 0.7
    heteroskedastic_rate: float = 0.2
    heavy_tail_rate: float = 0.2
    snr_dispersion: float = 0.5
    corr_dispersion: float = 0.15
    scm: ScmRouteConfig = ScmRouteConfig()
    bnn: BnnRouteConfig = BnnRouteConfig()
    tree: TreeRouteConfig = TreeRouteConfig()
    compositional: CompositionalRouteConfig = CompositionalRouteConfig()

    @field_validator("generator_weights", mode="before")
    @classmethod
    def freeze_generator_weights(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "HyperPriorConfig":
        if len(self.generator_weights) != 4:
            raise ValueError("generator_weights must have one weight per route (scm, bnn, tree, compositional)")
        if any(w < 0.0 for w in self.generator_weights) or abs(sum(self.generator_weights) - 1.0) > 1e-6:
            raise ValueError("generator_weights must be nonnegative and sum to one")
        for rate in (self.heteroskedastic_rate, self.heavy_tail_rate, self.corr_strength_mean):
            if not 0.0 <= rate <= 1.0:
                raise ValueError("rates and correlation mean must be fractions")
        if self.snr_dispersion <= 0.0 or self.corr_dispersion <= 0.0:
            raise ValueError("dispersions must be positive")
        return self


class ShapeJitterConfig(StrictConfigModel):
    sigma_n: float = 0.4
    sigma_p: float = 0.2
    n_min: int = 32
    n_max: int = 4096
    p_min: int = 1
    p_max: int = 100

    @model_validator(mode="after")
    def validate_values(self) -> "ShapeJitterConfig":
        if self.sigma_n <= 0.0 or self.sigma_p <= 0.0:
            raise ValueError("shape jitter widths must be positive")
        if self.sigma_p >= self.sigma_n:
            raise ValueError("sigma_p must be smaller than sigma_n")
        if not 2 <= self.n_min <= self.n_max:
            raise ValueError("row bounds must satisfy 2 <= n_min <= n_max")
        if not 1 <= self.p_min <= self.p_max <= 100:
            raise ValueError("feature bounds must satisfy 1 <= p_min <= p_max <= 100")
        return self


class PriorStudyModeConfig(StrictConfigModel):
    name: Literal["fast", "audit"]
    output_dir: str
    feature_grid: tuple[int, ...]
    n_probe_fit: int
    n_probe_score: int
    n_tasks: int

    @field_validator("feature_grid", mode="before")
    @classmethod
    def freeze_feature_grid(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "PriorStudyModeConfig":
        if not self.output_dir or self.n_tasks < 1:
            raise ValueError("study output must be nonempty and task count positive")
        if min(self.n_probe_fit, self.n_probe_score) < 2:
            raise ValueError("study partitions must be at least two")
        if not self.feature_grid or any(not 1 <= p <= 100 for p in self.feature_grid):
            raise ValueError("feature grid must be nonempty within 1..100")
        if tuple(sorted(set(self.feature_grid))) != self.feature_grid:
            raise ValueError("feature grid must be unique and increasing")
        return self


class PriorStudyConfig(StrictConfigModel):
    mode: PriorStudyModeConfig
    prior: HyperPriorConfig = HyperPriorConfig()
    jitter: ShapeJitterConfig = ShapeJitterConfig()
    seed: int = 0
    decision_owner: str
    decision_date: str

    @model_validator(mode="after")
    def validate_values(self) -> "PriorStudyConfig":
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not self.decision_owner or not self.decision_date:
            raise ValueError("decision owner and date must be nonempty")
        return self
