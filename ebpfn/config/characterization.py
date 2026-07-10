"""Strict configuration for fixed-map regression characterization."""

from typing import Literal

from pydantic import field_validator, model_validator

from ebpfn.config.base import StrictConfigModel


class RowBudgetConfig(StrictConfigModel):
    minimum: int = 256
    spacing: Literal["geometric", "sqrt"] = "geometric"
    weight: Literal["uniform", "row_count"] = "uniform"
    feature_view: Literal["frozen", "local"] = "frozen"

    @model_validator(mode="after")
    def validate_values(self) -> "RowBudgetConfig":
        if self.minimum < 2:
            raise ValueError("minimum row budget must be at least two")
        return self


class MapConfig(StrictConfigModel):
    bin_quantiles: tuple[float, ...] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
    max_products: int = 128
    max_conjunctions: int = 128
    max_rff: int = 256
    rff_distance_rows: int = 1024
    conjunction_min_prevalence: float = 0.05
    conjunction_max_prevalence: float = 0.95

    @field_validator("bin_quantiles", mode="before")
    @classmethod
    def freeze_bin_quantiles(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "MapConfig":
        if not self.bin_quantiles or any(not 0.0 < q < 1.0 for q in self.bin_quantiles):
            raise ValueError("bin quantiles must be nonempty and strictly internal")
        if tuple(sorted(set(self.bin_quantiles))) != self.bin_quantiles:
            raise ValueError("bin quantiles must be strictly increasing")
        if min(self.max_products, self.max_conjunctions, self.max_rff) < 0 or self.rff_distance_rows < 2:
            raise ValueError("map limits must be nonnegative and distance rows at least two")
        if not 0.0 <= self.conjunction_min_prevalence < self.conjunction_max_prevalence <= 1.0:
            raise ValueError("invalid conjunction prevalence interval")
        return self


class RidgeConfig(StrictConfigModel):
    lambda_: float = 1e-2
    gain_epsilon: float = 1e-12
    column_tolerance: float = 1e-12

    @model_validator(mode="after")
    def validate_values(self) -> "RidgeConfig":
        if self.lambda_ <= 0.0 or self.gain_epsilon <= 0.0 or self.column_tolerance <= 0.0:
            raise ValueError("ridge configuration values must be positive")
        return self


class CharacterizationConfig(StrictConfigModel):
    version: str = "characterization-1"
    seed: int = 0
    repeat: int = 0
    representation: Literal["raw", "contrast"] = "raw"
    include_observation_coordinates: bool = True
    target_clip: float = 5.0
    target_scale_epsilon: float = 1e-12
    row_budgets: RowBudgetConfig = RowBudgetConfig()
    maps: MapConfig = MapConfig()
    ridge: RidgeConfig = RidgeConfig()

    @model_validator(mode="after")
    def validate_values(self) -> "CharacterizationConfig":
        if not self.version or self.seed < 0 or self.repeat < 0:
            raise ValueError("version must be nonempty and seeds/repeats nonnegative")
        if self.target_clip <= 0.0 or self.target_scale_epsilon <= 0.0:
            raise ValueError("target scaling values must be positive")
        return self


class CharacterizationStudyModeConfig(StrictConfigModel):
    name: Literal["fast", "audit"]
    output_dir: str
    repeats: int
    n_probe_fit: int
    n_probe_score: int
    n_features: int

    @model_validator(mode="after")
    def validate_values(self) -> "CharacterizationStudyModeConfig":
        if not self.output_dir or self.repeats < 1:
            raise ValueError("study output must be nonempty and repeats positive")
        if min(self.n_probe_fit, self.n_probe_score) < 2 or self.n_features < 2:
            raise ValueError("study partitions and feature count must be at least two")
        return self


class CharacterizationStudyConfig(StrictConfigModel):
    mode: CharacterizationStudyModeConfig
    characterization: CharacterizationConfig
    ridge_candidates: tuple[float, ...]
    decision_owner: str
    decision_date: str
    applicability_max_rows: int
    applicability_max_features: int

    @field_validator("ridge_candidates", mode="before")
    @classmethod
    def freeze_ridge_candidates(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "CharacterizationStudyConfig":
        if not self.decision_owner or not self.decision_date:
            raise ValueError("decision owner and date must be nonempty")
        if self.applicability_max_rows < 2 or not 2 <= self.applicability_max_features <= 100:
            raise ValueError(
                "study applicability bounds must cover at least two rows/features and at most 100 features"
            )
        if not self.ridge_candidates or any(value <= 0.0 for value in self.ridge_candidates):
            raise ValueError("ridge candidates must be nonempty and positive")
        if tuple(sorted(set(self.ridge_candidates))) != self.ridge_candidates:
            raise ValueError("ridge candidates must be unique and increasing")
        return self
