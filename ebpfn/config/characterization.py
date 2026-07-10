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

    @model_validator(mode="after")
    def validate_values(self) -> "CharacterizationStudyModeConfig":
        if not self.name:
            raise ValueError("study mode name must be nonempty")
        return self


class CharacterizationStudyConfig(StrictConfigModel):
    mode: CharacterizationStudyModeConfig
    dataset: str
    output_root: str
    repeats: int
    max_rows: int
    max_features: int
    # Ambient feature count for synthetic mechanism tasks. Decoupled from max_features because the
    # hand-built DGP places signal in a fixed 1-4 features and pads the rest with noise: at the
    # real-data applicability width (100) a sparse mechanism is buried under noise columns and reads
    # as unrecoverable. None falls back to max_features (used by the p-complexity cost sweep).
    synthetic_max_features: int | None = None
    probe_score_fraction: float = 0.25
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
        if not self.dataset or not self.output_root or not self.decision_owner or not self.decision_date:
            raise ValueError("dataset, output root, and decision metadata must be nonempty")
        if self.repeats < 1:
            raise ValueError("study repeats must be positive")
        if self.max_rows < 4 or self.max_features < 4:
            raise ValueError("study row budget and feature count must both be at least four")
        if self.synthetic_max_features is not None and self.synthetic_max_features < 4:
            raise ValueError("synthetic_max_features must be at least four (the 'mixed' mechanism uses x0..x3)")
        if not 0.0 < self.probe_score_fraction < 1.0:
            raise ValueError("probe_score_fraction must be in (0, 1)")
        if self.applicability_max_rows < 2 or not 2 <= self.applicability_max_features <= 100:
            raise ValueError(
                "study applicability bounds must cover at least two rows/features and at most 100 features"
            )
        if not self.ridge_candidates or any(value <= 0.0 for value in self.ridge_candidates):
            raise ValueError("ridge candidates must be nonempty and positive")
        if tuple(sorted(set(self.ridge_candidates))) != self.ridge_candidates:
            raise ValueError("ridge candidates must be unique and increasing")
        return self

    @property
    def n_probe_score(self) -> int:
        score_rows = round(self.max_rows * self.probe_score_fraction)
        return min(max(score_rows, 2), self.max_rows - 2)

    @property
    def n_probe_fit(self) -> int:
        return self.max_rows - self.n_probe_score
