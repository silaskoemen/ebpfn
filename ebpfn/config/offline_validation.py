"""Strict configuration for the Step-5 PFN learning-curve panel."""

from typing import Literal

from pydantic import field_validator, model_validator

from ebpfn.config.base import StrictConfigModel
from ebpfn.config.pfn import PfnArchConfig, PfnTrainConfig


class OfflineValidationModeConfig(StrictConfigModel):
    """Scale and frozen identities for a Step-5 training-panel run."""

    name: Literal["fast", "pilot"]
    output_dir: str
    pairing_id: str
    baseline_eta_path: str
    source_roles_path: str
    seeds: tuple[int, ...]
    eta_labels: tuple[Literal["eta_0", "corr_strength_perturbed"], ...] = (
        "eta_0",
        "corr_strength_perturbed",
    )
    perturbed_corr_strength_mean: float

    @field_validator("seeds", "eta_labels", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "OfflineValidationModeConfig":
        if not self.output_dir or not self.pairing_id or not self.baseline_eta_path or not self.source_roles_path:
            raise ValueError("offline-validation paths and pairing_id must be nonempty")
        if not self.seeds or any(seed < 0 for seed in self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("offline-validation seeds must be unique nonnegative integers")
        if not self.eta_labels or len(set(self.eta_labels)) != len(self.eta_labels):
            raise ValueError("offline-validation eta labels must be nonempty and unique")
        if not 0.0 <= self.perturbed_corr_strength_mean <= 1.0:
            raise ValueError("perturbed_corr_strength_mean must be in [0, 1]")
        return self


class OfflineValidationConfig(StrictConfigModel):
    """Baseline-plus-perturbation learning-curve panel configuration."""

    version: str = "offline-validation-training-panel-1"
    mode: OfflineValidationModeConfig
    arch: PfnArchConfig = PfnArchConfig()
    train: PfnTrainConfig = PfnTrainConfig()
    characterization_dir: str
    source_role: Literal["pilot", "confirmatory"]
    coverage_levels: tuple[float, ...]
    crps_grid_size: int
    metric_row_chunk_size: int
    decision_owner: str
    decision_date: str

    @field_validator("coverage_levels", mode="before")
    @classmethod
    def freeze_coverage_levels(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "OfflineValidationConfig":
        if not self.version or not self.decision_owner or not self.decision_date:
            raise ValueError("offline-validation version and decision metadata must be nonempty")
        if not self.characterization_dir:
            raise ValueError("characterization_dir must be nonempty")
        if not self.coverage_levels or any(not 0.0 < level < 1.0 for level in self.coverage_levels):
            raise ValueError("coverage levels must be nonempty and lie in (0, 1)")
        if len(set(self.coverage_levels)) != len(self.coverage_levels):
            raise ValueError("coverage levels must be unique")
        if self.crps_grid_size < 2 or self.metric_row_chunk_size < 1:
            raise ValueError("CRPS grid and metric row chunk sizes must be positive")
        return self
