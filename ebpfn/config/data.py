"""Validated configuration for source preparation."""

from pydantic import model_validator

from ebpfn.config.base import StrictConfigModel


class SplitConfig(StrictConfigModel):
    seed: int
    policy_version: str
    final_test_fraction: float
    probe_score_fraction_of_train: float
    min_probe_fit: int
    min_probe_score: int
    min_final_test: int

    @model_validator(mode="after")
    def validate_values(self) -> "SplitConfig":
        if self.seed < 0:
            raise ValueError("split seed must be nonnegative")
        if not self.policy_version:
            raise ValueError("split policy_version must be nonempty")
        if not (0.0 < self.final_test_fraction < 1.0):
            raise ValueError("final_test_fraction must be in (0, 1)")
        if not (0.0 < self.probe_score_fraction_of_train < 1.0):
            raise ValueError("probe_score_fraction_of_train must be in (0, 1)")
        if min(self.min_probe_fit, self.min_probe_score, self.min_final_test) < 1:
            raise ValueError("minimum role counts must be positive")
        return self


class PreprocessingConfig(StrictConfigModel):
    max_features: int
    clip: float
    constant_atol: float
    constant_rtol: float
    scale_epsilon: float
    version: str

    @model_validator(mode="after")
    def validate_values(self) -> "PreprocessingConfig":
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.clip <= 0 or self.scale_epsilon <= 0:
            raise ValueError("clip and scale_epsilon must be positive")
        if self.constant_atol < 0 or self.constant_rtol < 0:
            raise ValueError("constant tolerances must be nonnegative")
        if not self.version:
            raise ValueError("preprocessing version must be nonempty")
        return self


class RotationConfig(StrictConfigModel):
    enabled: bool


class DataPipelineConfig(StrictConfigModel):
    split: SplitConfig
    preprocessing: PreprocessingConfig
    rotations: RotationConfig


class OpenMLConfig(StrictConfigModel):
    task_id: int
    cache_dir: str
    repeat: int
    fold: int
    sample: int

    @model_validator(mode="after")
    def validate_values(self) -> "OpenMLConfig":
        if self.task_id < 1:
            raise ValueError("OpenML task_id must be positive")
        if not self.cache_dir:
            raise ValueError("OpenML cache_dir must be nonempty")
        if min(self.repeat, self.fold, self.sample) < 0:
            raise ValueError("OpenML split coordinates must be nonnegative")
        return self


class DataPreparationModeConfig(StrictConfigModel):
    name: str
    output_dir: str


class PrepareDataConfig(StrictConfigModel):
    mode: DataPreparationModeConfig
    openml: OpenMLConfig
    data: DataPipelineConfig
