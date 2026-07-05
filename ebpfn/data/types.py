"""Validated task, partition, schema, and split contracts."""

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import TypeAlias

import numpy as np
import polars as pl

from ebpfn.data.hashing import content_hash
from ebpfn.data.hashing import is_json_value

RowId: TypeAlias = int
TaskType: TypeAlias = Literal["regression", "classification"]
FeatureKind: TypeAlias = Literal["numeric", "categorical", "binary"]
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _copy_vector(values: np.ndarray, *, name: str, dtype: Any | None = None) -> np.ndarray:
    vector = np.array(values, dtype=dtype, copy=True)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return vector


def _validate_regression_target(y: np.ndarray, *, allow_missing: bool) -> None:
    if not np.issubdtype(y.dtype, np.number):
        raise TypeError("regression target must be numeric")
    numeric = y.astype(float, copy=False)
    if np.isinf(numeric).any():
        raise ValueError("regression target cannot contain infinity")
    if not allow_missing and not np.isfinite(numeric).all():
        raise ValueError("task partition target must be finite")


def _validate_target_variation(y: np.ndarray, *, role: str) -> None:
    if len(y) < 2 or np.all(y == y[0]):
        raise ValueError(f"{role} regression target must contain finite variation")


@dataclass(frozen=True)
class FeatureSchema:
    names: tuple[str, ...]
    kinds: tuple[FeatureKind, ...]
    groups: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.kinds):
            raise ValueError("feature names and kinds must be nonempty and aligned")
        if len(set(self.names)) != len(self.names) or any(not name for name in self.names):
            raise ValueError("feature names must be unique and nonempty")
        if any(kind not in ("numeric", "categorical", "binary") for kind in self.kinds):
            raise ValueError("unsupported feature kind")
        if self.groups is not None and len(self.groups) != len(self.names):
            raise ValueError("feature groups must align with feature names")

    def select(self, names: tuple[str, ...]) -> "FeatureSchema":
        positions = {name: index for index, name in enumerate(self.names)}
        if unknown := [name for name in names if name not in positions]:
            raise ValueError(f"unknown schema features: {unknown}")
        indices = [positions[name] for name in names]
        groups = None if self.groups is None else tuple(self.groups[index] for index in indices)
        return FeatureSchema(names, tuple(self.kinds[index] for index in indices), groups)


@dataclass(frozen=True)
class RawTabularTask:
    task_id: str
    source_id: str
    target_name: str
    X: pl.DataFrame
    y: np.ndarray
    row_ids: np.ndarray
    task_type: TaskType
    schema: FeatureSchema
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frame = self.X.clone()
        y = _copy_vector(self.y, name="y")
        row_ids = _copy_vector(self.row_ids, name="row_ids", dtype=np.int64)
        if not self.task_id or not self.source_id or not self.target_name:
            raise ValueError("task_id, source_id, and target_name must be nonempty")
        if self.task_type != "regression":
            raise NotImplementedError("classification task construction is reserved for a later version")
        if frame.height != len(y) or len(y) != len(row_ids):
            raise ValueError("X, y, and row_ids must have aligned lengths")
        if tuple(frame.columns) != self.schema.names:
            raise ValueError("frame column order must exactly match the feature schema")
        if self.target_name in frame.columns:
            raise ValueError("target column cannot be present in X")
        if not np.array_equal(row_ids, np.arange(len(row_ids), dtype=np.int64)):
            raise ValueError("raw row_ids must be canonical zero-based positional offsets")
        _validate_regression_target(y, allow_missing=True)
        if not is_json_value(self.metadata):
            raise TypeError("metadata must be JSON-compatible and finite")
        object.__setattr__(self, "X", frame)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SourceSplit:
    source_id: str
    probe_fit_ids: tuple[RowId, ...]
    probe_score_ids: tuple[RowId, ...]
    final_test_ids: tuple[RowId, ...]
    outer_split_id: str
    policy_version: str
    seed: int

    def __post_init__(self) -> None:
        roles = (self.probe_fit_ids, self.probe_score_ids, self.final_test_ids)
        if not self.source_id or not self.outer_split_id:
            raise ValueError("source and split identities must be nonempty")
        if any(tuple(sorted(role)) != role or len(set(role)) != len(role) for role in roles):
            raise ValueError("split roles must be sorted and internally unique")
        if any(row_id < 0 for role in roles for row_id in role):
            raise ValueError("row IDs must be nonnegative")
        if (
            set(self.probe_fit_ids) & set(self.probe_score_ids)
            or set(self.probe_fit_ids) & set(self.final_test_ids)
            or set(self.probe_score_ids) & set(self.final_test_ids)
        ):
            raise ValueError("split roles must be disjoint")


@dataclass(frozen=True)
class TaskPartition:
    X: pl.DataFrame
    y: np.ndarray
    row_ids: np.ndarray

    def __post_init__(self) -> None:
        frame = self.X.clone()
        y = _copy_vector(self.y, name="y")
        row_ids = _copy_vector(self.row_ids, name="row_ids", dtype=np.int64)
        if frame.height != len(y) or len(y) != len(row_ids):
            raise ValueError("partition arrays must have aligned lengths")
        if len(np.unique(row_ids)) != len(row_ids):
            raise ValueError("partition row IDs must be unique")
        _validate_regression_target(y, allow_missing=False)
        object.__setattr__(self, "X", frame)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "row_ids", row_ids)


@dataclass(frozen=True)
class TuningTask:
    task_id: str
    source_id: str
    task_type: TaskType
    outer_split_id: str
    characterization_split_id: str
    probe_fit: TaskPartition
    probe_score: TaskPartition
    schema: FeatureSchema
    preprocessing_id: str

    def __post_init__(self) -> None:
        if self.task_type != "regression":
            raise NotImplementedError("classification is not implemented")
        expected = self.schema.names
        if tuple(self.probe_fit.X.columns) != expected or tuple(self.probe_score.X.columns) != expected:
            raise ValueError("tuning partitions must match the task schema")
        _validate_target_variation(self.probe_fit.y, role="probe-fit")


@dataclass(frozen=True)
class EvaluationTask:
    tuning: TuningTask
    final_test: TaskPartition

    def __post_init__(self) -> None:
        if tuple(self.final_test.X.columns) != self.tuning.schema.names:
            raise ValueError("final-test features must match the tuning schema")


@dataclass(frozen=True)
class CharacterizationShape:
    n_probe_fit: int
    n_probe_score: int
    p_numeric: int
    p_categorical: int
    task_type: TaskType
    n_classes: int | None = None


def characterization_shape(task: TuningTask, row_budget: int | None = None) -> CharacterizationShape:
    if row_budget is not None and row_budget < 2:
        raise ValueError("row_budget must be at least two")
    n_fit = task.probe_fit.X.height
    n_score = task.probe_score.X.height
    if row_budget is not None:
        total = n_fit + n_score
        effective = min(row_budget, total)
        n_fit = min(n_fit, max(1, round(effective * n_fit / total)))
        n_score = effective - n_fit
        if n_score < 1:
            n_score = 1
            n_fit = effective - 1
    kinds = task.schema.kinds
    return CharacterizationShape(
        n_fit, n_score, kinds.count("numeric") + kinds.count("binary"), kinds.count("categorical"), task.task_type
    )


def tuning_task_hash(task: TuningTask, resolved_tuning_config: Any) -> str:
    return content_hash(task, resolved_tuning_config, namespace="tuning-task-1")


def evaluation_task_hash(task: EvaluationTask, resolved_evaluation_config: Any) -> str:
    return content_hash(task, resolved_evaluation_config, namespace="evaluation-task-1")
