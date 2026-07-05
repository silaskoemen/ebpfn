"""Explicit, default-off construction of canonical targets and rotations."""

from dataclasses import dataclass
from typing import Any
from typing import cast

import numpy as np
import polars as pl

from ebpfn.data.hashing import content_hash
from ebpfn.data.types import FeatureKind
from ebpfn.data.types import FeatureSchema
from ebpfn.data.types import RawTabularTask
from ebpfn.data.types import SourceSplit


@dataclass(frozen=True)
class RotationDefinition:
    task_id: str
    target: str
    predictors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.target or not self.predictors:
            raise ValueError("rotation task, target, and predictors must be nonempty")
        if self.target in self.predictors or len(set(self.predictors)) != len(self.predictors):
            raise ValueError("target cannot be a predictor and predictors must be unique")


@dataclass(frozen=True)
class RotationDiagnostics:
    task_id: str
    target: str
    finite_target_counts: dict[str, int]
    target_variance_probe: float | None
    maximum_absolute_feature_correlation: float | None


def infer_feature_schema(frame: pl.DataFrame, names: tuple[str, ...]) -> FeatureSchema:
    kinds: list[str] = []
    for name in names:
        series = frame.get_column(name)
        numeric_values = (
            set(series.cast(pl.Float64, strict=False).drop_nulls().drop_nans().unique().to_list())
            if series.dtype.is_numeric()
            else set()
        )
        if series.dtype == pl.Boolean or (numeric_values and numeric_values <= {0.0, 1.0}):
            kinds.append("binary")
        elif series.dtype.is_numeric():
            kinds.append("numeric")
        else:
            kinds.append("categorical")
    return FeatureSchema(names, cast(tuple[FeatureKind, ...], tuple(kinds)))


def materialize_tasks(
    frame: pl.DataFrame,
    source_id: str,
    definitions: tuple[RotationDefinition, ...],
    *,
    rotations_enabled: bool,
    metadata: dict[str, Any] | None = None,
) -> tuple[RawTabularTask, ...]:
    if not definitions:
        raise ValueError("at least the canonical target definition is required")
    selected = definitions if rotations_enabled else definitions[:1]
    row_ids = np.arange(frame.height, dtype=np.int64)
    tasks: list[RawTabularTask] = []
    for definition in selected:
        required = {definition.target, *definition.predictors}
        if unknown := sorted(required - set(frame.columns)):
            raise ValueError(f"rotation references unknown columns: {unknown}")
        y = frame.get_column(definition.target).cast(pl.Float64, strict=False).to_numpy()
        X = frame.select(definition.predictors)
        schema = infer_feature_schema(frame, definition.predictors)
        task_metadata = dict(metadata or {})
        task_metadata["rotation_definition_id"] = content_hash(definition, namespace="rotation-1")
        tasks.append(
            RawTabularTask(
                definition.task_id, source_id, definition.target, X, y, row_ids, "regression", schema, task_metadata
            )
        )
    return tuple(tasks)


def rotation_diagnostics(task: RawTabularTask, split: SourceSplit) -> RotationDiagnostics:
    by_id = {int(row_id): index for index, row_id in enumerate(task.row_ids)}
    roles = {"probe_fit": split.probe_fit_ids, "probe_score": split.probe_score_ids, "final_test": split.final_test_ids}
    counts: dict[str, int] = {}
    for name, ids in roles.items():
        values = [task.y[by_id[row_id]] for row_id in ids if row_id in by_id]
        counts[name] = int(np.isfinite(np.asarray(values, dtype=float)).sum())
    probe_ids = [row_id for row_id in (*split.probe_fit_ids, *split.probe_score_ids) if row_id in by_id]
    indices = [by_id[row_id] for row_id in probe_ids]
    y = task.y[indices].astype(float, copy=False)
    valid_y = np.isfinite(y)
    variance = float(np.var(y[valid_y])) if valid_y.sum() >= 2 else None
    correlations: list[float] = []
    for name, kind in zip(task.schema.names, task.schema.kinds, strict=True):
        if kind == "categorical":
            continue
        x = task.X.get_column(name)[indices].cast(pl.Float64, strict=False).to_numpy().astype(float)
        valid = valid_y & np.isfinite(x)
        if valid.sum() >= 3 and np.std(x[valid]) > 0 and np.std(y[valid]) > 0:
            correlations.append(abs(float(np.corrcoef(x[valid], y[valid])[0, 1])))
    return RotationDiagnostics(task.task_id, task.target_name, counts, variance, max(correlations, default=None))
