"""Construction of leakage-safe tuning and evaluation tasks."""

from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import polars as pl

from ebpfn.config import DataPipelineConfig
from ebpfn.data.preprocessing import FeatureTransform
from ebpfn.data.preprocessing import fit_feature_transform
from ebpfn.data.splits import EligibilityReport
from ebpfn.data.splits import characterization_split_id
from ebpfn.data.splits import eligible_role_ids
from ebpfn.data.types import EvaluationTask
from ebpfn.data.types import RawTabularTask
from ebpfn.data.types import SourceSplit
from ebpfn.data.types import TaskPartition
from ebpfn.data.types import TuningTask


@dataclass(frozen=True)
class TaskBuildResult:
    task: EvaluationTask | None
    eligibility: EligibilityReport
    transform: FeatureTransform | None

    def __post_init__(self) -> None:
        built = self.task is not None
        if built != self.eligibility.admitted or built != (self.transform is not None):
            raise ValueError("task, transform, and eligibility admission must agree")


def _raw_partition(task: RawTabularTask, ids: tuple[int, ...]) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    by_id = {int(row_id): index for index, row_id in enumerate(task.row_ids)}
    indices = [by_id[row_id] for row_id in ids]
    return task.X[indices], task.y[indices].astype(float, copy=True), np.asarray(ids, dtype=np.int64)


def build_evaluation_task(task: RawTabularTask, split: SourceSplit, config: DataPipelineConfig) -> TaskBuildResult:
    if task.source_id != split.source_id:
        raise ValueError("task and split must refer to the same source")
    roles, report = eligible_role_ids(task, split, config.split)
    if not report.admitted:
        return TaskBuildResult(None, report, None)
    fit_frame, fit_y, fit_ids = _raw_partition(task, roles["probe_fit"])
    score_frame, score_y, score_ids = _raw_partition(task, roles["probe_score"])
    final_frame, final_y, final_ids = _raw_partition(task, roles["final_test"])
    try:
        transform = fit_feature_transform(fit_frame, task.schema, config.preprocessing)
        fit = TaskPartition(transform.apply(fit_frame), fit_y, fit_ids)
        score = TaskPartition(transform.apply(score_frame), score_y, score_ids)
        final = TaskPartition(transform.apply(final_frame), final_y, final_ids)
        split_id = characterization_split_id(task, split, roles)
        tuning = TuningTask(
            task.task_id,
            task.source_id,
            task.task_type,
            split.outer_split_id,
            split_id,
            fit,
            score,
            transform.output_schema,
            transform.transform_id,
            transform.probe_fit_missing_rates,
        )
    except (TypeError, ValueError) as error:
        rejected = replace(report, admitted=False, reasons=(*report.reasons, str(error)))
        return TaskBuildResult(None, rejected, None)
    return TaskBuildResult(EvaluationTask(tuning, final), report, transform)
