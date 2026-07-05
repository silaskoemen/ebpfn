"""Leakage-safe tabular task contracts and constructors."""

from ebpfn.data.hashing import content_hash
from ebpfn.data.preprocessing import FeatureTransform
from ebpfn.data.preprocessing import fit_feature_transform
from ebpfn.data.rotations import RotationDefinition
from ebpfn.data.rotations import RotationDiagnostics
from ebpfn.data.rotations import infer_feature_schema
from ebpfn.data.rotations import materialize_tasks
from ebpfn.data.rotations import rotation_diagnostics
from ebpfn.data.splits import EligibilityReport
from ebpfn.data.splits import create_source_split
from ebpfn.data.tasks import TaskBuildResult
from ebpfn.data.tasks import build_evaluation_task
from ebpfn.data.types import CharacterizationShape
from ebpfn.data.types import EvaluationTask
from ebpfn.data.types import FeatureSchema
from ebpfn.data.types import RawTabularTask
from ebpfn.data.types import SourceSplit
from ebpfn.data.types import TaskPartition
from ebpfn.data.types import TuningTask
from ebpfn.data.types import characterization_shape
from ebpfn.data.types import evaluation_task_hash
from ebpfn.data.types import tuning_task_hash

__all__ = [
    "CharacterizationShape",
    "EligibilityReport",
    "EvaluationTask",
    "FeatureSchema",
    "FeatureTransform",
    "RawTabularTask",
    "RotationDefinition",
    "RotationDiagnostics",
    "SourceSplit",
    "TaskBuildResult",
    "TaskPartition",
    "TuningTask",
    "build_evaluation_task",
    "characterization_shape",
    "content_hash",
    "create_source_split",
    "evaluation_task_hash",
    "fit_feature_transform",
    "infer_feature_schema",
    "materialize_tasks",
    "rotation_diagnostics",
    "tuning_task_hash",
]
