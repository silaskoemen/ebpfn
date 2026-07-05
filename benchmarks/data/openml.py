"""OpenML acquisition boundary for the Polars-based core data API."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from ebpfn.config import OpenMLConfig
from ebpfn.config import SplitConfig
from ebpfn.data import RawTabularTask
from ebpfn.data import RotationDefinition
from ebpfn.data import SourceSplit
from ebpfn.data import create_source_split
from ebpfn.data import materialize_tasks


@dataclass(frozen=True)
class OpenMLSource:
    frame: pl.DataFrame
    source_id: str
    canonical_target: str
    split: SourceSplit
    metadata: dict[str, Any]


def _to_polars(frame: Any) -> pl.DataFrame:
    return pl.DataFrame({str(name): frame[name].to_numpy() for name in frame.columns})


def load_openml_source(config: OpenMLConfig, split_config: SplitConfig) -> OpenMLSource:
    import openml

    openml.config.set_root_cache_directory(config.cache_dir)
    task = openml.tasks.get_task(config.task_id, download_splits=True)
    dataset = task.get_dataset()
    target_attribute = getattr(task, "target_name")
    features, target, _, _ = dataset.get_data(target=target_attribute, dataset_format="dataframe")
    feature_frame = _to_polars(features)
    target_name = str(target_attribute)
    frame = feature_frame.with_columns(pl.Series(target_name, np.asarray(target)))
    train_ids, test_ids = task.get_train_test_split_indices(
        repeat=config.repeat,
        fold=config.fold,
        sample=config.sample,
    )
    source_id = f"openml-task-{config.task_id}"
    split = create_source_split(
        source_id,
        frame.height,
        split_config,
        official_train_ids=tuple(int(value) for value in train_ids),
        official_test_ids=tuple(int(value) for value in test_ids),
    )
    if dataset.dataset_id is None:
        raise ValueError("OpenML dataset has no stable dataset ID")
    metadata = {
        "adapter": "openml",
        "dataset_id": int(dataset.dataset_id),
        "dataset_name": str(dataset.name),
        "openml_task_id": config.task_id,
    }
    return OpenMLSource(frame, source_id, target_name, split, metadata)


def canonical_openml_task(source: OpenMLSource) -> RawTabularTask:
    predictors = tuple(name for name in source.frame.columns if name != source.canonical_target)
    definition = RotationDefinition(f"{source.source_id}-canonical", source.canonical_target, predictors)
    return materialize_tasks(
        source.frame,
        source.source_id,
        (definition,),
        rotations_enabled=False,
        metadata=source.metadata,
    )[0]
