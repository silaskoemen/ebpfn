import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from benchmarks.data import canonical_openml_task
from benchmarks.data import load_openml_source
from ebpfn.config import DataPipelineConfig
from ebpfn.config import OpenMLConfig
from ebpfn.config import PreprocessingConfig
from ebpfn.config import RotationConfig
from ebpfn.config import SplitConfig
from ebpfn.data import FeatureSchema
from ebpfn.data import RawTabularTask
from ebpfn.data import RotationDefinition
from ebpfn.data import build_evaluation_task
from ebpfn.data import characterization_shape
from ebpfn.data import create_source_split
from ebpfn.data import evaluation_task_hash
from ebpfn.data import infer_feature_schema
from ebpfn.data import materialize_tasks
from ebpfn.data import rotation_diagnostics
from ebpfn.data import tuning_task_hash


def _frame(n: int = 200) -> pl.DataFrame:
    x = np.linspace(-3.0, 3.0, n)
    return pl.DataFrame(
        {
            "x": x,
            "binary": np.arange(n) % 2,
            "category": ["a", "b"] * (n // 2),
            "y": x**2 + 0.1 * x,
        }
    )


def _task(n: int = 200) -> RawTabularTask:
    frame = _frame(n)
    schema = FeatureSchema(("x", "binary", "category"), ("numeric", "binary", "categorical"))
    return RawTabularTask(
        "task",
        "source",
        "y",
        frame.select(schema.names),
        frame["y"].to_numpy(),
        np.arange(n),
        "regression",
        schema,
    )


def _config() -> DataPipelineConfig:
    return DataPipelineConfig(
        split=_split_config(min_probe_fit=20, min_probe_score=10, min_final_test=10),
        preprocessing=_preprocessing_config(),
        rotations=RotationConfig(enabled=False),
    )


def _split_config(
    *,
    seed: int = 0,
    min_probe_fit: int = 64,
    min_probe_score: int = 32,
    min_final_test: int = 32,
) -> SplitConfig:
    return SplitConfig(
        seed=seed,
        policy_version="split-2",
        final_test_fraction=0.2,
        probe_score_fraction_of_train=0.25,
        min_probe_fit=min_probe_fit,
        min_probe_score=min_probe_score,
        min_final_test=min_final_test,
    )


def _preprocessing_config(*, max_features: int = 100) -> PreprocessingConfig:
    return PreprocessingConfig(
        max_features=max_features,
        clip=4.0,
        constant_atol=1e-12,
        constant_rtol=1e-12,
        scale_epsilon=1e-12,
        version="preprocess-1",
    )


def test_raw_task_defensively_copies_inputs_and_rejects_target_leakage():
    frame = _frame()
    y = frame["y"].to_numpy().copy()
    task = _task()
    y[0] = 999.0
    frame = frame.with_columns(pl.lit(999.0).alias("x"))
    assert task.y[0] != 999.0
    assert task.X["x"][0] != 999.0
    schema = FeatureSchema(("y",), ("numeric",))
    with pytest.raises(ValueError, match="target column"):
        RawTabularTask(
            "t",
            "s",
            "y",
            pl.DataFrame({"y": [1.0, 2.0]}),
            np.array([1.0, 2.0]),
            np.arange(2),
            "regression",
            schema,
        )


def test_raw_task_rejects_noncanonical_positional_ids():
    task = _task()
    shifted_ids = task.row_ids + 1
    with pytest.raises(ValueError, match="canonical zero-based"):
        RawTabularTask(
            task.task_id,
            task.source_id,
            task.target_name,
            task.X,
            task.y,
            shifted_ids,
            task.task_type,
            task.schema,
        )


def test_fallback_split_is_deterministic_disjoint_and_exhaustive():
    config = _split_config(seed=4, min_probe_fit=1, min_probe_score=1, min_final_test=1)
    first = create_source_split("source", 101, config)
    second = create_source_split("source", 101, config)
    assert first == second
    roles = [set(first.probe_fit_ids), set(first.probe_score_ids), set(first.final_test_ids)]
    assert set.union(*roles) == set(range(101))
    assert not roles[0] & roles[1]
    assert not roles[0] & roles[2]
    assert not roles[1] & roles[2]


def test_fallback_split_applies_hierarchical_fractions():
    config = _split_config(min_probe_fit=1, min_probe_score=1, min_final_test=1)
    split = create_source_split("source", 100, config)
    assert len(split.probe_fit_ids) == 60
    assert len(split.probe_score_ids) == 20
    assert len(split.final_test_ids) == 20


def test_official_test_is_preserved_and_train_is_split_between_probe_roles():
    config = _split_config(min_probe_fit=1, min_probe_score=1, min_final_test=1)
    split = create_source_split(
        "source",
        100,
        config,
        official_train_ids=tuple(range(80)),
        official_test_ids=tuple(range(80, 100)),
    )
    assert split.final_test_ids == tuple(range(80, 100))
    assert len(split.probe_fit_ids) == 60
    assert len(split.probe_score_ids) == 20
    assert set(split.probe_fit_ids) | set(split.probe_score_ids) == set(range(80))


def test_build_task_excludes_categorical_and_separates_tuning_identity():
    task = _task()
    config = _config()
    split = create_source_split("source", len(task.y), config.split)
    result = build_evaluation_task(task, split, config)
    assert result.task is not None
    assert result.transform is not None
    assert result.task.tuning.schema.names == ("x", "binary")
    assert result.transform.excluded_categorical == ("category",)

    changed_y = task.y.copy()
    changed_y[[int(value) for value in split.final_test_ids]] += 1000.0
    changed = RawTabularTask(
        task.task_id,
        task.source_id,
        task.target_name,
        task.X,
        changed_y,
        task.row_ids,
        task.task_type,
        task.schema,
    )
    changed_result = build_evaluation_task(changed, split, config)
    assert changed_result.task is not None
    assert tuning_task_hash(result.task.tuning, config) == tuning_task_hash(changed_result.task.tuning, config)
    assert evaluation_task_hash(result.task, config) != evaluation_task_hash(changed_result.task, config)


def test_missing_targets_remove_rows_without_moving_roles():
    task = _task()
    config = _config()
    split = create_source_split("source", len(task.y), config.split)
    y = task.y.copy()
    missing_id = split.probe_score_ids[0]
    y[missing_id] = np.nan
    missing = RawTabularTask(
        task.task_id,
        task.source_id,
        task.target_name,
        task.X,
        y,
        task.row_ids,
        task.task_type,
        task.schema,
    )
    result = build_evaluation_task(missing, split, config)
    assert result.task is not None
    assert missing_id not in result.task.tuning.probe_score.row_ids
    assert result.eligibility.missing_targets["probe_score"] == 1


def test_constant_final_test_target_does_not_affect_admission():
    task = _task()
    config = _config()
    split = create_source_split("source", len(task.y), config.split)
    y = task.y.copy()
    y[list(split.final_test_ids)] = 1.0
    changed = RawTabularTask(
        task.task_id,
        task.source_id,
        task.target_name,
        task.X,
        y,
        task.row_ids,
        task.task_type,
        task.schema,
    )
    result = build_evaluation_task(changed, split, config)
    assert result.task is not None


def test_final_test_missingness_does_not_affect_tuning_identity():
    task = _task()
    config = _config()
    split = create_source_split("source", len(task.y), config.split)
    baseline = build_evaluation_task(task, split, config)
    assert baseline.task is not None

    y = task.y.copy()
    y[split.final_test_ids[0]] = np.nan
    changed_task = RawTabularTask(
        task.task_id,
        task.source_id,
        task.target_name,
        task.X,
        y,
        task.row_ids,
        task.task_type,
        task.schema,
    )
    changed = build_evaluation_task(changed_task, split, config)
    assert changed.task is not None
    assert tuning_task_hash(baseline.task.tuning, config) == tuning_task_hash(changed.task.tuning, config)
    assert evaluation_task_hash(baseline.task, config) != evaluation_task_hash(changed.task, config)
    assert changed.task.final_test.X.height == baseline.task.final_test.X.height - 1


def test_probe_fit_target_requires_variation():
    task = _task()
    config = _config()
    split = create_source_split("source", len(task.y), config.split)
    y = task.y.copy()
    y[list(split.probe_fit_ids)] = 1.0
    changed = RawTabularTask(
        task.task_id,
        task.source_id,
        task.target_name,
        task.X,
        y,
        task.row_ids,
        task.task_type,
        task.schema,
    )
    result = build_evaluation_task(changed, split, config)
    assert result.task is None
    assert result.eligibility.reasons == ("probe-fit regression target must contain finite variation",)


@pytest.mark.parametrize(("features", "admitted"), [(100, True), (101, False)])
def test_feature_width_boundary(features: int, admitted: bool):
    n = 100
    rng = np.random.default_rng(2)
    names = tuple(f"x{index}" for index in range(features))
    frame = pl.DataFrame({name: rng.normal(size=n) for name in names})
    task = RawTabularTask(
        "t",
        "s",
        "y",
        frame,
        rng.normal(size=n),
        np.arange(n),
        "regression",
        FeatureSchema(names, ("numeric",) * features),
    )
    config = DataPipelineConfig(
        split=_split_config(min_probe_fit=10, min_probe_score=5, min_final_test=5),
        preprocessing=_preprocessing_config(max_features=100),
        rotations=RotationConfig(enabled=False),
    )
    result = build_evaluation_task(task, create_source_split("s", n, config.split), config)
    assert (result.task is not None) is admitted
    assert result.eligibility.admitted is admitted
    if not admitted:
        assert result.eligibility.reasons == ("task has 101 usable predictors; maximum is 100",)


def test_explicit_rotations_default_to_canonical_and_share_source_positions():
    frame = _frame()
    definitions = (
        RotationDefinition("canonical", "y", ("x", "binary")),
        RotationDefinition("rotated", "x", ("y", "binary")),
    )
    canonical = materialize_tasks(frame, "source", definitions, rotations_enabled=False)
    rotated = materialize_tasks(frame, "source", definitions, rotations_enabled=True)
    assert len(canonical) == 1
    assert len(rotated) == 2
    np.testing.assert_array_equal(rotated[0].row_ids, rotated[1].row_ids)
    split = create_source_split(
        "source",
        frame.height,
        _split_config(min_probe_fit=1, min_probe_score=1, min_final_test=1),
    )
    diagnostics = rotation_diagnostics(rotated[1], split)
    assert diagnostics.target_variance_probe is not None
    assert diagnostics.finite_target_counts["final_test"] == len(split.final_test_ids)


def test_characterization_shape_uses_canonical_task_view():
    config = _config()
    result = build_evaluation_task(_task(), create_source_split("source", 200, config.split), config)
    assert result.task is not None
    shape = characterization_shape(result.task.tuning, row_budget=64)
    assert shape.n_probe_fit + shape.n_probe_score == 64
    assert shape.p_numeric == 2
    assert shape.p_categorical == 0


def test_only_zero_one_numeric_features_are_inferred_as_binary():
    frame = pl.DataFrame({"indicator": [0, 1, 0], "two_values": [1, 2, 1]})
    schema = infer_feature_schema(frame, tuple(frame.columns))
    assert schema.kinds == ("binary", "numeric")


def test_openml_adapter_converts_data_and_preserves_official_test(monkeypatch):
    class Column:
        def __init__(self, values):
            self.values = np.asarray(values)

        def to_numpy(self):
            return self.values

    class Frame:
        def __init__(self):
            self.columns = ("x", "indicator")
            self.data = {"x": Column(np.arange(10.0)), "indicator": Column(np.arange(10) % 2)}

        def __getitem__(self, name):
            return self.data[name]

    class Dataset:
        dataset_id = 42
        name = "offline"

        def get_data(self, *, target, dataset_format):
            assert target == "y"
            assert dataset_format == "dataframe"
            return Frame(), np.linspace(0.0, 1.0, 10), None, None

    class Task:
        target_name = "y"

        def get_dataset(self):
            return Dataset()

        def get_train_test_split_indices(self, *, repeat, fold, sample):
            assert (repeat, fold, sample) == (0, 0, 0)
            return np.arange(8), np.arange(8, 10)

    fake_openml = SimpleNamespace(
        config=SimpleNamespace(set_root_cache_directory=lambda _: None),
        tasks=SimpleNamespace(get_task=lambda task_id, download_splits: Task()),
    )
    monkeypatch.setitem(sys.modules, "openml", fake_openml)
    split_config = _split_config(min_probe_fit=1, min_probe_score=1, min_final_test=1)
    source = load_openml_source(
        OpenMLConfig(task_id=123, cache_dir="data/raw/openml", repeat=0, fold=0, sample=0),
        split_config,
    )
    assert source.frame.columns == ["x", "indicator", "y"]
    assert source.split.final_test_ids == (8, 9)
    task = canonical_openml_task(source)
    assert task.schema.names == ("x", "indicator")
    assert task.metadata["dataset_id"] == 42
