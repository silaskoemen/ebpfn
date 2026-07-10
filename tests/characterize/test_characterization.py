from dataclasses import replace
from itertools import pairwise

import numpy as np
import polars as pl
import pytest
from ebpfn.characterize import CharacterizationSchema
from ebpfn.characterize import Coordinate
from ebpfn.characterize import build_feature_maps
from ebpfn.characterize import build_row_budget_manifests
from ebpfn.characterize import characterize
from ebpfn.characterize import characterize_multiresolution
from ebpfn.characterize import fit_ridge_probe
from ebpfn.characterize import solve_ridge_coefficients
from ebpfn.characterize import target_functionals
from ebpfn.config import CharacterizationConfig
from ebpfn.config import MapConfig
from ebpfn.config import RidgeConfig
from ebpfn.config import RowBudgetConfig
from ebpfn.data import FeatureSchema
from ebpfn.data import TaskPartition
from ebpfn.data import TuningTask


def _task(*, n_fit: int = 180, n_score: int = 80, p: int = 4, seed: int = 3) -> TuningTask:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_fit + n_score, p))
    target = features[:, 0] + 0.7 * (features[:, 1] > 0.0) + rng.normal(scale=0.3, size=len(features))
    names = tuple(f"x{index}" for index in range(p))
    schema = FeatureSchema(names, ("numeric",) * p)
    fit = TaskPartition(pl.DataFrame(features[:n_fit], schema=names), target[:n_fit], np.arange(n_fit))
    score = TaskPartition(
        pl.DataFrame(features[n_fit:], schema=names), target[n_fit:], np.arange(n_fit, n_fit + n_score)
    )
    return TuningTask("task", "source", "regression", "outer", "inner", fit, score, schema, "prep", (0.0,) * p)


def _config(**changes) -> CharacterizationConfig:
    config = CharacterizationConfig(
        row_budgets=RowBudgetConfig(minimum=128, spacing="geometric"),
        maps=MapConfig(max_products=16, max_conjunctions=16, max_rff=24),
    )
    return config.model_copy(update=changes)


def test_row_budget_manifests_are_nested_reproducible_and_weighted():
    task = _task(n_fit=300, n_score=140)
    manifests = build_row_budget_manifests(task, _config())
    assert [manifest.row_budget for manifest in manifests] == [128, 256, 440]
    assert sum(manifest.weight for manifest in manifests) == pytest.approx(1.0)
    assert manifests == build_row_budget_manifests(task, _config())
    for smaller, larger in pairwise(manifests):
        assert set(smaller.probe_fit_indices) <= set(larger.probe_fit_indices)
        assert set(smaller.probe_score_indices) <= set(larger.probe_score_indices)


def test_explicit_random_identity_decouples_manifests_from_task_identity():
    task = _task()
    renamed = replace(task, task_id="renamed-task", characterization_split_id="renamed-split")
    identity = ("selection", 2, "target", 4)
    first = build_row_budget_manifests(task, _config(), random_identity=identity)
    second = build_row_budget_manifests(renamed, _config(), random_identity=identity)
    assert first == second
    first_characterization = characterize_multiresolution(task, _config(), random_identity=identity)
    second_characterization = characterize_multiresolution(renamed, _config(), random_identity=identity)
    assert np.array_equal(first_characterization.values, second_characterization.values)


def test_row_budget_allocation_retains_a_single_available_score_row():
    task = _task(n_fit=1000, n_score=1)
    manifest = build_row_budget_manifests(task, _config())[0]
    assert len(manifest.probe_fit_indices) == manifest.row_budget - 1
    assert len(manifest.probe_score_indices) == 1


def test_target_functionals_use_fit_statistics_and_reject_degenerate_scale():
    result = target_functionals(np.arange(10.0), np.array([-100.0, 100.0]), clip=5.0, scale_epsilon=1e-12)
    assert result.score[0, 0] == -5.0
    assert result.score[1, 0] == 5.0
    with pytest.raises(ValueError, match="degenerate"):
        target_functionals(np.ones(10), np.ones(3), clip=5.0, scale_epsilon=1e-12)


def test_map_dimensions_obey_v1_caps_and_interactions_use_distinct_features():
    rng = np.random.default_rng(9)
    fit = rng.normal(size=(150, 100))
    score = rng.normal(size=(40, 100))
    names = tuple(f"x{index}" for index in range(100))
    maps = build_feature_maps(fit, score, names, MapConfig(), seed_identity=("task", 128))
    by_name = {feature_map.name: feature_map for feature_map in maps}
    assert by_name["linear"].fit.shape[1] == 100
    assert by_name["bins"].fit.shape[1] == 800
    assert by_name["pairwise"].fit.shape[1] <= 1056
    assert by_name["rff"].fit.shape[1] == 356
    product_names = [name for name in by_name["pairwise"].column_names if "*" in name]
    assert all(left != right for left, right in (name.split("*") for name in product_names))


def test_ridge_gain_is_bounded_and_retains_negative_values():
    rng = np.random.default_rng(4)
    fit = rng.normal(size=(60, 3))
    score = rng.normal(size=(30, 3))
    target_fit = rng.normal(size=(60, 2))
    target_score = rng.normal(size=(30, 2))
    result = fit_ridge_probe(fit, score, target_fit, target_score, RidgeConfig())
    assert np.all(result.gains >= -1.0)
    assert np.all(result.gains <= 1.0)
    assert np.any(result.gains < 0.0)


def test_primal_and_dual_ridge_predictions_agree():
    rng = np.random.default_rng(12)
    design = rng.normal(size=(24, 9))
    targets = rng.normal(size=(24, 3))
    primal = solve_ridge_coefficients(design, targets, 0.01, solver="primal")
    dual = solve_ridge_coefficients(design, targets, 0.01, solver="dual")
    assert design @ primal == pytest.approx(design @ dual, abs=1e-10)


def test_raw_and_contrast_characterizations_align_and_reconstruct():
    task = _task()
    manifest = build_row_budget_manifests(task, _config())[0]
    raw = characterize(task, manifest, _config(representation="raw", include_observation_coordinates=False))
    contrast = characterize(task, manifest, _config(representation="contrast", include_observation_coordinates=False))
    assert raw.valid.all()
    assert contrast.valid.all()
    raw_matrix = raw.values.reshape(5, 4)
    contrast_matrix = contrast.values.reshape(5, 4)
    assert contrast_matrix[:, 0] == pytest.approx(raw_matrix[:, 0])
    assert contrast_matrix[:, 1] == pytest.approx((raw_matrix[:, 1] - raw_matrix[:, 0]) / 2.0)
    assert contrast_matrix[:, 2] == pytest.approx((raw_matrix[:, 2] - raw_matrix[:, 1]) / 2.0)
    assert contrast_matrix[:, 3] == pytest.approx((raw_matrix[:, 3] - raw_matrix[:, 0]) / 2.0)


def test_multiresolution_is_finite_deterministic_and_probe_score_is_not_fit_state():
    task = _task(n_fit=300, n_score=140)
    config = _config()
    first = characterize_multiresolution(task, config)
    second = characterize_multiresolution(task, config)
    assert np.array_equal(first.values, second.values)
    changed_score = replace(
        task,
        probe_score=TaskPartition(task.probe_score.X, task.probe_score.y + 100.0, task.probe_score.row_ids),
    )
    changed = characterize_multiresolution(changed_score, config)
    assert not np.array_equal(first.values, changed.values)
    assert first.valid.all()
    assert np.isfinite(first.values).all()


def test_characterization_seed_changes_rows_and_random_maps():
    task = _task(n_fit=300, n_score=140)
    first = characterize_multiresolution(task, _config(seed=1, include_observation_coordinates=False))
    second = characterize_multiresolution(task, _config(seed=2, include_observation_coordinates=False))
    assert not np.array_equal(first.values, second.values)


def test_schema_rejects_missing_parents_and_cycles():
    with pytest.raises(ValueError, match="missing coordinate parent"):
        CharacterizationSchema("v", "contrast", (Coordinate("child", "b", parent="absent"),))
    with pytest.raises(ValueError, match="cycle"):
        CharacterizationSchema(
            "v",
            "contrast",
            (Coordinate("a", "b", parent="b"), Coordinate("b", "b", parent="a")),
        )
    with pytest.raises(ValueError, match="child raw gains"):
        CharacterizationSchema(
            "v",
            "contrast",
            (Coordinate("raw-child", "b", learner="bins", statistic="gain"),),
        )


def test_constant_map_uses_total_valued_baseline_probe():
    fit = np.ones((20, 3))
    score = np.ones((10, 3))
    targets_fit = np.arange(20.0)[:, None]
    targets_score = np.arange(10.0)[:, None]
    result = fit_ridge_probe(fit, score, targets_fit, targets_score, RidgeConfig())
    assert result.dimension == 0
    assert result.solver == "baseline"
    assert result.gains.tolist() == [0.0]


def test_missingness_rates_are_emitted_as_raw_diagnostics():
    task = replace(_task(), probe_fit_missing_rates=(0.0, 0.1, 0.2, 0.3))
    result = characterize(task, build_row_budget_manifests(task, _config())[0], _config())
    raw = dict(zip((coordinate.statistic for coordinate in result.coordinates), result.raw_values, strict=True))
    assert raw["feature_missingness_mean"] == pytest.approx(0.15)
    assert raw["feature_missingness_max"] == pytest.approx(0.3)


def test_disabling_observation_coordinates_retains_raw_diagnostics():
    task = replace(_task(), probe_fit_missing_rates=(0.0, 0.1, 0.2, 0.3))
    result = characterize(
        task,
        build_row_budget_manifests(task, _config())[0],
        _config(include_observation_coordinates=False),
    )
    assert not any(coordinate.block == "observation" for coordinate in result.coordinates)
    assert result.metadata["observation_raw"]["feature_missingness_mean"] == pytest.approx(0.15)


def test_production_map_characterization_smoke():
    task = _task(n_fit=192, n_score=64, p=6)
    config = CharacterizationConfig()
    result = characterize(task, build_row_budget_manifests(task, config)[0], config)
    assert result.valid.all()
    dimensions = result.metadata["map_dimensions"]
    assert dimensions["linear"] <= 6
    assert dimensions["pairwise"] <= 304
    assert dimensions["rff"] <= 262
