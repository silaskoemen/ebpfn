"""End-to-end fixed-map characterization."""

from typing import Any

import numpy as np

from ebpfn.config import CharacterizationConfig
from ebpfn.data import TuningTask

from .budgets import build_row_budget_manifests
from .contracts import CharacterizationSchema
from .contracts import Coordinate
from .contracts import RowBudgetManifest
from .contracts import TaskCharacterization
from .maps import build_feature_maps
from .observation import observation_statistics
from .ridge import fit_ridge_probe
from .targets import TARGET_NAMES
from .targets import target_functionals


def _gain_block(map_name: str, target_name: str) -> str:
    if map_name == "pairwise":
        return "interaction"
    if map_name in {"bins", "rff"}:
        return "nonlinear"
    return "location" if target_name == "location" else "scale_tail"


def _gain_coordinates(budget: int, representation: str, map_names: tuple[str, ...]) -> tuple[Coordinate, ...]:
    coordinates: list[Coordinate] = []
    for target in TARGET_NAMES:
        if representation == "raw":
            for learner in map_names:
                coordinates.append(
                    Coordinate(
                        f"m{budget}.{target}.{learner}.gain",
                        _gain_block(learner, target),
                        learner=learner,
                        target=target,
                        row_budget=budget,
                        bounds=(-1.0, 1.0),
                    )
                )
            continue
        linear_name = f"m{budget}.{target}.linear.gain"
        coordinates.append(
            Coordinate(
                linear_name,
                _gain_block("linear", target),
                learner="linear",
                target=target,
                row_budget=budget,
                bounds=(-1.0, 1.0),
            )
        )
        bins_name = f"m{budget}.{target}.bins_vs_linear.contrast"
        coordinates.append(
            Coordinate(
                bins_name,
                "nonlinear",
                learner="bins",
                target=target,
                row_budget=budget,
                statistic="contrast",
                bounds=(-1.0, 1.0),
                parent=linear_name,
            )
        )
        pair_name = f"m{budget}.{target}.pairwise_vs_bins.contrast"
        coordinates.append(
            Coordinate(
                pair_name,
                "interaction",
                learner="pairwise",
                target=target,
                row_budget=budget,
                statistic="contrast",
                bounds=(-1.0, 1.0),
                parent=bins_name,
            )
        )
        coordinates.append(
            Coordinate(
                f"m{budget}.{target}.rff_vs_linear.contrast",
                "nonlinear",
                learner="rff",
                target=target,
                row_budget=budget,
                statistic="contrast",
                bounds=(-1.0, 1.0),
                parent=linear_name,
            )
        )
    return tuple(coordinates)


def _observation_coordinates(budget: int, names: tuple[str, ...]) -> tuple[Coordinate, ...]:
    concentration = {"feature_correlation_mean", "feature_correlation_max", "effective_rank_fraction"}
    fraction = {
        "feature_correlation_mean",
        "feature_correlation_max",
        "effective_rank_fraction",
        "feature_outlier_fraction",
        "feature_uniqueness_median",
        "feature_missingness_mean",
        "feature_missingness_max",
        "lower_tail_prevalence",
        "upper_tail_prevalence",
    }
    return tuple(
        Coordinate(
            f"m{budget}.observation.{name}",
            "feature_concentration" if name in concentration else "observation",
            row_budget=budget,
            statistic=name,
            bounds=(-1.0, 1.0),
            transform="fraction_to_signed" if name in fraction else "fixed_saturation",
        )
        for name in names
    )


def characterize(
    task: TuningTask,
    manifest: RowBudgetManifest,
    config: CharacterizationConfig,
) -> TaskCharacterization:
    fit_rows = list(manifest.probe_fit_indices)
    score_rows = list(manifest.probe_score_indices)
    feature_indices = list(manifest.feature_indices)
    feature_names = tuple(task.schema.names[index] for index in feature_indices)
    fit = task.probe_fit.X[fit_rows].to_numpy()[:, feature_indices].astype(np.float64)
    score = task.probe_score.X[score_rows].to_numpy()[:, feature_indices].astype(np.float64)
    targets = target_functionals(
        task.probe_fit.y[fit_rows],
        task.probe_score.y[score_rows],
        clip=config.target_clip,
        scale_epsilon=config.target_scale_epsilon,
    )
    maps = build_feature_maps(
        fit,
        score,
        feature_names,
        config.maps,
        seed_identity=(
            task.task_id,
            task.characterization_split_id,
            config.version,
            config.seed,
            config.repeat,
            manifest.row_budget,
        ),
    )
    gains: dict[str, np.ndarray] = {}
    dimensions: dict[str, int] = {}
    solvers: dict[str, str] = {}
    for feature_map in maps:
        result = fit_ridge_probe(feature_map.fit, feature_map.score, targets.fit, targets.score, config.ridge)
        gains[feature_map.name] = result.gains
        dimensions[feature_map.name] = result.dimension
        solvers[feature_map.name] = result.solver
    coordinates = list(_gain_coordinates(manifest.row_budget, config.representation, tuple(gains)))
    values: list[float] = []
    raw_values: list[float] = []
    for target_index in range(len(TARGET_NAMES)):
        linear = float(gains["linear"][target_index])
        bins = float(gains["bins"][target_index])
        pairwise = float(gains["pairwise"][target_index])
        rff = float(gains["rff"][target_index])
        represented = (
            (linear, bins, pairwise, rff)
            if config.representation == "raw"
            else (
                linear,
                (bins - linear) / 2.0,
                (pairwise - bins) / 2.0,
                (rff - linear) / 2.0,
            )
        )
        values.extend(represented)
        raw_values.extend(represented)
    missing = np.asarray(task.probe_fit_missing_rates)[feature_indices]
    observation_raw, observation_values = observation_statistics(fit, targets.fit, missing)
    if config.include_observation_coordinates:
        observation_coordinates = _observation_coordinates(manifest.row_budget, tuple(observation_raw))
        coordinates.extend(observation_coordinates)
        raw_values.extend(observation_raw.values())
        values.extend(observation_values.values())
    metadata: dict[str, Any] = {
        "schema_version": config.version,
        "representation": config.representation,
        "manifest_id": manifest.manifest_id,
        "row_budget": manifest.row_budget,
        "map_dimensions": dimensions,
        "ridge_solvers": solvers,
        "target_tail_prevalence": targets.tail_prevalence,
        "observation_raw": observation_raw,
        "log_n": float(np.log(manifest.row_budget)),
        "log_p": float(np.log(len(feature_indices))),
        "weight": manifest.weight,
    }
    valid = np.ones(len(coordinates), dtype=np.bool_)
    return TaskCharacterization(
        task.task_id,
        np.asarray(values),
        np.asarray(raw_values),
        valid,
        tuple(coordinates),
        metadata,
    )


def characterize_multiresolution(task: TuningTask, config: CharacterizationConfig) -> TaskCharacterization:
    results = [characterize(task, manifest, config) for manifest in build_row_budget_manifests(task, config)]
    coordinates = tuple(coordinate for result in results for coordinate in result.coordinates)
    schema = CharacterizationSchema(config.version, config.representation, coordinates)
    metadata = {
        "schema_version": schema.version,
        "representation": schema.representation,
        "budgets": [result.metadata for result in results],
    }
    return TaskCharacterization(
        task.task_id,
        np.concatenate([result.values for result in results]),
        np.concatenate([result.raw_values for result in results]),
        np.concatenate([result.valid for result in results]),
        schema.coordinates,
        metadata,
    )
