"""Hand-built characterization stability and decision study."""

import json
import re
import resource
import sqlite3
import sys
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
from benchmarks.studies.study_logging import configure_study_logging
from ebpfn.characterize import TaskCharacterization, characterize_multiresolution
from ebpfn.config import CharacterizationStudyConfig, PreprocessingConfig, RowBudgetConfig
from ebpfn.data import FeatureSchema, TaskPartition, TuningTask, content_hash
from ebpfn.data.preprocessing import fit_feature_transform
from ebpfn.data.rotations import infer_feature_schema
from ebpfn.utils import environment_provenance
from loguru import logger

MECHANISMS = (
    "null",
    "sparse_linear",
    "diffuse_linear",
    "threshold",
    "smooth",
    "interaction",
    "heteroskedastic",
    "heavy_tail",
    "rare_feature",
    "mixed",
)
_OPENML_CACHE_DIR = "data/raw/openml"


@dataclass(frozen=True)
class _OpenMLRegressionSpec:
    dataset_id: int
    target: str | None = None


@dataclass(frozen=True)
class _RegressionDataset:
    source_id: str
    target_name: str
    X: pl.DataFrame
    y: np.ndarray
    schema: FeatureSchema


_OPENML_REGRESSION_DATASETS: dict[str, _OpenMLRegressionSpec] = {
    "airfoil": _OpenMLRegressionSpec(43919, "pressure"),
    "concrete": _OpenMLRegressionSpec(4353, "Concrete compressive strength(MPa. megapascals)"),
    "energy": _OpenMLRegressionSpec(43338, "Y1"),
    "energy_cooling": _OpenMLRegressionSpec(43338, "Y2"),
    "energy_heating": _OpenMLRegressionSpec(43338, "Y1"),
    "kin8nm": _OpenMLRegressionSpec(189, "y"),
    "naval": _OpenMLRegressionSpec(44969, "gt_compressor_decay_state_coefficient"),
    "protein": _OpenMLRegressionSpec(42903, "RMSD"),
    "superconduct": _OpenMLRegressionSpec(43174, "critical_temp"),
    "yacht": _OpenMLRegressionSpec(42370, "Residuary.resistance"),
}


class _CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rows (
                    unit_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, unit_id: str) -> list[dict[str, Any]] | None:
        with self._connect() as connection:
            record = connection.execute("SELECT payload FROM rows WHERE unit_id = ?", (unit_id,)).fetchone()
        if record is None:
            return None
        payload = json.loads(record[0])
        if not isinstance(payload, list):
            raise TypeError(f"checkpoint payload for {unit_id!r} is not a row list")
        return payload

    def put(self, unit_id: str, rows: list[dict[str, Any]]) -> None:
        payload = json.dumps(rows, sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO rows(unit_id, payload) VALUES (?, ?)",
                (unit_id, payload),
            )


def _slug_value(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return slug or "none"


def _config_hash(config: CharacterizationStudyConfig) -> str:
    payload = config.model_dump(mode="json")
    payload.pop("output_root")
    return content_hash(payload, namespace="characterization-study-config-1")[:8]


def characterization_output_dir(config: CharacterizationStudyConfig, project_root: Path) -> Path:
    parts = (
        ("dataset", config.dataset),
        ("mode", config.mode.name),
        ("rows", config.max_rows),
        ("features", config.max_features),
        ("repeats", config.repeats),
        ("seed", config.characterization.seed),
    )
    stem = "__".join(f"{name}_{_slug_value(value)}" for name, value in parts)
    return project_root / config.output_root / f"{stem}__{_config_hash(config)}"


def _dataframe_to_polars(frame: Any) -> pl.DataFrame:
    return pl.DataFrame({str(name): frame[name].to_numpy() for name in frame.columns})


def _openml_spec(dataset: str) -> _OpenMLRegressionSpec | None:
    if dataset in _OPENML_REGRESSION_DATASETS:
        return _OPENML_REGRESSION_DATASETS[dataset]
    if dataset.startswith("openml_"):
        try:
            return _OpenMLRegressionSpec(int(dataset.removeprefix("openml_")))
        except ValueError:
            return None
    return None


@cache
def _load_openml_regression_dataset(dataset: str) -> _RegressionDataset:
    spec = _openml_spec(dataset)
    if spec is None:
        raise ValueError(f"unsupported characterization dataset {dataset!r}")
    import openml

    openml.config.set_root_cache_directory(_OPENML_CACHE_DIR)
    source = openml.datasets.get_dataset(spec.dataset_id)
    target = spec.target or source.default_target_attribute
    if target is None:
        _, _, _, names = source.get_data(dataset_format="dataframe")
        target = str(names[-1])
    features, target_values, _, _ = source.get_data(target=target, dataset_format="dataframe")
    X = _dataframe_to_polars(features)
    y = np.asarray(target_values, dtype=float)
    schema = infer_feature_schema(X, tuple(X.columns))
    return _RegressionDataset(f"openml-dataset-{spec.dataset_id}", str(target), X, y, schema)


def _split_rows(n_rows: int, config: CharacterizationStudyConfig, repeat: int) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 4:
        raise ValueError("real characterization datasets need at least four finite-target rows")
    n = min(config.max_rows, n_rows)
    score_rows = min(max(round(n * config.probe_score_fraction), 2), n - 2)
    fit_rows = n - score_rows
    rng = np.random.default_rng(
        np.random.SeedSequence([config.characterization.seed, repeat, *config.dataset.encode()])
    )
    selected = rng.permutation(n_rows)[:n]
    return selected[:fit_rows], selected[fit_rows:]


def _make_real_task(config: CharacterizationStudyConfig, repeat: int) -> TuningTask:
    dataset = _load_openml_regression_dataset(config.dataset)
    finite_target = np.flatnonzero(np.isfinite(dataset.y.astype(float, copy=False)))
    fit_indices, score_indices = _split_rows(len(finite_target), config, repeat)
    source_indices = finite_target[np.concatenate((fit_indices, score_indices))]
    frame = dataset.X[source_indices]
    y = dataset.y[source_indices].astype(float, copy=True)
    schema = dataset.schema
    numeric_names = tuple(name for name, kind in zip(schema.names, schema.kinds, strict=True) if kind != "categorical")
    if not numeric_names:
        raise ValueError(f"dataset {config.dataset!r} has no numeric or binary predictors")
    selected_names = numeric_names[: config.max_features]
    raw_schema = schema.select(selected_names)
    raw_frame = frame.select(selected_names)
    split = len(fit_indices)
    fit_frame = raw_frame[:split]
    score_frame = raw_frame[split:]
    preprocessing = PreprocessingConfig(
        max_features=config.max_features,
        clip=4.0,
        constant_atol=1e-12,
        constant_rtol=1e-12,
        scale_epsilon=1e-12,
        version="characterization-study-preprocess-1",
    )
    transform = fit_feature_transform(fit_frame, raw_schema, preprocessing)
    fit = TaskPartition(transform.apply(fit_frame), y[:split], source_indices[:split])
    score = TaskPartition(transform.apply(score_frame), y[split:], source_indices[split:])
    split_id = content_hash(
        config.dataset,
        repeat,
        tuple(int(index) for index in source_indices),
        config.max_features,
        config.probe_score_fraction,
        namespace="characterization-real-split-1",
    )
    return TuningTask(
        f"{config.dataset}-{repeat}",
        dataset.source_id,
        "regression",
        split_id,
        split_id,
        fit,
        score,
        transform.output_schema,
        transform.transform_id,
        transform.probe_fit_missing_rates,
    )


def _load_or_compute_rows(
    *,
    checkpoints: _CheckpointStore | None,
    unit_id: str,
    description: str,
    compute: Callable[[], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if checkpoints is None:
        return compute()
    try:
        rows = checkpoints.get(unit_id)
    except Exception:
        logger.warning(f"⚠️ checkpoint read failed: {description}")
        raise
    if rows is not None:
        logger.info(f"  ♻️ checkpoint hit: {description}")
        return rows
    try:
        rows = compute()
        checkpoints.put(unit_id, rows)
    except Exception:
        logger.warning(f"⚠️ failed: {description}")
        raise
    logger.success(f"✅ finished: {description}")
    return rows


def _is_synthetic_dataset(config: CharacterizationStudyConfig) -> bool:
    return config.dataset.startswith("synthetic_")


def _make_synthetic_task(
    mechanism: str,
    config: CharacterizationStudyConfig,
    repeat: int,
    *,
    strength: float = 1.0,
) -> TuningTask:
    n = config.max_rows
    rng = np.random.default_rng(
        np.random.SeedSequence([config.characterization.seed, repeat, MECHANISMS.index(mechanism)])
    )
    features = np.clip(rng.normal(size=(n, config.max_features)), -4.0, 4.0)
    noise = rng.normal(size=n)
    if mechanism == "null":
        target = noise
    elif mechanism == "sparse_linear":
        target = strength * 2.0 * features[:, 0] + noise
    elif mechanism == "diffuse_linear":
        target = strength * np.sum(features, axis=1) / np.sqrt(config.max_features) + noise
    elif mechanism == "threshold":
        target = strength * 2.5 * (features[:, 0] > 0.0) + noise
    elif mechanism == "smooth":
        target = strength * np.sin(2.0 * features[:, 0]) + noise * 0.5
    elif mechanism == "interaction":
        target = strength * features[:, 0] * features[:, 1] + noise * 0.5
    elif mechanism == "heteroskedastic":
        target = strength * features[:, 0] + noise * (0.2 + np.abs(features[:, 1]))
    elif mechanism == "heavy_tail":
        target = strength * features[:, 0] + rng.standard_t(3, size=n)
    elif mechanism == "rare_feature":
        features[:, -1] = rng.binomial(1, 0.015, size=n)
        target = strength * 3.0 * features[:, -1] + noise
    elif mechanism == "mixed":
        signal = features[:, 0] + (features[:, 1] > 0.0) + features[:, 2] * features[:, 3]
        target = strength * signal + noise * 0.5
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    names = tuple(f"x{index}" for index in range(config.max_features))
    schema = FeatureSchema(names, ("numeric",) * config.max_features)
    fit_ids = np.arange(config.n_probe_fit)
    score_ids = np.arange(config.n_probe_fit, n)
    fit = TaskPartition(
        pl.DataFrame(features[: config.n_probe_fit], schema=names),
        target[: config.n_probe_fit],
        fit_ids,
    )
    score = TaskPartition(
        pl.DataFrame(features[config.n_probe_fit :], schema=names),
        target[config.n_probe_fit :],
        score_ids,
    )
    return TuningTask(
        f"{mechanism}-{repeat}",
        "hand-built",
        "regression",
        "study-outer",
        f"study-inner-{repeat}",
        fit,
        score,
        schema,
        "study-preprocessed",
        (0.0,) * config.max_features,
    )


def make_task(label: str, config: CharacterizationStudyConfig, repeat: int, *, strength: float = 1.0) -> TuningTask:
    if _is_synthetic_dataset(config):
        return _make_synthetic_task(label, config, repeat, strength=strength)
    if label != "real":
        raise ValueError(f"real dataset {config.dataset!r} does not provide label {label!r}")
    if strength != 1.0:
        raise ValueError("response-strength sweeps are only defined for synthetic datasets")
    return _make_real_task(config, repeat)


def _process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _measure(call: Callable[[], TaskCharacterization]) -> tuple[TaskCharacterization, float, int, int]:
    # ru_maxrss is a whole-process high-water mark that never resets between calls, so the raw
    # reading cannot distinguish a small call from a large one made earlier in the same run.
    # Report the watermark's growth across this call instead of its absolute value.
    baseline_rss = _process_peak_rss_bytes()
    tracemalloc.start()
    started = perf_counter()
    try:
        result = call()
        runtime = perf_counter() - started
        _, traced_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_growth = max(0, _process_peak_rss_bytes() - baseline_rss)
    return result, runtime, traced_peak, rss_growth


def _result_rows(
    result: TaskCharacterization,
    *,
    dataset: str,
    mechanism: str,
    repeat: int,
    lambda_: float,
    policy: str,
    runtime: float,
    traced_peak: int,
    process_peak_rss_growth: int,
    strength: float = 1.0,
) -> list[dict[str, Any]]:
    rows = [
        {
            "dataset": dataset,
            "mechanism": mechanism,
            "strength": strength,
            "repeat": repeat,
            "lambda": lambda_,
            "policy": policy,
            "representation": "raw",
            "coordinate": coordinate.name,
            "block": coordinate.block,
            "learner": coordinate.learner,
            "target": coordinate.target,
            "statistic": coordinate.statistic,
            "row_budget": coordinate.row_budget,
            "value": float(value),
            "raw_value": float(raw),
            "valid": bool(valid),
            "runtime_seconds": runtime,
            "tracemalloc_peak_bytes": traced_peak,
            "process_peak_rss_growth_bytes": process_peak_rss_growth,
        }
        for coordinate, value, raw, valid in zip(
            result.coordinates, result.values, result.raw_values, result.valid, strict=True
        )
    ]
    for budget in result.metadata["budgets"]:
        for name, raw_value in budget["observation_raw"].items():
            rows.append(
                {
                    "mechanism": mechanism,
                    "dataset": dataset,
                    "strength": strength,
                    "repeat": repeat,
                    "lambda": lambda_,
                    "policy": policy,
                    "representation": "diagnostic",
                    "coordinate": f"m{budget['row_budget']}.diagnostic.{name}",
                    "block": "observation_diagnostic",
                    "learner": None,
                    "target": None,
                    "statistic": name,
                    "row_budget": budget["row_budget"],
                    "value": float(raw_value),
                    "raw_value": float(raw_value),
                    "valid": bool(np.isfinite(raw_value)),
                    "runtime_seconds": runtime,
                    "tracemalloc_peak_bytes": traced_peak,
                    "process_peak_rss_growth_bytes": process_peak_rss_growth,
                }
            )
    return rows


def _contrast_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gains = {
        (row["row_budget"], row["target"], row["learner"]): float(row["value"])
        for row in raw_rows
        if row["statistic"] == "gain"
    }
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if row["representation"] != "raw":
            continue
        contrast = dict(row)
        contrast["representation"] = "contrast"
        if row["statistic"] != "gain":
            rows.append(contrast)
            continue
        budget = row["row_budget"]
        target = row["target"]
        learner = row["learner"]
        linear = gains[(budget, target, "linear")]
        if learner == "linear":
            rows.append(contrast)
        elif learner == "bins":
            contrast["coordinate"] = str(row["coordinate"]).replace("bins.gain", "bins_vs_linear.contrast")
            contrast["statistic"] = "contrast"
            contrast["value"] = contrast["raw_value"] = (float(row["value"]) - linear) / 2.0
            rows.append(contrast)
        elif learner == "pairwise":
            bins = gains[(budget, target, "bins")]
            contrast["coordinate"] = str(row["coordinate"]).replace("pairwise.gain", "pairwise_vs_bins.contrast")
            contrast["statistic"] = "contrast"
            contrast["value"] = contrast["raw_value"] = (float(row["value"]) - bins) / 2.0
            rows.append(contrast)
        elif learner == "rff":
            contrast["coordinate"] = str(row["coordinate"]).replace("rff.gain", "rff_vs_linear.contrast")
            contrast["statistic"] = "contrast"
            contrast["value"] = contrast["raw_value"] = (float(row["value"]) - linear) / 2.0
            rows.append(contrast)
    return rows


def _append_both_representations(rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> None:
    rows.extend(result_rows)
    rows.extend(_contrast_rows(result_rows))


def _study_mechanisms(config: CharacterizationStudyConfig) -> tuple[str, ...]:
    if config.dataset.startswith("synthetic_") and config.dataset != "synthetic_handbuilt":
        mechanism = config.dataset.removeprefix("synthetic_")
        if mechanism not in MECHANISMS:
            raise ValueError(f"unknown synthetic characterization dataset {config.dataset!r}")
        return (mechanism,)
    if not _is_synthetic_dataset(config):
        if _openml_spec(config.dataset) is None:
            raise ValueError(f"unsupported characterization dataset {config.dataset!r}")
        return ("real",)
    if config.mode.name == "audit":
        return MECHANISMS
    return ("null", "sparse_linear", "threshold", "interaction", "mixed")


def _row_policy_name(policy: RowBudgetConfig) -> str:
    return f"{policy.spacing}/{policy.weight}/{policy.feature_view}"


def _row_policy_candidates(config: CharacterizationStudyConfig) -> tuple[RowBudgetConfig, ...]:
    selected = RowBudgetConfig(minimum=256, spacing="geometric", weight="uniform", feature_view="frozen")
    sqrt_challenger = RowBudgetConfig(minimum=256, spacing="sqrt", weight="uniform", feature_view="frozen")
    local_challenger = RowBudgetConfig(minimum=256, spacing="geometric", weight="uniform", feature_view="local")
    if config.mode.name == "audit":
        return (selected, sqrt_challenger, local_challenger)
    return (selected, local_challenger)


def _characterization_rows(
    *,
    config: CharacterizationStudyConfig,
    candidate: Any,
    mechanism: str,
    repeat: int,
    lambda_: float,
    policy: str,
    strength: float = 1.0,
) -> list[dict[str, Any]]:
    result, runtime, traced_peak, process_peak_rss = _measure(
        lambda: characterize_multiresolution(make_task(mechanism, config, repeat, strength=strength), candidate)
    )
    result_rows = _result_rows(
        result,
        dataset=config.dataset,
        mechanism=mechanism,
        repeat=repeat,
        lambda_=lambda_,
        policy=policy,
        runtime=runtime,
        traced_peak=traced_peak,
        process_peak_rss_growth=process_peak_rss,
        strength=strength,
    )
    rows: list[dict[str, Any]] = []
    _append_both_representations(rows, result_rows)
    return rows


def _p_complexity_row(
    config: CharacterizationStudyConfig,
    grid_config: CharacterizationStudyConfig,
    feature_count: int,
) -> dict[str, Any]:
    label = "sparse_linear" if _is_synthetic_dataset(config) else "real"
    result, runtime, traced_peak, process_peak_rss_growth = _measure(
        lambda: characterize_multiresolution(
            make_task(label, grid_config, 0),
            config.characterization.model_copy(update={"include_observation_coordinates": False}),
        )
    )
    return {
        "p": feature_count,
        "runtime_seconds": runtime,
        "tracemalloc_peak_bytes": traced_peak,
        "process_peak_rss_growth_bytes": process_peak_rss_growth,
        "budgets": json.dumps([item["map_dimensions"] for item in result.metadata["budgets"]], sort_keys=True),
    }


def _rank_stability(table: pl.DataFrame, policy: str) -> list[dict[str, Any]]:
    selected = table.filter((pl.col("policy") == policy) & pl.col("representation").is_in(["raw", "contrast"]))
    summaries: list[dict[str, Any]] = []
    for mechanism in selected["mechanism"].unique().sort():
        for representation in ("raw", "contrast"):
            group = selected.filter((pl.col("mechanism") == mechanism) & (pl.col("representation") == representation))
            repeats = group["repeat"].unique().sort().to_list()
            by_repeat = {
                repeat: dict(
                    zip(
                        group.filter(pl.col("repeat") == repeat)["coordinate"].to_list(),
                        group.filter(pl.col("repeat") == repeat)["value"].to_list(),
                        strict=True,
                    )
                )
                for repeat in repeats
            }
            correlations: list[float] = []
            for left_index, left_repeat in enumerate(repeats):
                for right_repeat in repeats[left_index + 1 :]:
                    names = sorted(set(by_repeat[left_repeat]) & set(by_repeat[right_repeat]))
                    if len(names) < 2:
                        continue
                    left = np.asarray([by_repeat[left_repeat][name] for name in names])
                    right = np.asarray([by_repeat[right_repeat][name] for name in names])
                    left_rank = np.argsort(np.argsort(left))
                    right_rank = np.argsort(np.argsort(right))
                    correlations.append(float(np.corrcoef(left_rank, right_rank)[0, 1]))
            summaries.append(
                {
                    "mechanism": mechanism,
                    "representation": representation,
                    "median_spearman": None if not correlations else float(np.median(correlations)),
                    "pair_count": len(correlations),
                }
            )
    return summaries


def _coordinate_stability(table: pl.DataFrame, policy: str) -> list[dict[str, Any]]:
    selected = table.filter(
        (pl.col("policy") == policy) & pl.col("representation").is_in(["raw", "contrast"])
    ).with_columns(pl.col("value").mean().over("mechanism", "representation", "coordinate").alias("_mean"))
    return (
        selected.group_by("mechanism", "representation", "coordinate")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std(ddof=0).alias("std"),
            ((pl.col("value") * pl.col("_mean")) >= 0.0).mean().alias("sign_stability"),
        )
        .sort("mechanism", "representation", "coordinate")
        .to_dicts()
    )


def _structure_diagnostics(table: pl.DataFrame, policy: str) -> dict[str, Any]:
    selected = table.filter((pl.col("policy") == policy) & pl.col("representation").is_in(["raw", "contrast"]))
    block_summary = (
        selected.group_by("representation", "mechanism", "block")
        .agg(pl.col("value").abs().mean().alias("mean_absolute"))
        .with_columns(
            (pl.col("mean_absolute") / pl.col("mean_absolute").sum().over("representation", "mechanism")).alias(
                "absolute_share"
            )
        )
        .sort("representation", "mechanism", "block")
    )
    dominant_blocks = (
        block_summary.sort("absolute_share", descending=True)
        .group_by("representation", "mechanism", maintain_order=True)
        .first()
        .select("representation", "mechanism", pl.col("block").alias("dominant_block"), "absolute_share")
        .sort("representation", "mechanism")
    )
    redundancy: list[dict[str, Any]] = []
    for representation in ("raw", "contrast"):
        group = selected.filter(pl.col("representation") == representation)
        samples = sorted(set(zip(group["mechanism"].to_list(), group["repeat"].to_list(), strict=True)))
        coordinates = sorted(group["coordinate"].unique().to_list())
        if len(samples) < 2 or len(coordinates) < 2:
            redundancy.append(
                {
                    "representation": representation,
                    "median_absolute_correlation": None,
                    "maximum_absolute_correlation": None,
                    "top_pairs": [],
                }
            )
            continue
        lookup = {
            (row["mechanism"], row["repeat"], row["coordinate"]): row["value"]
            for row in group.select("mechanism", "repeat", "coordinate", "value").to_dicts()
        }
        matrix = np.asarray(
            [[lookup[(mechanism, repeat, coordinate)] for coordinate in coordinates] for mechanism, repeat in samples]
        )
        correlation = np.nan_to_num(np.corrcoef(matrix, rowvar=False))
        pairs = sorted(
            (
                (abs(float(correlation[left, right])), coordinates[left], coordinates[right])
                for left in range(len(coordinates))
                for right in range(left + 1, len(coordinates))
            ),
            reverse=True,
        )
        redundancy.append(
            {
                "representation": representation,
                "median_absolute_correlation": float(np.median([pair[0] for pair in pairs])),
                "maximum_absolute_correlation": pairs[0][0],
                "top_pairs": [
                    {"absolute_correlation": value, "left": left, "right": right} for value, left, right in pairs[:20]
                ],
            }
        )
    return {
        "block_summary": block_summary.to_dicts(),
        "dominant_blocks": dominant_blocks.to_dicts(),
        "coordinate_redundancy": redundancy,
    }


def derive_study_status(mode: str, checks: dict[str, bool]) -> tuple[str, list[str]]:
    if mode == "fast":
        return "provisional", []
    missing_checks = {"representative_real_task_repeats"}
    failed = sorted(name for name, passed in checks.items() if not passed and name not in missing_checks)
    missing = sorted(name for name, passed in checks.items() if not passed and name in missing_checks)
    if failed:
        return "failed", [f"failed:{name}" for name in failed]
    if missing:
        return "incomplete", [f"missing:{name}" for name in missing]
    return "frozen", []


def run_study(
    config: CharacterizationStudyConfig,
    *,
    checkpoint_path: Path | None = None,
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = config.characterization
    mechanisms = _study_mechanisms(config)
    ridge_candidates = config.ridge_candidates if config.mode.name == "audit" else (base.ridge.lambda_,)
    checkpoints = None if checkpoint_path is None else _CheckpointStore(checkpoint_path)
    logger.info(
        f"🧭 characterization study | dataset={config.dataset} | mode={config.mode.name} | "
        f"tasks={len(mechanisms)} | ridge_candidates={len(ridge_candidates)}"
    )
    logger.info(f"🧪 ridge sweep start | {len(ridge_candidates)} candidate(s)")
    for lambda_ in ridge_candidates:
        candidate = base.model_copy(
            update={
                "ridge": base.ridge.model_copy(update={"lambda_": lambda_}),
                "include_observation_coordinates": False,
            }
        )
        for repeat in range(config.repeats):
            candidate = candidate.model_copy(update={"repeat": repeat})
            for mechanism in mechanisms:
                rows.extend(
                    _load_or_compute_rows(
                        checkpoints=checkpoints,
                        unit_id=f"ridge/{lambda_}/{repeat}/{mechanism}",
                        description=f"ridge lambda={lambda_} repeat={repeat} mechanism={mechanism}",
                        compute=lambda candidate=candidate, mechanism=mechanism, repeat=repeat, lambda_=lambda_: (
                            _characterization_rows(
                                config=config,
                                candidate=candidate,
                                mechanism=mechanism,
                                repeat=repeat,
                                lambda_=lambda_,
                                policy="ridge",
                            )
                        ),
                    )
                )
    logger.success("✅ ridge sweep complete")
    policies = _row_policy_candidates(config)
    logger.info(f"🧮 row-policy sweep start | {[_row_policy_name(policy) for policy in policies]}")
    for row_policy in policies:
        policy_name = _row_policy_name(row_policy)
        candidate = base.model_copy(update={"row_budgets": row_policy, "include_observation_coordinates": False})
        for repeat in range(config.repeats):
            candidate = candidate.model_copy(update={"repeat": repeat})
            for mechanism in mechanisms:
                rows.extend(
                    _load_or_compute_rows(
                        checkpoints=checkpoints,
                        unit_id=f"row_policy/{policy_name}/{repeat}/{mechanism}",
                        description=f"row_policy={policy_name} repeat={repeat} mechanism={mechanism}",
                        compute=lambda candidate=candidate, mechanism=mechanism, repeat=repeat, policy_name=policy_name: (
                            _characterization_rows(
                                config=config,
                                candidate=candidate,
                                mechanism=mechanism,
                                repeat=repeat,
                                lambda_=base.ridge.lambda_,
                                policy=policy_name,
                            )
                        ),
                    )
                )
    logger.success("✅ row-policy sweep complete")
    table = pl.DataFrame(rows)
    gain_rows = table.filter(
        (pl.col("policy") == "ridge") & (pl.col("representation") == "raw") & (pl.col("statistic") == "gain")
    )
    summaries = (
        gain_rows.group_by("lambda", "mechanism")
        .agg(pl.col("value").mean().alias("mean"), pl.col("value").std(ddof=0).alias("std"))
        .sort("lambda", "mechanism")
    )
    lambda_scores: list[tuple[bool, float, float]] = []
    has_null_reference = "null" in mechanisms
    for lambda_ in ridge_candidates:
        selected = summaries.filter(pl.col("lambda") == lambda_)
        if has_null_reference:
            null_mean = float(np.mean(selected.filter(pl.col("mechanism") == "null")["mean"].to_numpy()))
            signals = selected.filter(pl.col("mechanism") != "null")
            ratios = np.abs(signals["mean"].to_numpy() - null_mean) / np.maximum(signals["std"].to_numpy(), 1e-8)
            lambda_scores.append((null_mean <= 0.0, float(np.median(ratios)), lambda_))
        else:
            ratios = np.abs(selected["mean"].to_numpy()) / np.maximum(selected["std"].to_numpy(), 1e-8)
            lambda_scores.append((True, float(np.median(ratios)), lambda_))
    eligible = [item for item in lambda_scores if item[0]]
    diagnostic_lambda = max(eligible or lambda_scores, key=lambda item: (item[1], item[2]))[2]
    selected_lambda = diagnostic_lambda if eligible else None
    policy_rows = table.filter((pl.col("policy") != "ridge") & (pl.col("representation") == "raw"))
    policy_summary = (
        policy_rows.group_by("policy", "coordinate")
        .agg(pl.col("value").std(ddof=0).alias("std"), pl.col("runtime_seconds").mean().alias("runtime"))
        .group_by("policy")
        .agg(pl.col("std").median().alias("median_std"), pl.col("runtime").mean().alias("runtime"))
        .sort("median_std", "runtime", "policy")
    )
    minimum_std = float(np.min(policy_summary["median_std"].to_numpy()))
    competitive = set(policy_summary.filter(pl.col("median_std") <= minimum_std + 1e-8).get_column("policy").to_list())
    preference = (
        "geometric/uniform/frozen",
        "sqrt/uniform/frozen",
        "geometric/uniform/local",
        "sqrt/uniform/local",
    )
    selected_policy = next(policy for policy in preference if policy in competitive)
    selected_row_policy = next(policy for policy in policies if _row_policy_name(policy) == selected_policy)
    response_rows: list[dict[str, Any]] = []
    response_strengths = (0.0, 0.5, 1.0, 2.0) if config.mode.name == "audit" else (0.0, 1.0)
    response_repeats = range(config.repeats) if config.mode.name == "audit" else range(1)
    response_candidate = base.model_copy(
        update={
            "ridge": base.ridge.model_copy(update={"lambda_": diagnostic_lambda}),
            "row_budgets": selected_row_policy,
            "include_observation_coordinates": False,
        }
    )
    if _is_synthetic_dataset(config):
        logger.info("📈 response curves start")
        for repeat in response_repeats:
            candidate = response_candidate.model_copy(update={"repeat": repeat})
            for mechanism in ("sparse_linear", "threshold", "interaction"):
                for strength in response_strengths:
                    response_rows.extend(
                        _load_or_compute_rows(
                            checkpoints=checkpoints,
                            unit_id=f"response/{repeat}/{mechanism}/{strength}",
                            description=f"response repeat={repeat} mechanism={mechanism} strength={strength}",
                            compute=lambda candidate=candidate, mechanism=mechanism, repeat=repeat, strength=strength: (
                                _characterization_rows(
                                    config=config,
                                    candidate=candidate,
                                    mechanism=mechanism,
                                    repeat=repeat,
                                    lambda_=diagnostic_lambda,
                                    policy="response",
                                    strength=strength,
                                )
                            ),
                        )
                    )
        logger.success("✅ response curves complete")
    else:
        logger.info("⏭️ response curves skipped | real dataset")
    observation_rows: list[dict[str, Any]] = []
    observation_candidate = base.model_copy(
        update={
            "ridge": base.ridge.model_copy(update={"lambda_": diagnostic_lambda}),
            "row_budgets": selected_row_policy,
            "include_observation_coordinates": True,
        }
    )
    logger.info("🔎 observation diagnostics start")
    for repeat in range(config.repeats):
        candidate = observation_candidate.model_copy(update={"repeat": repeat})
        for mechanism in mechanisms:
            observation_rows.extend(
                _load_or_compute_rows(
                    checkpoints=checkpoints,
                    unit_id=f"observation_on/{repeat}/{mechanism}",
                    description=f"observation=on repeat={repeat} mechanism={mechanism}",
                    compute=lambda candidate=candidate, mechanism=mechanism, repeat=repeat: _characterization_rows(
                        config=config,
                        candidate=candidate,
                        mechanism=mechanism,
                        repeat=repeat,
                        lambda_=diagnostic_lambda,
                        policy="observation/on",
                    ),
                )
            )
    logger.success("✅ observation diagnostics complete")
    extra_frames = [table]
    if response_rows:
        extra_frames.append(pl.DataFrame(response_rows))
    if observation_rows:
        extra_frames.append(pl.DataFrame(observation_rows))
    table = pl.concat(extra_frames, how="vertical_relaxed")
    expected_learners = {"sparse_linear": "linear", "threshold": "bins", "interaction": "pairwise"}
    if response_rows:
        response_summary = (
            table.filter(
                (pl.col("policy") == "response")
                & (pl.col("target") == "location")
                & pl.struct("mechanism", "learner").map_elements(
                    lambda item: expected_learners.get(item["mechanism"]) == item["learner"],
                    return_dtype=pl.Boolean,
                )
            )
            .group_by("representation", "mechanism", "strength", "row_budget", "statistic")
            .agg(pl.col("value").mean().alias("mean"), pl.col("value").std(ddof=0).alias("std"))
            .sort("representation", "mechanism", "strength", "row_budget")
            .to_dicts()
        )
    else:
        response_summary = []
    coordinate_stability = _coordinate_stability(table, selected_policy)
    rank_stability = _rank_stability(table, selected_policy)
    structure_diagnostics = _structure_diagnostics(table, selected_policy)
    observation_summary = (
        table.filter(pl.col("policy").is_in([selected_policy, "observation/on"]))
        .group_by("policy", "representation", "block")
        .agg(
            pl.col("value").abs().mean().alias("mean_absolute"),
            pl.col("runtime_seconds").mean().alias("mean_runtime_seconds"),
            pl.len().alias("rows"),
        )
        .sort("policy", "representation", "block")
        .to_dicts()
    )
    p_complexity: list[dict[str, Any]] = []
    p_grid = (2, 8, 32, 100) if config.mode.name == "audit" else (2, config.max_features)
    logger.info(f"📏 p-complexity start | {p_grid}")
    for feature_count in p_grid:
        grid_config = config.model_copy(update={"max_features": feature_count})
        p_complexity.extend(
            _load_or_compute_rows(
                checkpoints=checkpoints,
                unit_id=f"p_complexity/{feature_count}",
                description=f"p_complexity p={feature_count}",
                compute=lambda grid_config=grid_config, feature_count=feature_count: [
                    _p_complexity_row(config, grid_config, feature_count)
                ],
            )
        )
    logger.success("✅ p-complexity complete")
    p_complexity = [
        {**row, "budgets": json.loads(row["budgets"]) if isinstance(row["budgets"], str) else row["budgets"]}
        for row in p_complexity
    ]
    raw_rows = table.filter(pl.col("representation") == "raw")
    checks = {
        "five_complete_repeats": config.repeats >= 5,
        "all_vectors_valid": bool(table["valid"].all()),
        "selected_lambda_null_nonpositive": bool(eligible) if has_null_reference else True,
        "complete_domain_feasible_within_scope": (config.max_rows >= config.applicability_max_rows),
        "feature_width_feasible_within_scope": max(p_grid) >= config.applicability_max_features,
        "memory_recorded": bool(
            (raw_rows["tracemalloc_peak_bytes"] > 0).all()
            and (raw_rows["process_peak_rss_growth_bytes"] >= 0).all()
            and any(row["process_peak_rss_growth_bytes"] > 0 for row in p_complexity)
        ),
        "stability_reported": bool(coordinate_stability and any(item["pair_count"] > 0 for item in rank_stability)),
        "response_curves_reported": bool(response_summary) if _is_synthetic_dataset(config) else True,
        "structure_diagnostics_reported": bool(
            structure_diagnostics["block_summary"] and structure_diagnostics["coordinate_redundancy"]
        ),
        # Real-arm evidence: this run must BE a real-dataset run with >= 5 repeats. Synthetic
        # runs deliberately fail this (their job is the null/response arm via the null mechanism),
        # so a synth audit reads "incomplete" until paired with a real audit run. The two decision
        # logs together are the real+synth confirmation (plans/v1/02_characterization.md, "Fast and
        # audit modes").
        "representative_real_task_repeats": (not _is_synthetic_dataset(config)) and config.repeats >= 5,
    }
    status, blocking_reasons = derive_study_status(config.mode.name, checks)
    decision = {
        "status": status,
        "ridge_lambda": selected_lambda,
        "row_budget_policy": selected_policy,
        "selection_rule": "valid/null-nonpositive, then response-to-repeat-SD; D2 by median repeat SD with preregistered simplicity tie-break",
        "repeats": config.repeats,
        "owner": config.decision_owner,
        "date": config.decision_date,
        "gate_checks": checks,
        "blocking_reasons": blocking_reasons,
        "applicability": {
            "max_rows": config.applicability_max_rows,
            "max_features": config.applicability_max_features,
            "full_domain_feasibility": "validated only within these construction-study bounds",
        },
    }
    evidence = {
        "lambda_scores": [
            {"null_nonpositive": valid, "response_noise_score": score, "lambda": lambda_}
            for valid, score, lambda_ in lambda_scores
        ],
        "policy_summary": policy_summary.to_dicts(),
        "p_complexity": p_complexity,
        "coordinate_stability": coordinate_stability,
        "rank_stability": rank_stability,
        "response_summary": response_summary,
        "observation_summary": observation_summary,
        "structure_diagnostics": structure_diagnostics,
    }
    return table, decision, evidence


def schema_payload(result: TaskCharacterization) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "coordinates": [asdict(coordinate) for coordinate in result.coordinates],
        "metadata": result.metadata,
    }


def _markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        if value == 0.0:
            return "0"
        if abs(value) >= 1000.0 or abs(value) < 0.001:
            return f"{value:.3g}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(rows: list[dict[str, Any]], columns: tuple[tuple[str, str], ...]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_markdown_value(row[key]) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _top_coordinate_summary(table: pl.DataFrame, selected_policy: str, *, per_group: int = 3) -> list[dict[str, Any]]:
    selected = table.filter(
        (pl.col("policy") == selected_policy) & pl.col("representation").is_in(["raw", "contrast"]) & pl.col("valid")
    )
    if selected.is_empty():
        return []
    summaries = (
        selected.group_by("mechanism", "representation", "coordinate", "block", "learner", "target", "statistic")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").abs().mean().alias("mean_absolute"),
            pl.col("value").std(ddof=0).alias("std"),
        )
        .to_dicts()
    )
    summaries.sort(
        key=lambda row: (
            str(row["mechanism"]),
            str(row["representation"]),
            -float(row["mean_absolute"] or 0.0),
            str(row["coordinate"]),
        )
    )
    counts: dict[tuple[str, str], int] = {}
    rows: list[dict[str, Any]] = []
    for row in summaries:
        key = (str(row["mechanism"]), str(row["representation"]))
        seen = counts[key] if key in counts else 0
        if seen >= per_group:
            continue
        counts[key] = seen + 1
        rows.append(row)
    return rows


# Fingerprint presentation knobs. These couple to the probe design: _MOMENT_ORDER is
# display-only (unknown targets sort to the end, so new moments degrade gracefully), but
# _NONLINEAR_LEARNERS is load-bearing - a renamed/added learner family must be listed here
# or it silently drops out of the linear->nonlinear gap. Revisit both if the probe set changes.
_MOMENT_ORDER = ("location", "scale_abs", "scale_square", "lower_tail", "upper_tail")
_NONLINEAR_LEARNERS = ("bins", "rff", "pairwise")


def _fingerprint_rows(table: pl.DataFrame, selected_policy: str) -> list[dict[str, Any]]:
    """Per-moment recoverability: gain by target x learner at the strongest, widest probe.

    This is the dataset-facing readout - how much of each conditional moment (mean,
    spread, tails) each learner family recovers - plus the linear->nonlinear gap.
    """
    gains = table.filter(
        (pl.col("policy") == selected_policy)
        & (pl.col("representation") == "raw")
        & (pl.col("statistic") == "gain")
        & pl.col("valid")
    )
    if gains.is_empty():
        return []
    gains = gains.filter(pl.col("strength") == gains.get_column("strength").max())
    gains = gains.filter(pl.col("row_budget") == gains.get_column("row_budget").max())
    means = gains.group_by("mechanism", "target", "learner").agg(pl.col("value").mean().alias("gain")).to_dicts()
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in means:
        grouped.setdefault((str(row["mechanism"]), str(row["target"])), {})[str(row["learner"])] = float(row["gain"])

    def moment_index(target: str) -> int:
        return _MOMENT_ORDER.index(target) if target in _MOMENT_ORDER else len(_MOMENT_ORDER)

    rows: list[dict[str, Any]] = []
    for mechanism, target in sorted(grouped, key=lambda key: (key[0], moment_index(key[1]), key[1])):
        learner_gains = grouped[(mechanism, target)]
        nonlinear = [learner_gains[name] for name in _NONLINEAR_LEARNERS if name in learner_gains]
        best_nonlinear = max(nonlinear) if nonlinear else None
        linear = learner_gains.get("linear")
        gap = best_nonlinear - linear if (best_nonlinear is not None and linear is not None) else None
        rows.append(
            {
                "mechanism": mechanism,
                "target": target,
                "linear": linear,
                "bins": learner_gains.get("bins"),
                "rff": learner_gains.get("rff"),
                "pairwise": learner_gains.get("pairwise"),
                "best_nonlinear": best_nonlinear,
                "gap": gap,
            }
        )
    return rows


def _fingerprint_headline(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    mechanisms = [row["mechanism"] for row in rows]
    primary = "real" if "real" in mechanisms else mechanisms[0]
    subset = [row for row in rows if row["mechanism"] == primary and row["best_nonlinear"] is not None]
    if not subset:
        return []
    best = max(subset, key=lambda row: row["best_nonlinear"])
    lines = [
        f"Fingerprint ({primary}): best-recovered moment **{best['target']}** "
        f"(gain {_markdown_value(best['best_nonlinear'])})."
    ]
    gap_rows = [row for row in subset if row["gap"] is not None and row["gap"] > 0.0]
    if gap_rows:
        widest = max(gap_rows, key=lambda row: row["gap"])
        lines.append(
            f"Widest linear->nonlinear gap: **{widest['target']}** "
            f"(linear {_markdown_value(widest['linear'])} -> nonlinear {_markdown_value(widest['best_nonlinear'])}, "
            f"+{_markdown_value(widest['gap'])})."
        )
    return lines


def _response_summary_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(evidence["response_summary"])
    if not rows:
        return []
    max_strength = max(float(row["strength"]) for row in rows)
    selected = [
        row
        for row in rows
        if float(row["strength"]) == max_strength
        and row["representation"] in {"raw", "contrast"}
        and row["statistic"] in {"gain", "contrast"}
    ]
    selected.sort(key=lambda row: (str(row["representation"]), str(row["mechanism"]), int(row["row_budget"])))
    return selected


def build_study_summary_markdown(
    config: CharacterizationStudyConfig,
    table: pl.DataFrame,
    decision: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    selected_policy = str(decision["row_budget_policy"])
    runtimes = (
        table.select("policy", "mechanism", "repeat", "lambda", "strength", "runtime_seconds")
        .unique()
        .get_column("runtime_seconds")
        .to_numpy()
    )
    valid_count = table.filter(pl.col("valid")).height
    total_count = table.height
    valid_pct = 100.0 * valid_count / total_count if total_count else 0.0
    median_runtime = float(np.median(runtimes)) if runtimes.size else None
    dominant_blocks = list(evidence["structure_diagnostics"]["dominant_blocks"])
    raw_dominant = [row for row in dominant_blocks if row["representation"] == "raw"]
    rank_values = [
        row["median_spearman"]
        for row in evidence["rank_stability"]
        if row["median_spearman"] is not None and row["pair_count"] > 0
    ]
    fingerprint = _fingerprint_rows(table, selected_policy)

    # Headline: dataset-facing answer first (what is this / did it work), decision line last.
    headline = [
        f"Status: **{decision['status']}**.",
        f"Dataset: **{config.dataset}** ({config.mode.name} mode, "
        f"{config.repeats} repeat(s), <= {config.max_rows} rows x {config.max_features} features).",
        f"Coordinates valid: **{_markdown_value(valid_pct)}%** ({valid_count}/{total_count}).",
    ]
    headline.extend(_fingerprint_headline(fingerprint))
    if raw_dominant:
        top = max(raw_dominant, key=lambda row: float(row["absolute_share"] or 0.0))
        headline.append(
            f"Dominant block (raw): **{top['dominant_block']}** on **{top['mechanism']}** "
            f"({_markdown_value(top['absolute_share'])} of absolute mass)."
        )
    decision_line = (
        f"Decision: ridge lambda **{_markdown_value(decision['ridge_lambda'])}**, " f"row policy **{selected_policy}**"
    )
    if median_runtime is not None:
        decision_line += f", median call **{_markdown_value(median_runtime)}s**"
    headline.append(decision_line + ".")
    if rank_values:
        headline.append(
            "Rank-stability range: "
            f"**{_markdown_value(float(np.min(rank_values)))}** to "
            f"**{_markdown_value(float(np.max(rank_values)))}**."
        )

    # Structurally-empty sections get a reason, not a bare "_No rows._".
    response_rows = _response_summary_rows(evidence)
    response_note = (
        None
        if response_rows
        else "_Requires a signal-strength sweep; real-dataset runs fix strength at 1.0, so no curve is emitted._"
    )
    rank_rows = list(evidence["rank_stability"])
    rank_note = (
        None
        if rank_values
        else f"_Requires >= 2 complete repeats for a rank correlation; this run had {config.repeats}._"
    )

    run_config = [
        {"field": "dataset", "value": config.dataset},
        {"field": "mode", "value": config.mode.name},
        {"field": "repeats", "value": config.repeats},
        {"field": "max_rows", "value": config.max_rows},
        {"field": "max_features", "value": config.max_features},
        {"field": "probe_score_fraction", "value": config.probe_score_fraction},
    ]
    gate_checks = [{"check": name, "passed": passed} for name, passed in sorted(decision["gate_checks"].items())]
    p_complexity = [
        {
            "p": row["p"],
            "runtime_seconds": row["runtime_seconds"],
            "tracemalloc_mb": float(row["tracemalloc_peak_bytes"]) / 1_000_000.0,
            "rss_growth_mb": float(row["process_peak_rss_growth_bytes"]) / 1_000_000.0,
        }
        for row in evidence["p_complexity"]
    ]
    sections = [
        "# Characterization Study Summary",
        "",
        "## Headline",
        *[f"- {item}" for item in headline],
        "",
        "## Dataset Fingerprint",
        "_Gain by conditional moment (rows) and learner family (columns) at the selected"
        " policy, strongest signal, and widest row budget. `gap` = best nonlinear - linear:"
        " how much curvature/interaction the linear learner misses._",
        "",
        _markdown_table(
            fingerprint,
            (
                ("mechanism", "Mechanism"),
                ("target", "Moment"),
                ("linear", "linear"),
                ("bins", "bins"),
                ("rff", "rff"),
                ("pairwise", "pairwise"),
                ("best_nonlinear", "best nonlinear"),
                ("gap", "gap"),
            ),
        ),
        "",
        "## Decision & Machinery",
        "",
        "### Run",
        _markdown_table(run_config, (("field", "Field"), ("value", "Value"))),
        "",
        "### Gate Checks",
        _markdown_table(gate_checks, (("check", "Check"), ("passed", "Passed"))),
        "",
        "### Ridge Sweep",
        _markdown_table(
            list(evidence["lambda_scores"]),
            (
                ("lambda", "Lambda"),
                ("null_nonpositive", "Null <= 0"),
                ("response_noise_score", "Response/noise score"),
            ),
        ),
        "",
        "### Row Policy Sweep",
        _markdown_table(
            list(evidence["policy_summary"]),
            (("policy", "Policy"), ("median_std", "Median repeat SD"), ("runtime", "Mean runtime s")),
        ),
        "",
        "## Structure Diagnostics",
        "",
        "### Dominant Blocks",
        _markdown_table(
            dominant_blocks,
            (
                ("representation", "Representation"),
                ("mechanism", "Mechanism"),
                ("dominant_block", "Dominant block"),
                ("absolute_share", "Absolute share"),
            ),
        ),
        "",
        "### Strongest Coordinates",
        _markdown_table(
            _top_coordinate_summary(table, selected_policy),
            (
                ("mechanism", "Mechanism"),
                ("representation", "Representation"),
                ("coordinate", "Coordinate"),
                ("block", "Block"),
                ("mean", "Mean"),
                ("mean_absolute", "Mean abs"),
                ("std", "SD"),
            ),
        ),
        "",
        "### Observation Diagnostics",
        _markdown_table(
            list(evidence["observation_summary"]),
            (
                ("policy", "Policy"),
                ("representation", "Representation"),
                ("block", "Block"),
                ("mean_absolute", "Mean abs"),
                ("mean_runtime_seconds", "Mean runtime s"),
            ),
        ),
        "",
        "## Stability & Response",
        "",
        "### Rank Stability",
        (
            rank_note
            if rank_note
            else _markdown_table(
                rank_rows,
                (
                    ("representation", "Representation"),
                    ("mechanism", "Mechanism"),
                    ("median_spearman", "Median Spearman"),
                    ("pair_count", "Pairs"),
                ),
            )
        ),
        "",
        "### Response Curves",
        (
            response_note
            if response_note
            else _markdown_table(
                response_rows,
                (
                    ("representation", "Representation"),
                    ("mechanism", "Mechanism"),
                    ("strength", "Strength"),
                    ("row_budget", "Rows"),
                    ("statistic", "Statistic"),
                    ("mean", "Mean"),
                    ("std", "SD"),
                ),
            )
        ),
        "",
        "## Cost",
        "",
        "### P Complexity",
        _markdown_table(
            p_complexity,
            (
                ("p", "Features"),
                ("runtime_seconds", "Runtime s"),
                ("tracemalloc_mb", "Tracemalloc MB"),
                ("rss_growth_mb", "RSS growth MB"),
            ),
        ),
        "",
    ]
    return "\n".join(sections)


def write_study_artifacts(
    config: CharacterizationStudyConfig,
    project_root: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    logger.info("🚀 Starting characterization study")
    destination = output or characterization_output_dir(config, project_root)
    destination.mkdir(parents=True, exist_ok=True)
    configure_study_logging(destination, study="characterization")
    table, decision, evidence = run_study(config, checkpoint_path=destination / "checkpoints.sqlite")
    table.write_parquet(destination / "coordinates.parquet")
    (destination / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
    (destination / "decision_log.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    (destination / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))
    (destination / "summary.md").write_text(build_study_summary_markdown(config, table, decision, evidence))
    (destination / "environment.json").write_text(
        json.dumps(environment_provenance(project_root), indent=2, sort_keys=True)
    )
    for representation in ("raw", "contrast"):
        characterizer = config.characterization.model_copy(update={"representation": representation})
        schema_label = "mixed" if _is_synthetic_dataset(config) else "real"
        result = characterize_multiresolution(make_task(schema_label, config, 0), characterizer)
        (destination / f"schema_{representation}.json").write_text(
            json.dumps(schema_payload(result), indent=2, sort_keys=True)
        )
    logger.success(f"🏁 characterization study complete, artifacts saved to {destination}")
    return decision
