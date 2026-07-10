"""Step 4 planted/null recovery matrix and search-protocol evidence."""

import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, SupportsFloat, cast

import numpy as np
import polars as pl
from benchmarks.studies.study_logging import configure_study_logging
from ebpfn.cache import EvaluationCache
from ebpfn.config import TuningConfig, TuningStudyConfig
from ebpfn.data import CharacterizationShape
from ebpfn.priors import EtaVectorizer, HyperPrior, build_hyperprior, sample_task
from ebpfn.tune import CandidateRecord, EvaluationResult, evaluate_candidate, make_panel, run_search
from ebpfn.utils import RandomStreams, environment_provenance
from loguru import logger

_REPRESENTATIONS = ("raw", "contrast")
_OBJECTIVES = ("directed", "energy")
_SCENARIOS = ("null", "planted")
_TABLE_NAMES = ("evaluations", "candidates", "failure_events", "rank_stability", "recovery")


def _float_metric(value: object) -> float:
    if value is None:
        raise ValueError("Expected a numeric metric value, got None.")
    return float(cast(SupportsFloat, value))


def _vector_key(vector: np.ndarray | tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), 12) for value in vector)


def _cell_config(
    config: TuningStudyConfig,
    representation: str,
    objective: str,
    cloud_size: int,
    regularization: str,
) -> TuningConfig:
    characterization = config.tuning.characterization.model_copy(update={"representation": representation})
    cloud = config.tuning.cloud.model_copy(update={"n_members": cloud_size})
    search = config.tuning.search.model_copy(
        update={
            "single_task_regularization": regularization,
            "trust_region_radius": config.trust_region_radius if regularization == "trust_region" else None,
            "competitive_tolerance": (
                config.competitive_tolerance if regularization == "closest_to_baseline" else None
            ),
        }
    )
    return config.tuning.model_copy(
        update={
            "objective": objective,
            "characterization": characterization,
            "cloud": cloud,
            "search": search,
        }
    )


def _planted_eta(config: TuningStudyConfig, scenario: str) -> HyperPrior:
    baseline = build_hyperprior(config.tuning.prior)
    if scenario == "null":
        return baseline
    return dataclasses.replace(baseline, log_snr_mean=baseline.log_snr_mean + config.planted_log_snr_shift)


def _real_tasks(
    config: TuningStudyConfig,
    eta: HyperPrior,
    streams: RandomStreams,
    cell_identity: tuple[str | int, ...],
) -> list:
    shape = CharacterizationShape(
        config.mode.n_probe_fit,
        config.mode.n_probe_score,
        config.mode.n_features,
        0,
        "regression",
    )
    return [
        sample_task(eta, shape, streams, *cell_identity, "real", index).tuning
        for index in range(config.mode.n_real_tasks)
    ]


def _evaluation_row(
    result: EvaluationResult,
    *,
    cell: dict[str, Any],
    phase: str,
    vector: tuple[float, ...],
    selected: bool,
) -> dict[str, Any]:
    return {
        **cell,
        "phase": phase,
        "stage": result.stage,
        "fidelity": result.fidelity,
        "candidate_vector": json.dumps(vector),
        "total": result.total,
        "per_block": json.dumps(result.per_block, sort_keys=True),
        "objective_terms": json.dumps(result.objective_terms, sort_keys=True),
        "failures": result.failures,
        "runtime_seconds": result.runtime_s,
        "cache_key": result.cache_key,
        "selected": selected,
    }


def _failure_rows(result: EvaluationResult, cell: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    rows = []
    for event in result.failure_events:
        payload = event.to_payload()
        payload["shape"] = json.dumps(payload["shape"], sort_keys=True)
        rows.append({**cell, "phase": phase, **payload})
    return rows


def _rank_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_rank = np.argsort(np.argsort(np.asarray(left)))
    right_rank = np.argsort(np.argsort(np.asarray(right)))
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _empty_failures() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "representation": pl.String,
            "objective": pl.String,
            "scenario": pl.String,
            "repeat": pl.Int64,
            "cloud_size": pl.Int64,
            "regularization": pl.String,
            "phase": pl.String,
            "task_id": pl.String,
            "source_id": pl.String,
            "member_index": pl.Int64,
            "fidelity": pl.String,
            "row_budget": pl.Int64,
            "route": pl.String,
            "shape": pl.String,
            "exception_type": pl.String,
            "message": pl.String,
        }
    )


@dataclasses.dataclass
class _StudyRows:
    evaluations: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    candidates: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    failures: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    ranks: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    recovery: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class _CellSpec:
    cell_id: int
    representation: str
    objective: str
    scenario: str
    repeat: int
    cloud_size: int
    regularization: str


def _extend_rows(target: _StudyRows, source: _StudyRows) -> None:
    target.evaluations.extend(source.evaluations)
    target.candidates.extend(source.candidates)
    target.failures.extend(source.failures)
    target.ranks.extend(source.ranks)
    target.recovery.extend(source.recovery)


def _rows_to_frames(rows: _StudyRows) -> dict[str, pl.DataFrame]:
    return {
        "evaluations": pl.DataFrame(rows.evaluations),
        "candidates": pl.DataFrame(rows.candidates),
        "failure_events": pl.DataFrame(rows.failures) if rows.failures else _empty_failures(),
        "rank_stability": pl.DataFrame(rows.ranks),
        "recovery": pl.DataFrame(rows.recovery),
    }


def _cell_part_dir(parts_dir: Path, cell_id: int) -> Path:
    return parts_dir / f"cell_{cell_id:04d}"


def _part_complete(parts_dir: Path, cell_id: int) -> bool:
    part = _cell_part_dir(parts_dir, cell_id)
    return (part / "complete.json").exists() and all((part / f"{name}.parquet").exists() for name in _TABLE_NAMES)


def _write_part(parts_dir: Path, cell_id: int, rows: _StudyRows) -> None:
    part = _cell_part_dir(parts_dir, cell_id)
    part.mkdir(parents=True, exist_ok=True)
    for name, frame in _rows_to_frames(rows).items():
        frame.write_parquet(part / f"{name}.parquet")
    (part / "complete.json").write_text(json.dumps({"cell_id": cell_id}, sort_keys=True))


def _read_parts(parts_dir: Path, specs: list[_CellSpec]) -> dict[str, pl.DataFrame]:
    frames: dict[str, list[pl.DataFrame]] = {name: [] for name in _TABLE_NAMES}
    for spec in specs:
        part = _cell_part_dir(parts_dir, spec.cell_id)
        for name in _TABLE_NAMES:
            frames[name].append(pl.read_parquet(part / f"{name}.parquet"))
    return {
        name: pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame() for name, parts in frames.items()
    }


def _cell_specs(config: TuningStudyConfig) -> list[_CellSpec]:
    specs: list[_CellSpec] = []
    for representation in _REPRESENTATIONS:
        for objective in _OBJECTIVES:
            for scenario in _SCENARIOS:
                for cloud_size in config.mode.cloud_sizes:
                    for regularization in config.mode.regularization_policies:
                        for repeat in range(config.mode.repeats):
                            specs.append(
                                _CellSpec(
                                    len(specs),
                                    representation,
                                    objective,
                                    scenario,
                                    repeat,
                                    cloud_size,
                                    regularization,
                                )
                            )
    return specs


def _run_cell(
    config: TuningStudyConfig,
    tuning: TuningConfig,
    planted_eta: HyperPrior,
    vectorizer: EtaVectorizer,
    planted_vector: np.ndarray,
    cell: dict[str, Any],
    rows: _StudyRows | None = None,
) -> _StudyRows:
    if rows is None:
        rows = _StudyRows()
    baseline_eta = build_hyperprior(tuning.prior)
    baseline_vector = tuple(float(value) for value in vectorizer.encode(baseline_eta))
    baseline_key = _vector_key(baseline_vector)
    repeat = int(cell["repeat"])
    scenario = str(cell["scenario"])
    streams = RandomStreams(tuning.seed + repeat)
    real_tasks = _real_tasks(config, planted_eta, streams, ("step4-recovery", scenario, repeat))
    search = run_search(tuning, real_tasks, streams, vectorizer=vectorizer)

    selected_vector = np.asarray(search.finalist_vector)
    selected_key = _vector_key(search.finalist_vector)
    for record in search.search_records:
        rows.evaluations.append(
            _evaluation_row(
                record.result,
                cell=cell,
                phase=record.origin,
                vector=record.vector,
                selected=False,
            )
        )
        rows.failures.extend(_failure_rows(record.result, cell, record.origin))
    for record in search.selection_records:
        rows.evaluations.append(
            _evaluation_row(
                record.result,
                cell=cell,
                phase="selection",
                vector=record.vector,
                selected=record.vector == selected_key,
            )
        )
        rows.failures.extend(_failure_rows(record.result, cell, "selection"))

    targets = list(search.real_targets_by_fidelity["full"])
    audit_panel = make_panel("final_audit", repeat, tuning, streams)
    cache = EvaluationCache(tuning.cache.root, tuning.cache.enabled) if tuning.cache.enabled else None
    audit_records: list[CandidateRecord] = []
    audit_vectors = [np.asarray(record.vector) for record in search.selection_ranking]
    if not any(_vector_key(vector) == baseline_key for vector in audit_vectors):
        audit_vectors.append(np.asarray(baseline_vector))
    for vector in audit_vectors:
        result = evaluate_candidate(
            vectorizer.decode(vector),
            targets,
            tuning,
            streams,
            audit_panel,
            "full",
            cache=cache,
            vectorizer=vectorizer,
            baseline_vector=baseline_vector,
        )
        vector_key = _vector_key(vector)
        record = CandidateRecord(vector_key, "final_audit", result)
        audit_records.append(record)
        rows.evaluations.append(
            _evaluation_row(
                result,
                cell=cell,
                phase="final_audit",
                vector=vector_key,
                selected=vector_key == selected_key,
            )
        )
        rows.failures.extend(_failure_rows(result, cell, "final_audit"))

    audit_by_vector = {record.vector: record.result.total for record in audit_records}
    selection_totals = [record.result.total for record in search.selection_ranking]
    audit_totals = [audit_by_vector[record.vector] for record in search.selection_ranking]
    panel_noise = [
        float(np.std([result.total for result in record.panel_results])) for record in search.selection_ranking
    ]
    rows.ranks.append(
        {
            **cell,
            "candidate_count": len(search.selection_ranking),
            "selection_audit_spearman": _rank_correlation(selection_totals, audit_totals),
            "median_selection_panel_std": float(np.median(panel_noise)),
        }
    )
    selected_audit = audit_by_vector[selected_key]
    baseline_audit = audit_by_vector[baseline_key]
    rows.recovery.append(
        {
            **cell,
            "selected_audit_loss": selected_audit,
            "baseline_audit_loss": baseline_audit,
            "fresh_seed_loss_reduction": baseline_audit - selected_audit,
            "null_movement": float(np.linalg.norm(selected_vector - np.asarray(baseline_vector))),
            "parameter_error": float(np.linalg.norm(selected_vector - planted_vector)),
            "failure_count": sum(record.result.failures for record in audit_records),
            "runtime_seconds": sum(record.result.runtime_s for record in audit_records),
        }
    )
    for rank, record in enumerate(search.selection_ranking):
        rows.candidates.append(
            {
                **cell,
                "selection_rank": rank,
                "candidate_vector": json.dumps(record.vector),
                "selection_loss": record.result.total,
                "audit_loss": audit_by_vector[record.vector],
                "selected": rank == 0,
                "origin": record.origin,
            }
        )
    return rows


def _run_cell_spec(config: TuningStudyConfig, spec: _CellSpec) -> tuple[int, _StudyRows, float]:
    start = time.perf_counter()
    planted_eta = _planted_eta(config, spec.scenario)
    tuning = _cell_config(config, spec.representation, spec.objective, spec.cloud_size, spec.regularization)
    baseline_eta = build_hyperprior(tuning.prior)
    vectorizer = EtaVectorizer(baseline_eta, tuning.active)
    planted_vector = np.asarray(vectorizer.encode(planted_eta))
    cell = {
        "representation": spec.representation,
        "objective": spec.objective,
        "scenario": spec.scenario,
        "repeat": spec.repeat,
        "cloud_size": spec.cloud_size,
        "regularization": spec.regularization,
    }
    return (
        spec.cell_id,
        _run_cell(config, tuning, planted_eta, vectorizer, planted_vector, cell),
        time.perf_counter() - start,
    )


def _worker_count(total: int, requested: int | None) -> int:
    if total <= 1:
        return 1
    if requested is not None:
        if requested < 1:
            raise ValueError("max_workers must be positive")
        return min(requested, total)
    configured = os.environ.get("EBPFN_TUNING_WORKERS")
    if configured is not None:
        workers = int(configured)
        if workers < 1:
            raise ValueError("EBPFN_TUNING_WORKERS must be positive")
        return min(workers, total)
    return min(os.cpu_count() or 1, total)


def _finalize_study(config: TuningStudyConfig, frames: dict[str, pl.DataFrame]) -> dict[str, Any]:
    evaluations = frames["evaluations"]
    candidates = frames["candidates"]
    failures = frames["failure_events"]
    ranks = frames["rank_stability"]
    recovery = frames["recovery"]
    combinations = set(zip(evaluations["representation"], evaluations["objective"], strict=True))
    scenarios = set(evaluations["scenario"])
    cloud_sizes = set(evaluations["cloud_size"])
    regularization_policies = set(evaluations["regularization"])
    checks = {
        "all_four_construction_cells": combinations
        == {(representation, objective) for representation in _REPRESENTATIONS for objective in _OBJECTIVES},
        "planted_and_null": scenarios == set(_SCENARIOS),
        "cloud_size_grid_complete": cloud_sizes == set(config.mode.cloud_sizes),
        "regularization_grid_complete": regularization_policies == set(config.mode.regularization_policies),
        "independent_final_audit": bool(evaluations.filter(pl.col("stage") == "final_audit").height),
        "full_evaluation_trace": evaluations.height > candidates.height,
    }
    status, missing = derive_study_status(config, checks)
    rank_values = ranks["selection_audit_spearman"].drop_nulls()
    evidence = {
        "checks": checks,
        "median_fresh_seed_loss_reduction": _float_metric(recovery["fresh_seed_loss_reduction"].median()),
        "median_null_movement": _float_metric(recovery.filter(pl.col("scenario") == "null")["null_movement"].median()),
        "median_rank_stability": None if rank_values.is_empty() else _float_metric(rank_values.median()),
        "total_failures": int(recovery["failure_count"].sum()),
    }
    decision = {
        "status": status,
        "missing_checks": missing,
        "decision_owner": config.decision_owner,
        "decision_date": config.decision_date,
        "multiresolution": config.multiresolution_decision,
        "synthetic_failures": config.synthetic_failure_decision,
        "single_task_regularization": config.single_task_regularization_decision,
        "optimizer": config.tuning.search.optimizer,
    }
    return {
        "evaluations": evaluations,
        "candidates": candidates,
        "failure_events": failures,
        "rank_stability": ranks,
        "recovery": recovery,
        "evidence": evidence,
        "decision": decision,
    }


def run_study(
    config: TuningStudyConfig,
    *,
    parts_dir: Path | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run all four construction cells under planted and null recovery."""
    specs = _cell_specs(config)
    if parts_dir is None:
        rows = _StudyRows()
        for spec in specs:
            _, cell_rows, _ = _run_cell_spec(config, spec)
            _extend_rows(rows, cell_rows)
        return _finalize_study(config, _rows_to_frames(rows))

    parts_dir.mkdir(parents=True, exist_ok=True)
    pending = [spec for spec in specs if not _part_complete(parts_dir, spec.cell_id)]
    completed = len(specs) - len(pending)
    if completed:
        logger.info(f"  resuming | {completed}/{len(specs)} cells already complete")
    if pending:
        workers = _worker_count(len(pending), max_workers)
        logger.info(f"  running | {len(pending)} tuning cells | {workers} worker(s)")
        if workers == 1:
            for spec in pending:
                cell_id, rows, elapsed = _run_cell_spec(config, spec)
                _write_part(parts_dir, cell_id, rows)
                completed += 1
                logger.info(
                    f"  cell {completed}/{len(specs)} | "
                    f"{spec.representation}/{spec.objective}/{spec.scenario} "
                    f"cloud={spec.cloud_size} reg={spec.regularization} repeat={spec.repeat} | "
                    f"{elapsed:.1f}s"
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_cell_spec, config, spec): spec for spec in pending}
                for future in as_completed(futures):
                    spec = futures[future]
                    cell_id, rows, elapsed = future.result()
                    _write_part(parts_dir, cell_id, rows)
                    completed += 1
                    logger.info(
                        f"  cell {completed}/{len(specs)} | "
                        f"{spec.representation}/{spec.objective}/{spec.scenario} "
                        f"cloud={spec.cloud_size} reg={spec.regularization} repeat={spec.repeat} | "
                        f"{elapsed:.1f}s"
                    )
    frames = _read_parts(parts_dir, specs)
    return _finalize_study(config, frames)


def derive_study_status(config: TuningStudyConfig, checks: dict[str, bool]) -> tuple[str, list[str]]:
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        return "failed", failed
    if config.mode.name == "fast":
        return "provisional", []
    pending = []
    if config.multiresolution_decision == "pending":
        pending.append("multiresolution_decision")
    if config.synthetic_failure_decision == "pending":
        pending.append("synthetic_failure_decision")
    if config.single_task_regularization_decision == "pending":
        pending.append("single_task_regularization_decision")
    return ("frozen", []) if not pending else ("incomplete", pending)


def write_study_artifacts(
    config: TuningStudyConfig,
    project_root: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    destination = output or (project_root / config.mode.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    configure_study_logging(destination, study="tuning")
    result = run_study(config, parts_dir=destination / "parts")
    for name in ("evaluations", "candidates", "failure_events", "rank_stability", "recovery"):
        result[name].write_parquet(destination / f"{name}.parquet")
    (destination / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
    (destination / "evidence.json").write_text(json.dumps(result["evidence"], indent=2, sort_keys=True))
    (destination / "decision_log.json").write_text(json.dumps(result["decision"], indent=2, sort_keys=True))
    (destination / "environment.json").write_text(
        json.dumps(environment_provenance(project_root), indent=2, sort_keys=True)
    )
    return {"status": result["decision"]["status"], "evaluations": result["evaluations"].height}
