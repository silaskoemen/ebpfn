"""Step 4 planted/null recovery matrix and search-protocol evidence."""

import dataclasses
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, SupportsFloat, cast

import numpy as np
import polars as pl
from benchmarks.studies.study_logging import configure_study_logging
from ebpfn.cache import EvaluationCache
from ebpfn.config import TuningConfig, TuningStudyConfig
from ebpfn.data import CharacterizationShape, content_hash
from ebpfn.priors import EtaVectorizer, HyperPrior, build_hyperprior, sample_task
from ebpfn.tune import CandidateRecord, EvaluationResult, characterize_task, evaluate_candidate, make_panel, run_search
from ebpfn.utils import RandomStreams, environment_provenance
from loguru import logger

_REPRESENTATIONS = ("raw", "contrast")
_OBJECTIVES = ("directed", "energy")


def _scenarios(config: TuningStudyConfig) -> tuple[str, ...]:
    """`base` (unperturbed baseline prior) + one planted scenario per active knob under test. The
    planted scenario name is the knob it perturbs, so the `scenario` column reads directly as which
    eta was used: `base`, or `base` shifted along one knob. (`base`, not `null`, to avoid collision
    with the Step-2 null *mechanism*; this is the no-planted-shift control on the search.)"""
    return ("base", *config.planted_knobs)


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
            "prior_distance_penalty": (config.prior_distance_penalty if regularization == "prior_distance" else None),
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


def _planted_eta(
    config: TuningStudyConfig, scenario: str, vectorizer: EtaVectorizer, baseline_eta: HyperPrior
) -> HyperPrior:
    """`base` returns the baseline; a knob scenario shifts that one coordinate by planted_unit_shift
    in vectorized [0,1] space (toward the interior so it stays feasible) and decodes back to an eta."""
    if scenario == "base":
        return baseline_eta
    index = vectorizer.active.index(scenario)
    vector = vectorizer.encode(baseline_eta)
    base = float(vector[index])
    vector[index] = (
        base + config.planted_unit_shift
        if base + config.planted_unit_shift <= 1.0
        else base - config.planted_unit_shift
    )
    return vectorizer.decode(vector)


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


def _checkpoint_identity(config: TuningStudyConfig) -> str:
    payload = config.model_dump(mode="json")
    for field in (
        "decision_owner",
        "decision_date",
        "multiresolution_decision",
        "synthetic_failure_decision",
        "single_task_regularization_decision",
    ):
        payload.pop(field)
    payload["mode"].pop("output_dir")
    return content_hash(payload, namespace="tuning-study-checkpoint-2")


class _CellStore:
    """Single-file SQLite checkpoint: one JSON payload per completed cell, keyed by cell_id.

    Replaces the old per-cell parquet directories (thousands of tiny files with per-file footer
    overhead) with one file. The main process is the only writer (workers return rows, the
    ``as_completed`` loop persists them), so a single connection per call needs no locking.
    """

    def __init__(self, path: Path, config_identity: str) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS cells (cell_id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            stored = connection.execute("SELECT value FROM metadata WHERE key = 'config_identity'").fetchone()
            if stored is None:
                if connection.execute("SELECT EXISTS(SELECT 1 FROM cells)").fetchone()[0]:
                    raise ValueError("existing tuning checkpoint has no config identity; start a fresh checkpoint")
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('config_identity', ?)",
                    (config_identity,),
                )
            elif stored[0] != config_identity:
                raise ValueError("tuning checkpoint config does not match the resolved study config")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def completed_ids(self) -> set[int]:
        with self._connect() as connection:
            return {int(row[0]) for row in connection.execute("SELECT cell_id FROM cells")}

    def put(self, cell_id: int, rows: _StudyRows) -> None:
        payload = json.dumps(dataclasses.asdict(rows))
        with self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO cells(cell_id, payload) VALUES (?, ?)", (cell_id, payload))

    def read_frames(self) -> dict[str, pl.DataFrame]:
        merged = _StudyRows()
        with self._connect() as connection:
            for (payload,) in connection.execute("SELECT payload FROM cells ORDER BY cell_id"):
                _extend_rows(merged, _StudyRows(**json.loads(payload)))
        return _rows_to_frames(merged)


def _cell_specs(config: TuningStudyConfig) -> list[_CellSpec]:
    specs: list[_CellSpec] = []
    for representation in _REPRESENTATIONS:
        for objective in _OBJECTIVES:
            for scenario in _scenarios(config):
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
            "movement_from_base": float(np.linalg.norm(selected_vector - np.asarray(baseline_vector))),
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
    tuning = _cell_config(config, spec.representation, spec.objective, spec.cloud_size, spec.regularization)
    baseline_eta = build_hyperprior(tuning.prior)
    vectorizer = EtaVectorizer(baseline_eta, tuning.active)
    planted_eta = _planted_eta(config, spec.scenario, vectorizer, baseline_eta)
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
        "base_and_planted": scenarios == set(_scenarios(config)),
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
        "median_base_movement": _float_metric(
            recovery.filter(pl.col("scenario") == "base")["movement_from_base"].median()
        ),
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
    checkpoint_path: Path | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run all four construction cells under base and planted recovery.

    With ``checkpoint_path`` set, completed cells are persisted to a single SQLite file and skipped
    on resume; without it the run is in-memory and serial.
    """
    specs = _cell_specs(config)
    logger.info(
        f"🧭 tuning recovery | mode={config.mode.name} | {len(specs)} cells "
        f"({len(_REPRESENTATIONS)} rep x {len(_OBJECTIVES)} obj x {len(_scenarios(config))} scenario "
        f"x {len(config.mode.cloud_sizes)} cloud x {len(config.mode.regularization_policies)} reg "
        f"x {config.mode.repeats} repeat)"
    )
    logger.info(f"🎛️ scenarios: {', '.join(_scenarios(config))}")
    logger.info(
        f"🧵 parallelism | cpu_count={os.cpu_count()} | "
        f"EBPFN_TUNING_WORKERS={os.environ.get('EBPFN_TUNING_WORKERS', '<unset>')} | "
        f"max_workers={max_workers if max_workers is not None else '<auto>'}"
    )
    if checkpoint_path is None:
        rows = _StudyRows()
        for spec in specs:
            _, cell_rows, _ = _run_cell_spec(config, spec)
            _extend_rows(rows, cell_rows)
        return _finalize_study(config, _rows_to_frames(rows))

    store = _CellStore(checkpoint_path, _checkpoint_identity(config))
    done = store.completed_ids()
    pending = [spec for spec in specs if spec.cell_id not in done]
    completed = len(specs) - len(pending)
    if completed:
        logger.info(f"  resuming | {completed}/{len(specs)} cells already checkpointed")
    if pending:
        workers = _worker_count(len(pending), max_workers)
        logger.info(f"⚙️ run | {len(pending)} pending cells | {workers} worker(s) resolved")
        if workers == 1:
            for spec in pending:
                cell_id, rows, elapsed = _run_cell_spec(config, spec)
                store.put(cell_id, rows)
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
                    store.put(cell_id, rows)
                    completed += 1
                    logger.info(
                        f"  cell {completed}/{len(specs)} | "
                        f"{spec.representation}/{spec.objective}/{spec.scenario} "
                        f"cloud={spec.cloud_size} reg={spec.regularization} repeat={spec.repeat} | "
                        f"{elapsed:.1f}s"
                    )
    return _finalize_study(config, store.read_frames())


def derive_study_status(config: TuningStudyConfig, checks: dict[str, bool]) -> tuple[str, list[str]]:
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        return "failed", failed
    if config.mode.name != "audit":  # fast + sweep are provisional evidence, not decision-freezing
        return "provisional", []
    pending = []
    if config.multiresolution_decision == "pending":
        pending.append("multiresolution_decision")
    if config.synthetic_failure_decision == "pending":
        pending.append("synthetic_failure_decision")
    if config.single_task_regularization_decision == "pending":
        pending.append("single_task_regularization_decision")
    return ("frozen", []) if not pending else ("incomplete", pending)


def _best_location_gain(characterization: Any) -> float:
    """Apparent SNR proxy: the best held-out gain any learner achieves on the conditional mean.

    This is 'how learnable is this task by the map family' — the operationally correct target for a
    prior, and (unlike generative SNR) directly measurable, since generative SNR is confounded with
    map-incompleteness."""
    best = float("-inf")
    for coord, value in zip(characterization.coordinates, characterization.raw_values, strict=True):
        if coord.statistic == "gain" and coord.target == "location":
            best = max(best, float(value))
    return best if best != float("-inf") else float("nan")


def _real_best_location_gains(characterization_dir: Path) -> list[float]:
    gains: list[float] = []
    for path in sorted(characterization_dir.glob("*mode_audit*/coordinates.parquet")):
        if "synthetic" in path.parent.name:  # the real reference must exclude the synthetic baseline arm
            continue
        frame = pl.read_parquet(path)
        raw = frame.filter(
            (pl.col("representation") == "raw")
            & (pl.col("statistic") == "gain")
            & (pl.col("target") == "location")
            & pl.col("valid")
        )
        if raw.is_empty():
            continue
        raw = raw.filter(pl.col("row_budget") == raw["row_budget"].max())
        gains.append(float(raw["value"].max()))  # ty:ignore[invalid-argument-type]
    return gains


def apparent_snr_report(
    config: TuningStudyConfig,
    project_root: Path,
    *,
    n_tasks: int = 64,
    characterization_dir: Path | None = None,
) -> dict[str, Any]:
    """One-time calibration diagnostic: does the baseline prior's apparent-SNR (best location gain)
    distribution match the real corpus? This is how SNR-as-cleanliness is handled — by matching the
    distribution as fixed config, not by tuning a non-identifiable knob."""
    baseline_eta = build_hyperprior(config.tuning.prior)
    char_raw = config.tuning.characterization.model_copy(update={"representation": "raw"})
    shape = CharacterizationShape(
        config.mode.n_probe_fit, config.mode.n_probe_score, config.mode.n_features, 0, "regression"
    )
    streams = RandomStreams(config.tuning.seed)
    synthetic: list[float] = []
    for index in range(n_tasks):
        task = sample_task(baseline_eta, shape, streams, "apparent-snr", index, "synthetic", index).tuning
        gain = _best_location_gain(characterize_task(task, char_raw, "full", random_identity=("apparent-snr", index)))
        if gain == gain:  # drop NaN
            synthetic.append(gain)
    real = _real_best_location_gains(characterization_dir or (project_root / "benchmarks/results/characterization"))

    def quantiles(values: list[float]) -> dict[str, float] | None:
        return {str(p): float(np.quantile(values, p)) for p in (0.1, 0.5, 0.9)} if values else None

    report: dict[str, Any] = {
        "n_synthetic": len(synthetic),
        "n_real": len(real),
        "synthetic_mean": float(np.mean(synthetic)) if synthetic else None,
        "real_mean": float(np.mean(real)) if real else None,
        "synthetic_quantiles": quantiles(synthetic),
        "real_quantiles": quantiles(real),
    }
    if synthetic and real:
        grid = np.sort(np.concatenate([synthetic, real]))
        cdf_synth = np.searchsorted(np.sort(synthetic), grid, side="right") / len(synthetic)
        cdf_real = np.searchsorted(np.sort(real), grid, side="right") / len(real)
        report["ks_statistic"] = float(np.max(np.abs(cdf_synth - cdf_real)))
        report["mean_gap_real_minus_synthetic"] = float(np.mean(real) - np.mean(synthetic))
    return report


def _md_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return "0" if value == 0.0 else f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|")


def _md_table(rows: list[dict[str, Any]], columns: tuple[tuple[str, str], ...]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_md_value(row.get(key)) for key, _ in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _identifiability_rows(config: TuningStudyConfig, recovery: pl.DataFrame) -> list[dict[str, Any]]:
    """Per policy/objective/scenario recovery signal, noise, movement, and error.

    A knob is identifiable when its S/N clears ~1 with small parameter error; `base` is the control.
    """
    needed = {
        "regularization",
        "objective",
        "scenario",
        "fresh_seed_loss_reduction",
        "parameter_error",
        "movement_from_base",
    }
    if recovery.is_empty() or not needed.issubset(set(recovery.columns)):
        return []
    rows = (
        recovery.group_by("regularization", "objective", "scenario")
        .agg(
            pl.col("fresh_seed_loss_reduction").mean().alias("mean_loss_red"),
            pl.col("fresh_seed_loss_reduction").std(ddof=0).alias("sd_loss_red"),
            pl.col("parameter_error").mean().alias("mean_param_err"),
            pl.col("movement_from_base").mean().alias("mean_move"),
            pl.len().alias("n"),
        )
        .to_dicts()
    )
    order = ["base", *config.planted_knobs]
    for row in rows:
        sd = row["sd_loss_red"]
        row["sn"] = row["mean_loss_red"] / sd if sd and sd > 1e-12 else None
    rows.sort(
        key=lambda row: (
            str(row["regularization"]),
            str(row["objective"]),
            order.index(row["scenario"]) if row["scenario"] in order else 99,
        )
    )
    return rows


def build_tuning_summary_markdown(
    config: TuningStudyConfig, result: dict[str, Any], apparent_snr: dict[str, Any]
) -> str:
    decision = result["decision"]
    evidence = result["evidence"]
    ident = _identifiability_rows(config, result["recovery"])
    scenarios = _scenarios(config)
    cells = (
        len(_REPRESENTATIONS)
        * len(_OBJECTIVES)
        * len(scenarios)
        * len(config.mode.cloud_sizes)
        * len(config.mode.regularization_policies)
        * config.mode.repeats
    )
    headline = [
        f"Status: **{decision['status']}**.",
        f"Mode: **{config.mode.name}** — {cells} cells "
        f"(cloud {list(config.mode.cloud_sizes)}, reg {list(config.mode.regularization_policies)}, "
        f"{config.mode.repeats} repeats, {config.mode.n_features} features).",
        f"Planted knobs: {', '.join(config.planted_knobs)}.",
    ]
    if evidence.get("median_base_movement") is not None:
        headline.append(f"Median `base` movement: **{_md_value(evidence['median_base_movement'])}** (want ≈ 0).")
    planted = [row for row in ident if row["scenario"] != "base" and row["sn"] is not None]
    if planted:
        top = max(planted, key=lambda row: row["sn"])
        headline.append(
            f"Most-identifiable knob: **{top['scenario']}** ({top['regularization']}, {top['objective']}, S/N "
            f"{_md_value(top['sn'])}, param error {_md_value(top['mean_param_err'])})."
        )
    if apparent_snr.get("synthetic_mean") is not None and apparent_snr.get("real_mean") is not None:
        headline.append(
            f"Apparent-SNR coverage: synthetic mean **{_md_value(apparent_snr['synthetic_mean'])}** vs "
            f"real **{_md_value(apparent_snr['real_mean'])}** (KS {_md_value(apparent_snr.get('ks_statistic'))})."
        )
    if evidence.get("total_failures") is not None:
        headline.append(f"Cloud-member failures: {evidence['total_failures']}.")

    gate_rows = [{"check": name, "passed": passed} for name, passed in sorted(evidence.get("checks", {}).items())]
    snr_rows = [
        {"metric": "n tasks", "synthetic": apparent_snr.get("n_synthetic"), "real": apparent_snr.get("n_real")},
        {"metric": "mean", "synthetic": apparent_snr.get("synthetic_mean"), "real": apparent_snr.get("real_mean")},
    ]
    for label, key in (("q10", "0.1"), ("q50", "0.5"), ("q90", "0.9")):
        sq = (apparent_snr.get("synthetic_quantiles") or {}).get(key)
        rq = (apparent_snr.get("real_quantiles") or {}).get(key)
        snr_rows.append({"metric": label, "synthetic": sq, "real": rq})

    sections = [
        "# Tuning Recovery Study Summary",
        "",
        "## Headline",
        *[f"- {item}" for item in headline],
        "",
        "## Knob Identifiability",
        "_Per objective x scenario, aggregated over repeats+representations. `sn` = mean loss-reduction /"
        " its SD (recoverable when ~ 1); `mean_param_err` = ‖selected - planted‖ (small = recovered);"
        " `mean_move` = ‖selected - base‖. The `base` row is the no-planted-shift control._",
        "",
        _md_table(
            ident,
            (
                ("regularization", "Regularization"),
                ("objective", "Objective"),
                ("scenario", "Scenario"),
                ("mean_loss_red", "Mean loss-red"),
                ("sd_loss_red", "SD"),
                ("sn", "S/N"),
                ("mean_param_err", "Param err"),
                ("mean_move", "Move from base"),
                ("n", "n"),
            ),
        ),
        "",
        "## Apparent-SNR Calibration",
        "_Best location gain (how learnable the mean is by the map family). Synthetic = baseline prior;"
        " real = Step-2 frozen corpus. A large gap/KS means the prior mis-covers real learnability._",
        "",
        _md_table(snr_rows, (("metric", "Metric"), ("synthetic", "Synthetic"), ("real", "Real"))),
        "",
        "## Gate Checks",
        _md_table(gate_rows, (("check", "Check"), ("passed", "Passed"))),
        "",
    ]
    return "\n".join(sections)


def write_study_artifacts(
    config: TuningStudyConfig,
    project_root: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    destination = output or (project_root / config.mode.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    configure_study_logging(destination, study="tuning")
    result = run_study(config, checkpoint_path=destination / "checkpoints.sqlite")
    for name in ("evaluations", "candidates", "failure_events", "rank_stability", "recovery"):
        result[name].write_parquet(destination / f"{name}.parquet")
    (destination / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
    (destination / "evidence.json").write_text(json.dumps(result["evidence"], indent=2, sort_keys=True))
    (destination / "decision_log.json").write_text(json.dumps(result["decision"], indent=2, sort_keys=True))
    logger.info("📐 apparent-SNR calibration | characterizing baseline-prior tasks")
    apparent_snr = apparent_snr_report(config, project_root)
    (destination / "apparent_snr.json").write_text(json.dumps(apparent_snr, indent=2, sort_keys=True))
    (destination / "summary.md").write_text(build_tuning_summary_markdown(config, result, apparent_snr))
    (destination / "environment.json").write_text(
        json.dumps(environment_provenance(project_root), indent=2, sort_keys=True)
    )
    status = result["decision"]["status"]
    logger.success(f"✅ tuning recovery complete | status={status} | artifacts → {destination}")
    return {"status": status, "evaluations": result["evaluations"].height}
