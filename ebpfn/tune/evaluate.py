"""The simulator-only candidate evaluator.

This module imports no PFN symbol: candidate evaluation is likelihood-free. For a
hyperprior ``eta`` it draws a matched synthetic cloud per real task, characterizes
each member at the requested fidelity, and scores the cloud against the real
characterization with the directed-coverage or energy objective. Row budgets are
aggregated inside the objective; tasks and sources are aggregated here by the
objective's explicit-weight rule.
"""

import dataclasses
import math
import time
from collections.abc import Sequence

import numpy as np

from ebpfn.cache import EvaluationCache, evaluation_cache_key
from ebpfn.characterize import (
    BudgetCharacterizationError,
    TaskCharacterization,
    build_row_budget_manifests,
    characterize,
    characterize_multiresolution,
)
from ebpfn.compare import directed_coverage, energy_score, validity_report
from ebpfn.config import CharacterizationConfig, TuningConfig
from ebpfn.data import TuningTask, characterization_shape
from ebpfn.priors import EtaVectorizer, GeneratedTask, HyperPrior, sample_task
from ebpfn.utils import RandomStreams

from .contracts import EvaluationResult, FailureEvent, Panel, RealTarget

# Soft penalty scale for the optional D3 trust-region regularization.
_TRUST_REGION_PENALTY = 1.0e3


def characterize_task(
    task: TuningTask,
    char_config: CharacterizationConfig,
    fidelity: str,
    *,
    random_identity: tuple[str | int, ...] | None = None,
) -> TaskCharacterization:
    """Characterize one task at ``min`` (smallest single budget) or ``full``."""
    if fidelity == "full":
        return characterize_multiresolution(task, char_config, random_identity=random_identity)
    if fidelity == "min":
        manifests = build_row_budget_manifests(task, char_config, random_identity=random_identity)
        try:
            return characterize(task, manifests[0], char_config, random_identity=random_identity)
        except Exception as error:
            if random_identity is None:
                raise
            raise BudgetCharacterizationError(manifests[0].row_budget, error) from error
    raise ValueError(f"unknown fidelity {fidelity!r}")


def _combine(objective: str, values: np.ndarray) -> float:
    # Directed coverage combines by root-mean-square (it aggregates squared
    # coverages); energy score combines by an ordinary mean (a proper scoring
    # rule). Applied identically within a source and across sources.
    if objective == "directed":
        return float(math.sqrt(float(np.mean(np.square(values)))))
    return float(np.mean(values))


def _aggregate_hierarchical(objective: str, by_source: dict[str, list[float]]) -> float:
    """Combine losses within each source, then across sources (spec hierarchy).

    ``L_source`` combines tasks within a source; ``L`` combines the source
    scores. With a single source this reduces to a flat task combination.
    """
    source_scores = np.array([_combine(objective, np.asarray(losses, dtype=float)) for losses in by_source.values()])
    return _combine(objective, source_scores)


def _aggregate_blocks(objective: str, blocks_by_source: dict[str, list[dict[str, float]]]) -> dict[str, float]:
    source_blocks: list[dict[str, float]] = []
    for task_blocks in blocks_by_source.values():
        names = set(task_blocks[0])
        if any(set(block) != names for block in task_blocks[1:]):
            raise ValueError("block schemas must agree across tasks within a source")
        source_blocks.append(
            {name: _combine(objective, np.array([block[name] for block in task_blocks])) for name in names}
        )
    names = set(source_blocks[0])
    if any(set(block) != names for block in source_blocks[1:]):
        raise ValueError("block schemas must agree across sources")
    return {name: _combine(objective, np.array([block[name] for block in source_blocks])) for name in names}


def _sample_and_characterize(
    eta: HyperPrior,
    task: TuningTask,
    config: TuningConfig,
    streams: RandomStreams,
    panel: Panel,
    fidelity: str,
) -> tuple[list[TaskCharacterization], tuple[FailureEvent, ...]]:
    """Draw a matched cloud and characterize it, applying the D1 failure policy.

    Under ``on_failure='raise'`` any generation/characterization failure
    propagates; under ``'exclude'`` the member is dropped and counted (never
    resampled). Returns valid member characterizations and structured failures.
    """
    shape = characterization_shape(task)
    member_chars: list[TaskCharacterization] = []
    failure_events: list[FailureEvent] = []
    for member_index in range(config.cloud.n_members):
        member: GeneratedTask | None = None
        phase = "generation"
        try:
            member = sample_task(
                eta,
                shape,
                streams,
                panel.stage,
                panel.token,
                task.task_id,
                member_index,
                common_random_numbers=True,
            )
            if characterization_shape(member.tuning) != shape:
                raise ValueError("synthetic member does not exactly match the requested characterization shape")
            phase = "characterization"
            random_identity = ("tuning-panel", panel.stage, panel.token, task.task_id, member_index)
            member_chars.append(
                characterize_task(
                    member.tuning,
                    config.characterization,
                    fidelity,
                    random_identity=random_identity,
                )
            )
        except Exception as error:
            if config.cloud.on_failure == "raise":
                raise
            original = error.original if isinstance(error, BudgetCharacterizationError) else error
            failure_events.append(
                FailureEvent(
                    task_id=task.task_id,
                    source_id=task.source_id,
                    member_index=member_index,
                    phase=phase,
                    fidelity=fidelity,
                    row_budget=error.row_budget if isinstance(error, BudgetCharacterizationError) else None,
                    route=None if member is None else str(member.diagnostics["route"]),
                    shape={
                        "n_probe_fit": shape.n_probe_fit,
                        "n_probe_score": shape.n_probe_score,
                        "p_numeric": shape.p_numeric,
                        "p_categorical": shape.p_categorical,
                    },
                    exception_type=type(original).__name__,
                    message=str(original),
                )
            )
    if not member_chars:
        raise ValueError("every cloud member failed; cannot score the task")
    return member_chars, tuple(failure_events)


def _validity_diagnostics(
    real: TaskCharacterization,
    cloud: list[TaskCharacterization],
    config: TuningConfig,
) -> dict[str, object]:
    real_report = validity_report(real)
    cloud_reports = [validity_report(member) for member in cloud]
    block_names = set(real_report.within_block_fraction)
    if any(set(report.within_block_fraction) != block_names for report in cloud_reports):
        raise ValueError("validity block schemas must agree across the real target and cloud")
    minimum_blocks = {
        block: min(
            real_report.within_block_fraction[block],
            *(report.within_block_fraction[block] for report in cloud_reports),
        )
        for block in block_names
    }
    minimum_overall = min(real_report.overall_fraction, *(report.overall_fraction for report in cloud_reports))
    return {
        "minimum_overall_fraction": minimum_overall,
        "minimum_within_block_fraction": minimum_blocks,
        "passes_overall_qc": minimum_overall >= config.compare.qc_overall,
        "passes_within_block_qc": all(
            fraction >= config.compare.qc_within_block for fraction in minimum_blocks.values()
        ),
    }


def evaluate_candidate(
    eta: HyperPrior,
    targets: Sequence[RealTarget],
    config: TuningConfig,
    streams: RandomStreams,
    panel: Panel,
    fidelity: str,
    *,
    cache: EvaluationCache | None = None,
    vectorizer: EtaVectorizer | None = None,
    baseline_vector: tuple[float, ...] | None = None,
) -> EvaluationResult:
    if not targets:
        raise ValueError("evaluation requires at least one real target")
    real_tasks = [target.task for target in targets]
    pair_ids = panel.energy_pair_ids if config.objective == "energy" else None
    key = evaluation_cache_key(
        config, eta, real_tasks, streams.base_seed, panel.stage, fidelity, panel.identity(), energy_pair_ids=pair_ids
    )
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            result = EvaluationResult.from_payload(hit)
            vector = result.candidate_vector
            n_tasks = int(result.objective_terms["n_tasks"])
            return _regularized_result(result, config, vector, baseline_vector, n_tasks)

    vector = tuple(float(value) for value in vectorizer.encode(eta)) if vectorizer is not None else ()
    start = time.perf_counter()
    by_source: dict[str, list[float]] = {}
    blocks_by_source: dict[str, list[dict[str, float]]] = {}
    per_task_terms: list[dict[str, object]] = []
    observation_by_source: dict[str, list[float]] = {}
    ensemble_by_source: dict[str, list[float]] = {}
    failure_events: list[FailureEvent] = []
    for task_index, target in enumerate(targets):
        source = target.task.source_id
        member_chars, task_failures = _sample_and_characterize(eta, target.task, config, streams, panel, fidelity)
        failure_events.extend(task_failures)
        task_terms: dict[str, object] = {
            "task_index": task_index,
            "source_id": source,
            "validity": _validity_diagnostics(target.characterization, member_chars, config),
        }
        if config.objective == "directed":
            directed = directed_coverage(target.characterization, member_chars, config.compare)
            loss, per_block = directed.total, directed.per_block
            task_terms.update(
                {
                    "per_budget": {str(b): v for b, v in directed.per_budget.items()},
                    "k_by_budget": {str(b): k for b, k in directed.k_by_budget.items()},
                    "neighbors_by_budget": {str(b): list(n) for b, n in directed.neighbors_by_budget.items()},
                }
            )
        else:
            energy = energy_score(target.characterization, member_chars, config.compare, pair_ids=pair_ids)
            loss, per_block = energy.total, energy.per_block
            observation_by_source.setdefault(source, []).append(energy.observation_term)
            ensemble_by_source.setdefault(source, []).append(energy.ensemble_term)
            task_terms.update(
                {
                    "per_budget": {str(b): v for b, v in energy.per_budget.items()},
                    "observation_term": energy.observation_term,
                    "ensemble_term": energy.ensemble_term,
                }
            )
        per_task_terms.append(task_terms)
        by_source.setdefault(source, []).append(loss)
        blocks_by_source.setdefault(source, []).append(dict(per_block))

    n_tasks = len(targets)
    raw_total = _aggregate_hierarchical(config.objective, by_source)
    per_block = _aggregate_blocks(config.objective, blocks_by_source)

    objective_terms: dict[str, object] = {
        "objective": config.objective,
        "n_tasks": n_tasks,
        "n_sources": len(by_source),
        "per_task": per_task_terms,
    }
    if config.objective == "energy":
        objective_terms["observation_term"] = _aggregate_hierarchical("energy", observation_by_source)
        objective_terms["ensemble_term"] = _aggregate_hierarchical("energy", ensemble_by_source)
        objective_terms["energy_pair_ids"] = [[int(i), int(j)] for i, j in pair_ids] if pair_ids is not None else None

    result = EvaluationResult(
        total=raw_total,
        per_block=per_block,
        objective_terms=objective_terms,
        failures=len(failure_events),
        failure_events=tuple(failure_events),
        runtime_s=time.perf_counter() - start,
        candidate_vector=vector,
        eta=eta,
        stage=panel.stage,
        fidelity=fidelity,
        seeds={"base_seed": streams.base_seed, "stage": panel.stage, "panel_token": panel.token},
        cache_key=key,
    )
    if cache is not None:
        cache.put(key, result.to_payload())
    return _regularized_result(result, config, vector, baseline_vector, n_tasks)


def _regularized_result(
    result: EvaluationResult,
    config: TuningConfig,
    vector: tuple[float, ...],
    baseline_vector: tuple[float, ...] | None,
    n_tasks: int,
) -> EvaluationResult:
    total = _apply_regularization(result.total, config, vector, baseline_vector, n_tasks)
    if total == result.total:
        return result
    return dataclasses.replace(result, total=total)


def _apply_regularization(
    total: float,
    config: TuningConfig,
    vector: tuple[float, ...],
    baseline_vector: tuple[float, ...] | None,
    n_tasks: int,
) -> float:
    policy = config.search.single_task_regularization
    if policy != "trust_region" or n_tasks != 1 or baseline_vector is None or not vector:
        return total
    radius = config.search.trust_region_radius
    if radius is None:
        raise ValueError("trust_region regularization requires a configured radius")
    distance = float(np.linalg.norm(np.asarray(vector) - np.asarray(baseline_vector)))
    if distance <= radius:
        return total
    return total + _TRUST_REGION_PENALTY * (distance - radius)
