"""Multifidelity simulator-only search.

The loop screens feasible Sobol candidates at the cheap minimum-budget fidelity,
advances the strong and deliberately parameter-diverse candidates to the full
multiresolution grid, refines with a population optimizer, then reranks the
frozen finalists on independent selection panels and selects once. It contains no
PFN import, construction, training, inference, or metric.
"""

import dataclasses
from collections.abc import Callable
from collections.abc import Sequence

import numpy as np

from ebpfn.cache import EvaluationCache
from ebpfn.config import TuningConfig
from ebpfn.data import TuningTask
from ebpfn.data import content_hash
from ebpfn.priors import EtaVectorizer
from ebpfn.priors import build_hyperprior
from ebpfn.utils import RandomRole
from ebpfn.utils import RandomStreams

from .contracts import CandidateRecord
from .contracts import Panel
from .contracts import RealTarget
from .contracts import SearchResult
from .evaluate import characterize_task
from .evaluate import evaluate_candidate
from .optimizer import optimize_population
from .panels import make_panel
from .panels import make_panels

_FIDELITIES = ("min", "full")


def _vector_key(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(value), 12) for value in vector)


def _farthest_point_diverse(candidates: list[np.ndarray], selected: list[np.ndarray], count: int) -> list[np.ndarray]:
    """Greedily pick ``count`` candidates maximizing distance to chosen vectors."""
    chosen: list[np.ndarray] = []
    reference = list(selected)
    pool = list(candidates)
    while pool and len(chosen) < count:
        if reference:
            distances = [min(float(np.linalg.norm(vector - other)) for other in reference) for vector in pool]
        else:
            distances = [float(np.linalg.norm(vector)) for vector in pool]
        index = int(np.argmax(distances))
        picked = pool.pop(index)
        chosen.append(picked)
        reference.append(picked)
    return chosen


def run_search(
    config: TuningConfig,
    real_tasks: Sequence[TuningTask],
    streams: RandomStreams,
    *,
    cache: EvaluationCache | None = None,
    vectorizer: EtaVectorizer | None = None,
) -> SearchResult:
    if not real_tasks:
        raise ValueError("search requires at least one real task")
    if len(real_tasks) != 1 and config.search.single_task_regularization != "none":
        raise ValueError("single-task regularization requires exactly one real task")
    if cache is None and config.cache.enabled:
        # Honor the configured run-local cache without requiring the caller to
        # wire it up. Pass cache=EvaluationCache(..., enabled=False) to opt out.
        cache = EvaluationCache(config.cache.root, config.cache.enabled)
    base_eta = build_hyperprior(config.prior)
    vectorizer = vectorizer or EtaVectorizer(base_eta, config.active)
    baseline_vector = tuple(float(value) for value in vectorizer.encode(base_eta))

    real_chars = {
        fidelity: [characterize_task(task, config.characterization, fidelity) for task in real_tasks]
        for fidelity in _FIDELITIES
    }

    def targets(fidelity: str) -> list[RealTarget]:
        return [RealTarget(task, real_chars[fidelity][index]) for index, task in enumerate(real_tasks)]

    def evaluate(vector: np.ndarray, fidelity: str, panel: Panel, origin: str) -> CandidateRecord:
        eta = vectorizer.decode(vector)
        result = evaluate_candidate(
            eta,
            targets(fidelity),
            config,
            streams,
            panel,
            fidelity,
            cache=cache,
            vectorizer=vectorizer,
            baseline_vector=baseline_vector,
        )
        return CandidateRecord(vector=_vector_key(vector), origin=origin, result=result)

    search_panel = make_panel("search", 0, config, streams)

    # 1. Screen feasible Sobol candidates (plus the baseline) at min fidelity.
    sobol_rng = streams.generator(RandomRole.SEARCH, "sobol")
    sobol_vectors = list(vectorizer.sobol(config.search.sobol_candidates, sobol_rng))
    screen_vectors = [np.asarray(baseline_vector), *sobol_vectors]
    screen_records = [evaluate(vector, "min", search_panel, "sobol") for vector in screen_vectors]

    # 2. Retain strong + deliberately parameter-diverse candidates.
    order = sorted(range(len(screen_records)), key=lambda i: screen_records[i].result.total)
    strong = [screen_vectors[i] for i in order[: config.search.retain_strong]]
    remaining = [screen_vectors[i] for i in order[config.search.retain_strong :]]
    diverse = _farthest_point_diverse(remaining, strong, config.search.retain_diverse)

    # 3. Re-evaluate the advanced set over the full multiresolution grid.
    advanced_vectors = _dedup([*strong, *diverse])
    advanced_records = [evaluate(vector, "full", search_panel, "advanced") for vector in advanced_vectors]
    finalist_vectors = list(advanced_vectors)

    # 4. Population optimizer with its configured fidelity schedule.
    optimizer_records: list[CandidateRecord] = []
    if config.search.optimizer == "differential_evolution":

        def objective(vector: np.ndarray) -> float:
            record = evaluate(vector, config.search.de_fidelity, search_panel, "optimizer")
            optimizer_records.append(record)
            return record.result.total

        de_rng = streams.generator(RandomRole.SEARCH, "differential-evolution")
        best = optimize_population(
            objective,
            vectorizer.is_feasible,
            vectorizer.dimension,
            de_rng,
            maxiter=config.search.de_maxiter,
            popsize=config.search.de_popsize,
        )
        if vectorizer.is_feasible(best):
            finalist_vectors = _dedup([*finalist_vectors, best])

    # 5. Freeze finalists and rerank on independent selection panels.
    selection_panels = make_panels("selection", config.search.selection_panel_size, config, streams)
    ranking, selection_records = _rerank(finalist_vectors, selection_panels, evaluate, baseline_vector, config)

    # 6. Select once; the final-audit stage may never revise this.
    finalist = ranking[0]
    finalist_vector = np.asarray(finalist.vector)
    return SearchResult(
        finalist_eta=vectorizer.decode(finalist_vector),
        finalist_vector=finalist.vector,
        selection_ranking=ranking,
        search_records=[*screen_records, *advanced_records, *optimizer_records],
        optimizer_records=optimizer_records,
        selection_records=selection_records,
    )


def _dedup(vectors: list[np.ndarray]) -> list[np.ndarray]:
    seen: set[tuple[float, ...]] = set()
    unique: list[np.ndarray] = []
    for vector in vectors:
        key = _vector_key(vector)
        if key not in seen:
            seen.add(key)
            unique.append(vector)
    return unique


def _rerank(
    finalist_vectors: list[np.ndarray],
    panels: list[Panel],
    evaluate: Callable[[np.ndarray, str, Panel, str], CandidateRecord],
    baseline_vector: tuple[float, ...],
    config: TuningConfig,
) -> tuple[list[CandidateRecord], list[CandidateRecord]]:
    scored: list[tuple[float, float, CandidateRecord]] = []
    all_panel_records: list[CandidateRecord] = []
    for vector in finalist_vectors:
        panel_records = [evaluate(vector, "full", panel, "finalist") for panel in panels]
        all_panel_records.extend(panel_records)
        mean_total = float(np.mean([record.result.total for record in panel_records]))
        baseline_distance = float(np.linalg.norm(np.asarray(vector) - np.asarray(baseline_vector)))
        panel_results = tuple(record.result for record in panel_records)
        block_names = set(panel_results[0].per_block)
        if any(set(result.per_block) != block_names for result in panel_results[1:]):
            raise ValueError("selection panels returned inconsistent block schemas")
        selection_result = dataclasses.replace(
            panel_results[0],
            total=mean_total,
            per_block={
                block: float(np.mean([result.per_block[block] for result in panel_results])) for block in block_names
            },
            objective_terms={
                "objective": config.objective,
                "aggregation": "selection_panel_mean",
                "panel_totals": [result.total for result in panel_results],
                "panel_cache_keys": [result.cache_key for result in panel_results],
            },
            failures=sum(result.failures for result in panel_results),
            failure_events=tuple(event for result in panel_results for event in result.failure_events),
            runtime_s=sum(result.runtime_s for result in panel_results),
            seeds={"panels": [result.seeds for result in panel_results]},
            cache_key=content_hash(*(result.cache_key for result in panel_results), namespace="selection-aggregate-1"),
        )
        aggregate_record = dataclasses.replace(panel_records[0], result=selection_result, panel_results=panel_results)
        scored.append((mean_total, baseline_distance, aggregate_record))

    scored.sort(key=lambda entry: (entry[0], entry[1]))
    if config.search.single_task_regularization == "closest_to_baseline":
        tolerance = config.search.competitive_tolerance
        if tolerance is None:
            raise ValueError("closest_to_baseline requires a competitive tolerance")
        best_loss = scored[0][0]
        competitive = [entry for entry in scored if entry[0] <= best_loss + tolerance]
        selected = min(competitive, key=lambda entry: entry[1])
        scored = [selected, *(entry for entry in scored if entry is not selected)]
    return [record for _, _, record in scored], all_panel_records
