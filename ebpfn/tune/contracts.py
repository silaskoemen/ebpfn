"""Result and panel contracts for simulator-only tuning.

``EvaluationResult`` owns JSON (de)serialization so the cache store can stay a
dependency leaf that only handles plain payload dicts.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ebpfn.characterize import TaskCharacterization
from ebpfn.data import TuningTask
from ebpfn.priors import BnnHyperPrior, CompositionalHyperPrior, HyperPrior, ScmHyperPrior, TreeHyperPrior


def _eta_from_dict(payload: dict[str, Any]) -> HyperPrior:
    return HyperPrior(
        generator_weights={str(name): float(weight) for name, weight in payload["generator_weights"].items()},
        corr_strength_mean=payload["corr_strength_mean"],
        log_snr_mean=payload["log_snr_mean"],
        heteroskedastic_rate=payload["heteroskedastic_rate"],
        heavy_tail_rate=payload["heavy_tail_rate"],
        snr_dispersion=payload["snr_dispersion"],
        corr_dispersion=payload["corr_dispersion"],
        scm=ScmHyperPrior(**payload["scm"]),
        bnn=BnnHyperPrior(**payload["bnn"]),
        tree=TreeHyperPrior(**payload["tree"]),
        compositional=CompositionalHyperPrior(**payload["compositional"]),
    )


@dataclass(frozen=True)
class RealTarget:
    """A real tuning task paired with its characterization at one fidelity."""

    task: TuningTask
    characterization: TaskCharacterization


@dataclass(frozen=True)
class Panel:
    """A reproducible generation-slot manifest for one random stage.

    Candidate evaluations within a panel reuse the same generation slots (the
    ``stage``/``token`` identity threaded into cloud sampling, plus any common
    energy pair sample), so only ``eta`` varies across candidates.
    """

    stage: str
    token: int
    energy_pair_ids: tuple[tuple[int, int], ...] | None = None

    def identity(self) -> tuple[str, int]:
        return (self.stage, self.token)


@dataclass(frozen=True)
class FailureEvent:
    """One excluded synthetic member failure with enough context for D1."""

    task_id: str
    source_id: str
    member_index: int
    phase: str
    fidelity: str
    row_budget: int | None
    route: str | None
    shape: dict[str, int]
    exception_type: str
    message: str

    def to_payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FailureEvent":
        return cls(
            task_id=str(payload["task_id"]),
            source_id=str(payload["source_id"]),
            member_index=int(payload["member_index"]),
            phase=str(payload["phase"]),
            fidelity=str(payload["fidelity"]),
            row_budget=None if payload["row_budget"] is None else int(payload["row_budget"]),
            route=None if payload["route"] is None else str(payload["route"]),
            shape={str(name): int(value) for name, value in payload["shape"].items()},
            exception_type=str(payload["exception_type"]),
            message=str(payload["message"]),
        )


@dataclass(frozen=True)
class EvaluationResult:
    """The stored outcome of one candidate/stage evaluation."""

    total: float
    per_block: dict[str, float]
    objective_terms: dict[str, Any]
    failures: int
    failure_events: tuple[FailureEvent, ...]
    runtime_s: float
    candidate_vector: tuple[float, ...]
    eta: HyperPrior
    stage: str
    fidelity: str
    seeds: dict[str, Any]
    cache_key: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "per_block": dict(self.per_block),
            "objective_terms": self.objective_terms,
            "failures": self.failures,
            "failure_events": [event.to_payload() for event in self.failure_events],
            "runtime_s": self.runtime_s,
            "candidate_vector": list(self.candidate_vector),
            "eta": dataclasses.asdict(self.eta),
            "stage": self.stage,
            "fidelity": self.fidelity,
            "seeds": self.seeds,
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EvaluationResult":
        return cls(
            total=float(payload["total"]),
            per_block={str(block): float(value) for block, value in payload["per_block"].items()},
            objective_terms=payload["objective_terms"],
            failures=int(payload["failures"]),
            failure_events=tuple(FailureEvent.from_payload(event) for event in payload["failure_events"]),
            runtime_s=float(payload["runtime_s"]),
            candidate_vector=tuple(float(value) for value in payload["candidate_vector"]),
            eta=_eta_from_dict(payload["eta"]),
            stage=str(payload["stage"]),
            fidelity=str(payload["fidelity"]),
            seeds=payload["seeds"],
            cache_key=str(payload["cache_key"]),
        )


@dataclass(frozen=True)
class CandidateRecord:
    """A candidate vector, its origin, and its evaluation result."""

    vector: tuple[float, ...]
    origin: str
    result: EvaluationResult
    panel_results: tuple[EvaluationResult, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    """The frozen outcome of a search: one selected finalist plus evidence."""

    finalist_eta: HyperPrior
    finalist_vector: tuple[float, ...]
    selection_ranking: list[CandidateRecord]
    search_records: list[CandidateRecord] = field(default_factory=list)
    optimizer_records: list[CandidateRecord] = field(default_factory=list)
    selection_records: list[CandidateRecord] = field(default_factory=list)
    real_targets_by_fidelity: dict[str, tuple[RealTarget, ...]] = field(default_factory=dict)
