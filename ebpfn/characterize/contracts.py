"""Public characterization contracts and schema validation."""

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Coordinate:
    name: str
    block: str
    learner: str | None = None
    target: str | None = None
    row_budget: int | None = None
    feature_budget: str | None = None
    statistic: str = "gain"
    bounds: tuple[float, float] | None = None
    transform: str = "identity"
    magnitude_semantics: str | None = None
    parent: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.block:
            raise ValueError("coordinate name and block must be nonempty")
        if self.bounds is not None and self.bounds[0] >= self.bounds[1]:
            raise ValueError("coordinate bounds must be increasing")


@dataclass(frozen=True)
class CharacterizationSchema:
    version: str
    representation: str
    coordinates: tuple[Coordinate, ...]

    def __post_init__(self) -> None:
        if not self.version or self.representation not in {"raw", "contrast"}:
            raise ValueError("invalid characterization schema identity")
        names = tuple(coordinate.name for coordinate in self.coordinates)
        if len(set(names)) != len(names):
            raise ValueError("coordinate names must be unique")
        by_name = {coordinate.name: coordinate for coordinate in self.coordinates}
        for coordinate in self.coordinates:
            if self.representation == "raw" and coordinate.statistic == "contrast":
                raise ValueError("raw schemas cannot contain contrast coordinates")
            if (
                self.representation == "contrast"
                and coordinate.statistic == "gain"
                and coordinate.learner not in {None, "linear"}
            ):
                raise ValueError("contrast schemas cannot contain child raw gains")
            if coordinate.parent is not None and coordinate.parent not in by_name:
                raise ValueError(f"missing coordinate parent {coordinate.parent!r}")
            seen: set[str] = set()
            current = coordinate
            while current.parent is not None:
                if current.name in seen:
                    raise ValueError("coordinate parent graph contains a cycle")
                seen.add(current.name)
                current = by_name[current.parent]


@dataclass(frozen=True)
class RowBudgetManifest:
    row_budget: int
    probe_fit_indices: tuple[int, ...]
    probe_score_indices: tuple[int, ...]
    feature_indices: tuple[int, ...]
    weight: float
    manifest_id: str

    def __post_init__(self) -> None:
        if self.row_budget != len(self.probe_fit_indices) + len(self.probe_score_indices):
            raise ValueError("row budget must equal the selected fit and score rows")
        if not self.probe_fit_indices or not self.probe_score_indices or not self.feature_indices:
            raise ValueError("manifest selections must be nonempty")
        if self.weight <= 0.0 or not np.isfinite(self.weight) or not self.manifest_id:
            raise ValueError("manifest weight and identity must be valid")


@dataclass(frozen=True)
class CharacterizationDiagnostics:
    map_dimensions: dict[str, int]
    target_tail_prevalence: dict[str, float]
    ridge_solvers: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskCharacterization:
    task_id: str
    values: np.ndarray
    raw_values: np.ndarray
    valid: np.ndarray
    coordinates: tuple[Coordinate, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be nonempty")
        values = np.array(self.values, dtype=np.float64, copy=True)
        raw = np.array(self.raw_values, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=np.bool_, copy=True)
        expected = (len(self.coordinates),)
        if values.shape != expected or raw.shape != expected or valid.shape != expected:
            raise ValueError("characterization arrays must align with coordinates")
        if not np.isfinite(values).all() or not np.isfinite(raw).all():
            raise ValueError("characterization arrays must be finite")
        for value, coordinate in zip(values, self.coordinates, strict=True):
            if (
                coordinate.bounds is not None
                and not coordinate.bounds[0] - 1e-12 <= value <= coordinate.bounds[1] + 1e-12
            ):
                raise ValueError(f"coordinate {coordinate.name!r} violates its bounds")
        values.flags.writeable = False
        raw.flags.writeable = False
        valid.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "raw_values", raw)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "metadata", dict(self.metadata))
