"""Persistent source roles and task-level eligibility intersections."""

from dataclasses import dataclass

import numpy as np

from ebpfn.config import SplitConfig
from ebpfn.data.hashing import content_hash
from ebpfn.data.types import RawTabularTask
from ebpfn.data.types import SourceSplit


@dataclass(frozen=True)
class EligibilityReport:
    task_id: str
    admitted: bool
    counts: dict[str, int]
    missing_targets: dict[str, int]
    reasons: tuple[str, ...]


def _partition(
    ids: np.ndarray, first_fraction: float, rng: np.random.Generator
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shuffled = rng.permutation(ids)
    cut = round(len(ids) * first_fraction)
    return tuple(sorted(int(value) for value in shuffled[:cut])), tuple(sorted(int(value) for value in shuffled[cut:]))


def create_source_split(
    source_id: str,
    n_rows: int,
    config: SplitConfig,
    *,
    official_train_ids: tuple[int, ...] | None = None,
    official_test_ids: tuple[int, ...] | None = None,
) -> SourceSplit:
    """Assign positional source rows once, preserving an official final-test role."""

    if n_rows < 3:
        raise ValueError("a source needs at least three rows")
    all_ids = set(range(n_rows))
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, *source_id.encode()]))
    if (official_train_ids is None) != (official_test_ids is None):
        raise ValueError("official train and test IDs must be provided together")
    if official_train_ids is not None and official_test_ids is not None:
        train = tuple(sorted(official_train_ids))
        test = tuple(sorted(official_test_ids))
        if set(train) | set(test) != all_ids or set(train) & set(test):
            raise ValueError("official train/test IDs must be a disjoint partition of the source")
        probe_fit, probe_score = _partition(
            np.asarray(train),
            1.0 - config.probe_score_fraction_of_train,
            rng,
        )
        final_test = test
    else:
        train, final_test = _partition(
            np.arange(n_rows),
            1.0 - config.final_test_fraction,
            rng,
        )
        probe_fit, probe_score = _partition(
            np.asarray(train),
            1.0 - config.probe_score_fraction_of_train,
            rng,
        )
    manifest = (source_id, probe_fit, probe_score, final_test, config.policy_version, config.seed)
    return SourceSplit(
        source_id,
        probe_fit,
        probe_score,
        final_test,
        content_hash(manifest, namespace="outer-split-1"),
        config.policy_version,
        config.seed,
    )


def eligible_role_ids(
    task: RawTabularTask, split: SourceSplit, config: SplitConfig
) -> tuple[dict[str, tuple[int, ...]], EligibilityReport]:
    positions = {int(row_id): index for index, row_id in enumerate(task.row_ids)}
    finite_ids = {int(task.row_ids[index]) for index in np.flatnonzero(np.isfinite(task.y.astype(float)))}
    roles = {
        "probe_fit": tuple(row_id for row_id in split.probe_fit_ids if row_id in finite_ids and row_id in positions),
        "probe_score": tuple(
            row_id for row_id in split.probe_score_ids if row_id in finite_ids and row_id in positions
        ),
        "final_test": tuple(row_id for row_id in split.final_test_ids if row_id in finite_ids and row_id in positions),
    }
    original = {
        "probe_fit": sum(row_id in positions for row_id in split.probe_fit_ids),
        "probe_score": sum(row_id in positions for row_id in split.probe_score_ids),
        "final_test": sum(row_id in positions for row_id in split.final_test_ids),
    }
    counts = {name: len(ids) for name, ids in roles.items()}
    missing = {name: original[name] - counts[name] for name in roles}
    minimums = {
        "probe_fit": config.min_probe_fit,
        "probe_score": config.min_probe_score,
        "final_test": config.min_final_test,
    }
    reasons = tuple(f"{name}_below_minimum" for name in roles if counts[name] < minimums[name])
    return roles, EligibilityReport(task.task_id, not reasons, counts, missing, reasons)


def characterization_split_id(task: RawTabularTask, split: SourceSplit, roles: dict[str, tuple[int, ...]]) -> str:
    return content_hash(
        task.task_id,
        split.outer_split_id,
        roles["probe_fit"],
        roles["probe_score"],
        split.policy_version,
        split.seed,
        namespace="characterization-split-1",
    )
