"""Deterministic nested row-budget manifests."""

import hashlib
import math

import numpy as np

from ebpfn.config import CharacterizationConfig
from ebpfn.data import TuningTask
from ebpfn.data import content_hash

from .contracts import RowBudgetManifest


def _seed(*parts: str | int) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _budget_grid(total: int, minimum: int, spacing: str) -> tuple[int, ...]:
    start = min(total, minimum)
    if start == total:
        return (total,)
    geometric = [start]
    while geometric[-1] * 2 < total:
        geometric.append(geometric[-1] * 2)
    levels = len(geometric) + 1
    if spacing == "geometric":
        return (*geometric, total)
    roots = np.linspace(math.sqrt(start), math.sqrt(total), levels)
    return tuple(sorted({start, *(round(value * value) for value in roots[1:-1]), total}))


def build_row_budget_manifests(task: TuningTask, config: CharacterizationConfig) -> tuple[RowBudgetManifest, ...]:
    n_fit = task.probe_fit.X.height
    n_score = task.probe_score.X.height
    total = n_fit + n_score
    budgets = _budget_grid(total, config.row_budgets.minimum, config.row_budgets.spacing)
    identity = (task.task_id, task.characterization_split_id, config.version, config.seed, config.repeat)
    fit_order = np.random.default_rng(_seed(*identity, "fit-order")).permutation(n_fit)
    score_order = np.random.default_rng(_seed(*identity, "score-order")).permutation(n_score)
    manifests: list[RowBudgetManifest] = []
    for budget in budgets:
        minimum_fit = max(1, budget - n_score)
        maximum_fit = min(n_fit, budget - 1)
        if minimum_fit > maximum_fit:
            raise ValueError(f"row budget {budget} cannot retain both fit and score rows")
        selected_fit = min(maximum_fit, max(minimum_fit, round(budget * n_fit / total)))
        selected_score = budget - selected_fit
        fit_indices = tuple(int(index) for index in fit_order[:selected_fit])
        score_indices = tuple(int(index) for index in score_order[:selected_score])
        feature_indices = tuple(range(len(task.schema.names)))
        if config.row_budgets.feature_view == "local":
            frame = task.probe_fit.X[list(fit_indices)].to_numpy()
            keep = np.ptp(frame, axis=0) > config.ridge.column_tolerance
            feature_indices = tuple(int(index) for index in np.flatnonzero(keep))
            if not feature_indices:
                raise ValueError(f"row budget {budget} has no locally usable features")
        weight = 1.0 if config.row_budgets.weight == "uniform" else float(budget)
        manifest_id = content_hash(
            identity,
            budget,
            fit_indices,
            score_indices,
            feature_indices,
            config.row_budgets,
            namespace="row-budget-manifest-1",
        )
        manifests.append(RowBudgetManifest(budget, fit_indices, score_indices, feature_indices, weight, manifest_id))
    total_weight = sum(manifest.weight for manifest in manifests)
    return tuple(
        RowBudgetManifest(
            manifest.row_budget,
            manifest.probe_fit_indices,
            manifest.probe_score_indices,
            manifest.feature_indices,
            manifest.weight / total_weight,
            manifest.manifest_id,
        )
        for manifest in manifests
    )
