"""Fixed multi-output ridge probes and normalized held-out gains."""

import hashlib
from dataclasses import dataclass

import numpy as np

from ebpfn.config import RidgeConfig


@dataclass(frozen=True)
class RidgeResult:
    predictions: np.ndarray
    gains: np.ndarray
    dimension: int
    solver: str


def solve_ridge_coefficients(design: np.ndarray, targets: np.ndarray, lambda_: float, *, solver: str) -> np.ndarray:
    if design.ndim != 2 or targets.ndim != 2 or design.shape[0] != targets.shape[0]:
        raise ValueError("ridge design and targets must be aligned matrices")
    if lambda_ <= 0.0 or solver not in {"primal", "dual"}:
        raise ValueError("ridge lambda and solver must be valid")
    n, dimension = design.shape
    if solver == "primal":
        system = design.T @ design + n * lambda_ * np.eye(dimension)
        return np.linalg.solve(system, design.T @ targets)
    system = design @ design.T + n * lambda_ * np.eye(n)
    return design.T @ np.linalg.solve(system, targets)


def _prepared_design(fit: np.ndarray, score: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(fit, axis=0)
    scale = np.std(fit, axis=0)
    keep = np.isfinite(scale) & (scale > tolerance)
    if not np.any(keep):
        return np.empty((len(fit), 0)), np.empty((len(score), 0))
    fit_scaled = (fit[:, keep] - center[keep]) / scale[keep]
    score_scaled = (score[:, keep] - center[keep]) / scale[keep]
    seen: dict[bytes, list[int]] = {}
    unique_indices: list[int] = []
    for index in range(fit_scaled.shape[1]):
        digest = hashlib.blake2b(np.ascontiguousarray(fit_scaled[:, index]).view(np.uint8), digest_size=16).digest()
        matching = seen.setdefault(digest, [])
        if not any(np.array_equal(fit_scaled[:, prior], fit_scaled[:, index]) for prior in matching):
            matching.append(index)
            unique_indices.append(index)
    fit_scaled = fit_scaled[:, unique_indices]
    score_scaled = score_scaled[:, unique_indices]
    normalization = np.sqrt(fit_scaled.shape[1])
    return fit_scaled / normalization, score_scaled / normalization


def fit_ridge_probe(
    map_fit: np.ndarray,
    map_score: np.ndarray,
    targets_fit: np.ndarray,
    targets_score: np.ndarray,
    config: RidgeConfig,
) -> RidgeResult:
    design_fit, design_score = _prepared_design(map_fit, map_score, config.column_tolerance)
    target_center = np.mean(targets_fit, axis=0)
    centered_targets = targets_fit - target_center
    n, dimension = design_fit.shape
    if dimension == 0:
        predictions = np.broadcast_to(target_center, targets_score.shape).copy()
        return RidgeResult(predictions, np.zeros(targets_score.shape[1]), 0, "baseline")
    if dimension <= n:
        solver = "primal"
    else:
        solver = "dual"
    coefficients = solve_ridge_coefficients(design_fit, centered_targets, config.lambda_, solver=solver)
    predictions = target_center + design_score @ coefficients
    baseline = np.broadcast_to(target_center, targets_score.shape)
    baseline_error = np.mean((targets_score - baseline) ** 2, axis=0)
    probe_error = np.mean((targets_score - predictions) ** 2, axis=0)
    gains = (baseline_error - probe_error) / (baseline_error + probe_error + config.gain_epsilon)
    return RidgeResult(predictions, gains, dimension, solver)
