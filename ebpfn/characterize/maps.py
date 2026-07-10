"""Deterministic fixed feature maps."""

import hashlib
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from ebpfn.config import MapConfig


@dataclass(frozen=True)
class FeatureMap:
    name: str
    fit: np.ndarray
    score: np.ndarray
    column_names: tuple[str, ...]


def _rng(seed_parts: tuple[str | int, ...]) -> np.random.Generator:
    digest = hashlib.sha256("\0".join(map(str, seed_parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _bins(
    fit: np.ndarray, score: np.ndarray, names: tuple[str, ...], quantiles: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    thresholds = np.quantile(fit, quantiles, axis=0).T
    fit_columns = [fit]
    score_columns = [score]
    output_names = list(names)
    for feature, name in enumerate(names):
        for quantile, threshold in zip(quantiles, thresholds[feature], strict=True):
            fit_columns.append((fit[:, feature] >= threshold)[:, None])
            score_columns.append((score[:, feature] >= threshold)[:, None])
            output_names.append(f"{name}>=q{quantile:g}")
    return np.column_stack(fit_columns), np.column_stack(score_columns), tuple(output_names), thresholds


def _pairwise(
    fit: np.ndarray,
    score: np.ndarray,
    names: tuple[str, ...],
    bins_fit: np.ndarray,
    bins_score: np.ndarray,
    bin_names: tuple[str, ...],
    thresholds: np.ndarray,
    config: MapConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    fit_columns = [bins_fit]
    score_columns = [bins_score]
    output_names = list(bin_names)
    pairs = list(combinations(range(fit.shape[1]), 2))
    if pairs:
        for index in rng.permutation(len(pairs))[: config.max_products]:
            left, right = pairs[int(index)]
            fit_columns.append((fit[:, left] * fit[:, right])[:, None])
            score_columns.append((score[:, left] * score[:, right])[:, None])
            output_names.append(f"{names[left]}*{names[right]}")
        candidates = list(
            product(range(len(pairs)), range(len(config.bin_quantiles)), range(len(config.bin_quantiles)))
        )
        accepted = 0
        for index in rng.permutation(len(candidates)):
            pair_index, left_q, right_q = candidates[int(index)]
            left, right = pairs[pair_index]
            fit_column = (fit[:, left] >= thresholds[left, left_q]) & (fit[:, right] >= thresholds[right, right_q])
            prevalence = float(np.mean(fit_column))
            if not config.conjunction_min_prevalence <= prevalence <= config.conjunction_max_prevalence:
                continue
            score_column = (score[:, left] >= thresholds[left, left_q]) & (
                score[:, right] >= thresholds[right, right_q]
            )
            fit_columns.append(fit_column[:, None])
            score_columns.append(score_column[:, None])
            output_names.append(
                f"{names[left]}>=q{config.bin_quantiles[left_q]:g}&{names[right]}>=q{config.bin_quantiles[right_q]:g}"
            )
            accepted += 1
            if accepted == config.max_conjunctions:
                break
    return np.column_stack(fit_columns), np.column_stack(score_columns), tuple(output_names)


def _median_bandwidth(fit: np.ndarray, maximum_rows: int, rng: np.random.Generator) -> float | None:
    if len(fit) > maximum_rows:
        fit = fit[rng.choice(len(fit), maximum_rows, replace=False)]
    squared = np.sum((fit[:, None, :] - fit[None, :, :]) ** 2, axis=2)
    distances = np.sqrt(squared[np.triu_indices(len(fit), 1)])
    positive = distances[distances > 0.0]
    return None if not len(positive) else float(np.median(positive))


def build_feature_maps(
    fit: np.ndarray,
    score: np.ndarray,
    names: tuple[str, ...],
    config: MapConfig,
    *,
    seed_identity: tuple[str | int, ...],
) -> tuple[FeatureMap, ...]:
    if fit.ndim != 2 or score.ndim != 2 or fit.shape[1] != score.shape[1] or fit.shape[1] != len(names):
        raise ValueError("feature map inputs must be aligned matrices")
    bins_fit, bins_score, bin_names, thresholds = _bins(fit, score, names, config.bin_quantiles)
    pair_rng = _rng((*seed_identity, "pairwise"))
    pair_fit, pair_score, pair_names = _pairwise(
        fit, score, names, bins_fit, bins_score, bin_names, thresholds, config, pair_rng
    )
    maps = [
        FeatureMap("linear", fit.copy(), score.copy(), names),
        FeatureMap("bins", bins_fit, bins_score, bin_names),
        FeatureMap("pairwise", pair_fit, pair_score, pair_names),
    ]
    rff_rng = _rng((*seed_identity, "rff"))
    bandwidth = _median_bandwidth(fit, config.rff_distance_rows, rff_rng)
    if bandwidth is None or config.max_rff == 0:
        rff_fit, rff_score, rff_names = fit.copy(), score.copy(), names
    else:
        weights = rff_rng.normal(size=(fit.shape[1], config.max_rff)) / bandwidth
        phases = rff_rng.uniform(0.0, 2.0 * np.pi, size=config.max_rff)
        rff_fit = np.column_stack((fit, np.sqrt(2.0) * np.cos(fit @ weights + phases)))
        rff_score = np.column_stack((score, np.sqrt(2.0) * np.cos(score @ weights + phases)))
        rff_names = (*names, *(f"rff_{index}" for index in range(config.max_rff)))
    maps.append(FeatureMap("rff", rff_fit, rff_score, tuple(rff_names)))
    return tuple(maps)
