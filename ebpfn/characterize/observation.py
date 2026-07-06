"""Fixed, probe-fit-only observation diagnostics."""

import numpy as np


def _saturate(value: float, scale: float) -> float:
    return value / (scale + abs(value))


def _moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    safe = np.where(scale > 0.0, scale, 1.0)
    standardized = centered / safe
    return np.mean(standardized**3, axis=0), np.mean(standardized**4, axis=0) - 3.0


def observation_statistics(
    features: np.ndarray, targets: np.ndarray, missing_rates: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    if features.shape[1] == 1:
        correlations = np.array([0.0])
    else:
        matrix = np.corrcoef(features, rowvar=False)
        correlations = np.abs(matrix[np.triu_indices(features.shape[1], 1)])
        correlations = np.nan_to_num(correlations)
    covariance = np.cov(features, rowvar=False)
    eigenvalues = np.atleast_1d(np.linalg.eigvalsh(np.atleast_2d(covariance)))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if float(np.sum(eigenvalues)) == 0.0:
        effective_rank_fraction = 1.0 / features.shape[1]
    else:
        probabilities = eigenvalues / np.sum(eigenvalues)
        positive = probabilities[probabilities > 0.0]
        effective_rank_fraction = float(np.exp(-np.sum(positive * np.log(positive))) / features.shape[1])
    skew, kurtosis = _moments(features)
    target_skew, target_kurtosis = _moments(targets[:, [0]])
    uniqueness = np.array([len(np.unique(features[:, index])) / len(features) for index in range(features.shape[1])])
    raw = {
        "feature_correlation_mean": float(np.mean(correlations)),
        "feature_correlation_max": float(np.max(correlations)),
        "effective_rank_fraction": effective_rank_fraction,
        "feature_abs_skew_median": float(np.median(np.abs(skew))),
        "feature_abs_kurtosis_median": float(np.median(np.abs(kurtosis))),
        "feature_outlier_fraction": float(np.mean(np.abs(features) >= 3.0)),
        "feature_uniqueness_median": float(np.median(uniqueness)),
        "feature_missingness_mean": float(np.mean(missing_rates)),
        "feature_missingness_max": float(np.max(missing_rates)),
        "target_skew": float(target_skew[0]),
        "target_excess_kurtosis": float(target_kurtosis[0]),
        "lower_tail_prevalence": float(np.mean(targets[:, 3])),
        "upper_tail_prevalence": float(np.mean(targets[:, 4])),
        "tail_prevalence_asymmetry": float(np.mean(targets[:, 4]) - np.mean(targets[:, 3])),
    }
    values = {
        "feature_correlation_mean": 2.0 * raw["feature_correlation_mean"] - 1.0,
        "feature_correlation_max": 2.0 * raw["feature_correlation_max"] - 1.0,
        "effective_rank_fraction": 2.0 * raw["effective_rank_fraction"] - 1.0,
        "feature_abs_skew_median": _saturate(raw["feature_abs_skew_median"], 2.0),
        "feature_abs_kurtosis_median": _saturate(raw["feature_abs_kurtosis_median"], 6.0),
        "feature_outlier_fraction": 2.0 * raw["feature_outlier_fraction"] - 1.0,
        "feature_uniqueness_median": 2.0 * raw["feature_uniqueness_median"] - 1.0,
        "feature_missingness_mean": 2.0 * raw["feature_missingness_mean"] - 1.0,
        "feature_missingness_max": 2.0 * raw["feature_missingness_max"] - 1.0,
        "target_skew": _saturate(raw["target_skew"], 2.0),
        "target_excess_kurtosis": _saturate(raw["target_excess_kurtosis"], 6.0),
        "lower_tail_prevalence": 2.0 * raw["lower_tail_prevalence"] - 1.0,
        "upper_tail_prevalence": 2.0 * raw["upper_tail_prevalence"] - 1.0,
        "tail_prevalence_asymmetry": raw["tail_prevalence_asymmetry"],
    }
    return raw, values
