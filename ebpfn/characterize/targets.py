"""Probe-fit-only regression target functionals."""

from dataclasses import dataclass

import numpy as np

TARGET_NAMES = ("location", "scale_abs", "scale_square", "lower_tail", "upper_tail")


@dataclass(frozen=True)
class TargetFunctionals:
    fit: np.ndarray
    score: np.ndarray
    tail_prevalence: dict[str, float]


def target_functionals(
    y_fit: np.ndarray, y_score: np.ndarray, *, clip: float, scale_epsilon: float
) -> TargetFunctionals:
    center = float(np.median(y_fit))
    scale = float(1.4826 * np.median(np.abs(y_fit - center)))
    if not np.isfinite(scale) or scale <= scale_epsilon:
        raise ValueError("probe-fit target has degenerate robust scale")
    lower, upper = np.quantile(y_fit, (0.2, 0.8))

    def transform(values: np.ndarray) -> np.ndarray:
        z = np.clip((values - center) / scale, -clip, clip)
        return np.column_stack((z, np.abs(z), z**2, values <= lower, values >= upper)).astype(np.float64)

    fit = transform(y_fit)
    score = transform(y_score)
    prevalence = {
        "lower_fit": float(np.mean(fit[:, 3])),
        "upper_fit": float(np.mean(fit[:, 4])),
        "lower_score": float(np.mean(score[:, 3])),
        "upper_score": float(np.mean(score[:, 4])),
    }
    return TargetFunctionals(fit, score, prevalence)
