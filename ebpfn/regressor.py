"""Probabilistic regressors for p(Y|X) (spec §3.3).

Point predictors are useless here by construction (the means match); only proper
scoring / calibration moves. Two heads share a `Predictive` interface so they
plug into the same calibration path:

- catboost_gauss: CatBoost `RMSEWithUncertainty` -> per-point Gaussian (mean, var).
  Gives an *exact* Gaussian NLL and closed-form CRPS. Primary head.
- qgbm: lightgbm quantile grid -> a sorted quantile function. NLL via a Gaussian
  surrogate from the quantiles; CRPS from the pinball integral. Second opinion.
"""

from __future__ import annotations

import warnings
from abc import ABC
from abc import abstractmethod

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import norm

from ebpfn.config import ModelConfig
from ebpfn.priors import Dataset

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


class Predictive(ABC):
    """A fitted predictive distribution over a fixed test X. All methods return
    per-point arrays (or a per-point quantile) so calibration can aggregate."""

    @abstractmethod
    def nll(self, y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def crps(self, y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def cdf(self, y: np.ndarray) -> np.ndarray:
        """Predictive CDF at y -> PIT values."""

    @abstractmethod
    def quantile(self, level: float) -> np.ndarray:
        """Predicted quantile at `level` for each test point."""


class GaussianPredictive(Predictive):
    def __init__(self, mean: np.ndarray, sigma: np.ndarray, eps: float = 1e-6):
        self.mean = mean
        self.sigma = np.maximum(sigma, eps)

    def nll(self, y: np.ndarray) -> np.ndarray:
        s = self.sigma
        return 0.5 * np.log(2 * np.pi) + np.log(s) + (y - self.mean) ** 2 / (2 * s**2)

    def crps(self, y: np.ndarray) -> np.ndarray:
        # CRPS(N(mu,sigma), y) = sigma * [ w(2 Phi(w) - 1) + 2 phi(w) - 1/sqrt(pi) ]
        w = (y - self.mean) / self.sigma
        return self.sigma * (w * (2 * norm.cdf(w) - 1) + 2 * norm.pdf(w) - _INV_SQRT_PI)

    def cdf(self, y: np.ndarray) -> np.ndarray:
        return norm.cdf((y - self.mean) / self.sigma)

    def quantile(self, level: float) -> np.ndarray:
        return self.mean + self.sigma * norm.ppf(level)


class QuantilePredictive(Predictive):
    def __init__(self, Q: np.ndarray, levels: np.ndarray):
        self.Q = Q  # (n, K), sorted across K
        self.levels = levels

    def quantile(self, level: float) -> np.ndarray:
        return np.array([np.interp(level, self.levels, q) for q in self.Q])

    def cdf(self, y: np.ndarray) -> np.ndarray:
        return np.array([np.interp(y[i], self.Q[i], self.levels, left=0.0, right=1.0) for i in range(y.size)])

    def nll(self, y: np.ndarray) -> np.ndarray:
        mu = self.quantile(0.5)
        sigma = np.maximum((self.quantile(0.8413) - self.quantile(0.1587)) / 2.0, 1e-6)
        return 0.5 * np.log(2 * np.pi) + np.log(sigma) + (y - mu) ** 2 / (2 * sigma**2)

    def crps(self, y: np.ndarray) -> np.ndarray:
        # CRPS = 2 * integral_0^1 pinball_alpha d_alpha, grid-averaged over levels.
        diff = y[:, None] - self.Q
        pin = np.maximum(self.levels[None, :] * diff, (self.levels[None, :] - 1.0) * diff)
        return 2.0 * pin.mean(axis=1)


class ProbModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> ProbModel: ...

    @abstractmethod
    def predictive(self, X: np.ndarray) -> Predictive: ...


class GaussianCatBoost(ProbModel):
    """CatBoost RMSEWithUncertainty -> per-point Gaussian (data uncertainty)."""

    def __init__(self, config: ModelConfig):
        self._cfg = config
        self.model: CatBoostRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianCatBoost:
        cfg = self._cfg
        self.model = CatBoostRegressor(
            loss_function="RMSEWithUncertainty",
            iterations=cfg.catboost_iterations,
            learning_rate=cfg.catboost_learning_rate,
            depth=cfg.catboost_depth,
            verbose=0,
            allow_writing_files=False,
        )
        self.model.fit(X, y)
        return self

    def predictive(self, X: np.ndarray) -> GaussianPredictive:
        assert self.model is not None, "fit before predictive"
        p = self.model.predict(X)  # (n, 2): mean, variance
        return GaussianPredictive(mean=p[:, 0], sigma=np.sqrt(np.maximum(p[:, 1], 1e-12)))


class QuantileGBM(ProbModel):
    """A grid of lightgbm quantile regressors approximating p(Y|X)."""

    def __init__(self, config: ModelConfig):
        self.quantiles = np.asarray(sorted(config.quantiles), dtype=float)
        self._cfg = config
        self.models: list[LGBMRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> QuantileGBM:
        cfg = self._cfg
        self.models = [
            LGBMRegressor(
                objective="quantile",
                alpha=float(q),
                n_estimators=cfg.n_estimators,
                learning_rate=cfg.learning_rate,
                num_leaves=cfg.num_leaves,
                min_child_samples=cfg.min_child_samples,
                verbose=-1,
            ).fit(X, y)
            for q in self.quantiles
        ]
        return self

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """(n, Q) predicted quantiles, sorted across Q to avoid crossing."""
        with warnings.catch_warnings():
            # lightgbm fits on ndarrays; sklearn's predict-time feature-name check is noise.
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            Q = np.column_stack([m.predict(X) for m in self.models])
        return np.sort(Q, axis=1)

    def predictive(self, X: np.ndarray) -> QuantilePredictive:
        return QuantilePredictive(self.predict_quantiles(X), self.quantiles)


_REGISTRY = {"catboost_gauss": GaussianCatBoost, "qgbm": QuantileGBM}


def train_prob_regressor(D: Dataset, kind: str, config: ModelConfig) -> ProbModel:
    """Train a probabilistic regressor on a dataset (spec §6 signature)."""
    if kind not in _REGISTRY:
        raise NotImplementedError(f"kind {kind!r} not in {sorted(_REGISTRY)}")
    return _REGISTRY[kind](config).fit(D.X, D.Y)
