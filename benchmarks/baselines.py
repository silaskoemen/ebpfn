"""Strong probabilistic regression baselines + Gaussian scoring, shared by the P0
headroom diagnostic and (later) the larger bank/validation studies.

Each baseline fits on standardized targets and returns per-row ``(mean, std)`` so a
Gaussian NLL is directly comparable to the PFN's own NLL in the same standardized
space. sklearn + catboost live in the ``bench`` feature (also in the
``default`` env); nothing here imports the PFN.
"""

import numpy as np
from scipy.special import erf, ndtri
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

_STD_FLOOR = 1e-6
_Z90 = 1.6448536269514722  # central-90% Gaussian half-width


def gp_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, max_samples: int = 3000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """GP regressor (RBF + white-noise), subsampled to ``max_samples`` train rows."""
    if len(x_train) > max_samples:
        idx = np.random.default_rng(seed).choice(len(x_train), max_samples, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=False, n_restarts_optimizer=1, alpha=1e-6, random_state=seed
    )
    gp.fit(x_train, y_train)
    mean, std = gp.predict(x_test, return_std=True)
    return mean, np.maximum(std, _STD_FLOOR)


def catboost_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, iterations: int = 500, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """CatBoost with ``RMSEWithUncertainty`` — predicts per-row mean + data variance."""
    from catboost import CatBoostRegressor  # lazy: catboost's conda build can lag numpy's ABI

    model = CatBoostRegressor(
        loss_function="RMSEWithUncertainty",
        iterations=iterations,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_test))  # (n, 2): [mean, variance]
    return pred[:, 0], np.sqrt(np.maximum(pred[:, 1], _STD_FLOOR**2))


def gaussian_nll(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    std = np.maximum(std, _STD_FLOOR)
    return float(np.mean(0.5 * np.log(2 * np.pi * std**2) + (y - mean) ** 2 / (2 * std**2)))


def rmse(y: np.ndarray, mean: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - mean) ** 2)))


def gaussian_crps(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """CRPS of a Gaussian forecast, in closed form.

    The PFN's CRPS comes from its own predictive distribution; a Gaussian baseline needs
    this to be scored on the same axis, and the analytic form avoids the quadrature grid
    being a second source of difference between the two.
    """
    z = (y - mean) / std
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return float(np.mean(std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))))


def gaussian_coverage(y: np.ndarray, mean: np.ndarray, std: np.ndarray, level: float) -> float:
    """Empirical coverage of a Gaussian forecast's central ``level`` interval."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    half = float(ndtri(0.5 + level / 2.0)) * std
    return float(np.mean(np.abs(y - mean) <= half))


def coverage90(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    lo, hi = mean - _Z90 * std, mean + _Z90 * std
    return float(np.mean((y >= lo) & (y <= hi)))
