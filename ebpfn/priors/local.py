"""Local prior: tasks from a cloud of DGPs fitted to one target's rows.

η is a population-level knob and a single dataset barely identifies it. The task-level
latent φ is what one dataset *does* pin down, so this prior is shaped by an
approximate p(φ | D): every task is a fresh draw of a plausible DGP for the target, rather than a
selection from the origin prior.

* **x** -- a Gaussian copula fitted to the rows: empirical marginals (kept discrete
  where a column is), Ledoit-Wolf-shrunk normal-score correlation.
* **y** -- half the tasks draw a joint GP posterior sample at the new x (RBF + white
  noise, so the draw carries the fitted noise); the other half fit extra-trees to a
  bootstrap of the rows and add resampled out-of-bag residuals. Two inductive biases,
  so the cloud is not one surrogate's opinion.

Fine-tuning on the rows themselves is the degenerate case: a point mass on the
empirical distribution. This prior instead smooths it into unlimited fresh tasks.
"""

import dataclasses

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

DISCRETE_MAX_UNIQUE = 10
N_TREES = 32
TREE_MIN_LEAF = 3
TREE_MAX_FEATURES = 0.5
GP_RESTARTS = 1


@dataclasses.dataclass(frozen=True)
class LocalPrior:
    x: np.ndarray  # (n, p) rows the prior is fitted to
    y: np.ndarray  # (n,)
    correlation: np.ndarray  # (p, p) normal-score correlation
    discrete: np.ndarray  # (p,) bool, column keeps its observed support
    gp: GaussianProcessRegressor
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    residuals: np.ndarray  # out-of-bag tree residuals

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, *, seed: int) -> "LocalPrior":
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        n = len(y)
        scores = stats.norm.ppf((stats.rankdata(x, axis=0) - 0.5) / n)
        covariance = LedoitWolf().fit(scores).covariance_
        sd = np.sqrt(np.diag(covariance))
        sd[sd == 0] = 1.0  # a constant column: no dependence to carry
        correlation = covariance / np.outer(sd, sd)
        np.fill_diagonal(correlation, 1.0)
        discrete = np.array([len(np.unique(column)) <= DISCRETE_MAX_UNIQUE for column in x.T])

        x_mean, x_scale = x.mean(0), x.std(0)
        x_scale[x_scale == 0] = 1.0
        y_mean, y_scale = float(y.mean()), float(y.std()) or 1.0
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-5, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=GP_RESTARTS, random_state=seed)
        gp.fit((x - x_mean) / x_scale, (y - y_mean) / y_scale)

        forest = _forest(seed).set_params(oob_score=True).fit(x, y)
        residuals = y - forest.oob_prediction_
        return cls(x, y, correlation, discrete, gp, x_mean, x_scale, y_mean, y_scale, residuals[np.isfinite(residuals)])

    def sample_x(self, m: int, rng: np.random.Generator) -> np.ndarray:
        u = stats.norm.cdf(rng.multivariate_normal(np.zeros(len(self.correlation)), self.correlation, size=m))
        columns = [
            np.quantile(column, q, method="inverted_cdf" if discrete else "linear")
            for column, q, discrete in zip(self.x.T, u.T, self.discrete, strict=True)
        ]
        return np.column_stack(columns)

    def sample_gp(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        draw = self.gp.sample_y((x - self.x_mean) / self.x_scale, random_state=int(rng.integers(2**31)))
        return draw[:, 0] * self.y_scale + self.y_mean

    def sample_trees(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        rows = rng.integers(len(self.y), size=len(self.y))
        forest = _forest(int(rng.integers(2**31))).fit(self.x[rows], self.y[rows])
        return forest.predict(x) + rng.choice(self.residuals, size=len(x))

    def sample_task(self, m: int, rng: np.random.Generator, *, surrogate: str) -> tuple[np.ndarray, np.ndarray]:
        """``m`` fresh rows ``(x, y)`` from one DGP drawn under ``surrogate`` ("gp" or "trees")."""
        x = self.sample_x(m, rng)
        if surrogate == "gp":
            return x, self.sample_gp(x, rng)
        if surrogate == "trees":
            return x, self.sample_trees(x, rng)
        raise ValueError(f"unknown surrogate {surrogate!r}")


def _forest(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=N_TREES,
        min_samples_leaf=TREE_MIN_LEAF,
        max_features=TREE_MAX_FEATURES,
        bootstrap=True,
        random_state=seed,
    )
