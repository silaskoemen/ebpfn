"""A fixed-length, feature-permutation-invariant summary of one regression dataset.

This is the input to the amortized posterior q(eta | D), not a distance space: no
coordinate is compared across datasets directly, a learned map decides what each one
says about eta. Statistics are aggregated over columns so the vector length depends on
nothing but this module, and every entry is finite for any task the generator accepts.
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor

SUMMARY_VERSION = "dataset-summary-1"

N_FOLDS = 5
N_TREES = 64
RIDGE_ALPHA = 1e-2
KNN_KS = (1, 10)
TREE_DEPTHS = (2, None)
# Floor on a standard deviation before it is treated as a constant column.
_EPS = 1e-8


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(scale > _EPS, scale, 1.0)


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(1.0 - np.mean((y - prediction) ** 2) / max(float(np.var(y)), _EPS))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() <= _EPS or b.std() <= _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _aggregate(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}.mean": float(values.mean()),
        f"{prefix}.std": float(values.std()),
        f"{prefix}.max": float(values.max()),
    }


def _marginals(prefix: str, columns: np.ndarray) -> dict[str, float]:
    """Shape of each column's marginal: skew, tails, and how discrete it is."""
    n = columns.shape[0]
    skew = np.nan_to_num(stats.skew(columns, axis=0))
    kurt = np.nan_to_num(stats.kurtosis(columns, axis=0))
    unique = np.array([np.unique(col).size / n for col in columns.T])
    mode = np.array([np.max(np.unique(col, return_counts=True)[1]) / n for col in columns.T])
    return (
        _aggregate(f"{prefix}.abs_skew", np.abs(skew))
        | _aggregate(f"{prefix}.log_kurt", np.log1p(np.maximum(kurt + 3.0, 0.0)))
        | _aggregate(f"{prefix}.unique_frac", unique)
        | _aggregate(f"{prefix}.mode_frac", mode)
    )


def _dependence(x: np.ndarray) -> dict[str, float]:
    """How the features covary: spectrum of the correlation matrix and its nonlinearity."""
    p = x.shape[1]
    if p == 1:
        return {"x.eff_rank": 1.0, "x.top_eig": 1.0, "x.abs_corr": 0.0, "x.rank_gap": 0.0}
    pearson = np.nan_to_num(np.corrcoef(x, rowvar=False))
    spearman = np.nan_to_num(stats.spearmanr(x).statistic)
    eig = np.clip(np.linalg.eigvalsh(pearson), 0.0, None)
    share = eig / max(float(eig.sum()), _EPS)
    entropy = -float(np.sum(share[share > 0] * np.log(share[share > 0])))
    off = ~np.eye(p, dtype=bool)
    return {
        "x.eff_rank": float(np.exp(entropy) / p),
        "x.top_eig": float(share.max()),
        "x.abs_corr": float(np.abs(pearson[off]).mean()),
        "x.rank_gap": float(np.abs(np.abs(spearman[off]) - np.abs(pearson[off])).mean()),
    }


def _relevance(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Marginal association of each feature with the target, linear and monotone."""
    pearson = np.abs([_corr(col, y) for col in x.T])
    ranks_y = stats.rankdata(y)
    spearman = np.abs([_corr(stats.rankdata(col), ranks_y) for col in x.T])
    return _aggregate("xy.abs_corr", pearson) | _aggregate("xy.abs_spearman", spearman)


def _learners(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    """Out-of-sample fit of learners with different inductive biases, and the residual shape."""
    folds = KFold(N_FOLDS, shuffle=True, random_state=seed)
    out = {"fit.ridge_r2": _r2(y, cross_val_predict(Ridge(alpha=RIDGE_ALPHA), x, y, cv=folds))}
    for k in KNN_KS:
        out[f"fit.knn{k}_r2"] = _r2(y, cross_val_predict(KNeighborsRegressor(k), x, y, cv=folds))

    importances = np.zeros(x.shape[1])
    residual = np.zeros_like(y)
    prediction = np.zeros_like(y)
    for depth in TREE_DEPTHS:
        forest = ExtraTreesRegressor(
            N_TREES, max_depth=depth, bootstrap=True, oob_score=True, random_state=seed, n_jobs=1
        ).fit(x, y)
        oob = np.nan_to_num(forest.oob_prediction_, nan=float(y.mean()))
        out[f"fit.trees_d{depth or 'full'}_r2"] = _r2(y, oob)
        if depth is None:
            importances, prediction, residual = forest.feature_importances_, oob, y - oob

    ordered = np.sort(importances)[::-1] / max(float(importances.sum()), _EPS)
    out["fit.top_importance"] = float(ordered[0])
    out["fit.importance_90_frac"] = float((np.searchsorted(np.cumsum(ordered), 0.9) + 1) / x.shape[1])
    out["res.abs_skew"] = float(abs(np.nan_to_num(stats.skew(residual))))
    out["res.log_kurt"] = float(np.log1p(max(float(np.nan_to_num(stats.kurtosis(residual))) + 3.0, 0.0)))
    out["res.hetero"] = _corr(np.abs(residual), prediction)
    return out


def summarize(x: np.ndarray, y: np.ndarray, *, seed: int = 0) -> dict[str, float]:
    """Summary of one dataset ``(x, y)``; shape enters as ``log n`` and ``log p``."""
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError(f"expected x (n, p) and y (n,), got {x.shape} and {y.shape}")
    xs = _standardize(np.asarray(x, dtype=np.float64))
    ys = _standardize(np.asarray(y, dtype=np.float64)[:, None])[:, 0]
    out = {"shape.log_n": float(np.log(x.shape[0])), "shape.log_p": float(np.log(x.shape[1]))}
    out |= _marginals("x", xs) | _marginals("y", ys[:, None]) | _dependence(xs) | _relevance(xs, ys)
    out |= _learners(xs, ys, seed)
    if not all(np.isfinite(value) for value in out.values()):
        raise ValueError(f"non-finite summary entries: {[k for k, v in out.items() if not np.isfinite(v)]}")
    return out
