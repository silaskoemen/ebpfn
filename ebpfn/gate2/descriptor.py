"""The conditional-structure descriptor (plans/gate2.md §1).

"How a learner sees the problem", made into a measurement. For a task we z-score
X and rank-Gaussianize Y (affine/monotone nuisances removed),
project X onto random unit directions u, and per direction read off
affine-invariant features of the conditional Y | u^T X:

  - cond_r2     : fraction of Var(Y) explained by a binned conditional mean
                  along t = u^T X  (conditional signal-to-noise / learnability)
  - nonlinearity: cond_r2 minus the linear-fit R^2  (conditional-mean curvature)
  - hetero      : |rank-corr(|residual|, t)|         (heteroskedasticity)
  - skew        : |skew(residual)|                    (conditional asymmetry)
  - kurt        : excess kurtosis(residual)           (conditional tail weight)

Each is aggregated across projections by mean + quantiles -> fixed length,
dimension-adaptive, affine-invariant. Two MULTIVARIATE features are appended,
because the 1D-projection profile is provably blind to interaction structure and
effective dimensionality (the documented pushback that "this vector *is* how a
learner sees the problem" was too strong):

  - eff_dim         : participation ratio of per-feature univariate cond-R^2
                      (effective number of informative features)
  - interaction_gain: held-out R^2 of a GBM minus that of the best additive model
                      (signal that *requires* feature interactions)

Before the operator block, standardized-raw-Y marginal diagnostics record skew,
excess kurtosis, tail spreads and outlier masses. These preserve target tail
behavior that rank Gaussianization deliberately removes.

Finally, an operator-spectrum block approximates the conditional expectation
operator directly: first Hermite moment functions of rank-Gaussianized Y are
crossed with raw, random-quadratic, local-RBF and global-RBF feature maps of X.
For each X block we append regularized CCA modes plus CKA/energy summaries, and
then per-Hermite-moment ridge multiple-correlations. This keeps the cheap
random-feature interface while exposing whether dependence lives in mean, scale
or higher shape modes of Y, not just hand-picked scalar bin statistics.

The vector is deterministic given (task, config.seed). DESCRIPTOR_NAMES freezes
the layout; nothing below may be tuned against calibration.
"""
from __future__ import annotations

from math import factorial

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ebpfn.gate2.config import DescriptorConfig
from ebpfn.priors import Dataset

# The five per-projection conditional features, aggregated across projections.
_PROJ_FEATURES = ("cond_r2", "nonlinearity", "hetero", "skew", "kurt")
_MULTIVARIATE = ("eff_dim", "interaction_gain")
_OPERATOR_BLOCKS = ("raw", "poly", "rbf_local", "rbf_global")
_OPERATOR_SUMMARIES = ("energy", "stable_rank", "cka")
_RAW_Y_TAIL_FEATURES = (
    "y_raw_skew",
    "y_raw_excess_kurt",
    "y_raw_q99_q95",
    "y_raw_q05_q01",
    "y_raw_mass_abs_gt2",
    "y_raw_mass_abs_gt3",
    "y_raw_mass_abs_gt4",
    "y_raw_mass_upper_gt2",
    "y_raw_mass_lower_ltneg2",
)


def descriptor_names(cfg: DescriptorConfig) -> list[str]:
    """Frozen descriptor layout for the projection and operator-spectrum blocks."""
    aggs = ["mean"] + [f"q{int(round(q * 100)):02d}" for q in cfg.quantiles]
    names = [f"{feat}_{agg}" for feat in _PROJ_FEATURES for agg in aggs]
    names += list(_MULTIVARIATE)
    names += list(_RAW_Y_TAIL_FEATURES)
    for block in _OPERATOR_BLOCKS:
        names += [f"{block}_cca_mode_{i + 1}" for i in range(cfg.cca_modes)]
        names += [f"{block}_cca_{summary}" for summary in _OPERATOR_SUMMARIES]
        names += [f"{block}_y_moment_{i + 1}_r2" for i in range(cfg.y_moments)]
    return names


def _zscore(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (M - M.mean(axis=0)) / (M.std(axis=0) + eps)


def _rank_gaussian(y: np.ndarray) -> np.ndarray:
    """Rank -> inverse-normal CDF: maps y to standard-normal scores via its ranks.

    Invariant to *any* strictly-monotone reparametrization of Y (a superset of the
    affine invariance the old z-score gave), and tail-robust by construction --
    the marginal is exactly Gaussian, so the residual skew/kurt features can no
    longer be hijacked by a single heavy Y tail (the Y^3 blow-up failure mode).
    Ties get the average rank (`rankdata` default)."""
    r = rankdata(y)  # 1..n, average ties
    return norm.ppf((r - 0.5) / r.size)


def _binned_mean(t: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """Predict y from t by its mean within quantile bins of t (a smooth,
    monotone-agnostic conditional-mean estimator). Returns per-point prediction."""
    edges = np.unique(np.quantile(t, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:  # t nearly constant -> no conditional structure to read
        return np.full_like(y, y.mean())
    idx = np.clip(np.digitize(t, edges[1:-1]), 0, edges.size - 2)
    pred = np.empty_like(y)
    overall = y.mean()
    for b in np.unique(idx):
        m = idx == b
        pred[m] = y[m].mean() if m.sum() >= 2 else overall
    return pred


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return max(0.0, 1.0 - sse / sst)


def _proj_features(t: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[float, ...]:
    """The five affine-invariant conditional features for one projection t."""
    pred = _binned_mean(t, y, n_bins)
    resid = y - pred
    cond_r2 = _r2(y, pred)
    # linear R^2 = squared Pearson corr of (t, y)
    ts, ys = t.std(), y.std()
    lin_r2 = 0.0 if ts == 0 or ys == 0 else float(np.corrcoef(t, y)[0, 1]) ** 2
    nonlin = max(0.0, cond_r2 - lin_r2)
    # heteroskedasticity: rank-corr of |resid| with t (Spearman via ranks)
    rt = np.argsort(np.argsort(t)).astype(float)
    ra = np.argsort(np.argsort(np.abs(resid))).astype(float)
    hetero = 0.0 if rt.std() == 0 or ra.std() == 0 else abs(float(np.corrcoef(rt, ra)[0, 1]))
    rstd = resid.std()
    sk = 0.0 if rstd == 0 else abs(float(skew(resid)))
    ku = 0.0 if rstd == 0 else float(kurtosis(resid))  # excess (Fisher)
    return cond_r2, nonlin, hetero, sk, ku


def _aggregate(values: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    """mean + requested quantiles across projections, for one feature."""
    return [float(values.mean())] + [float(np.quantile(values, q)) for q in quantiles]


def _standardize_cols(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (M - M.mean(axis=0, keepdims=True)) / (M.std(axis=0, keepdims=True) + eps)


def _hermite_y_features(yz: np.ndarray, degree: int) -> np.ndarray:
    """First orthonormal Hermite moment functions of rank-Gaussianized Y.

    For Z ~ N(0, 1), H_k(Z) / sqrt(k!) has unit variance and is orthogonal to
    lower orders. These columns are a finite basis for functions of Y: k=1 reads
    location, k=2 scale, k=3 asymmetry, and higher k tail/multimodal structure.
    """
    cols = []
    h_prev = np.ones_like(yz)
    h = yz.copy()
    for k in range(1, degree + 1):
        if k == 1:
            hk = h
        else:
            h_next = yz * h - (k - 1) * h_prev
            h_prev, h = h, h_next
            hk = h
        cols.append(hk / np.sqrt(float(factorial(k))))
    return _standardize_cols(np.column_stack(cols))


def _median_pairwise_distance(Xz: np.ndarray, rng: np.random.Generator, max_points: int = 192) -> float:
    n = Xz.shape[0]
    Xs = Xz[rng.choice(n, size=max_points, replace=False)] if n > max_points else Xz
    diff = Xs[:, None, :] - Xs[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    med = float(np.median(tri)) if tri.size else 1.0
    return max(med, 1e-6)


def _rff_features(Xz: np.ndarray, n_features: int, bandwidth: float, rng: np.random.Generator) -> np.ndarray:
    W = rng.standard_normal((Xz.shape[1], n_features)) / bandwidth
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    return np.sqrt(2.0 / n_features) * np.cos(Xz @ W + b)


def _operator_x_blocks(Xz: np.ndarray, cfg: DescriptorConfig,
                       rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Feature blocks for the finite-feature conditional-operator estimate."""
    U = rng.standard_normal((Xz.shape[1], cfg.n_poly))
    U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
    poly = ((Xz @ U) ** 2 - 1.0) / np.sqrt(2.0)

    med = _median_pairwise_distance(Xz, rng)
    return {
        "raw": Xz,
        "poly": poly,
        "rbf_local": _rff_features(Xz, cfg.n_rff, cfg.rbf_local_scale * med, rng),
        "rbf_global": _rff_features(Xz, cfg.n_rff, cfg.rbf_global_scale * med, rng),
    }


def _inv_sqrt_spd(C: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-12)
    return (vecs / np.sqrt(vals)) @ vecs.T


def _operator_spectrum_features(Xfeat: np.ndarray, Yfeat: np.ndarray,
                                cfg: DescriptorConfig) -> list[float]:
    """Regularized CCA modes plus CKA summaries for one X feature block."""
    Xc = _standardize_cols(Xfeat)
    Yc = _standardize_cols(Yfeat)
    denom = max(1, Xc.shape[0] - 1)
    Cxx0 = (Xc.T @ Xc) / denom
    Cyy0 = (Yc.T @ Yc) / denom
    Cxy = (Xc.T @ Yc) / denom

    Cxx = Cxx0 + cfg.cca_ridge * np.eye(Cxx0.shape[0])
    Cyy = Cyy0 + cfg.cca_ridge * np.eye(Cyy0.shape[0])
    whitened = _inv_sqrt_spd(Cxx) @ Cxy @ _inv_sqrt_spd(Cyy)
    modes = np.clip(np.linalg.svd(whitened, compute_uv=False), 0.0, 1.0)
    padded = np.zeros(cfg.cca_modes)
    padded[:min(cfg.cca_modes, modes.size)] = modes[:cfg.cca_modes]

    energy = float(np.sum(modes ** 2))
    stable_rank = 0.0 if modes.size == 0 or modes[0] <= 1e-12 else float(energy / (modes[0] ** 2))
    x_norm = float(np.sum(Cxx0 * Cxx0))
    y_norm = float(np.sum(Cyy0 * Cyy0))
    cka = 0.0 if x_norm <= 1e-12 or y_norm <= 1e-12 else float(np.sum(Cxy * Cxy) / np.sqrt(x_norm * y_norm))
    return [float(v) for v in padded] + [energy, stable_rank, cka]


def _raw_y_tail_features(y: np.ndarray) -> list[float]:
    """Marginal raw-Y tail diagnostics after ordinary standard scaling.

    The rank-Gaussian operator block intentionally removes marginal target shape.
    This block keeps that information separately, so heavy tails, one-sided
    outliers and bounded/short-tailed targets remain visible without destabilizing
    the conditional-geometry features.
    """
    ys = np.asarray(y, dtype=float)
    sd = float(ys.std())
    if sd <= 1e-12:
        return [0.0] * len(_RAW_Y_TAIL_FEATURES)
    yz = (ys - float(ys.mean())) / sd
    q01, q05, q95, q99 = np.quantile(yz, [0.01, 0.05, 0.95, 0.99])
    return [
        float(skew(yz)),
        float(kurtosis(yz)),
        float(q99 - q95),
        float(q05 - q01),
        float(np.mean(np.abs(yz) > 2.0)),
        float(np.mean(np.abs(yz) > 3.0)),
        float(np.mean(np.abs(yz) > 4.0)),
        float(np.mean(yz > 2.0)),
        float(np.mean(yz < -2.0)),
    ]


def _moment_r2_features(Xfeat: np.ndarray, Yfeat: np.ndarray, cfg: DescriptorConfig) -> list[float]:
    """Ridge multiple-correlation of one X block with each Hermite Y moment.

    This preserves what the CCA/CKA summaries intentionally collapse: whether the
    dependence is mostly in the conditional mean (moment 1), scale (moment 2), or
    higher shape/tail modes. For a single Y column this is the squared canonical
    correlation against the X block.
    """
    Xc = _standardize_cols(Xfeat)
    Yc = _standardize_cols(Yfeat)
    denom = max(1, Xc.shape[0] - 1)
    Cxx = (Xc.T @ Xc) / denom + cfg.cca_ridge * np.eye(Xc.shape[1])
    Cxx_inv = np.linalg.pinv(Cxx)

    vals = []
    for j in range(Yc.shape[1]):
        yj = Yc[:, j]
        cxy = (Xc.T @ yj) / denom
        cyy = float((yj @ yj) / denom)
        r2 = 0.0 if cyy <= 1e-12 else float(cxy @ Cxx_inv @ cxy / cyy)
        vals.append(float(np.clip(r2, 0.0, 1.0)))
    return vals


def _effective_dimension(Xz: np.ndarray, yz: np.ndarray, n_bins: int) -> float:
    """Participation ratio of per-feature univariate conditional-R^2.

    PR = (sum r)^2 / sum r^2 in [1, d]: ~1 if one feature carries all the signal,
    ~d if many share it. Captures effective dimensionality (additively); the
    interaction term below captures what additivity misses."""
    r = np.array([_r2(yz, _binned_mean(Xz[:, j], yz, n_bins)) for j in range(Xz.shape[1])])
    s1, s2 = r.sum(), float(np.sum(r ** 2))
    if s2 <= 1e-12:
        return 1.0
    return float(s1 * s1 / s2)


def _interaction_gain(Xz: np.ndarray, yz: np.ndarray, cfg: DescriptorConfig, rng: np.random.Generator) -> float:
    """Held-out R^2 of a GBM minus that of the best additive model.

    Additive baseline: linear regression on each feature's univariate binned
    conditional mean (a GAM-style additive fit, no interactions). The GBM can use
    interactions. The positive gap is interaction-driven structure -- exactly the
    conditional structure random 1D projections cannot represent. Scored on a
    held-out split so neither R^2 is in-sample-inflated."""
    n, d = Xz.shape
    perm = rng.permutation(n)
    n_val = max(4, int(round(cfg.gbm_val_frac * n)))
    val, tr = perm[:n_val], perm[n_val:]
    if tr.size < 8:
        return 0.0
    # additive transform: replace each feature by its train-fit binned conditional mean
    def _transform(idx: np.ndarray) -> np.ndarray:
        cols = []
        for j in range(d):
            edges = np.unique(np.quantile(Xz[tr, j], np.linspace(0.0, 1.0, cfg.n_bins + 1)))
            if edges.size < 3:
                cols.append(np.full(idx.size, yz[tr].mean()))
                continue
            bt = np.clip(np.digitize(Xz[tr, j], edges[1:-1]), 0, edges.size - 2)
            means = np.array([yz[tr][bt == b].mean() if (bt == b).any() else yz[tr].mean()
                              for b in range(edges.size - 1)])
            bi = np.clip(np.digitize(Xz[idx, j], edges[1:-1]), 0, edges.size - 2)
            cols.append(means[bi])
        return np.column_stack(cols)

    add = LinearRegression().fit(_transform(tr), yz[tr])
    r2_add = r2_score(yz[val], add.predict(_transform(val)))
    gbm = HistGradientBoostingRegressor(max_iter=cfg.gbm_max_iter, random_state=cfg.seed)
    gbm.fit(Xz[tr], yz[tr])
    r2_gbm = r2_score(yz[val], gbm.predict(Xz[val]))
    return max(0.0, float(r2_gbm) - max(0.0, float(r2_add)))


def _descriptor_once(X: np.ndarray, y: np.ndarray, cfg: DescriptorConfig,
                     rng: np.random.Generator) -> np.ndarray:
    """The descriptor on one (already row-matched) sample. X z-scored per feature,
    Y rank-Gaussian transformed; both make the vector affine/monotone-invariant."""
    Xz, yz = _zscore(X), _rank_gaussian(y)
    d = Xz.shape[1]

    # per-projection conditional features
    U = rng.standard_normal((cfg.n_proj, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    feats = np.array([_proj_features(Xz @ U[i], yz, cfg.n_bins) for i in range(cfg.n_proj)])  # (n_proj, 5)

    vec: list[float] = []
    for j in range(feats.shape[1]):
        vec += _aggregate(feats[:, j], cfg.quantiles)
    vec.append(_effective_dimension(Xz, yz, cfg.n_bins))
    vec.append(_interaction_gain(Xz, yz, cfg, rng))
    vec += _raw_y_tail_features(y)
    Yop = _hermite_y_features(yz, cfg.y_moments)
    Xblocks = _operator_x_blocks(Xz, cfg, rng)
    for block in _OPERATOR_BLOCKS:
        Xop = Xblocks[block]
        vec += _operator_spectrum_features(Xop, Yop, cfg)
        vec += _moment_r2_features(Xop, Yop, cfg)
    return np.array(vec, dtype=float)


def conditional_descriptor(task: Dataset, cfg: DescriptorConfig | None = None,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """The frozen conditional-structure descriptor of one task, measured at a
    matched row budget `n0` for cross-task comparability.

    If n <= n0 the descriptor is computed once on all rows. If n > n0 it is
    computed on `n_sub` independent size-n0 subsamples and averaged -- equalizing
    the estimator's bias/variance across tasks of different n (the Gate-1
    n-confound, removed rather than merely capped)."""
    cfg = cfg or DescriptorConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    X, y = task.X, task.Y
    n = X.shape[0]
    if n < cfg.min_rows:
        raise ValueError(f"task has {n} rows; need >= {cfg.min_rows} for a stable descriptor")

    if n <= cfg.n0:
        return _descriptor_once(X, y, cfg, rng)
    vecs = []
    for _ in range(cfg.n_sub):
        sel = rng.choice(n, size=cfg.n0, replace=False)
        vecs.append(_descriptor_once(X[sel], y[sel], cfg, rng))
    return np.mean(vecs, axis=0)
