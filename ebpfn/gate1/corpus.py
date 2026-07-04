"""Real-task corpus via column rotation over TabArena (plans/gate1_revised.md §3.3).

To test whether prior-coverage predicts the PFN's calibration we need *real*
tabular tasks. Each source table is column-rotated: every eligible continuous
column becomes a regression target Y in turn, the rest are features X. Two
filters keep tasks honest -- learnability (a GBM must beat the marginal, so the
target is predictable) and redundancy (drop near-deterministic targets) -- and
every task is clamped to the PFN's (n, d) regime with n, d, schema recorded for
the n,d confound control (§4).

The frame logic (`encode_frame`, `rotate_frame`) is pure and unit-tested
offline; `load_corpus` is the thin OpenML fetch on top. Categorical features are
ordinal-encoded (label-free, keeps d small); missing values are median/mode
imputed; rows missing the *target* are dropped (never imputed into Y).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from ebpfn.gate1.config import CorpusConfig
from ebpfn.priors import Dataset


@dataclass(frozen=True)
class RealTask:
    """One column-rotated regression task plus the schema the gate test needs."""

    data: Dataset
    source_did: int
    source_name: str
    target: str
    n: int
    d: int
    n_cat_features: int
    learnability_r2: float


def encode_frame(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Encode every column to a float matrix for use as features.

    Numeric columns are median-imputed; categorical/object columns are ordinal-
    encoded (missing -> its own code) then mode-imputed. Returns (M, names,
    is_cat, n_unique) where M is (n, p); the raw frame is kept by the caller for
    pulling un-imputed targets.
    """
    cols = list(df.columns)
    n, p = len(df), len(cols)
    M = np.empty((n, p), dtype=float)
    is_cat = np.zeros(p, dtype=bool)
    n_unique = np.zeros(p, dtype=int)
    for j, c in enumerate(cols):
        s = df[c]
        n_unique[j] = int(s.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
            M[:, j] = np.where(np.isfinite(v), v, med)
        else:
            is_cat[j] = True
            codes, _ = pd.factorize(s, use_na_sentinel=True)  # NaN -> -1
            codes = codes.astype(float)
            if (codes < 0).any():  # mode-impute the missing sentinel
                valid = codes[codes >= 0]
                mode = float(np.bincount(valid.astype(int)).argmax()) if valid.size else 0.0
                codes = np.where(codes < 0, mode, codes)
            M[:, j] = codes
    return M, cols, is_cat, n_unique


def _learnability_r2(X: np.ndarray, y: np.ndarray, cfg: CorpusConfig, rng: np.random.Generator) -> float:
    """Held-out R^2 of a GBM (R^2 of the marginal/mean predictor is 0)."""
    n = len(y)
    perm = rng.permutation(n)
    n_test = max(1, int(round(cfg.test_frac * n)))
    te, tr = perm[:n_test], perm[n_test:]
    model = HistGradientBoostingRegressor(max_iter=cfg.learnability_max_iter, random_state=cfg.seed)
    model.fit(X[tr], y[tr])
    return float(r2_score(y[te], model.predict(X[te])))


def rotate_frame(df: pd.DataFrame, did: int, name: str, cfg: CorpusConfig, rng: np.random.Generator) -> list[RealTask]:
    """Column-rotate one table into filtered, (n,d)-clamped RealTasks."""
    M, names, is_cat, n_unique = encode_frame(df)
    p = M.shape[1]
    tasks: list[RealTask] = []
    for c in range(p):
        if is_cat[c] or n_unique[c] < cfg.target_min_unique:
            continue  # only continuous columns are regression targets
        raw = pd.to_numeric(df[names[c]], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(raw)
        if mask.sum() < cfg.n_min:
            continue
        feat_cols = [j for j in range(p) if j != c]
        X_all, Y_all = M[mask][:, feat_cols], raw[mask]
        if Y_all.std() == 0:
            continue
        # clamp d: keep the highest-variance features if over the cap
        d = X_all.shape[1]
        if d < 1:
            continue
        if d > cfg.d_max:
            keep = np.argsort(X_all.var(axis=0))[::-1][: cfg.d_max]
            X_all, feat_cols = X_all[:, keep], [feat_cols[i] for i in keep]
            d = cfg.d_max
        # clamp n: subsample without replacement
        idx = np.arange(X_all.shape[0])
        if idx.size > cfg.n_max:
            idx = rng.choice(idx, size=cfg.n_max, replace=False)
        X, Y = X_all[idx], Y_all[idx]
        r2 = _learnability_r2(X, Y, cfg, rng)
        if r2 < cfg.learnability_min or r2 > cfg.redundancy_max:
            continue
        tasks.append(
            RealTask(
                data=Dataset(X=X, Y=Y),
                source_did=did,
                source_name=name,
                target=names[c],
                n=int(X.shape[0]),
                d=int(d),
                n_cat_features=int(sum(is_cat[j] for j in feat_cols)),
                learnability_r2=r2,
            )
        )
    tasks.sort(key=lambda t: t.learnability_r2, reverse=True)
    return tasks[: cfg.max_tasks_per_dataset]


def load_corpus(
    cfg: CorpusConfig | None = None, rng: np.random.Generator | None = None, verbose: bool = False
) -> list[RealTask]:
    """Fetch TabArena datasets and column-rotate them into the filtered corpus.

    A dataset that fails to download or parse is skipped (logged), so one bad
    table never aborts the whole corpus.
    """
    import openml

    cfg = cfg or CorpusConfig()
    rng = rng or np.random.default_rng(cfg.seed)
    openml.config.set_root_cache_directory(os.path.abspath(cfg.cache_dir))
    suite = openml.study.get_suite(cfg.suite_id)
    dids = sorted(suite.data)[: cfg.max_datasets]

    corpus: list[RealTask] = []
    for i, did in enumerate(dids):
        try:
            ds = openml.datasets.get_dataset(
                did, download_data=True, download_qualities=False, download_features_meta_data=False
            )
            df, _, _, _ = ds.get_data()
            tasks = rotate_frame(df, did, ds.name, cfg, rng)
        except Exception as e:  # a single bad dataset must not abort the corpus
            print(f"[corpus] skip did={did}: {type(e).__name__}: {str(e)[:120]}")
            continue
        corpus.extend(tasks)
        if verbose:
            print(
                f"[corpus] {i + 1}/{len(dids)} did={did} {ds.name[:30]:30s} +{len(tasks)} tasks "
                f"(total {len(corpus)})"
            )
    return corpus
