"""Corpus column-rotation logic, tested offline on a synthetic frame with a
known structure so the learnability / redundancy / type filters are checkable
without hitting OpenML. plans/gate1_revised.md §3.3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ebpfn.gate1.config import CorpusConfig
from ebpfn.gate1.corpus import encode_frame
from ebpfn.gate1.corpus import rotate_frame


def _frame(n=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    target = 1.5 * a - 0.7 * b + 0.3 * rng.normal(size=n)  # learnable, continuous
    pure_noise = rng.normal(size=n)  # independent: unlearnable as a target
    cat = rng.choice(["x", "y", "z"], size=n)  # categorical -> never a target
    few = rng.integers(0, 3, size=n).astype(float)  # < target_min_unique -> never a target
    const = np.ones(n)  # constant -> never a target
    df = pd.DataFrame(
        {"a": a, "b": b, "c": c, "target": target, "pure_noise": pure_noise, "cat": cat, "few": few, "const": const}
    )
    df.loc[df.index[:10], "a"] = np.nan  # inject missingness to exercise imputation
    df["cat"] = df["cat"].astype("category")
    return df


def test_encode_frame_types_and_imputation():
    df = _frame()
    M, names, is_cat, n_unique = encode_frame(df)
    assert M.shape == (len(df), len(names))
    assert np.isfinite(M).all()  # NaNs imputed
    assert is_cat[names.index("cat")] and not is_cat[names.index("a")]
    assert n_unique[names.index("const")] == 1


def test_rotation_excludes_non_continuous_and_unlearnable_targets():
    df = _frame()
    cfg = CorpusConfig(n_min=200, max_tasks_per_dataset=10, learnability_min=0.05)
    tasks = rotate_frame(df, did=1, name="synthetic", cfg=cfg, rng=np.random.default_rng(0))
    targets = {t.target for t in tasks}
    assert "target" in targets  # the learnable continuous column survives
    assert {"cat", "few", "const"}.isdisjoint(targets)  # type/cardinality filters
    assert "pure_noise" not in targets  # learnability filter
    for t in tasks:
        assert t.learnability_r2 >= cfg.learnability_min
        assert t.data.X.shape == (t.n, t.d)
        assert t.data.Y.shape == (t.n,)


def test_max_tasks_per_dataset_and_sorted_by_learnability():
    df = _frame()
    cfg = CorpusConfig(n_min=200, max_tasks_per_dataset=2)
    tasks = rotate_frame(df, 1, "s", cfg, np.random.default_rng(0))
    assert len(tasks) <= 2
    r2s = [t.learnability_r2 for t in tasks]
    assert r2s == sorted(r2s, reverse=True)


def test_n_and_d_caps():
    df = _frame(n=600)
    cfg = CorpusConfig(n_min=100, n_max=300, d_max=3, max_tasks_per_dataset=10)
    tasks = rotate_frame(df, 1, "s", cfg, np.random.default_rng(0))
    assert tasks, "expected at least one task"
    for t in tasks:
        assert t.n <= 300
        assert t.d <= 3


def test_config_validation():
    with pytest.raises(ValueError):
        CorpusConfig(n_min=1)
    with pytest.raises(ValueError):
        CorpusConfig(learnability_min=0.9, redundancy_max=0.5)
