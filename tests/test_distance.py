"""s-OTDD sanity: agrees roughly with exact OT, and self-distance ~ 0."""

import numpy as np
from ebpfn.config import DataConfig
from ebpfn.config import Prior
from ebpfn.distance import exact_otdd
from ebpfn.distance import s_otdd
from ebpfn.priors import sample_task


def test_sotdd_self_distance_small():
    rng = np.random.default_rng(0)
    D = sample_task(Prior("A", "real", 0.5, DataConfig()), rng, n=600)
    d_self = s_otdd(D, D, lam=1.0, n_proj=300, rng=rng)
    d_other = s_otdd(
        D,
        sample_task(Prior("A", "decoy", 0.5, DataConfig()), rng, n=600),
        lam=1.0,
        n_proj=300,
        rng=rng,
    )
    assert d_self < d_other


def test_sotdd_tracks_exact_ot():
    rng = np.random.default_rng(0)
    dc = DataConfig()
    Da = sample_task(Prior("A", "real", 0.5, dc), rng, n=400)
    Db = sample_task(Prior("A", "decoy", 0.5, dc), rng, n=400)
    sd = s_otdd(Da, Db, lam=1.0, n_proj=500, rng=rng)
    ed = exact_otdd(Da, Db, lam=1.0)
    # Sliced-W lower-bounds W but should be the same order of magnitude.
    assert sd > 0 and ed > 0
    assert 0.2 * ed < sd < 1.5 * ed, (sd, ed)
