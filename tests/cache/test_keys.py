import dataclasses

from ebpfn.cache import evaluation_cache_key
from ebpfn.config import CacheConfig
from ebpfn.config import CharacterizationConfig
from ebpfn.config import CloudConfig
from ebpfn.config import HyperPriorConfig
from ebpfn.config import TuningConfig
from ebpfn.data import CharacterizationShape
from ebpfn.priors import build_hyperprior
from ebpfn.priors import sample_cloud
from ebpfn.utils import RandomStreams


def _real_task():
    streams = RandomStreams(0)
    eta = build_hyperprior(HyperPriorConfig())
    return sample_cloud(eta, CharacterizationShape(120, 60, 4, 0, "regression"), 1, streams, "real")[0].tuning


def _key(config, eta, tasks, base_seed=0, stage="search", fidelity="min", identity=("search", 0), pair_ids=None):
    return evaluation_cache_key(config, eta, tasks, base_seed, stage, fidelity, identity, energy_pair_ids=pair_ids)


def test_identical_inputs_reproduce_the_key():
    task = _real_task()
    eta = build_hyperprior(HyperPriorConfig())
    config = TuningConfig()
    assert _key(config, eta, [task]) == _key(config, eta, [task])


def test_key_changes_for_every_semantic_input():
    task = _real_task()
    eta = build_hyperprior(HyperPriorConfig())
    config = TuningConfig()
    base = _key(config, eta, [task])

    moved_eta = build_hyperprior(HyperPriorConfig(log_snr_mean=0.9))
    other_config = TuningConfig(objective="directed")
    contrast_config = TuningConfig(characterization=CharacterizationConfig(representation="contrast"))
    cloud_config = TuningConfig(cloud=CloudConfig(n_members=32))
    version_config = TuningConfig(cache=CacheConfig(cache_version="tuning-cache-3"))

    assert _key(config, moved_eta, [task]) != base
    assert _key(config, eta, [task], stage="selection", identity=("selection", 0)) != base
    assert _key(config, eta, [task], fidelity="full") != base
    assert _key(other_config, eta, [task]) != base
    assert _key(contrast_config, eta, [task]) != base
    assert _key(cloud_config, eta, [task]) != base
    assert _key(config, eta, [task], identity=("search", 1)) != base
    assert _key(config, eta, [task], pair_ids=[(0, 1)]) != base
    assert _key(version_config, eta, [task]) != base
    # The generation base seed is distinct from config.seed and must be in the key.
    assert _key(config, eta, [task], base_seed=1) != base


def test_key_is_insensitive_to_storage_only_settings():
    task = _real_task()
    eta = build_hyperprior(HyperPriorConfig())
    base = _key(TuningConfig(), eta, [task])
    relocated = TuningConfig(cache=CacheConfig(root="/tmp/other", enabled=False))
    assert _key(relocated, eta, [task]) == base


def test_key_tracks_tuning_task_identity():
    eta = build_hyperprior(HyperPriorConfig())
    config = TuningConfig()
    first = _real_task()
    second = sample_cloud(eta, CharacterizationShape(120, 60, 4, 0, "regression"), 1, RandomStreams(99), "other")[
        0
    ].tuning
    assert _key(config, eta, [first]) != _key(config, eta, [second])


def test_key_consumes_only_tuning_tasks_not_final_test():
    # The key builder accepts TuningTask objects, which structurally exclude any
    # final-test partition, so final-test values cannot affect evaluation identity.
    import inspect

    signature = inspect.signature(evaluation_cache_key)
    assert "real_tasks" in signature.parameters
    task = _real_task()
    assert not any(field.name == "final_test" for field in dataclasses.fields(type(task)))
