import numpy as np
import pytest
from ebpfn.cache import EvaluationCache
from ebpfn.config import CacheConfig, CloudConfig, HyperPriorConfig, SearchConfig, TuningConfig
from ebpfn.data import CharacterizationShape
from ebpfn.priors import EtaVectorizer, build_hyperprior, sample_cloud
from ebpfn.tune import run_search
from ebpfn.utils import RandomStreams


def _real_tasks(n_tasks: int = 2, seed: int = 0):
    streams = RandomStreams(seed)
    eta = build_hyperprior(HyperPriorConfig())
    return [
        sample_cloud(eta, CharacterizationShape(300, 140, 5, 0, "regression"), 1, streams, "real", i)[0].tuning
        for i in range(n_tasks)
    ]


def _config(**search_overrides) -> TuningConfig:
    search = SearchConfig(
        sobol_candidates=6,
        retain_strong=3,
        retain_diverse=2,
        de_maxiter=2,
        de_popsize=3,
        de_fidelity="min",
        selection_panel_size=2,
        **search_overrides,
    )
    # Default cache off so tests that omit a cache do not write under the repo;
    # tests that exercise caching override the cache subconfig explicitly.
    return TuningConfig(
        objective="energy", cloud=CloudConfig(n_members=6), search=search, cache=CacheConfig(enabled=False)
    )


def test_search_selects_exactly_one_feasible_finalist(tmp_path):
    tasks = _real_tasks()
    config = _config().model_copy(update={"cache": CacheConfig(root=str(tmp_path))})
    result = run_search(config, tasks, RandomStreams(0), cache=EvaluationCache(tmp_path))
    vectorizer = EtaVectorizer(build_hyperprior(config.prior), config.active)
    assert vectorizer.is_feasible(np.asarray(result.finalist_vector))
    # The selection ranking is ordered; the finalist is its head.
    totals = [record.result.total for record in result.selection_ranking]
    assert totals == sorted(totals)
    assert result.finalist_vector == result.selection_ranking[0].vector
    assert len(result.selection_records) == len(result.selection_ranking) * config.search.selection_panel_size
    assert all(len(record.panel_results) == config.search.selection_panel_size for record in result.selection_ranking)


def test_search_is_reproducible(tmp_path):
    tasks = _real_tasks(seed=1)
    config = _config()
    first = run_search(config, tasks, RandomStreams(3))
    second = run_search(config, tasks, RandomStreams(3))
    assert np.allclose(first.finalist_vector, second.finalist_vector)
    assert first.selection_ranking[0].result.total == pytest.approx(second.selection_ranking[0].result.total)


def test_search_without_optimizer_still_selects(tmp_path):
    tasks = _real_tasks(seed=2)
    config = _config(optimizer="none")
    result = run_search(config, tasks, RandomStreams(4))
    assert result.finalist_eta.generator_weights  # a decoded hyperprior
    assert len(result.selection_ranking) >= 1
    assert result.optimizer_records == []


def test_single_task_regularization_rejects_multiple_tasks():
    tasks = _real_tasks(n_tasks=2, seed=5)
    config = _config(single_task_regularization="closest_to_baseline", competitive_tolerance=0.01)
    with pytest.raises(ValueError, match="exactly one real task"):
        run_search(config, tasks, RandomStreams(6))


def test_optimizer_evaluations_are_retained(monkeypatch):
    from ebpfn.tune import search as search_module

    def fake_optimizer(objective, feasible, dimension, rng, **kwargs):
        vector = np.zeros(dimension)
        assert feasible(vector)
        objective(vector)
        return vector

    monkeypatch.setattr(search_module, "optimize_population", fake_optimizer)
    config = _config(optimizer="differential_evolution")
    result = run_search(config, _real_tasks(n_tasks=1, seed=7), RandomStreams(8))
    assert len(result.optimizer_records) == 1
    assert result.optimizer_records[0].origin == "optimizer"
    assert result.optimizer_records[0] in result.search_records
