import pytest
from ebpfn.cache import EvaluationCache
from ebpfn.config import HyperPriorConfig
from ebpfn.priors import build_hyperprior
from ebpfn.tune import EvaluationResult


def _result(key: str = "abc123") -> EvaluationResult:
    return EvaluationResult(
        total=0.42,
        per_block={"location": 0.1, "nonlinear": 0.3},
        objective_terms={"objective": "energy", "observation_term": 0.5, "ensemble_term": 0.08},
        failures=0,
        failure_events=(),
        runtime_s=1.25,
        candidate_vector=(0.1, 0.2, 0.3),
        eta=build_hyperprior(HyperPriorConfig()),
        stage="search",
        fidelity="min",
        seeds={"base_seed": 0, "stage": "search", "panel_token": 0},
        cache_key=key,
    )


def test_put_get_round_trip_preserves_fields(tmp_path):
    cache = EvaluationCache(tmp_path)
    result = _result()
    cache.put(result.cache_key, result.to_payload())
    loaded = EvaluationResult.from_payload(cache.get(result.cache_key))
    assert loaded.total == pytest.approx(result.total)
    assert loaded.per_block == pytest.approx(result.per_block)
    assert loaded.candidate_vector == pytest.approx(result.candidate_vector)
    assert loaded.stage == result.stage
    assert loaded.fidelity == result.fidelity
    assert loaded.seeds == result.seeds
    assert loaded.eta.generator_weights == pytest.approx(result.eta.generator_weights)
    assert loaded.eta.scm.target_indegree_mean == pytest.approx(result.eta.scm.target_indegree_mean)


def test_missing_key_returns_none(tmp_path):
    cache = EvaluationCache(tmp_path)
    assert cache.get("absent") is None


def test_disabled_cache_is_a_noop(tmp_path):
    cache = EvaluationCache(tmp_path, enabled=False)
    result = _result()
    cache.put(result.cache_key, result.to_payload())
    assert cache.get(result.cache_key) is None
    assert not list(tmp_path.iterdir())
