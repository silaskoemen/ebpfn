import dataclasses
from collections import Counter

import numpy as np
import pytest
from ebpfn.config import HyperPriorConfig
from ebpfn.data import CharacterizationShape
from ebpfn.priors import ROUTE_ORDER
from ebpfn.priors import HyperPrior
from ebpfn.priors import build_hyperprior
from ebpfn.priors import sample_cloud
from ebpfn.priors import sample_task
from ebpfn.utils import RandomStreams


def _eta(**changes) -> HyperPrior:
    return build_hyperprior(HyperPriorConfig(**changes))


def _one_hot(eta: HyperPrior, route: str) -> HyperPrior:
    return dataclasses.replace(eta, generator_weights={name: (1.0 if name == route else 0.0) for name in ROUTE_ORDER})


def _shape(n_fit: int, n_score: int, p: int) -> CharacterizationShape:
    return CharacterizationShape(n_fit, n_score, p, 0, "regression")


def _knn_r2(train_x, train_y, test_x, test_y, k: int = 10) -> float:
    distances = ((test_x[:, None, :] - train_x[None, :, :]) ** 2).sum(-1)
    neighbours = np.argsort(distances, axis=1)[:, :k]
    prediction = train_y[neighbours].mean(axis=1)
    residual = float(((test_y - prediction) ** 2).sum())
    total = float(((test_y - test_y.mean()) ** 2).sum())
    return 1.0 - residual / total


def test_same_eta_shape_and_identity_reproduce_task_and_diagnostics():
    eta = _eta()
    streams = RandomStreams(3)
    shape = _shape(120, 60, 8)
    first = sample_task(eta, shape, streams, "task", 1)
    second = sample_task(eta, shape, streams, "task", 1)
    assert np.array_equal(first.tuning.probe_fit.y, second.tuning.probe_fit.y)
    assert first.tuning.probe_fit.X.equals(second.tuning.probe_fit.X)
    assert first.diagnostics == second.diagnostics


def test_distinct_identity_produces_distinct_tasks():
    eta = _eta()
    streams = RandomStreams(3)
    shape = _shape(120, 60, 8)
    first = sample_task(eta, shape, streams, "task", 1)
    other = sample_task(eta, shape, streams, "task", 2)
    assert not np.array_equal(first.tuning.probe_fit.y, other.tuning.probe_fit.y)


def test_route_frequencies_converge_to_direct_weights():
    eta = _eta(generator_weights=[0.4, 0.3, 0.2, 0.1])
    streams = RandomStreams(0)
    cloud = sample_cloud(eta, _shape(40, 20, 6), 4000, streams, "freq")
    counts = Counter(task.diagnostics["route"] for task in cloud)
    for route, weight in eta.generator_weights.items():
        assert abs(counts[route] / len(cloud) - weight) < 0.03


@pytest.mark.parametrize("route", ROUTE_ORDER)
@pytest.mark.parametrize("p", [1, 2, 50, 100])
def test_each_route_preserves_shape_and_feature_contract(route, p):
    eta = _one_hot(_eta(), route)
    streams = RandomStreams(1)
    task = sample_task(eta, _shape(64, 32, p), streams, route, p).tuning
    assert len(task.schema.names) == p
    assert task.probe_fit.X.width == p
    assert task.probe_score.X.width == p
    assert task.probe_fit.X.height == 64
    assert task.probe_score.X.height == 32
    assert np.isfinite(task.probe_fit.y).all()
    assert np.isfinite(task.probe_score.y).all()


def test_scm_expected_indegree_is_stable_across_feature_count():
    eta = _one_hot(_eta(), "scm")
    streams = RandomStreams(2)
    for p in (8, 32, 100):
        indegrees = [
            sample_task(eta, _shape(200, 80, p), streams, "ind", p, t).diagnostics["mean_indegree"] for t in range(12)
        ]
        assert abs(float(np.mean(indegrees)) - 2.0) < 0.5


def test_generated_tuning_task_is_valid_and_hides_diagnostics():
    task = sample_task(_eta(), _shape(80, 40, 5), RandomStreams(0), "valid")
    assert task.tuning.task_type == "regression"
    assert task.tuning.probe_fit_missing_rates == (0.0,) * 5
    # realized route/theta/SNR live only on diagnostics, never on the tuning task.
    assert "route" in task.diagnostics
    assert not hasattr(task.tuning, "diagnostics")
    assert "realized_snr" in task.diagnostics


@pytest.mark.parametrize("route", ROUTE_ORDER)
def test_each_route_is_learnable_under_high_snr(route):
    eta = _one_hot(_eta(log_snr_mean=2.0, snr_dispersion=0.2, heavy_tail_rate=0.0, heteroskedastic_rate=0.0), route)
    task = sample_task(eta, _shape(400, 200, 6), RandomStreams(5), route).tuning
    r2 = _knn_r2(task.probe_fit.X.to_numpy(), task.probe_fit.y, task.probe_score.X.to_numpy(), task.probe_score.y)
    assert r2 > 0.05


def test_distinct_base_seed_yields_distinct_task_identity():
    eta = _eta()
    shape = _shape(60, 30, 5)
    first = sample_task(eta, shape, RandomStreams(0), "task", 0)
    other = sample_task(eta, shape, RandomStreams(1), "task", 0)
    assert not np.array_equal(first.tuning.probe_fit.y, other.tuning.probe_fit.y)
    assert first.tuning.task_id != other.tuning.task_id
    assert first.tuning.characterization_split_id != other.tuning.characterization_split_id


def test_compositional_records_active_thresholds_and_distinct_pairs():
    eta = _one_hot(_eta(), "compositional")
    diagnostics = sample_task(eta, _shape(120, 60, 12), RandomStreams(0), "comp").diagnostics
    assert len(diagnostics["active_indices"]) == diagnostics["active_count"]
    assert len(diagnostics["thresholds"]) == diagnostics["active_count"]
    pairs = diagnostics["interaction_pairs"]
    assert diagnostics["n_interactions"] == len(pairs)
    assert all(a != b for a, b in pairs)  # no self-interactions
    assert len({tuple(sorted(pair)) for pair in pairs}) == len(pairs)  # distinct pairs


def test_scm_reserves_hidden_and_target_nodes():
    eta = _one_hot(_eta(), "scm")
    p = 6
    diagnostics = sample_task(eta, _shape(120, 60, p), RandomStreams(0), "scm").diagnostics
    assert diagnostics["n_nodes"] == p + diagnostics["n_hidden"] + 1
    assert diagnostics["n_active"] >= 1
    assert len(diagnostics["active_indices"]) == diagnostics["n_active"]


def test_categorical_shape_is_rejected():
    with pytest.raises(ValueError, match="categorical"):
        sample_task(_eta(), CharacterizationShape(40, 20, 4, 1, "regression"), RandomStreams(0), "cat")
