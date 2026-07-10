"""Coverage for the review-hardened evaluator semantics: source hierarchy,
failure instrumentation, and recorded objective diagnostics."""

import dataclasses

import numpy as np
import pytest
from ebpfn.config import CloudConfig
from ebpfn.config import HyperPriorConfig
from ebpfn.config import TuningConfig
from ebpfn.data import CharacterizationShape
from ebpfn.priors import build_hyperprior
from ebpfn.priors import sample_cloud
from ebpfn.tune import RealTarget
from ebpfn.tune import characterize_task
from ebpfn.tune import evaluate_candidate
from ebpfn.tune import make_panel
from ebpfn.tune.evaluate import _aggregate_hierarchical
from ebpfn.utils import RandomStreams


def _real_tasks(n_tasks: int, seed: int = 0):
    streams = RandomStreams(seed)
    eta = build_hyperprior(HyperPriorConfig())
    return [
        sample_cloud(eta, CharacterizationShape(300, 140, 5, 0, "regression"), 1, streams, "real", i)[0].tuning
        for i in range(n_tasks)
    ]


def _targets(tasks, config, fidelity, source_ids=None):
    targets = []
    for index, task in enumerate(tasks):
        if source_ids is not None:
            task = dataclasses.replace(task, source_id=source_ids[index])
        targets.append(RealTarget(task, characterize_task(task, config.characterization, fidelity)))
    return targets


def test_source_hierarchy_weights_sources_equally():
    # Directed within/across-source combination is root-mean-square. Two sources
    # with (1) and (3) tasks must weight each source equally, unlike a flat RMS.
    losses = {"a": [0.2], "b": [0.4, 0.4, 0.4]}
    hierarchical = _aggregate_hierarchical("directed", losses)
    expected = float(np.sqrt(np.mean([0.2**2, 0.4**2])))  # equal source weight
    flat = float(np.sqrt(np.mean([0.2**2, 0.4**2, 0.4**2, 0.4**2])))
    assert hierarchical == pytest.approx(expected)
    assert hierarchical != pytest.approx(flat)


def test_energy_hierarchy_is_mean_of_source_means():
    losses = {"a": [0.2], "b": [0.4, 0.6]}
    assert _aggregate_hierarchical("energy", losses) == pytest.approx(np.mean([0.2, np.mean([0.4, 0.6])]))


def test_single_source_reduces_to_flat_combination():
    losses_flat = {"only": [0.2, 0.4, 0.6]}
    assert _aggregate_hierarchical("energy", losses_flat) == pytest.approx(np.mean([0.2, 0.4, 0.6]))


def test_multi_source_evaluation_reports_source_count():
    tasks = _real_tasks(2, seed=1)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=5))
    streams = RandomStreams(2)
    panel = make_panel("search", 0, config, streams)
    eta = build_hyperprior(HyperPriorConfig())
    targets = _targets(tasks, config, "min", source_ids=["src_a", "src_b"])
    result = evaluate_candidate(eta, targets, config, streams, panel, "min")
    assert result.objective_terms["n_sources"] == 2
    assert result.objective_terms["n_tasks"] == 2


def test_objective_diagnostics_are_recorded_for_energy():
    tasks = _real_tasks(1, seed=3)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=6))
    streams = RandomStreams(4)
    panel = make_panel("search", 0, config, streams)
    eta = build_hyperprior(HyperPriorConfig())
    result = evaluate_candidate(eta, _targets(tasks, config, "full"), config, streams, panel, "full")
    per_task = result.objective_terms["per_task"]
    assert len(per_task) == 1
    assert per_task[0]["per_budget"]  # per-budget energy scores retained
    assert "observation_term" in per_task[0]
    assert "ensemble_term" in per_task[0]
    assert per_task[0]["validity"]["passes_overall_qc"]
    assert per_task[0]["validity"]["passes_within_block_qc"]
    assert result.objective_terms["energy_pair_ids"] is None  # exact path


def test_objective_diagnostics_are_recorded_for_directed():
    tasks = _real_tasks(1, seed=5)
    config = TuningConfig(objective="directed", cloud=CloudConfig(n_members=8))
    streams = RandomStreams(6)
    panel = make_panel("search", 0, config, streams)
    eta = build_hyperprior(HyperPriorConfig())
    result = evaluate_candidate(eta, _targets(tasks, config, "full"), config, streams, panel, "full")
    per_task = result.objective_terms["per_task"][0]
    assert per_task["k_by_budget"]  # neighborhood size retained
    assert per_task["neighbors_by_budget"]  # neighbor ids retained
    assert per_task["per_budget"]


def test_failure_policy_excludes_and_counts(monkeypatch):
    from ebpfn.tune import evaluate as ev

    tasks = _real_tasks(1, seed=9)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=6, on_failure="exclude"))
    streams = RandomStreams(10)
    panel = make_panel("search", 0, config, streams)
    eta = build_hyperprior(HyperPriorConfig())
    targets = _targets(tasks, config, "min")  # real characterization built with the real function

    real_characterize = ev.characterize_task
    calls = {"n": 0}

    def flaky(task, char_config, fidelity, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic characterization failure")
        return real_characterize(task, char_config, fidelity, **kwargs)

    monkeypatch.setattr(ev, "characterize_task", flaky)
    result = evaluate_candidate(eta, targets, config, streams, panel, "min")
    assert result.failures == 1
    assert result.failure_events[0].phase == "characterization"
    assert result.failure_events[0].exception_type == "RuntimeError"
    assert result.failure_events[0].member_index == 0


def test_failure_policy_raises_by_default(monkeypatch):
    from ebpfn.tune import evaluate as ev

    tasks = _real_tasks(1, seed=11)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=6))  # on_failure='raise'
    streams = RandomStreams(12)
    panel = make_panel("search", 0, config, streams)
    eta = build_hyperprior(HyperPriorConfig())
    targets = _targets(tasks, config, "min")

    def always_fails(task, char_config, fidelity, **kwargs):
        raise RuntimeError("synthetic characterization failure")

    monkeypatch.setattr(ev, "characterize_task", always_fails)
    with pytest.raises(RuntimeError, match="synthetic characterization failure"):
        evaluate_candidate(eta, targets, config, streams, panel, "min")


def test_generation_failure_is_excluded_and_recorded(monkeypatch):
    from ebpfn.tune import evaluate as ev

    tasks = _real_tasks(1, seed=13)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=5, on_failure="exclude"))
    streams = RandomStreams(14)
    panel = make_panel("search", 0, config, streams)
    eta = build_hyperprior(HyperPriorConfig())
    targets = _targets(tasks, config, "min")
    real_sample = ev.sample_task

    def flaky_sample(*args, **kwargs):
        identity = args[3:]
        if identity[-1] == 0:
            raise FloatingPointError("synthetic generation failure")
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(ev, "sample_task", flaky_sample)
    result = evaluate_candidate(eta, targets, config, streams, panel, "min")
    assert result.failures == 1
    assert result.failure_events[0].phase == "generation"
    assert result.failure_events[0].route is None


def test_energy_pair_sample_incompatible_with_exclude():
    from ebpfn.config import CompareConfig

    with pytest.raises(ValueError, match="incompatible with cloud on_failure"):
        TuningConfig(
            objective="energy",
            compare=CompareConfig(energy_pair_sample=8),
            cloud=CloudConfig(on_failure="exclude"),
        )


def test_diagnostics_survive_cache_round_trip(tmp_path):
    from ebpfn.cache import EvaluationCache

    tasks = _real_tasks(1, seed=7)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=5))
    streams = RandomStreams(8)
    panel = make_panel("search", 0, config, streams)
    eta = build_hyperprior(HyperPriorConfig())
    cache = EvaluationCache(tmp_path)
    targets = _targets(tasks, config, "min")
    fresh = evaluate_candidate(eta, targets, config, streams, panel, "min", cache=cache)
    cached = evaluate_candidate(eta, targets, config, streams, panel, "min", cache=cache)
    assert cached.objective_terms["per_task"] == fresh.objective_terms["per_task"]
    assert cached.objective_terms["observation_term"] == pytest.approx(fresh.objective_terms["observation_term"])
    assert cached.failure_events == fresh.failure_events


def test_energy_terms_follow_source_hierarchy():
    tasks = _real_tasks(3, seed=15)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=5))
    streams = RandomStreams(16)
    panel = make_panel("search", 0, config, streams)
    eta = build_hyperprior(HyperPriorConfig())
    targets = _targets(tasks, config, "min", source_ids=["one", "many", "many"])
    result = evaluate_candidate(eta, targets, config, streams, panel, "min")
    assert result.total == pytest.approx(
        result.objective_terms["observation_term"] - result.objective_terms["ensemble_term"]
    )
