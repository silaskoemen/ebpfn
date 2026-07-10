import sys

import numpy as np
import pytest
from ebpfn.cache import EvaluationCache
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
from ebpfn.utils import RandomStreams


def _real_tasks(n_tasks: int = 1, seed: int = 0):
    streams = RandomStreams(seed)
    eta = build_hyperprior(HyperPriorConfig())
    return [
        sample_cloud(eta, CharacterizationShape(300, 140, 5, 0, "regression"), 1, streams, "real", i)[0].tuning
        for i in range(n_tasks)
    ]


def _targets(tasks, config, fidelity):
    return [RealTarget(task, characterize_task(task, config.characterization, fidelity)) for task in tasks]


def test_min_and_full_fidelity_are_distinct():
    tasks = _real_tasks()
    config = TuningConfig()
    minimal = characterize_task(tasks[0], config.characterization, "min")
    full = characterize_task(tasks[0], config.characterization, "full")
    assert len(minimal.coordinates) < len(full.coordinates)


def test_fidelities_can_rerank_candidates():
    tasks = _real_tasks(seed=3)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=8))
    streams = RandomStreams(5)
    panel = make_panel("search", 0, config, streams)
    etas = [build_hyperprior(HyperPriorConfig(log_snr_mean=value)) for value in (-0.5, 0.7, 1.5)]

    def scores(fidelity):
        return [
            evaluate_candidate(eta, _targets(tasks, config, fidelity), config, streams, panel, fidelity).total
            for eta in etas
        ]

    min_scores = scores("min")
    full_scores = scores("full")
    # The cheap (single-budget) and full (multiresolution) fidelities score the
    # same candidates over different coordinate sets, so their score vectors
    # genuinely differ -- reranking between the stages is therefore possible.
    assert not np.allclose(min_scores, full_scores)
    assert min_scores != full_scores


def test_stages_are_disjoint_but_reproducible():
    tasks = _real_tasks(seed=1)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=6))
    streams = RandomStreams(2)
    eta = build_hyperprior(HyperPriorConfig())
    targets = _targets(tasks, config, "full")

    search_panel = make_panel("search", 0, config, streams)
    audit_panel = make_panel("final_audit", 0, config, streams)
    search_total = evaluate_candidate(eta, targets, config, streams, search_panel, "full").total
    audit_total = evaluate_candidate(eta, targets, config, streams, audit_panel, "full").total
    repeat_total = evaluate_candidate(eta, targets, config, streams, search_panel, "full").total

    assert search_total == pytest.approx(repeat_total)  # reproducible within a stage
    assert search_total != pytest.approx(audit_total)  # disjoint across stages


def test_evaluation_does_not_import_the_pfn_subsystem():
    tasks = _real_tasks(seed=4)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=4))
    streams = RandomStreams(6)
    eta = build_hyperprior(HyperPriorConfig())
    panel = make_panel("search", 0, config, streams)
    evaluate_candidate(eta, _targets(tasks, config, "min"), config, streams, panel, "min")
    assert not any(name == "ebpfn.pfn" or name.startswith("ebpfn.pfn.") for name in sys.modules)


def test_cache_hit_returns_an_equal_result(tmp_path):
    tasks = _real_tasks(seed=7)
    config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=5))
    streams = RandomStreams(8)
    eta = build_hyperprior(HyperPriorConfig())
    cache = EvaluationCache(tmp_path)
    panel = make_panel("search", 0, config, streams)
    targets = _targets(tasks, config, "min")
    first = evaluate_candidate(eta, targets, config, streams, panel, "min", cache=cache)
    second = evaluate_candidate(eta, targets, config, streams, panel, "min", cache=cache)
    assert first.cache_key == second.cache_key
    assert first.total == pytest.approx(second.total)
    assert len(list(tmp_path.iterdir())) == 1
