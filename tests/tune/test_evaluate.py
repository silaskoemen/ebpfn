import subprocess
import sys

import numpy as np
import pytest
from ebpfn.cache import EvaluationCache
from ebpfn.config import CloudConfig, HyperPriorConfig, SearchConfig, TuningConfig
from ebpfn.data import CharacterizationShape
from ebpfn.priors import EtaVectorizer, build_hyperprior, sample_cloud
from ebpfn.tune import RealTarget, characterize_task, evaluate_candidate, make_panel
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


_BOUNDARY_SCRIPT = """
import sys

from ebpfn.config import CloudConfig, HyperPriorConfig, TuningConfig
from ebpfn.data import CharacterizationShape
from ebpfn.priors import build_hyperprior, sample_cloud
from ebpfn.tune import RealTarget, characterize_task, evaluate_candidate, make_panel
from ebpfn.utils import RandomStreams

streams = RandomStreams(4)
eta = build_hyperprior(HyperPriorConfig())
task = sample_cloud(eta, CharacterizationShape(300, 140, 5, 0, "regression"), 1, streams, "real", 0)[0].tuning
config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=4))
targets = [RealTarget(task, characterize_task(task, config.characterization, "min"))]
run_streams = RandomStreams(6)
panel = make_panel("search", 0, config, run_streams)
evaluate_candidate(eta, targets, config, run_streams, panel, "min")
imported = sorted(n for n in sys.modules if n == "ebpfn.pfn" or n.startswith("ebpfn.pfn."))
assert not imported, f"evaluation imported PFN modules: {imported}"
"""


def test_evaluation_does_not_import_the_pfn_subsystem():
    # Run in a clean interpreter: other tests import ebpfn.pfn, so a shared-process
    # sys.modules check is unreliable. This verifies the evaluation path itself never
    # pulls in the PFN subsystem, keeping candidate evaluation likelihood-free.
    completed = subprocess.run([sys.executable, "-c", _BOUNDARY_SCRIPT], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


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


def test_regularization_reuses_raw_cache_and_applies_at_read_time(tmp_path):
    tasks = _real_tasks(seed=9)
    base_config = TuningConfig(objective="energy", cloud=CloudConfig(n_members=5))
    regularized_config = base_config.model_copy(
        update={
            "search": SearchConfig(
                single_task_regularization="prior_distance",
                prior_distance_penalty=0.08,
            )
        }
    )
    streams = RandomStreams(10)
    eta = build_hyperprior(HyperPriorConfig(log_snr_mean=2.0))
    vectorizer = EtaVectorizer(build_hyperprior(base_config.prior), base_config.active)
    baseline_vector = tuple(float(value) for value in vectorizer.encode(build_hyperprior(base_config.prior)))
    panel = make_panel("search", 0, base_config, streams)
    targets = _targets(tasks, base_config, "min")
    cache = EvaluationCache(tmp_path)

    raw = evaluate_candidate(
        eta,
        targets,
        base_config,
        streams,
        panel,
        "min",
        cache=cache,
        vectorizer=vectorizer,
        baseline_vector=baseline_vector,
    )
    regularized = evaluate_candidate(
        eta,
        targets,
        regularized_config,
        streams,
        panel,
        "min",
        cache=cache,
        vectorizer=vectorizer,
        baseline_vector=baseline_vector,
    )

    assert raw.cache_key == regularized.cache_key
    distance = float(np.linalg.norm(np.asarray(raw.candidate_vector) - np.asarray(baseline_vector)))
    assert regularized.total == pytest.approx(raw.total + 0.08 * distance**2)
    assert len(list(tmp_path.iterdir())) == 1
