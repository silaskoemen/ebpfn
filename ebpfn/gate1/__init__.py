"""Gate-1 (revised): own SCM+BNN prior + own PFN (plans/gate1_revised.md).

The released TabPFN prior is not samplable, so we build the prior ourselves and
train our own PFN on it -- the only way to get an exact prior<->model pairing for
the coverage->calibration test. This subpackage holds the prior substrate; the
PFN, corpus, coverage, and gate test land alongside it as they are built.
"""
from ebpfn.gate1.config import (
    CorpusConfig,
    CoverageConfig,
    DownstreamConfig,
    GateConfig,
    PFNConfig,
    PriorConfig,
)
from ebpfn.gate1.coverage import (
    corpus_coverage,
    corpus_null,
    knn_mean_recall,
    prior_self_null,
    task_coverage,
    x_only_sliced,
)
from ebpfn.gate1.dgp import DGP, BnnDgp, ScmDgp
from ebpfn.gate1.downstream import corpus_calibration, task_calibration
from ebpfn.gate1.gate import gate1_test, partial_spearman
from ebpfn.gate1.prior import MixturePrior, build_prior

__all__ = [
    "DGP",
    "BnnDgp",
    "ScmDgp",
    "MixturePrior",
    "CorpusConfig",
    "CoverageConfig",
    "DownstreamConfig",
    "GateConfig",
    "PFNConfig",
    "PriorConfig",
    "build_prior",
    "corpus_calibration",
    "corpus_coverage",
    "corpus_null",
    "gate1_test",
    "knn_mean_recall",
    "partial_spearman",
    "prior_self_null",
    "task_calibration",
    "task_coverage",
    "x_only_sliced",
]
