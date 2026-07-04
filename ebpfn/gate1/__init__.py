"""Gate-1 (revised): own SCM+BNN prior + own PFN (plans/gate1_revised.md).

The released TabPFN prior is not samplable, so we build the prior ourselves and
train our own PFN on it -- the only way to get an exact prior<->model pairing for
the coverage->calibration test. This subpackage holds the prior substrate; the
PFN, corpus, coverage, and gate test land alongside it as they are built.
"""

from ebpfn.gate1.config import CorpusConfig
from ebpfn.gate1.config import CoverageConfig
from ebpfn.gate1.config import DownstreamConfig
from ebpfn.gate1.config import GateConfig
from ebpfn.gate1.config import PFNConfig
from ebpfn.gate1.config import PriorConfig
from ebpfn.gate1.coverage import corpus_coverage
from ebpfn.gate1.coverage import corpus_null
from ebpfn.gate1.coverage import knn_mean_recall
from ebpfn.gate1.coverage import prior_self_null
from ebpfn.gate1.coverage import task_coverage
from ebpfn.gate1.coverage import x_only_sliced
from ebpfn.gate1.dgp import DGP
from ebpfn.gate1.dgp import BnnDgp
from ebpfn.gate1.dgp import ScmDgp
from ebpfn.gate1.downstream import corpus_calibration
from ebpfn.gate1.downstream import task_calibration
from ebpfn.gate1.gate import gate1_test
from ebpfn.gate1.gate import partial_spearman
from ebpfn.gate1.prior import MixturePrior
from ebpfn.gate1.prior import build_prior

__all__ = [
    "DGP",
    "BnnDgp",
    "CorpusConfig",
    "CoverageConfig",
    "DownstreamConfig",
    "GateConfig",
    "MixturePrior",
    "PFNConfig",
    "PriorConfig",
    "ScmDgp",
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
