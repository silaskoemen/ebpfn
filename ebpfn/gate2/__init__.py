"""Gate-2: does conditional-structure coverage predict a PFN's calibration?

Re-spec after Gate-1's joint s-OTDD coverage came out non-discriminating
(plans/gate2.md). Two pre-committed parts: (A) a variance go/no-go on the
coverage quantity itself, and (B) an across-prior fixed-effects ablation as the
primary test. The conditional-structure descriptor is frozen in `descriptor.py`.
"""
from ebpfn.gate2.ablation import ablation_test
from ebpfn.gate2.config import (
    DescriptorConfig,
    Gate2Config,
    Gate2CoverageConfig,
    prior_ladder,
)
from ebpfn.gate2.coverage import build_cloud, corpus_coverage, variance_check
from ebpfn.gate2.descriptor import conditional_descriptor, descriptor_names
from ebpfn.gate2.report import format_report, gate2_verdict

__all__ = [
    "DescriptorConfig",
    "Gate2Config",
    "Gate2CoverageConfig",
    "prior_ladder",
    "conditional_descriptor",
    "descriptor_names",
    "build_cloud",
    "corpus_coverage",
    "variance_check",
    "ablation_test",
    "gate2_verdict",
    "format_report",
]
