"""Gate-2: does conditional-structure coverage predict a PFN's calibration?

Re-spec after Gate-1's joint s-OTDD coverage came out non-discriminating
(plans/gate2.md). Two pre-committed parts: (A) a variance go/no-go on the
coverage quantity itself, and (B) an across-prior fixed-effects ablation as the
primary test. The conditional-structure descriptor is frozen in `descriptor.py`.
"""

from ebpfn.gate2.ablation import ablation_test
from ebpfn.gate2.config import DescriptorConfig
from ebpfn.gate2.config import Gate2Config
from ebpfn.gate2.config import Gate2CoverageConfig
from ebpfn.gate2.config import prior_ladder
from ebpfn.gate2.coverage import build_cloud
from ebpfn.gate2.coverage import corpus_coverage
from ebpfn.gate2.coverage import variance_check
from ebpfn.gate2.descriptor import conditional_descriptor
from ebpfn.gate2.descriptor import descriptor_names
from ebpfn.gate2.report import format_report
from ebpfn.gate2.report import gate2_verdict

__all__ = [
    "DescriptorConfig",
    "Gate2Config",
    "Gate2CoverageConfig",
    "ablation_test",
    "build_cloud",
    "conditional_descriptor",
    "corpus_coverage",
    "descriptor_names",
    "format_report",
    "gate2_verdict",
    "prior_ladder",
    "variance_check",
]
