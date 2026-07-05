"""Dataset acquisition adapters used by benchmark and diagnostic scripts."""

from benchmarks.data.openml import OpenMLSource
from benchmarks.data.openml import canonical_openml_task
from benchmarks.data.openml import load_openml_source

__all__ = ["OpenMLSource", "canonical_openml_task", "load_openml_source"]
