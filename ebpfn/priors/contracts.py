"""Frozen contract for generator output.

A generated task is split in two: the :class:`TuningTask` a consumer sees, and hidden
``diagnostics`` recording how it was made. Mechanism draws, seeds, and realized
parameters live in the latter and never leak into the former.
"""

from dataclasses import dataclass
from typing import Any

from ebpfn.data import TuningTask


@dataclass(frozen=True)
class GeneratedTask:
    """The generator output: a task the consumer sees plus hidden diagnostics."""

    tuning: TuningTask
    diagnostics: dict[str, Any]
