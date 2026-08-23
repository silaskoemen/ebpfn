"""The real-dataset corpus every diagnostic scores against.

Nine regression datasets, materialized at their own shapes by
``benchmarks.studies.characterization`` and read back from its artifacts. This lives
apart from any one diagnostic because the corpus is a property of the programme, not
of the script that happens to load it first.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from benchmarks.studies.characterization import make_task
from ebpfn.config import CharacterizationStudyConfig
from ebpfn.data import TuningTask

CHAR_DIR = Path("benchmarks/results/characterization")

# Our own trained PFN, superseded as the base of record by the released TabICLv2
# regressor; kept as the argparse default until the released export replaces it.
DEFAULT_BASE = Path(
    "benchmarks/results/offline_validation/learning_curve_pilot/checkpoints/"
    "28c6eac5e7702ca9/seed_0/1ab3167a4351f79906e3/checkpoint_step_00000950.pt"
)


def iter_real_tasks(char_dir: Path = CHAR_DIR) -> Iterator[tuple[str, int, TuningTask]]:
    """Yield ``(dataset, repeat, TuningTask)`` for every audit-corpus repeat."""
    for manifest_path in sorted(char_dir.glob("*mode_audit*/task_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        rows = manifest.get("tasks") or []
        if not rows:
            continue
        study = CharacterizationStudyConfig.model_validate_json((manifest_path.parent / "config.json").read_text())
        for row in rows:
            yield str(manifest["dataset"]), int(row["repeat"]), make_task(str(row["label"]), study, int(row["repeat"]))
