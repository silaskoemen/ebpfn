"""The base model of record: the released TabICLv2 regressor.

Studies and diagnostics take a ``--checkpoint`` path so any fine-tune can be scored the
same way, which means the released weights need a path too. They get one lazily: the
release is fetched and SHA-verified into the user cache, converted into the vendored
backbone, and written out once as an ordinary step-0 checkpoint. The file is gitignored
-- it is derived data, byte-reproducible from the pin, and too large to version.
"""

from pathlib import Path

from ebpfn.config import PfnTrainConfig
from ebpfn.pfn.checkpoints import TABICL_V2_REGRESSOR
from ebpfn.pfn.train import export_released_base

BASE_DIR = Path("assets")
BASE_CHECKPOINT = BASE_DIR / f"checkpoint_step_{0:08d}.pt"


def released_base_checkpoint(*, allow_download: bool = True) -> Path:
    """Path to the released base as a step-0 checkpoint, exporting it on first use."""
    if BASE_CHECKPOINT.exists():
        return BASE_CHECKPOINT
    path, _ = export_released_base(BASE_DIR, TABICL_V2_REGRESSOR, train=PfnTrainConfig(), allow_download=allow_download)
    return path
