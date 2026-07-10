"""Shared Loguru sinks for benchmark study runs."""

from pathlib import Path
from typing import Final

from loguru import logger

_SINK_IDS: list[int] = []
_RUN_LOG: Final = "run.log"


def configure_study_logging(destination: Path, *, study: str) -> None:
    """Add per-run file sinks while preserving Loguru's default stderr sink."""
    for sink_id in _SINK_IDS:
        logger.remove(sink_id)
    _SINK_IDS.clear()
    destination.mkdir(parents=True, exist_ok=True)
    run_sink = logger.add(destination / _RUN_LOG, level="DEBUG", rotation="10 MB", enqueue=True)
    _SINK_IDS.append(run_sink)
    logger.info(f"📝 study logging configured | {study} | {destination}")
