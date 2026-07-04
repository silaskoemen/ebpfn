"""Run persistence (spec §5): full config + git SHA + per-seed raw metrics to disk."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl

from ebpfn.config import ExperimentConfig


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def save_run(out_dir: str | Path, cfg: ExperimentConfig, frames: dict[str, pl.DataFrame]) -> Path:
    """Write parquet frames, the full config, and run metadata. Returns the dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.write_parquet(out / f"{name}.parquet")
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    meta = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "construction": cfg.sweep.construction,
        "n_seeds": cfg.sweep.n_seeds,
        "values": list(cfg.sweep.values),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out
