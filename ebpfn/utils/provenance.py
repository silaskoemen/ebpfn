"""Runtime provenance that is recorded but does not alter cache identity."""

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_revision(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty(project_root: Path) -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def environment_provenance(
    project_root: str | Path, packages: tuple[str, ...] = ("ebpfn", "numpy", "polars")
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git_revision": _git_revision(root),
        "git_dirty": _git_dirty(root),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": versions,
        "project_root": str(root),
    }
