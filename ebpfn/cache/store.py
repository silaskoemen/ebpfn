"""Run-local content-addressed store for evaluation payloads.

The store deals only in JSON-serializable payload dicts, so it stays a
dependency leaf below ``ebpfn.tune`` (which owns result (de)serialization). Cross
-run reuse is explicit: point two runs at the same ``root``.
"""

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


class EvaluationCache:
    """A directory of ``<key>.json`` evaluation payloads."""

    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        with path.open() as handle:
            payload: dict[str, Any] = json.load(handle)
        return payload

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)
