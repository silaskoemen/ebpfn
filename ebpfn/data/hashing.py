"""Canonical, type-aware content hashing for task and split identities."""

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel


def _canonical_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return float(value).hex()


def canonical_value(value: Any) -> Any:
    """Convert supported values into a deterministic, type-tagged JSON tree."""

    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, (int, np.integer)):
        return ["int", str(int(value))]
    if isinstance(value, (float, np.floating)):
        return ["float", _canonical_float(float(value))]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, Enum):
        return ["enum", type(value).__qualname__, canonical_value(value.value)]
    if isinstance(value, np.ndarray):
        return ["ndarray", str(value.dtype), list(value.shape), canonical_value(value.tolist())]
    if isinstance(value, pl.DataFrame):
        columns = [
            [name, str(dtype), canonical_value(value.get_column(name).to_list())]
            for name, dtype in value.schema.items()
        ]
        return ["polars.DataFrame", value.height, columns]
    if isinstance(value, BaseModel):
        fields = [[name, canonical_value(getattr(value, name))] for name in type(value).model_fields]
        return ["pydantic", type(value).__module__, type(value).__qualname__, fields]
    if dataclasses.is_dataclass(value):
        fields = [[field.name, canonical_value(getattr(value, field.name))] for field in dataclasses.fields(value)]
        return ["dataclass", type(value).__qualname__, fields]
    if isinstance(value, Mapping):
        items = sorted(
            ((json.dumps(canonical_value(key), separators=(",", ":")), key, item) for key, item in value.items()),
            key=lambda entry: entry[0],
        )
        return ["mapping", [[canonical_value(key), canonical_value(item)] for _, key, item in items]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [canonical_value(item) for item in value]]
    raise TypeError(f"unsupported value for canonical hashing: {type(value).__qualname__}")


def content_hash(*values: Any, namespace: str = "ebpfn-1") -> str:
    payload = canonical_value((namespace, *values))
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_json_value(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True
