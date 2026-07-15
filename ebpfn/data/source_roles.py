"""Frozen source-level roles for pilot and confirmatory studies."""

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from ebpfn.data.hashing import content_hash

SourceRole = Literal["pilot", "confirmatory"]


@dataclasses.dataclass(frozen=True)
class SourceRoleSplit:
    """A disjoint assignment of independent sources to study roles."""

    policy_version: str
    pilot_source_ids: tuple[str, ...]
    confirmatory_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.policy_version:
            raise ValueError("source-role policy_version must be nonempty")
        if not self.pilot_source_ids or not self.confirmatory_source_ids:
            raise ValueError("pilot and confirmatory source roles must both be nonempty")
        if len(set(self.pilot_source_ids)) != len(self.pilot_source_ids):
            raise ValueError("pilot source IDs must be unique")
        if len(set(self.confirmatory_source_ids)) != len(self.confirmatory_source_ids):
            raise ValueError("confirmatory source IDs must be unique")
        overlap = set(self.pilot_source_ids) & set(self.confirmatory_source_ids)
        if overlap:
            raise ValueError(f"source roles must be disjoint; overlap: {sorted(overlap)}")

    @property
    def split_id(self) -> str:
        return content_hash(
            self.policy_version,
            self.pilot_source_ids,
            self.confirmatory_source_ids,
            namespace="source-role-split-1",
        )

    def role_for(self, source_id: str) -> SourceRole:
        if source_id in self.pilot_source_ids:
            return "pilot"
        if source_id in self.confirmatory_source_ids:
            return "confirmatory"
        raise ValueError(f"source {source_id!r} is absent from the frozen source-role split")

    def to_payload(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "split_id": self.split_id,
            "pilot_source_ids": list(self.pilot_source_ids),
            "confirmatory_source_ids": list(self.confirmatory_source_ids),
        }


def source_role_split_from_dict(payload: dict[str, object]) -> SourceRoleSplit:
    required = {"policy_version", "pilot_source_ids", "confirmatory_source_ids"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"source-role split is missing fields: {sorted(missing)}")
    pilot = payload["pilot_source_ids"]
    confirmatory = payload["confirmatory_source_ids"]
    if (
        not isinstance(pilot, Sequence)
        or isinstance(pilot, str)
        or not isinstance(confirmatory, Sequence)
        or isinstance(confirmatory, str)
    ):
        raise TypeError("source-role assignments must be arrays of source IDs")
    split = SourceRoleSplit(
        policy_version=str(payload["policy_version"]),
        pilot_source_ids=tuple(str(value) for value in pilot),
        confirmatory_source_ids=tuple(str(value) for value in confirmatory),
    )
    stored_id = payload.get("split_id")
    if stored_id is not None and stored_id != split.split_id:
        raise ValueError("stored source-role split_id does not match the split contents")
    return split


def load_source_role_split(path: Path) -> SourceRoleSplit:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("source-role split must contain a JSON object")
    return source_role_split_from_dict(payload)
