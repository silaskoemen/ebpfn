import pytest

from ebpfn.data import SourceRoleSplit, source_role_split_from_dict


def test_source_role_split_groups_tasks_by_independent_source() -> None:
    split = SourceRoleSplit("roles-1", ("source-a", "source-b"), ("source-c",))

    assert split.role_for("source-a") == "pilot"
    assert split.role_for("source-c") == "confirmatory"
    assert split.split_id == SourceRoleSplit("roles-1", ("source-a", "source-b"), ("source-c",)).split_id


def test_source_role_split_rejects_overlap_and_unknown_sources() -> None:
    with pytest.raises(ValueError, match="overlap"):
        SourceRoleSplit("roles-1", ("shared",), ("shared",))

    split = SourceRoleSplit("roles-1", ("pilot",), ("confirmatory",))
    with pytest.raises(ValueError, match="absent"):
        split.role_for("unknown")


def test_source_role_split_validates_stored_identity() -> None:
    with pytest.raises(ValueError, match="split_id"):
        source_role_split_from_dict(
            {
                "policy_version": "roles-1",
                "split_id": "wrong",
                "pilot_source_ids": ["pilot"],
                "confirmatory_source_ids": ["confirmatory"],
            }
        )
