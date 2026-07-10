import subprocess

import ebpfn.utils.provenance as provenance
import numpy as np
import pytest
from ebpfn.config import DataPipelineConfig, StrictConfigModel
from ebpfn.utils import RandomRole, RandomStreams
from pydantic import ValidationError


def test_strict_model_rejects_unknown_nested_key():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        DataPipelineConfig.model_validate({"split": {"seed": 1, "typo": 2}})


class RequiredConfig(StrictConfigModel):
    value: int


def test_strict_model_rejects_missing_required_key():
    with pytest.raises(ValidationError, match="Field required"):
        RequiredConfig.model_validate({})


def test_strict_model_rejects_type_coercion():
    with pytest.raises(ValidationError, match="valid integer"):
        RequiredConfig.model_validate({"value": "1"})


def test_named_random_streams_are_reproducible_distinct_and_order_independent():
    streams = RandomStreams(17)
    search_a = streams.generator(RandomRole.SEARCH, "candidate", 3).normal(size=8)
    _ = streams.generator(RandomRole.SELECTION, "candidate", 3).normal(size=100)
    search_b = streams.generator(RandomRole.SEARCH, "candidate", 3).normal(size=8)
    audit = streams.generator(RandomRole.FINAL_AUDIT, "candidate", 3).normal(size=8)
    np.testing.assert_array_equal(search_a, search_b)
    assert not np.array_equal(search_a, audit)


def test_negative_base_seed_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        RandomStreams(-1)


@pytest.mark.parametrize(
    ("returncode", "output", "expected"), [(0, " M ebpfn/a.py\n", True), (0, "", False), (1, "", None)]
)
def test_git_dirty_provenance(monkeypatch, tmp_path, returncode: int, output: str, expected: bool | None):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], returncode, stdout=output, stderr="")

    monkeypatch.setattr(provenance.subprocess, "run", run)
    assert provenance._git_dirty(tmp_path) is expected
