import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from benchmarks.studies.characterization import (
    _load_or_compute_rows,
    _measure,
    derive_study_status,
    write_study_artifacts,
)
from ebpfn.config import CharacterizationStudyConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_fast_study_produces_provisional_decision_evidence(tmp_path):
    mode = os.environ.get("EBPFN_CHARACTERIZATION_STUDY_MODE", "fast")
    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        raw = compose(config_name="characterization", overrides=[f"characterization_mode={mode}"])
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(resolved, dict)
    config = CharacterizationStudyConfig.model_validate(resolved)
    output = tmp_path / "artifacts" if mode == "fast" else Path(config.mode.output_dir)
    decision = write_study_artifacts(config, Path.cwd(), output=output)
    evidence = json.loads((output / "evidence.json").read_text())
    assert decision["status"] == ("provisional" if mode == "fast" else "incomplete")
    if mode == "fast":
        assert decision["ridge_lambda"] == config.characterization.ridge.lambda_
    assert evidence["lambda_scores"]
    assert evidence["policy_summary"]
    coordinates = pl.read_parquet(output / "coordinates.parquet")
    assert set(coordinates["representation"].unique()) == {"raw", "contrast", "diagnostic"}
    assert (coordinates["tracemalloc_peak_bytes"] > 0).all()
    assert (coordinates["process_peak_rss_growth_bytes"] >= 0).all()
    assert evidence["coordinate_stability"]
    assert evidence["rank_stability"]
    assert evidence["response_summary"]
    assert evidence["observation_summary"]
    assert evidence["structure_diagnostics"]["block_summary"]
    assert evidence["structure_diagnostics"]["coordinate_redundancy"]
    expected = {
        "config.json",
        "coordinates.parquet",
        "decision_log.json",
        "environment.json",
        "evidence.json",
        "parts",
        "run.log",
        "schema_contrast.json",
        "schema_raw.json",
    }
    assert {path.name for path in output.iterdir()} == expected


def test_process_rss_growth_reflects_the_call_not_the_whole_process_watermark():
    def large_allocation() -> float:
        return float(np.zeros((4000, 4000), dtype=np.float64).sum())

    def tiny_allocation() -> float:
        return float(np.zeros((2, 2)).sum())

    _, _, _, large_growth = _measure(large_allocation)
    _, _, _, tiny_growth = _measure(tiny_allocation)
    assert large_growth >= 0
    assert tiny_growth >= 0
    assert tiny_growth < large_growth


def test_checkpoint_rows_skip_existing_parts(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return [{"unit": "a", "value": 1.0}]

    first = _load_or_compute_rows(
        parts_dir=tmp_path,
        unit_id="section/a",
        description="section a",
        compute=compute,
    )
    second = _load_or_compute_rows(
        parts_dir=tmp_path,
        unit_id="section/a",
        description="section a",
        compute=compute,
    )

    assert calls["n"] == 1
    assert first == second == [{"unit": "a", "value": 1.0}]


def test_audit_status_is_derived_from_evidence():
    complete = {
        "five_complete_repeats": True,
        "all_vectors_valid": True,
        "selected_lambda_null_nonpositive": True,
        "complete_domain_feasible_within_scope": True,
        "feature_width_feasible_within_scope": True,
        "memory_recorded": True,
        "stability_reported": True,
        "response_curves_reported": True,
        "structure_diagnostics_reported": True,
        "representative_real_task_repeats": True,
    }
    assert derive_study_status("audit", complete) == ("frozen", [])
    missing_real = {**complete, "representative_real_task_repeats": False}
    assert derive_study_status("audit", missing_real) == (
        "incomplete",
        ["missing:representative_real_task_repeats"],
    )
    failed_null = {**missing_real, "selected_lambda_null_nonpositive": False}
    status, reasons = derive_study_status("audit", failed_null)
    assert status == "failed"
    assert "failed:selected_lambda_null_nonpositive" in reasons
