import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from benchmarks.studies import characterization
from benchmarks.studies.characterization import (
    _CheckpointStore,
    _load_or_compute_rows,
    _measure,
    _RegressionDataset,
    characterization_output_dir,
    derive_study_status,
    make_task,
    write_study_artifacts,
)
from ebpfn.config import CharacterizationStudyConfig
from ebpfn.data import FeatureSchema
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
    output = tmp_path / "artifacts"
    decision = write_study_artifacts(config, Path.cwd(), output=output)
    evidence = json.loads((output / "evidence.json").read_text())
    assert decision["status"] == ("provisional" if mode == "fast" else "incomplete")
    if mode == "fast":
        assert decision["ridge_lambda"] == config.characterization.ridge.lambda_
    assert evidence["lambda_scores"]
    assert evidence["policy_summary"]
    coordinates = pl.read_parquet(output / "coordinates.parquet")
    assert set(coordinates["dataset"].unique()) == {config.dataset}
    assert set(coordinates["representation"].unique()) == {"raw", "contrast", "diagnostic"}
    assert (coordinates["tracemalloc_peak_bytes"] > 0).all()
    assert (coordinates["process_peak_rss_growth_bytes"] >= 0).all()
    assert evidence["coordinate_stability"]
    assert evidence["rank_stability"]
    assert evidence["response_summary"]
    assert evidence["observation_summary"]
    assert evidence["structure_diagnostics"]["block_summary"]
    assert evidence["structure_diagnostics"]["coordinate_redundancy"]
    summary = (output / "summary.md").read_text()
    assert "## Headline" in summary
    assert "## Dataset Fingerprint" in summary
    assert "### Strongest Coordinates" in summary
    assert "### Row Policy Sweep" in summary
    expected = {
        "checkpoints.sqlite",
        "config.json",
        "coordinates.parquet",
        "decision_log.json",
        "environment.json",
        "evidence.json",
        "run.log",
        "schema_contrast.json",
        "schema_raw.json",
        "summary.md",
    }
    assert {path.name for path in output.iterdir()} == expected


def test_default_output_directory_is_slugged_and_hashed(tmp_path):
    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        raw = compose(config_name="characterization", overrides=["dataset=synthetic_sparse_linear", "max_rows=128"])
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(resolved, dict)
    config = CharacterizationStudyConfig.model_validate(resolved)

    output = characterization_output_dir(config, tmp_path)

    assert output.parent == tmp_path / config.output_root
    assert output.name.startswith(
        "dataset_synthetic_sparse_linear__mode_fast__rows_128__features_12__repeats_1__seed_0__"
    )
    assert "=" not in output.name
    assert len(output.name.rsplit("__", maxsplit=1)[1]) == 8


def test_real_dataset_task_uses_named_registry_and_flat_size_knobs(monkeypatch):
    n = 80
    frame = pl.DataFrame(
        {
            "x0": np.linspace(-1.0, 1.0, n),
            "x1": np.arange(n) % 2,
            "x2": np.sin(np.arange(n)),
            "x3": np.cos(np.arange(n)),
            "x4": np.arange(n),
            "category": ["a", "b"] * (n // 2),
        }
    )
    y = frame["x0"].to_numpy() ** 2 + frame["x2"].to_numpy()
    schema = FeatureSchema(
        ("x0", "x1", "x2", "x3", "x4", "category"),
        ("numeric", "binary", "numeric", "numeric", "numeric", "categorical"),
    )
    dataset = _RegressionDataset("unit-source", "y", frame, y, schema)
    monkeypatch.setattr(characterization, "_openml_spec", lambda name: object() if name == "concrete" else None)
    monkeypatch.setattr(characterization, "_load_openml_regression_dataset", lambda name: dataset)

    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        raw = compose(config_name="characterization", overrides=["dataset=concrete", "max_rows=40", "max_features=4"])
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(resolved, dict)
    config = CharacterizationStudyConfig.model_validate(resolved)

    task = make_task("real", config, 0)

    assert task.source_id == "unit-source"
    assert task.probe_fit.X.height + task.probe_score.X.height == 40
    assert task.schema.names == ("x0", "x1", "x2", "x3")
    assert task.schema.kinds == ("numeric", "binary", "numeric", "numeric")


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


def test_checkpoint_rows_skip_existing_units(tmp_path):
    calls = {"n": 0}
    checkpoints = _CheckpointStore(tmp_path / "checkpoints.sqlite")

    def compute():
        calls["n"] += 1
        return [{"unit": "a", "value": 1.0}]

    first = _load_or_compute_rows(
        checkpoints=checkpoints,
        unit_id="section/a",
        description="section a",
        compute=compute,
    )
    second = _load_or_compute_rows(
        checkpoints=checkpoints,
        unit_id="section/a",
        description="section a",
        compute=compute,
    )

    assert calls["n"] == 1
    assert first == second == [{"unit": "a", "value": 1.0}]
    assert (tmp_path / "checkpoints.sqlite").exists()


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
