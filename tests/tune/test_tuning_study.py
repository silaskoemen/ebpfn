import json
from pathlib import Path

import polars as pl
from benchmarks.studies import tuning_recovery
from ebpfn.config import TuningStudyConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _config(mode: str = "fast") -> TuningStudyConfig:
    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        raw = compose(config_name="tuning", overrides=[f"tuning_mode={mode}"])
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(resolved, dict)
    return TuningStudyConfig.model_validate(resolved)


def test_tuning_study_configuration_is_strict_and_resolved():
    config = _config()
    assert config.mode.name == "fast"
    assert config.tuning.search.optimizer == "none"
    assert config.mode.cloud_sizes == (8,)


def test_status_requires_audit_decisions():
    checks = {"complete": True}
    fast = _config("fast")
    assert tuning_recovery.derive_study_status(fast, checks) == ("provisional", [])

    audit = _config("audit")
    status, missing = tuning_recovery.derive_study_status(audit, checks)
    assert status == "incomplete"
    assert set(missing) == {
        "multiresolution_decision",
        "synthetic_failure_decision",
        "single_task_regularization_decision",
    }

    resolved = audit.model_copy(
        update={
            "multiresolution_decision": "characterization-audit-1",
            "synthetic_failure_decision": "raise",
            "single_task_regularization_decision": "none",
        }
    )
    assert tuning_recovery.derive_study_status(resolved, checks) == ("frozen", [])


def test_artifact_writer_emits_complete_contract(tmp_path, monkeypatch):
    config = _config()
    tables = {
        "evaluations": pl.DataFrame({"total": [0.2]}),
        "candidates": pl.DataFrame({"selection_rank": [0]}),
        "failure_events": pl.DataFrame({"message": []}, schema={"message": pl.String}),
        "rank_stability": pl.DataFrame({"selection_audit_spearman": [1.0]}),
        "recovery": pl.DataFrame({"fresh_seed_loss_reduction": [0.1]}),
    }
    result = {
        **tables,
        "evidence": {"checks": {"complete": True}},
        "decision": {"status": "provisional", "missing_checks": []},
    }
    monkeypatch.setattr(tuning_recovery, "run_study", lambda _, **__: result)
    summary = tuning_recovery.write_study_artifacts(config, tmp_path, output=tmp_path / "out")
    assert summary == {"status": "provisional", "evaluations": 1}
    expected = {
        "evaluations.parquet",
        "candidates.parquet",
        "failure_events.parquet",
        "rank_stability.parquet",
        "recovery.parquet",
        "config.json",
        "evidence.json",
        "decision_log.json",
        "apparent_snr.json",
        "summary.md",
        "environment.json",
        "run.log",
    }
    assert {path.name for path in (tmp_path / "out").iterdir()} == expected
    assert json.loads((tmp_path / "out" / "decision_log.json").read_text())["status"] == "provisional"


def test_checkpointed_run_skips_completed_parts(tmp_path, monkeypatch):
    config = _config()
    specs = [
        tuning_recovery._CellSpec(0, "raw", "energy", "base", 0, 8, "none"),
        tuning_recovery._CellSpec(1, "raw", "energy", "log_snr_mean", 0, 8, "none"),
    ]

    def rows(repeat: int) -> tuning_recovery._StudyRows:
        cell = {
            "representation": "raw",
            "objective": "energy",
            "scenario": "base" if repeat == 0 else "log_snr_mean",
            "repeat": repeat,
            "cloud_size": 8,
            "regularization": "none",
        }
        return tuning_recovery._StudyRows(
            evaluations=[{**cell, "total": float(repeat)}],
            candidates=[{**cell, "selection_rank": 0}],
            ranks=[{**cell, "selection_audit_spearman": 1.0}],
            recovery=[{**cell, "fresh_seed_loss_reduction": 0.0}],
        )

    checkpoint_path = tmp_path / "checkpoints.sqlite"
    tuning_recovery._CellStore(checkpoint_path).put(0, rows(0))
    called = []

    def run_cell_spec(_, spec):
        called.append(spec.cell_id)
        return spec.cell_id, rows(spec.cell_id), 0.0

    monkeypatch.setattr(tuning_recovery, "_cell_specs", lambda _: specs)
    monkeypatch.setattr(tuning_recovery, "_run_cell_spec", run_cell_spec)
    monkeypatch.setattr(tuning_recovery, "_finalize_study", lambda _, frames: frames)

    result = tuning_recovery.run_study(config, checkpoint_path=checkpoint_path, max_workers=1)

    assert called == [1]
    assert result["evaluations"]["repeat"].to_list() == [0, 1]
