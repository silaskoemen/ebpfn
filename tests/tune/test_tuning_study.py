import json
from pathlib import Path

import polars as pl
import pytest
from benchmarks.studies import tuning_recovery
from ebpfn.config import CharacterizationStudyConfig, TuningStudyConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _config(mode: str = "fast") -> TuningStudyConfig:
    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        raw = compose(config_name="tuning", overrides=[f"tuning_mode={mode}"])
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(resolved, dict)
    return TuningStudyConfig.model_validate(resolved)


def _characterization_config() -> CharacterizationStudyConfig:
    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        raw = compose(
            config_name="characterization",
            overrides=["characterization_mode=audit", "dataset=airfoil", "repeats=2"],
        )
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(resolved, dict)
    return CharacterizationStudyConfig.model_validate(resolved)


def test_tuning_study_configuration_is_strict_and_resolved():
    config = _config()
    assert config.mode.name == "fast"
    assert config.tuning.search.optimizer == "none"
    assert config.tuning.characterization.row_budgets.minimum == 256
    assert config.mode.cloud_sizes == (8,)


def test_status_requires_audit_decisions():
    checks = {"complete": True}
    fast = _config("fast")
    assert tuning_recovery.derive_study_status(fast, checks) == ("provisional", [])

    audit = _config("audit")
    assert tuning_recovery.derive_study_status(audit, checks) == ("frozen", [])
    pending = audit.model_copy(update={"single_task_regularization_decision": "pending"})
    assert tuning_recovery.derive_study_status(pending, checks) == (
        "incomplete",
        ["single_task_regularization_decision"],
    )


def test_checkpoint_identity_excludes_decision_and_output_metadata():
    config = _config("audit")
    changed_metadata = config.model_copy(
        update={
            "decision_date": "2099-01-01",
            "single_task_regularization_decision": "pending",
            "mode": config.mode.model_copy(update={"output_dir": "somewhere/else"}),
        }
    )
    assert tuning_recovery._checkpoint_identity(changed_metadata) == tuning_recovery._checkpoint_identity(config)

    changed_penalty = config.model_copy(update={"prior_distance_penalty": 0.09})
    assert tuning_recovery._checkpoint_identity(changed_penalty) != tuning_recovery._checkpoint_identity(config)


def test_artifact_writer_emits_complete_contract(tmp_path, monkeypatch):
    config = _config()
    roles_dir = tmp_path / "configs"
    roles_dir.mkdir()
    (roles_dir / "source_roles.json").write_text(
        json.dumps(
            {
                "policy_version": "test-source-roles-1",
                "pilot_source_ids": ["pilot-source"],
                "confirmatory_source_ids": ["confirmatory-source"],
            }
        )
    )
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
        "source_split.json",
    }
    assert {path.name for path in (tmp_path / "out").iterdir()} == expected
    assert json.loads((tmp_path / "out" / "decision_log.json").read_text())["status"] == "provisional"
    assert json.loads((tmp_path / "out" / "source_split.json").read_text())["split_id"]


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
    tuning_recovery._CellStore(checkpoint_path, tuning_recovery._checkpoint_identity(config)).put(0, rows(0))
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


def test_checkpoint_rejects_a_different_study_config(tmp_path):
    config = _config()
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    tuning_recovery._CellStore(checkpoint_path, tuning_recovery._checkpoint_identity(config))
    changed = config.model_copy(update={"planted_unit_shift": 0.3})

    with pytest.raises(ValueError, match="does not match"):
        tuning_recovery._CellStore(checkpoint_path, tuning_recovery._checkpoint_identity(changed))


def test_apparent_snr_real_targets_use_frozen_decision_repeats_and_source_role(tmp_path):
    config = _characterization_config()
    output = tmp_path / "dataset_airfoil__mode_audit__test"
    output.mkdir()
    (output / "config.json").write_text(json.dumps(config.model_dump(mode="json")))
    selected_lambda = config.characterization.ridge.lambda_
    selected_policy = tuning_recovery._row_policy_name(config.characterization)
    (output / "decision_log.json").write_text(
        json.dumps({"ridge_lambda": selected_lambda, "row_budget_policy": selected_policy})
    )
    tasks = [
        {
            "repeat": repeat,
            "source_id": "source-a",
            "shape": {
                "n_probe_fit": 480,
                "n_probe_score": 160,
                "p_numeric": 5,
                "p_categorical": 0,
                "task_type": "regression",
            },
        }
        for repeat in range(2)
    ]
    (output / "task_manifest.json").write_text(json.dumps({"dataset": "airfoil", "tasks": tasks}))
    rows = []
    for repeat, gains in enumerate(((0.4, 0.5), (0.6, 0.7))):
        for learner, gain in zip(("linear", "rff"), gains, strict=True):
            rows.append(
                {
                    "repeat": repeat,
                    "representation": "raw",
                    "policy": "observation/on",
                    "lambda": selected_lambda,
                    "statistic": "gain",
                    "target": "location",
                    "learner": learner,
                    "valid": True,
                    "row_budget": 640,
                    "value": gain,
                }
            )
        rows.append(
            {
                **rows[-1],
                "lambda": 0.1 if selected_lambda != 0.1 else 0.01,
                "value": 0.99,
            }
        )
    pl.DataFrame(rows).write_parquet(output / "coordinates.parquet")
    roles_path = tmp_path / "source_roles.json"
    roles_path.write_text(
        json.dumps(
            {
                "policy_version": "test-roles-1",
                "pilot_source_ids": ["source-a"],
                "confirmatory_source_ids": ["source-b"],
            }
        )
    )

    targets, split_id = tuning_recovery._real_apparent_snr_targets(tmp_path, roles_path, role="pilot")

    assert split_id
    assert len(targets) == 1
    assert targets[0].source_id == "source-a"
    assert targets[0].real_gains == pytest.approx((0.5, 0.7))
    assert targets[0].shapes[0].p_numeric == 5
    confirmatory, _ = tuning_recovery._real_apparent_snr_targets(tmp_path, roles_path, role="confirmatory")
    assert confirmatory == []
