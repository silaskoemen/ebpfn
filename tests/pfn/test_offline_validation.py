import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from benchmarks.studies import offline_validation
from ebpfn.config import OfflineValidationConfig, OfflineValidationModeConfig, PfnArchConfig, PfnTrainConfig
from ebpfn.config.prior import HyperPriorConfig
from ebpfn.data import FeatureSchema, SourceRoleSplit, TaskPartition, TuningTask
from ebpfn.priors import build_hyperprior, hyperprior_to_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _config(tmp_path: Path) -> OfflineValidationConfig:
    eta_path = tmp_path / "eta_0.json"
    eta_path.write_text(json.dumps(hyperprior_to_dict(build_hyperprior(HyperPriorConfig()))))
    split = SourceRoleSplit(
        policy_version="test-source-roles-1",
        pilot_source_ids=("pilot",),
        confirmatory_source_ids=("confirmatory",),
    )
    split_path = tmp_path / "source_roles.json"
    split_path.write_text(json.dumps(split.to_payload()))
    return OfflineValidationConfig(
        mode=OfflineValidationModeConfig(
            name="fast",
            output_dir="output",
            pairing_id="test-pairing-1",
            baseline_eta_path=eta_path.name,
            source_roles_path=split_path.name,
            seeds=(0,),
            perturbed_corr_strength_mean=0.6,
        ),
        arch=PfnArchConfig(
            n_bins=16,
            embed_dim=8,
            col_num_blocks=1,
            row_num_blocks=1,
            icl_num_blocks=1,
            col_nhead=2,
            row_nhead=2,
            icl_nhead=2,
            n_cls_rows=4,
        ),
        train=PfnTrainConfig(
            seed=0,
            steps=2,
            tasks_per_step=1,
            gradient_accumulation_steps=1,
            lr=2e-3,
            warmup_steps=0,
            checkpoint_interval=1,
            anchor_probe_fit=16,
            anchor_probe_score=8,
            anchor_features=2,
            device="cpu",
        ),
        characterization_dir="characterization",
        source_role="pilot",
        coverage_levels=(0.5, 0.8),
        crps_grid_size=32,
        metric_row_chunk_size=4,
        decision_owner="test",
        decision_date="2026-07-15",
    )


def _evaluation_records() -> list[offline_validation.RealTaskRecord]:
    rng = np.random.default_rng(4)
    names = ("x0", "x1")
    features = rng.normal(size=(24, 2))
    target = features[:, 0] + rng.normal(scale=0.2, size=24)
    task = TuningTask(
        task_id="pilot-task-0",
        source_id="pilot",
        task_type="regression",
        outer_split_id="outer",
        characterization_split_id="characterization",
        probe_fit=TaskPartition(pl.DataFrame(features[:16], schema=names), target[:16], np.arange(16)),
        probe_score=TaskPartition(pl.DataFrame(features[16:], schema=names), target[16:], np.arange(16, 24)),
        schema=FeatureSchema(names, ("numeric", "numeric")),
        preprocessing_id="preprocessing",
        probe_fit_missing_rates=(0.0, 0.0),
    )
    return [offline_validation.RealTaskRecord("pilot_dataset", 0, "pilot", task)]


def test_fast_hydra_config_resolves_to_the_tiny_panel() -> None:
    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        raw = compose(config_name="offline_validation", overrides=["offline_validation_mode=fast"])
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(resolved, dict)

    config = OfflineValidationConfig.model_validate(resolved)
    assert config.mode.name == "fast"
    assert config.arch.embed_dim == 16
    assert config.train.steps == 4
    assert config.train.gradient_accumulation_steps == 1
    assert config.mode.eta_labels == ("eta_0", "corr_strength_perturbed")
    decision = json.loads((config_dir / "crps_quadrature_decision.json").read_text())
    assert config.crps_grid_size == decision["selected_grid_size"] == 256


def test_eta_panel_can_select_one_independently_resumable_job(tmp_path: Path) -> None:
    config = _config(tmp_path)
    mode = config.mode.model_copy(update={"eta_labels": ("eta_0",)})
    selected = offline_validation.build_eta_panel(config.model_copy(update={"mode": mode}), tmp_path)

    assert [member["label"] for member in selected] == ["eta_0"]


def test_training_panel_writes_step_checkpoints_and_resumes_from_terminal_jobs(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    destination = tmp_path / "output"
    records = _evaluation_records()
    summary = offline_validation.write_training_panel_artifacts(
        config,
        tmp_path,
        output=destination,
        evaluation_records=records,
    )

    assert summary == {
        "status": "pass",
        "n_eta": 2,
        "n_seeds": 1,
        "n_jobs": 2,
        "n_evaluations": 4,
        "n_prediction_rows": 32,
        "n_failures": 0,
        "pairing_id": "test-pairing-1",
    }
    manifests = pl.read_parquet(destination / "training_manifests.parquet")
    curves = pl.read_parquet(destination / "training_curves.parquet")
    assert manifests["status"].to_list() == ["complete", "complete"]
    assert curves.height == 4
    predictions = pl.read_parquet(destination / "pfn_predictions.parquet")
    row_metrics = pl.read_parquet(destination / "pfn_row_metrics.parquet")
    aggregates = pl.read_parquet(destination / "pfn_aggregate_metrics.parquet")
    assert predictions.height == row_metrics.height == 32
    assert set(aggregates["aggregation_level"]) == {"task", "source", "panel"}
    assert aggregates.height == 12
    first_task = aggregates.filter(pl.col("aggregation_level") == "task").row(0, named=True)
    contributing = row_metrics.filter(
        (pl.col("job_id") == first_task["job_id"])
        & (pl.col("checkpoint_step") == first_task["checkpoint_step"])
        & (pl.col("task_id") == first_task["task_id"])
    )
    assert first_task["nll_std"] == pytest.approx(contributing["nll_std"].mean())
    assert first_task["crps_std"] == pytest.approx(contributing["crps_std"].mean())
    assert first_task["rmse_std"] == pytest.approx(contributing["squared_error_std"].mean() ** 0.5)
    assert predictions["logits"].list.len().unique().to_list() == [16]
    assert len(list((destination / "checkpoints").rglob("checkpoint_step_00000001.pt"))) == 2
    assert len(list((destination / "checkpoints").rglob("checkpoint_step_00000002.pt"))) == 2

    original_job_ids = set(manifests["job_id"])
    extended_config = config.model_copy(update={"train": config.train.model_copy(update={"steps": 3})})
    extended = offline_validation.write_training_panel_artifacts(
        extended_config,
        tmp_path,
        output=destination,
        evaluation_records=records,
    )
    extended_manifests = pl.read_parquet(destination / "training_manifests.parquet")
    assert set(extended_manifests["job_id"]) == original_job_ids
    assert extended_manifests["completed_steps"].to_list() == [3, 3]
    assert extended["n_evaluations"] == 6
    assert extended["n_prediction_rows"] == 48
    assert len(list((destination / "checkpoints").rglob("checkpoint_step_00000003.pt"))) == 2

    def should_not_retrain(*args, **kwargs):
        raise AssertionError("completed jobs must replay without retraining")

    monkeypatch.setattr(offline_validation, "train_pfn", should_not_retrain)
    monkeypatch.setattr(offline_validation, "_evaluate_task", should_not_retrain)
    repeated = offline_validation.write_training_panel_artifacts(
        extended_config,
        tmp_path,
        output=destination,
        evaluation_records=records,
    )
    assert repeated == extended


def test_failed_job_is_retained_and_not_silently_reseeded(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    destination = tmp_path / "failed"
    calls = 0

    def fail_training(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("planned failure")

    monkeypatch.setattr(offline_validation, "train_pfn", fail_training)
    first = offline_validation.write_training_panel_artifacts(config, tmp_path, output=destination)
    second = offline_validation.write_training_panel_artifacts(config, tmp_path, output=destination)

    assert first["status"] == second["status"] == "failed"
    assert first["n_failures"] == second["n_failures"] == 2
    assert first["n_evaluations"] == second["n_evaluations"] == 0
    assert first["n_prediction_rows"] == second["n_prediction_rows"] == 0
    assert calls == 2
    failures = pl.read_parquet(destination / "failure_events.parquet")
    assert failures["seed"].to_list() == [0, 0]
    assert failures["error_type"].to_list() == ["RuntimeError", "RuntimeError"]
