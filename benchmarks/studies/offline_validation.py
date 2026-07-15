"""Resumable Step-5 PFN learning-curve training panel.

This first Step-5 slice trains the frozen baseline and one decision-relevant
correlation perturbation under a paired task stream. It deliberately stops at
training artifacts; real-source inference and surrogate analysis are added by
the complete offline-validation orchestrator.
"""

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import polars as pl
import torch
from benchmarks.studies.study_logging import configure_study_logging
from ebpfn.config import CharacterizationStudyConfig, OfflineValidationConfig
from ebpfn.data import TuningTask, characterization_shape, content_hash, load_source_role_split
from ebpfn.pfn import PairedPriorTaskSource, collate_tasks
from ebpfn.pfn.metrics import compute_row_metrics, to_raw
from ebpfn.pfn.train import load_checkpoint, select_device, train_pfn
from ebpfn.priors import HyperPrior, hyperprior_from_dict, hyperprior_to_dict
from ebpfn.utils import RandomStreams, environment_provenance
from loguru import logger


@dataclasses.dataclass(frozen=True)
class RealTaskRecord:
    """One frozen real task evaluated by every compatible checkpoint."""

    dataset: str
    repeat: int
    source_role: str
    task: TuningTask


def _load_eta(path: Path) -> HyperPrior:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("eta artifact must contain a JSON object")
    return hyperprior_from_dict(payload)


def build_eta_panel(config: OfflineValidationConfig, project_root: Path) -> list[dict[str, Any]]:
    """Construct the exact baseline-plus-correlation panel."""
    baseline = _load_eta(project_root / config.mode.baseline_eta_path)
    perturbed_value = config.mode.perturbed_corr_strength_mean
    if perturbed_value == baseline.corr_strength_mean:
        raise ValueError("the learning-curve perturbation must differ from baseline corr_strength_mean")
    perturbed = dataclasses.replace(baseline, corr_strength_mean=perturbed_value)
    members = (("eta_0", baseline), ("corr_strength_perturbed", perturbed))
    selected = set(config.mode.eta_labels)
    return [
        {
            "label": label,
            "eta_id": content_hash(eta, namespace="offline-validation-eta-1")[:16],
            "eta": hyperprior_to_dict(eta),
        }
        for label, eta in members
        if label in selected
    ]


def load_real_task_panel(config: OfflineValidationConfig, project_root: Path) -> list[RealTaskRecord]:
    """Reconstruct and verify the role-eligible characterization tasks."""
    from benchmarks.studies.characterization import make_task

    root = project_root / config.characterization_dir
    split = load_source_role_split(project_root / config.mode.source_roles_path)
    records: list[RealTaskRecord] = []
    for manifest_path in sorted(root.glob("*mode_audit*/task_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"{manifest_path} has no task records")
        source_ids = {str(row["source_id"]) for row in tasks}
        if len(source_ids) != 1:
            raise ValueError(f"{manifest_path} must describe exactly one independent source")
        source_id = source_ids.pop()
        role = split.role_for(source_id)
        if role != config.source_role:
            continue
        study = CharacterizationStudyConfig.model_validate_json((manifest_path.parent / "config.json").read_text())
        for row in tasks:
            task = make_task(str(row["label"]), study, int(row["repeat"]))
            shape = characterization_shape(task)
            expected = {
                "task_id": task.task_id,
                "source_id": task.source_id,
                "outer_split_id": task.outer_split_id,
                "characterization_split_id": task.characterization_split_id,
                "preprocessing_id": task.preprocessing_id,
                "shape": {
                    "n_probe_fit": shape.n_probe_fit,
                    "n_probe_score": shape.n_probe_score,
                    "p_numeric": shape.p_numeric,
                    "p_categorical": shape.p_categorical,
                    "task_type": shape.task_type,
                },
            }
            observed = {name: row[name] for name in expected}
            if observed != expected:
                raise ValueError(f"reconstructed task does not match {manifest_path}: repeat {row['repeat']}")
            records.append(
                RealTaskRecord(
                    dataset=str(manifest["dataset"]),
                    repeat=int(row["repeat"]),
                    source_role=role,
                    task=task,
                )
            )
    if not records:
        raise ValueError(f"no {config.source_role} real tasks found under {root}")
    return records


def _job_id(
    config: OfflineValidationConfig,
    eta: HyperPrior,
    seed: int,
) -> str:
    train = config.train.model_copy(update={"seed": seed})
    train_contract = train.model_dump(mode="json")
    train_contract.pop("steps")
    return content_hash(
        config.version,
        config.arch,
        train_contract,
        eta,
        config.mode.pairing_id,
        namespace="offline-validation-training-job-2",
    )[:20]


def _latest_checkpoint(directory: Path) -> Path | None:
    paths = sorted(directory.glob("checkpoint_step_*.pt"))
    return paths[-1] if paths else None


def _load_completed_job(
    job_dir: Path,
    job_id: str,
    requested_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    result_path = job_dir / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())
    if not isinstance(result, dict) or result.get("job_id") != job_id:
        raise ValueError(f"stored training result does not match job {job_id}")
    if result.get("status") == "running":
        return None
    if result.get("status") not in ("complete", "failed"):
        raise ValueError(f"stored training result for {job_id} has an unknown status")
    completed_steps = int(result.get("completed_steps", 0))
    if result["status"] == "complete" and completed_steps < requested_steps:
        return None
    if completed_steps > requested_steps:
        raise ValueError(
            f"stored training result for {job_id} has {completed_steps} steps, "
            f"beyond requested horizon {requested_steps}"
        )
    curve_path = job_dir / "training_curve.parquet"
    if result.get("status") == "complete":
        if not curve_path.exists():
            raise FileNotFoundError(f"completed job {job_id} has no training curve")
        curves = pl.read_parquet(curve_path).to_dicts()
    else:
        curves = []
    return result, curves


def _write_job_result(job_dir: Path, result: dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))


def _run_job(
    config: OfflineValidationConfig,
    project_root: Path,
    destination: Path,
    member: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eta = hyperprior_from_dict(member["eta"])
    job_id = _job_id(config, eta, seed)
    job_dir = destination / "checkpoints" / member["eta_id"] / f"seed_{seed}" / job_id
    if stored := _load_completed_job(job_dir, job_id, config.train.steps):
        logger.info(f"training job already recorded | {member['label']} | seed={seed} | job={job_id}")
        return stored

    train = config.train.model_copy(update={"seed": seed})
    source = PairedPriorTaskSource(eta, RandomStreams(seed), pairing_id=config.mode.pairing_id)
    resume_from = _latest_checkpoint(job_dir)
    started = time.perf_counter()
    base_result: dict[str, Any] = {
        "job_id": job_id,
        "eta_id": member["eta_id"],
        "eta_label": member["label"],
        "seed": seed,
        "pairing_id": config.mode.pairing_id,
        "requested_steps": train.steps,
        "source_stream": source.stream_provenance,
    }
    _write_job_result(job_dir, {**base_result, "status": "running", "completed_steps": 0})
    try:
        _, training = train_pfn(
            config.arch,
            train,
            source=source,
            checkpoint_dir=job_dir,
            resume_from=resume_from,
            log_every=max(1, train.steps // 10),
        )
        curves = [
            {
                "job_id": job_id,
                "eta_id": member["eta_id"],
                "eta_label": member["label"],
                "seed": seed,
                "step": step,
                "loss": loss,
            }
            for step, loss in enumerate(training.losses, start=1)
        ]
        pl.DataFrame(curves).write_parquet(job_dir / "training_curve.parquet")
        if training.checkpoint_path is None:
            raise RuntimeError("training job completed without its required checkpoint")
        result = {
            **base_result,
            "status": "complete",
            "completed_steps": training.steps,
            "checkpoint_path": str(training.checkpoint_path.relative_to(project_root)),
            "runtime_seconds": time.perf_counter() - started,
            "error_type": None,
            "error_message": None,
        }
    except Exception as error:
        latest_checkpoint = _latest_checkpoint(job_dir)
        result = {
            **base_result,
            "status": "failed",
            "completed_steps": int(latest_checkpoint.stem.rsplit("_", 1)[-1]) if latest_checkpoint else 0,
            "checkpoint_path": (str(latest_checkpoint.relative_to(project_root)) if latest_checkpoint else None),
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        curves = []
        logger.exception(f"training job failed and was retained | {member['label']} | seed={seed} | job={job_id}")
    _write_job_result(job_dir, result)
    return result, curves


def run_training_panel(
    config: OfflineValidationConfig,
    project_root: Path,
    destination: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    panel = build_eta_panel(config, project_root)
    manifests: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for member in panel:
        for seed in config.mode.seeds:
            manifest, job_curves = _run_job(config, project_root, destination, member, seed)
            manifests.append(manifest)
            curves.extend(job_curves)
    return manifests, curves


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError as error:
        raise ValueError(f"invalid step-checkpoint filename: {path.name}") from error


def _coverage_column(level: float) -> str:
    return f"coverage_p{round(100 * level):02d}"


def _evaluate_task(
    config: OfflineValidationConfig,
    checkpoint_path: Path,
    training_manifest: dict[str, Any],
    record: RealTaskRecord,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shape = characterization_shape(record.task)
    rows = shape.n_probe_fit + shape.n_probe_score
    if rows > config.arch.max_context or shape.p_numeric > 100 or shape.p_categorical:
        raise ValueError(
            f"task {record.task.task_id} is outside PFN support: rows={rows}, "
            f"numeric={shape.p_numeric}, categorical={shape.p_categorical}"
        )
    device = select_device(config.train.device)
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    checkpoint_eta = hyperprior_from_dict(checkpoint["source_eta"])
    checkpoint_eta_id = content_hash(checkpoint_eta, namespace="offline-validation-eta-1")[:16]
    if checkpoint_eta_id != training_manifest["eta_id"]:
        raise ValueError("checkpoint eta does not match its training manifest")
    batch = collate_tasks([record.task]).to(device)
    with torch.inference_mode():
        logits = model.predict_logits(batch.x, batch.y_train_std)[0]
    y_std = batch.y_test_std[0]
    target_mean = batch.target_mean[0]
    target_std = batch.target_std[0]
    raw_target = torch.as_tensor(record.task.probe_score.y, dtype=torch.float32, device=device)
    raw_mean = to_raw(model.distribution.mean(logits), target_mean, target_std)
    log_scale = torch.log(target_std)

    metric_chunks = []
    for start in range(0, len(y_std), config.metric_row_chunk_size):
        stop = min(start + config.metric_row_chunk_size, len(y_std))
        metric_chunks.append(
            compute_row_metrics(
                model.distribution,
                logits[start:stop],
                y_std[start:stop],
                coverage_levels=config.coverage_levels,
                crps_grid_size=config.crps_grid_size,
            )
        )

    checkpoint_step = _checkpoint_step(checkpoint_path)
    predictions: list[dict[str, Any]] = []
    row_metrics: list[dict[str, Any]] = []
    offset = 0
    for chunk in metric_chunks:
        size = len(chunk.nll)
        for local_index in range(size):
            index = offset + local_index
            identity = {
                "job_id": training_manifest["job_id"],
                "eta_id": training_manifest["eta_id"],
                "eta_label": training_manifest["eta_label"],
                "seed": training_manifest["seed"],
                "checkpoint_step": checkpoint_step,
                "dataset": record.dataset,
                "source_id": record.task.source_id,
                "source_role": record.source_role,
                "repeat": record.repeat,
                "task_id": record.task.task_id,
                "row_id": int(record.task.probe_score.row_ids[index]),
            }
            predictions.append(
                {
                    **identity,
                    "target_std": float(y_std[index]),
                    "target_raw": float(raw_target[index]),
                    "target_mean": float(target_mean),
                    "target_scale": float(target_std),
                    "predictive_mean_std": float(chunk.mean[local_index]),
                    "predictive_mean_raw": float(raw_mean[index]),
                    "logits": logits[index].detach().cpu().tolist(),
                }
            )
            absolute_raw = torch.abs(raw_mean[index] - raw_target[index])
            squared_raw = (raw_mean[index] - raw_target[index]) ** 2
            metric_row = {
                **identity,
                "nll_std": float(chunk.nll[local_index]),
                "nll_raw": float(chunk.nll[local_index] + log_scale),
                "crps_std": float(chunk.crps[local_index]),
                "crps_raw": float(chunk.crps[local_index] * target_std),
                "absolute_error_std": float(chunk.absolute_error[local_index]),
                "absolute_error_raw": float(absolute_raw),
                "squared_error_std": float(chunk.squared_error[local_index]),
                "squared_error_raw": float(squared_raw),
            }
            for level, (lower, upper, inside) in chunk.intervals.items():
                column = _coverage_column(float(level))
                lower_raw = to_raw(lower[local_index], target_mean, target_std)
                upper_raw = to_raw(upper[local_index], target_mean, target_std)
                metric_row[column] = bool(inside[local_index])
                metric_row[f"lower_{column}_std"] = float(lower[local_index])
                metric_row[f"upper_{column}_std"] = float(upper[local_index])
                metric_row[f"lower_{column}_raw"] = float(lower_raw)
                metric_row[f"upper_{column}_raw"] = float(upper_raw)
            row_metrics.append(metric_row)
        offset += size
    return predictions, row_metrics


def _evaluation_id(
    config: OfflineValidationConfig,
    training_manifest: dict[str, Any],
    checkpoint_path: Path,
    record: RealTaskRecord,
) -> str:
    return content_hash(
        config.version,
        training_manifest["job_id"],
        _checkpoint_step(checkpoint_path),
        record.dataset,
        record.repeat,
        record.task,
        config.coverage_levels,
        config.crps_grid_size,
        namespace="offline-validation-checkpoint-evaluation-1",
    )[:20]


def _run_evaluation_unit(
    config: OfflineValidationConfig,
    project_root: Path,
    checkpoint_path: Path,
    training_manifest: dict[str, Any],
    record: RealTaskRecord,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation_id = _evaluation_id(config, training_manifest, checkpoint_path, record)
    unit_dir = checkpoint_path.parent / "evaluations" / evaluation_id
    result_path = unit_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("evaluation_id") != evaluation_id:
            raise ValueError(f"stored evaluation result does not match {evaluation_id}")
        if result.get("status") == "complete":
            return (
                result,
                pl.read_parquet(unit_dir / "predictions.parquet").to_dicts(),
                pl.read_parquet(unit_dir / "row_metrics.parquet").to_dicts(),
            )
        if result.get("status") == "failed":
            return result, [], []
        if result.get("status") != "running":
            raise ValueError(f"stored evaluation {evaluation_id} has an unknown status")

    unit_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "evaluation_id": evaluation_id,
        "job_id": training_manifest["job_id"],
        "eta_id": training_manifest["eta_id"],
        "eta_label": training_manifest["eta_label"],
        "seed": training_manifest["seed"],
        "checkpoint_step": _checkpoint_step(checkpoint_path),
        "checkpoint_path": str(checkpoint_path.relative_to(project_root)),
        "dataset": record.dataset,
        "source_id": record.task.source_id,
        "source_role": record.source_role,
        "repeat": record.repeat,
        "task_id": record.task.task_id,
    }
    result_path.write_text(json.dumps({**base, "status": "running"}, indent=2, sort_keys=True))
    started = time.perf_counter()
    try:
        predictions, row_metrics = _evaluate_task(config, checkpoint_path, training_manifest, record)
        pl.DataFrame(predictions).write_parquet(unit_dir / "predictions.parquet")
        pl.DataFrame(row_metrics).write_parquet(unit_dir / "row_metrics.parquet")
        result = {
            **base,
            "status": "complete",
            "n_rows": len(row_metrics),
            "runtime_seconds": time.perf_counter() - started,
            "error_type": None,
            "error_message": None,
        }
    except Exception as error:
        result = {
            **base,
            "status": "failed",
            "n_rows": 0,
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        predictions, row_metrics = [], []
        logger.exception(
            f"checkpoint evaluation failed and was retained | job={training_manifest['job_id']} | "
            f"step={base['checkpoint_step']} | task={record.task.task_id}"
        )
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result, predictions, row_metrics


def run_checkpoint_evaluations(
    config: OfflineValidationConfig,
    project_root: Path,
    training_manifests: list[dict[str, Any]],
    records: list[RealTaskRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifests: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    row_metrics: list[dict[str, Any]] = []
    for training_manifest in training_manifests:
        if training_manifest["status"] != "complete":
            continue
        final_checkpoint = project_root / training_manifest["checkpoint_path"]
        checkpoint_paths = sorted(final_checkpoint.parent.glob("checkpoint_step_*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"completed training job has no checkpoints: {training_manifest['job_id']}")
        for checkpoint_path in checkpoint_paths:
            for record in records:
                manifest, unit_predictions, unit_metrics = _run_evaluation_unit(
                    config,
                    project_root,
                    checkpoint_path,
                    training_manifest,
                    record,
                )
                manifests.append(manifest)
                predictions.extend(unit_predictions)
                row_metrics.extend(unit_metrics)
    return manifests, predictions, row_metrics


_MEAN_METRICS = (
    "nll_std",
    "nll_raw",
    "crps_std",
    "crps_raw",
    "absolute_error_std",
    "absolute_error_raw",
    "squared_error_std",
    "squared_error_raw",
)


def aggregate_row_metrics(rows: list[dict[str, Any]], coverage_levels: tuple[float, ...]) -> list[dict[str, Any]]:
    """Build equal-row task, equal-task source, and equal-source panel aggregates."""
    coverage_columns = tuple(_coverage_column(level) for level in coverage_levels)
    value_columns = (*_MEAN_METRICS, *coverage_columns)
    checkpoint_keys = ("job_id", "eta_id", "eta_label", "seed", "checkpoint_step")

    def grouped(items: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for item in items:
            groups.setdefault(tuple(item[key] for key in keys), []).append(item)
        return groups

    task_keys = (*checkpoint_keys, "dataset", "source_id", "source_role", "repeat", "task_id")
    task_rows: list[dict[str, Any]] = []
    for key, members in grouped(rows, task_keys).items():
        row = dict(zip(task_keys, key, strict=True))
        row.update(
            {
                "aggregation_level": "task",
                "n_rows": len(members),
                "n_tasks": 1,
                **{column: sum(float(member[column]) for member in members) / len(members) for column in value_columns},
            }
        )
        task_rows.append(row)

    source_keys = (*checkpoint_keys, "source_id", "source_role")
    source_rows: list[dict[str, Any]] = []
    for key, members in grouped(task_rows, source_keys).items():
        row = dict(zip(source_keys, key, strict=True))
        row.update(
            {
                "aggregation_level": "source",
                "dataset": None,
                "repeat": None,
                "task_id": None,
                "n_rows": sum(int(member["n_rows"]) for member in members),
                "n_tasks": len(members),
                **{column: sum(float(member[column]) for member in members) / len(members) for column in value_columns},
            }
        )
        source_rows.append(row)

    panel_rows: list[dict[str, Any]] = []
    for key, members in grouped(source_rows, checkpoint_keys).items():
        row = dict(zip(checkpoint_keys, key, strict=True))
        row.update(
            {
                "aggregation_level": "panel",
                "dataset": None,
                "source_id": None,
                "source_role": members[0]["source_role"],
                "repeat": None,
                "task_id": None,
                "n_rows": sum(int(member["n_rows"]) for member in members),
                "n_tasks": sum(int(member["n_tasks"]) for member in members),
                **{column: sum(float(member[column]) for member in members) / len(members) for column in value_columns},
            }
        )
        panel_rows.append(row)

    aggregates = [*task_rows, *source_rows, *panel_rows]
    for row in aggregates:
        row["rmse_std"] = math.sqrt(float(row["squared_error_std"]))
        row["rmse_raw"] = math.sqrt(float(row["squared_error_raw"]))
    return aggregates


def write_training_panel_artifacts(
    config: OfflineValidationConfig,
    project_root: Path,
    *,
    output: Path | None = None,
    evaluation_records: list[RealTaskRecord] | None = None,
) -> dict[str, Any]:
    """Run or resume the paired training panel and persist aggregate tables."""
    destination = output or (project_root / config.mode.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    configure_study_logging(destination, study="offline-validation-training-panel")
    source_split = load_source_role_split(project_root / config.mode.source_roles_path)
    panel = build_eta_panel(config, project_root)
    manifests, curves = run_training_panel(config, project_root, destination)
    if any(row["status"] == "complete" for row in manifests):
        records = evaluation_records if evaluation_records is not None else load_real_task_panel(config, project_root)
    else:
        records = []
    evaluation_manifests, predictions, row_metrics = run_checkpoint_evaluations(
        config,
        project_root,
        manifests,
        records,
    )
    aggregates = aggregate_row_metrics(row_metrics, config.coverage_levels) if row_metrics else []
    training_failures = [
        {
            "stage": "training",
            **row,
            "evaluation_id": None,
            "dataset": None,
            "source_id": None,
            "repeat": None,
            "task_id": None,
        }
        for row in manifests
        if row["status"] == "failed"
    ]
    evaluation_failures = [{"stage": "evaluation", **row} for row in evaluation_manifests if row["status"] == "failed"]
    failures = [*training_failures, *evaluation_failures]

    (destination / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
    (destination / "eta_panel.json").write_text(json.dumps(panel, indent=2, sort_keys=True))
    (destination / "source_split.json").write_text(json.dumps(source_split.to_payload(), indent=2, sort_keys=True))
    (destination / "environment.json").write_text(
        json.dumps(environment_provenance(project_root), indent=2, sort_keys=True)
    )
    pl.DataFrame(manifests).sort("eta_label", "seed").write_parquet(destination / "training_manifests.parquet")
    if curves:
        curve_frame = pl.DataFrame(curves).sort("eta_label", "seed", "step")
    else:
        curve_frame = pl.DataFrame(
            schema={
                "job_id": pl.String,
                "eta_id": pl.String,
                "eta_label": pl.String,
                "seed": pl.Int64,
                "step": pl.Int64,
                "loss": pl.Float64,
            }
        )
    curve_frame.write_parquet(destination / "training_curves.parquet")
    source_manifests = [
        {
            "dataset": record.dataset,
            "source_id": record.task.source_id,
            "source_role": record.source_role,
            "repeat": record.repeat,
            "task_id": record.task.task_id,
            "outer_split_id": record.task.outer_split_id,
            "characterization_split_id": record.task.characterization_split_id,
            "preprocessing_id": record.task.preprocessing_id,
            "n_probe_fit": record.task.probe_fit.X.height,
            "n_probe_score": record.task.probe_score.X.height,
            "n_features": record.task.probe_fit.X.width,
        }
        for record in records
    ]
    table_payloads = {
        "source_manifests.parquet": source_manifests,
        "evaluation_manifests.parquet": evaluation_manifests,
        "pfn_predictions.parquet": predictions,
        "pfn_row_metrics.parquet": row_metrics,
        "pfn_aggregate_metrics.parquet": aggregates,
    }
    for filename, rows in table_payloads.items():
        if rows:
            pl.DataFrame(rows).write_parquet(destination / filename)
        else:
            pl.DataFrame(schema={"status": pl.String}).write_parquet(destination / filename)
    if failures:
        pl.DataFrame(failures, infer_schema_length=None).sort("stage", "eta_label", "seed").write_parquet(
            destination / "failure_events.parquet"
        )
    else:
        pl.DataFrame(
            schema={
                "job_id": pl.String,
                "eta_id": pl.String,
                "eta_label": pl.String,
                "seed": pl.Int64,
                "error_type": pl.String,
                "error_message": pl.String,
            }
        ).write_parquet(destination / "failure_events.parquet")
    status = (
        "pass"
        if not failures
        and all(row["status"] == "complete" for row in manifests)
        and all(row["status"] == "complete" for row in evaluation_manifests)
        else "failed"
    )
    summary = {
        "status": status,
        "n_eta": len(panel),
        "n_seeds": len(config.mode.seeds),
        "n_jobs": len(manifests),
        "n_evaluations": len(evaluation_manifests),
        "n_prediction_rows": len(predictions),
        "n_failures": len(failures),
        "pairing_id": config.mode.pairing_id,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.success(f"offline-validation training panel complete | status={status} | artifacts -> {destination}")
    return summary
