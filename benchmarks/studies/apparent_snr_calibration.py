"""Shape-matched apparent-SNR gate for freezing the baseline prior."""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from benchmarks.studies.study_logging import configure_study_logging
from benchmarks.studies.tuning_recovery import apparent_snr_report
from ebpfn.config import TuningStudyConfig
from ebpfn.data import CharacterizationShape, load_source_role_split
from ebpfn.priors import build_hyperprior, hyperprior_from_dict, hyperprior_to_dict, sample_task
from ebpfn.utils import RandomStreams, environment_provenance
from loguru import logger


def _candidate_panel(
    config: TuningStudyConfig,
    project_root: Path,
    baseline_report: dict[str, Any],
    existing_rows: list[dict[str, Any]],
    on_row: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    calibration = config.calibration
    baseline = build_hyperprior(config.tuning.prior)
    existing = {
        (float(row["log_snr_mean"]), float(row["snr_dispersion"])): row
        for row in existing_rows
        if row.get("report", {}).get("calibration_version") == baseline_report.get("calibration_version")
        and row.get("report", {}).get("source_split_id") == baseline_report.get("source_split_id")
    }
    rows: list[dict[str, Any]] = []
    for log_snr_mean in calibration.log_snr_mean_candidates:
        for snr_dispersion in calibration.snr_dispersion_candidates:
            eta = dataclasses.replace(
                baseline,
                log_snr_mean=log_snr_mean,
                snr_dispersion=snr_dispersion,
            )
            if log_snr_mean == baseline.log_snr_mean and snr_dispersion == baseline.snr_dispersion:
                report = baseline_report
            elif (log_snr_mean, snr_dispersion) in existing:
                report = existing[(log_snr_mean, snr_dispersion)]["report"]
            else:
                report = apparent_snr_report(
                    config,
                    project_root,
                    n_tasks_per_target=calibration.n_synthetic_per_target,
                    characterization_dir=project_root / calibration.characterization_dir,
                    source_roles_path=project_root / calibration.source_roles_path,
                    role=calibration.source_role,
                    eta=eta,
                )
            rows.append(
                {
                    "log_snr_mean": log_snr_mean,
                    "snr_dispersion": snr_dispersion,
                    "mean_gap_real_minus_synthetic": report.get("mean_gap_real_minus_synthetic"),
                    "source_balanced_energy_score": report.get("source_balanced_energy_score"),
                    "eta": hyperprior_to_dict(eta),
                    "report": report,
                }
            )
            if on_row is not None:
                on_row(rows)
            logger.info(
                f"calibration cell | log_snr_mean={log_snr_mean} | "
                f"snr_dispersion={snr_dispersion} | "
                f"gap={report.get('mean_gap_real_minus_synthetic')} | "
                f"energy={report.get('source_balanced_energy_score')}"
            )
    return rows


def _decision(
    config: TuningStudyConfig,
    baseline_report: dict[str, Any],
    panel: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gap = baseline_report.get("mean_gap_real_minus_synthetic")
    tolerance = config.calibration.max_abs_source_weighted_gap
    if gap is None:
        status = "incomplete"
        reason = "no role-eligible shape-matched real tasks"
        selected = {"eta": hyperprior_to_dict(build_hyperprior(config.tuning.prior))}
    elif abs(float(gap)) <= tolerance:
        status = "frozen_baseline"
        reason = "source-weighted apparent-SNR gap is within the predeclared tolerance"
        selected = next(
            row
            for row in panel
            if row["log_snr_mean"] == config.tuning.prior.log_snr_mean
            and row["snr_dispersion"] == config.tuning.prior.snr_dispersion
        )
    else:
        eligible = [
            row
            for row in panel
            if row["mean_gap_real_minus_synthetic"] is not None
            and abs(float(row["mean_gap_real_minus_synthetic"])) <= tolerance
            and row["source_balanced_energy_score"] is not None
        ]
        ranked = eligible or [
            row
            for row in panel
            if row["source_balanced_energy_score"] is not None and row["mean_gap_real_minus_synthetic"] is not None
        ]
        if not ranked:
            status = "incomplete"
            reason = "candidate panel produced no finite calibration scores"
            selected = {"eta": hyperprior_to_dict(build_hyperprior(config.tuning.prior))}
        else:
            if eligible:
                best_energy = min(float(row["source_balanced_energy_score"]) for row in eligible)
                competitive = [
                    row
                    for row in eligible
                    if float(row["source_balanced_energy_score"])
                    <= best_energy + config.calibration.energy_competitive_tolerance
                ]
                baseline_prior = config.tuning.prior

                def baseline_distance(row: dict[str, Any]) -> float:
                    mean_distance = (float(row["log_snr_mean"]) - baseline_prior.log_snr_mean) / 5.0
                    dispersion_distance = np.log(float(row["snr_dispersion"]) / baseline_prior.snr_dispersion)
                    return float(mean_distance**2 + dispersion_distance**2)

                selected = min(
                    competitive,
                    key=lambda row: (
                        baseline_distance(row),
                        float(row["source_balanced_energy_score"]),
                    ),
                )
            else:
                selected = min(
                    ranked,
                    key=lambda row: (
                        float(row["source_balanced_energy_score"]),
                        abs(float(row["mean_gap_real_minus_synthetic"])),
                    ),
                )
            if not eligible:
                status = "recalibration_failed"
                reason = "no fixed-SNR candidate met the predeclared mean-gap tolerance"
            elif selected["log_snr_mean"] in (
                config.calibration.log_snr_mean_candidates[0],
                config.calibration.log_snr_mean_candidates[-1],
            ) or selected["snr_dispersion"] in (
                config.calibration.snr_dispersion_candidates[0],
                config.calibration.snr_dispersion_candidates[-1],
            ):
                status = "candidate_grid_boundary"
                reason = "the selected candidate lies on the fixed-SNR grid boundary"
            else:
                status = "structural_check_required"
                reason = "a candidate met the calibration gate and awaits the structural sanity check"
    decision = {
        "status": status,
        "reason": reason,
        "source_role": config.calibration.source_role,
        "source_split_id": baseline_report.get("source_split_id"),
        "baseline_mean_gap_real_minus_synthetic": gap,
        "max_abs_source_weighted_gap": tolerance,
        "selection_rule": (
            "filter by mean-gap tolerance; retain source-balanced energy scores within the competitive "
            "tolerance; select the candidate closest to baseline"
        ),
        "energy_competitive_tolerance": config.calibration.energy_competitive_tolerance,
        "structural_sanity_decision": "not_required" if status == "frozen_baseline" else "pending",
        "selected_log_snr_mean": selected.get("log_snr_mean"),
        "selected_snr_dispersion": selected.get("snr_dispersion"),
        "selected_mean_gap_real_minus_synthetic": selected.get("mean_gap_real_minus_synthetic", gap),
        "selected_source_balanced_energy_score": selected.get("source_balanced_energy_score"),
        "decision_owner": config.decision_owner,
        "decision_date": config.decision_date,
    }
    return decision, selected["eta"]


def _structural_sanity(
    config: TuningStudyConfig,
    selected_eta_payload: dict[str, Any],
    calibration_report: dict[str, Any],
    *,
    tasks_per_target: int = 4,
) -> dict[str, Any]:
    baseline = build_hyperprior(config.tuning.prior)
    selected = hyperprior_from_dict(selected_eta_payload)
    baseline_structural = hyperprior_to_dict(baseline)
    selected_structural = hyperprior_to_dict(selected)
    for payload in (baseline_structural, selected_structural):
        payload.pop("log_snr_mean")
        payload.pop("snr_dispersion")
    targets = calibration_report.get("targets")
    if not isinstance(targets, list) or not targets:
        return {"passed": False, "reason": "calibration report has no target shapes", "n_pairs": 0}

    streams = RandomStreams(config.tuning.seed)
    rows: list[dict[str, Any]] = []
    for target in targets:
        shape_payload = target["shape"]
        shape = CharacterizationShape(
            n_probe_fit=int(shape_payload["n_probe_fit"]),
            n_probe_score=int(shape_payload["n_probe_score"]),
            p_numeric=int(shape_payload["p_numeric"]),
            p_categorical=int(shape_payload["p_categorical"]),
            task_type="regression",
        )
        for member in range(tasks_per_target):
            identity = ("calibration-structural-sanity", target["dataset"], member)
            first = sample_task(baseline, shape, streams, *identity, common_random_numbers=True)
            second = sample_task(selected, shape, streams, *identity, common_random_numbers=True)
            first_theta = dict(first.diagnostics["shared_theta"])
            second_theta = dict(second.diagnostics["shared_theta"])
            first_theta.pop("log_snr")
            second_theta.pop("log_snr")
            rows.append(
                {
                    "dataset": target["dataset"],
                    "member": member,
                    "route_equal": first.diagnostics["route"] == second.diagnostics["route"],
                    "structural_theta_equal": first_theta == second_theta,
                    "fit_features_equal": first.tuning.probe_fit.X.equals(second.tuning.probe_fit.X),
                    "score_features_equal": first.tuning.probe_score.X.equals(second.tuning.probe_score.X),
                    "task_identity_distinct": first.tuning.task_id != second.tuning.task_id,
                }
            )
    checks = {
        "only_snr_hyperparameters_changed": baseline_structural == selected_structural,
        "routes_paired": all(row["route_equal"] for row in rows),
        "structural_theta_paired": all(row["structural_theta_equal"] for row in rows),
        "fit_features_paired": all(row["fit_features_equal"] for row in rows),
        "score_features_paired": all(row["score_features_equal"] for row in rows),
        "eta_specific_task_identity": all(row["task_identity_distinct"] for row in rows),
    }
    return {
        "passed": all(checks.values()),
        "n_pairs": len(rows),
        "checks": checks,
        "rows": rows,
    }


def write_calibration_artifacts(
    config: TuningStudyConfig,
    project_root: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    calibration = config.calibration
    destination = output or (project_root / calibration.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    configure_study_logging(destination, study="apparent-snr-calibration")
    report = apparent_snr_report(
        config,
        project_root,
        n_tasks_per_target=calibration.n_synthetic_per_target,
        characterization_dir=project_root / calibration.characterization_dir,
        source_roles_path=project_root / calibration.source_roles_path,
        role=calibration.source_role,
    )
    baseline_gap = report.get("mean_gap_real_minus_synthetic")
    if baseline_gap is not None and abs(float(baseline_gap)) <= calibration.max_abs_source_weighted_gap:
        baseline = build_hyperprior(config.tuning.prior)
        panel = [
            {
                "log_snr_mean": baseline.log_snr_mean,
                "snr_dispersion": baseline.snr_dispersion,
                "mean_gap_real_minus_synthetic": baseline_gap,
                "source_balanced_energy_score": report.get("source_balanced_energy_score"),
                "eta": hyperprior_to_dict(baseline),
                "report": report,
            }
        ]
    else:
        panel_path = destination / "candidate_panel.json"
        if panel_path.exists():
            existing_panel = json.loads(panel_path.read_text())
            if not isinstance(existing_panel, list):
                raise TypeError("existing calibration candidate panel must be a JSON array")
        else:
            existing_panel = []

        def persist_panel(rows: list[dict[str, Any]]) -> None:
            panel_path.write_text(json.dumps(rows, indent=2, sort_keys=True))

        panel = _candidate_panel(
            config,
            project_root,
            report,
            existing_panel,
            on_row=persist_panel,
        )
    decision, eta = _decision(config, report, panel)
    if decision["status"] == "structural_check_required":
        structural_sanity = _structural_sanity(config, eta, report)
        if structural_sanity["passed"]:
            decision["status"] = "frozen_recalibrated"
            decision["reason"] = "the selected candidate passed calibration and the structural sanity check"
            decision["structural_sanity_decision"] = "passed"
        else:
            decision["status"] = "structural_check_failed"
            decision["reason"] = "the selected candidate failed the structural sanity check"
            decision["structural_sanity_decision"] = "failed"
    else:
        structural_sanity = {"passed": None, "reason": "not required", "n_pairs": 0}
    source_split = load_source_role_split(project_root / calibration.source_roles_path)

    (destination / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
    (destination / "apparent_snr.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (destination / "candidate_panel.json").write_text(json.dumps(panel, indent=2, sort_keys=True))
    (destination / "decision_log.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    (destination / "structural_sanity.json").write_text(json.dumps(structural_sanity, indent=2, sort_keys=True))
    (destination / "eta_0_candidate.json").write_text(json.dumps(eta, indent=2, sort_keys=True))
    (destination / "source_split.json").write_text(json.dumps(source_split.to_payload(), indent=2, sort_keys=True))
    (destination / "environment.json").write_text(
        json.dumps(environment_provenance(project_root), indent=2, sort_keys=True)
    )
    if decision["status"] in ("frozen_baseline", "frozen_recalibrated"):
        (destination / "eta_0.json").write_text(json.dumps(eta, indent=2, sort_keys=True))
    logger.success(f"apparent-SNR calibration complete | status={decision['status']} | artifacts -> {destination}")
    return decision
