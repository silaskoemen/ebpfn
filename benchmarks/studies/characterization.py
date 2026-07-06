"""Hand-built characterization stability and decision study."""

import json
import resource
import sys
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
from ebpfn.characterize import TaskCharacterization
from ebpfn.characterize import characterize_multiresolution
from ebpfn.config import CharacterizationStudyConfig
from ebpfn.config import RowBudgetConfig
from ebpfn.data import FeatureSchema
from ebpfn.data import TaskPartition
from ebpfn.data import TuningTask
from ebpfn.utils import environment_provenance

MECHANISMS = (
    "null",
    "sparse_linear",
    "diffuse_linear",
    "threshold",
    "smooth",
    "interaction",
    "heteroskedastic",
    "heavy_tail",
    "rare_feature",
    "mixed",
)


def make_task(mechanism: str, config: CharacterizationStudyConfig, repeat: int, *, strength: float = 1.0) -> TuningTask:
    mode = config.mode
    n = mode.n_probe_fit + mode.n_probe_score
    rng = np.random.default_rng(
        np.random.SeedSequence([config.characterization.seed, repeat, MECHANISMS.index(mechanism)])
    )
    features = np.clip(rng.normal(size=(n, mode.n_features)), -4.0, 4.0)
    noise = rng.normal(size=n)
    if mechanism == "null":
        target = noise
    elif mechanism == "sparse_linear":
        target = strength * 2.0 * features[:, 0] + noise
    elif mechanism == "diffuse_linear":
        target = strength * np.sum(features, axis=1) / np.sqrt(mode.n_features) + noise
    elif mechanism == "threshold":
        target = strength * 2.5 * (features[:, 0] > 0.0) + noise
    elif mechanism == "smooth":
        target = strength * np.sin(2.0 * features[:, 0]) + noise * 0.5
    elif mechanism == "interaction":
        target = strength * features[:, 0] * features[:, 1] + noise * 0.5
    elif mechanism == "heteroskedastic":
        target = strength * features[:, 0] + noise * (0.2 + np.abs(features[:, 1]))
    elif mechanism == "heavy_tail":
        target = strength * features[:, 0] + rng.standard_t(3, size=n)
    elif mechanism == "rare_feature":
        features[:, -1] = rng.binomial(1, 0.015, size=n)
        target = strength * 3.0 * features[:, -1] + noise
    elif mechanism == "mixed":
        signal = features[:, 0] + (features[:, 1] > 0.0) + features[:, 2] * features[:, 3]
        target = strength * signal + noise * 0.5
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    names = tuple(f"x{index}" for index in range(mode.n_features))
    schema = FeatureSchema(names, ("numeric",) * mode.n_features)
    fit_ids = np.arange(mode.n_probe_fit)
    score_ids = np.arange(mode.n_probe_fit, n)
    fit = TaskPartition(pl.DataFrame(features[: mode.n_probe_fit], schema=names), target[: mode.n_probe_fit], fit_ids)
    score = TaskPartition(
        pl.DataFrame(features[mode.n_probe_fit :], schema=names), target[mode.n_probe_fit :], score_ids
    )
    return TuningTask(
        f"{mechanism}-{repeat}",
        "hand-built",
        "regression",
        "study-outer",
        f"study-inner-{repeat}",
        fit,
        score,
        schema,
        "study-preprocessed",
        (0.0,) * mode.n_features,
    )


def _process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _measure(call: Callable[[], TaskCharacterization]) -> tuple[TaskCharacterization, float, int, int]:
    # ru_maxrss is a whole-process high-water mark that never resets between calls, so the raw
    # reading cannot distinguish a small call from a large one made earlier in the same run.
    # Report the watermark's growth across this call instead of its absolute value.
    baseline_rss = _process_peak_rss_bytes()
    tracemalloc.start()
    started = perf_counter()
    try:
        result = call()
        runtime = perf_counter() - started
        _, traced_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_growth = max(0, _process_peak_rss_bytes() - baseline_rss)
    return result, runtime, traced_peak, rss_growth


def _result_rows(
    result: TaskCharacterization,
    *,
    mechanism: str,
    repeat: int,
    lambda_: float,
    policy: str,
    runtime: float,
    traced_peak: int,
    process_peak_rss_growth: int,
    strength: float = 1.0,
) -> list[dict[str, Any]]:
    rows = [
        {
            "mechanism": mechanism,
            "strength": strength,
            "repeat": repeat,
            "lambda": lambda_,
            "policy": policy,
            "representation": "raw",
            "coordinate": coordinate.name,
            "block": coordinate.block,
            "learner": coordinate.learner,
            "target": coordinate.target,
            "statistic": coordinate.statistic,
            "row_budget": coordinate.row_budget,
            "value": float(value),
            "raw_value": float(raw),
            "valid": bool(valid),
            "runtime_seconds": runtime,
            "tracemalloc_peak_bytes": traced_peak,
            "process_peak_rss_growth_bytes": process_peak_rss_growth,
        }
        for coordinate, value, raw, valid in zip(
            result.coordinates, result.values, result.raw_values, result.valid, strict=True
        )
    ]
    for budget in result.metadata["budgets"]:
        for name, raw_value in budget["observation_raw"].items():
            rows.append(
                {
                    "mechanism": mechanism,
                    "strength": strength,
                    "repeat": repeat,
                    "lambda": lambda_,
                    "policy": policy,
                    "representation": "diagnostic",
                    "coordinate": f"m{budget['row_budget']}.diagnostic.{name}",
                    "block": "observation_diagnostic",
                    "learner": None,
                    "target": None,
                    "statistic": name,
                    "row_budget": budget["row_budget"],
                    "value": float(raw_value),
                    "raw_value": float(raw_value),
                    "valid": bool(np.isfinite(raw_value)),
                    "runtime_seconds": runtime,
                    "tracemalloc_peak_bytes": traced_peak,
                    "process_peak_rss_growth_bytes": process_peak_rss_growth,
                }
            )
    return rows


def _contrast_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gains = {
        (row["row_budget"], row["target"], row["learner"]): float(row["value"])
        for row in raw_rows
        if row["statistic"] == "gain"
    }
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if row["representation"] != "raw":
            continue
        contrast = dict(row)
        contrast["representation"] = "contrast"
        if row["statistic"] != "gain":
            rows.append(contrast)
            continue
        budget = row["row_budget"]
        target = row["target"]
        learner = row["learner"]
        linear = gains[(budget, target, "linear")]
        if learner == "linear":
            rows.append(contrast)
        elif learner == "bins":
            contrast["coordinate"] = str(row["coordinate"]).replace("bins.gain", "bins_vs_linear.contrast")
            contrast["statistic"] = "contrast"
            contrast["value"] = contrast["raw_value"] = (float(row["value"]) - linear) / 2.0
            rows.append(contrast)
        elif learner == "pairwise":
            bins = gains[(budget, target, "bins")]
            contrast["coordinate"] = str(row["coordinate"]).replace("pairwise.gain", "pairwise_vs_bins.contrast")
            contrast["statistic"] = "contrast"
            contrast["value"] = contrast["raw_value"] = (float(row["value"]) - bins) / 2.0
            rows.append(contrast)
        elif learner == "rff":
            contrast["coordinate"] = str(row["coordinate"]).replace("rff.gain", "rff_vs_linear.contrast")
            contrast["statistic"] = "contrast"
            contrast["value"] = contrast["raw_value"] = (float(row["value"]) - linear) / 2.0
            rows.append(contrast)
    return rows


def _append_both_representations(rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> None:
    rows.extend(result_rows)
    rows.extend(_contrast_rows(result_rows))


def _rank_stability(table: pl.DataFrame, policy: str) -> list[dict[str, Any]]:
    selected = table.filter((pl.col("policy") == policy) & pl.col("representation").is_in(["raw", "contrast"]))
    summaries: list[dict[str, Any]] = []
    for mechanism in selected["mechanism"].unique().sort():
        for representation in ("raw", "contrast"):
            group = selected.filter((pl.col("mechanism") == mechanism) & (pl.col("representation") == representation))
            repeats = group["repeat"].unique().sort().to_list()
            by_repeat = {
                repeat: dict(
                    zip(
                        group.filter(pl.col("repeat") == repeat)["coordinate"].to_list(),
                        group.filter(pl.col("repeat") == repeat)["value"].to_list(),
                        strict=True,
                    )
                )
                for repeat in repeats
            }
            correlations: list[float] = []
            for left_index, left_repeat in enumerate(repeats):
                for right_repeat in repeats[left_index + 1 :]:
                    names = sorted(set(by_repeat[left_repeat]) & set(by_repeat[right_repeat]))
                    if len(names) < 2:
                        continue
                    left = np.asarray([by_repeat[left_repeat][name] for name in names])
                    right = np.asarray([by_repeat[right_repeat][name] for name in names])
                    left_rank = np.argsort(np.argsort(left))
                    right_rank = np.argsort(np.argsort(right))
                    correlations.append(float(np.corrcoef(left_rank, right_rank)[0, 1]))
            summaries.append(
                {
                    "mechanism": mechanism,
                    "representation": representation,
                    "median_spearman": None if not correlations else float(np.median(correlations)),
                    "pair_count": len(correlations),
                }
            )
    return summaries


def _coordinate_stability(table: pl.DataFrame, policy: str) -> list[dict[str, Any]]:
    selected = table.filter(
        (pl.col("policy") == policy) & pl.col("representation").is_in(["raw", "contrast"])
    ).with_columns(pl.col("value").mean().over("mechanism", "representation", "coordinate").alias("_mean"))
    return (
        selected.group_by("mechanism", "representation", "coordinate")
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").std(ddof=0).alias("std"),
            ((pl.col("value") * pl.col("_mean")) >= 0.0).mean().alias("sign_stability"),
        )
        .sort("mechanism", "representation", "coordinate")
        .to_dicts()
    )


def _structure_diagnostics(table: pl.DataFrame, policy: str) -> dict[str, Any]:
    selected = table.filter((pl.col("policy") == policy) & pl.col("representation").is_in(["raw", "contrast"]))
    block_summary = (
        selected.group_by("representation", "mechanism", "block")
        .agg(pl.col("value").abs().mean().alias("mean_absolute"))
        .with_columns(
            (pl.col("mean_absolute") / pl.col("mean_absolute").sum().over("representation", "mechanism")).alias(
                "absolute_share"
            )
        )
        .sort("representation", "mechanism", "block")
    )
    dominant_blocks = (
        block_summary.sort("absolute_share", descending=True)
        .group_by("representation", "mechanism", maintain_order=True)
        .first()
        .select("representation", "mechanism", pl.col("block").alias("dominant_block"), "absolute_share")
        .sort("representation", "mechanism")
    )
    redundancy: list[dict[str, Any]] = []
    for representation in ("raw", "contrast"):
        group = selected.filter(pl.col("representation") == representation)
        samples = sorted(set(zip(group["mechanism"].to_list(), group["repeat"].to_list(), strict=True)))
        coordinates = sorted(group["coordinate"].unique().to_list())
        lookup = {
            (row["mechanism"], row["repeat"], row["coordinate"]): row["value"]
            for row in group.select("mechanism", "repeat", "coordinate", "value").to_dicts()
        }
        matrix = np.asarray(
            [[lookup[(mechanism, repeat, coordinate)] for coordinate in coordinates] for mechanism, repeat in samples]
        )
        correlation = np.nan_to_num(np.corrcoef(matrix, rowvar=False))
        pairs = sorted(
            (
                (abs(float(correlation[left, right])), coordinates[left], coordinates[right])
                for left in range(len(coordinates))
                for right in range(left + 1, len(coordinates))
            ),
            reverse=True,
        )
        redundancy.append(
            {
                "representation": representation,
                "median_absolute_correlation": float(np.median([pair[0] for pair in pairs])),
                "maximum_absolute_correlation": pairs[0][0],
                "top_pairs": [
                    {"absolute_correlation": value, "left": left, "right": right} for value, left, right in pairs[:20]
                ],
            }
        )
    return {
        "block_summary": block_summary.to_dicts(),
        "dominant_blocks": dominant_blocks.to_dicts(),
        "coordinate_redundancy": redundancy,
    }


def derive_study_status(mode: str, checks: dict[str, bool]) -> tuple[str, list[str]]:
    if mode == "fast":
        return "provisional", []
    missing_checks = {"representative_real_task_repeats"}
    failed = sorted(name for name, passed in checks.items() if not passed and name not in missing_checks)
    missing = sorted(name for name, passed in checks.items() if not passed and name in missing_checks)
    if failed:
        return "failed", [f"failed:{name}" for name in failed]
    if missing:
        return "incomplete", [f"missing:{name}" for name in missing]
    return "frozen", []


def run_study(config: CharacterizationStudyConfig) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = config.characterization
    mechanisms = (
        MECHANISMS if config.mode.name == "audit" else ("null", "sparse_linear", "threshold", "interaction", "mixed")
    )
    ridge_candidates = config.ridge_candidates if config.mode.name == "audit" else (base.ridge.lambda_,)
    for lambda_ in ridge_candidates:
        candidate = base.model_copy(
            update={
                "ridge": base.ridge.model_copy(update={"lambda_": lambda_}),
                "include_observation_coordinates": False,
            }
        )
        for repeat in range(config.mode.repeats):
            candidate = candidate.model_copy(update={"repeat": repeat})
            for mechanism in mechanisms:
                result, runtime, traced_peak, process_peak_rss = _measure(
                    lambda mechanism=mechanism, repeat=repeat, candidate=candidate: characterize_multiresolution(
                        make_task(mechanism, config, repeat), candidate
                    )
                )
                result_rows = _result_rows(
                    result,
                    mechanism=mechanism,
                    repeat=repeat,
                    lambda_=lambda_,
                    policy="ridge",
                    runtime=runtime,
                    traced_peak=traced_peak,
                    process_peak_rss_growth=process_peak_rss,
                )
                _append_both_representations(rows, result_rows)
    all_policies = tuple(
        RowBudgetConfig(minimum=256, spacing=spacing, weight=weight, feature_view=feature_view)
        for spacing in ("geometric", "sqrt")
        for weight in ("uniform", "row_count")
        for feature_view in ("frozen", "local")
    )
    policies = all_policies if config.mode.name == "audit" else (all_policies[0], all_policies[4])
    for row_policy in policies:
        policy_name = f"{row_policy.spacing}/{row_policy.weight}/{row_policy.feature_view}"
        candidate = base.model_copy(update={"row_budgets": row_policy, "include_observation_coordinates": False})
        for repeat in range(config.mode.repeats):
            candidate = candidate.model_copy(update={"repeat": repeat})
            for mechanism in mechanisms:
                result, runtime, traced_peak, process_peak_rss = _measure(
                    lambda mechanism=mechanism, repeat=repeat, candidate=candidate: characterize_multiresolution(
                        make_task(mechanism, config, repeat), candidate
                    )
                )
                result_rows = _result_rows(
                    result,
                    mechanism=mechanism,
                    repeat=repeat,
                    lambda_=base.ridge.lambda_,
                    policy=policy_name,
                    runtime=runtime,
                    traced_peak=traced_peak,
                    process_peak_rss_growth=process_peak_rss,
                )
                _append_both_representations(rows, result_rows)
    table = pl.DataFrame(rows)
    gain_rows = table.filter(
        (pl.col("policy") == "ridge") & (pl.col("representation") == "raw") & (pl.col("statistic") == "gain")
    )
    summaries = (
        gain_rows.group_by("lambda", "mechanism")
        .agg(pl.col("value").mean().alias("mean"), pl.col("value").std(ddof=0).alias("std"))
        .sort("lambda", "mechanism")
    )
    lambda_scores: list[tuple[bool, float, float]] = []
    for lambda_ in ridge_candidates:
        selected = summaries.filter(pl.col("lambda") == lambda_)
        null_mean = float(np.mean(selected.filter(pl.col("mechanism") == "null")["mean"].to_numpy()))
        signals = selected.filter(pl.col("mechanism") != "null")
        ratios = np.abs(signals["mean"].to_numpy() - null_mean) / np.maximum(signals["std"].to_numpy(), 1e-8)
        lambda_scores.append((null_mean <= 0.0, float(np.median(ratios)), lambda_))
    eligible = [item for item in lambda_scores if item[0]]
    diagnostic_lambda = max(eligible or lambda_scores, key=lambda item: (item[1], item[2]))[2]
    selected_lambda = diagnostic_lambda if eligible else None
    policy_rows = table.filter((pl.col("policy") != "ridge") & (pl.col("representation") == "raw"))
    policy_summary = (
        policy_rows.group_by("policy", "coordinate")
        .agg(pl.col("value").std(ddof=0).alias("std"), pl.col("runtime_seconds").mean().alias("runtime"))
        .group_by("policy")
        .agg(pl.col("std").median().alias("median_std"), pl.col("runtime").mean().alias("runtime"))
        .sort("median_std", "runtime", "policy")
    )
    minimum_std = float(np.min(policy_summary["median_std"].to_numpy()))
    competitive = set(policy_summary.filter(pl.col("median_std") <= minimum_std + 1e-8).get_column("policy").to_list())
    preference = (
        "geometric/uniform/frozen",
        "sqrt/uniform/frozen",
        "geometric/row_count/frozen",
        "sqrt/row_count/frozen",
        "geometric/uniform/local",
        "sqrt/uniform/local",
        "geometric/row_count/local",
        "sqrt/row_count/local",
    )
    selected_policy = next(policy for policy in preference if policy in competitive)
    selected_row_policy = next(
        policy
        for policy in all_policies
        if f"{policy.spacing}/{policy.weight}/{policy.feature_view}" == selected_policy
    )
    response_rows: list[dict[str, Any]] = []
    response_strengths = (0.0, 0.5, 1.0, 2.0) if config.mode.name == "audit" else (0.0, 1.0)
    response_repeats = range(config.mode.repeats) if config.mode.name == "audit" else range(1)
    response_candidate = base.model_copy(
        update={
            "ridge": base.ridge.model_copy(update={"lambda_": diagnostic_lambda}),
            "row_budgets": selected_row_policy,
            "include_observation_coordinates": False,
        }
    )
    for repeat in response_repeats:
        candidate = response_candidate.model_copy(update={"repeat": repeat})
        for mechanism in ("sparse_linear", "threshold", "interaction"):
            for strength in response_strengths:
                result, runtime, traced_peak, process_peak_rss = _measure(
                    lambda mechanism=mechanism, repeat=repeat, strength=strength, candidate=candidate: (
                        characterize_multiresolution(make_task(mechanism, config, repeat, strength=strength), candidate)
                    )
                )
                result_rows = _result_rows(
                    result,
                    mechanism=mechanism,
                    repeat=repeat,
                    lambda_=diagnostic_lambda,
                    policy="response",
                    runtime=runtime,
                    traced_peak=traced_peak,
                    process_peak_rss_growth=process_peak_rss,
                    strength=strength,
                )
                _append_both_representations(response_rows, result_rows)
    table = pl.concat((table, pl.DataFrame(response_rows)), how="vertical_relaxed")
    expected_learners = {"sparse_linear": "linear", "threshold": "bins", "interaction": "pairwise"}
    response_summary = (
        table.filter(
            (pl.col("policy") == "response")
            & (pl.col("target") == "location")
            & pl.struct("mechanism", "learner").map_elements(
                lambda item: expected_learners.get(item["mechanism"]) == item["learner"],
                return_dtype=pl.Boolean,
            )
        )
        .group_by("representation", "mechanism", "strength", "row_budget", "statistic")
        .agg(pl.col("value").mean().alias("mean"), pl.col("value").std(ddof=0).alias("std"))
        .sort("representation", "mechanism", "strength", "row_budget")
        .to_dicts()
    )
    coordinate_stability = _coordinate_stability(table, selected_policy)
    rank_stability = _rank_stability(table, selected_policy)
    structure_diagnostics = _structure_diagnostics(table, selected_policy)
    p_complexity: list[dict[str, Any]] = []
    p_grid = (2, 8, 32, 100) if config.mode.name == "audit" else (2, config.mode.n_features)
    for feature_count in p_grid:
        grid_mode = config.mode.model_copy(update={"n_features": feature_count})
        grid_config = config.model_copy(update={"mode": grid_mode})
        result, runtime, traced_peak, process_peak_rss_growth = _measure(
            lambda grid_config=grid_config: characterize_multiresolution(
                make_task("sparse_linear", grid_config, 0),
                base.model_copy(update={"include_observation_coordinates": False}),
            )
        )
        p_complexity.append(
            {
                "p": feature_count,
                "runtime_seconds": runtime,
                "tracemalloc_peak_bytes": traced_peak,
                "process_peak_rss_growth_bytes": process_peak_rss_growth,
                "budgets": [item["map_dimensions"] for item in result.metadata["budgets"]],
            }
        )
    raw_rows = table.filter(pl.col("representation") == "raw")
    checks = {
        "five_complete_repeats": config.mode.repeats >= 5,
        "all_vectors_valid": bool(table["valid"].all()),
        "selected_lambda_null_nonpositive": bool(eligible),
        "complete_domain_feasible_within_scope": (
            config.mode.n_probe_fit + config.mode.n_probe_score >= config.applicability_max_rows
        ),
        "feature_width_feasible_within_scope": max(p_grid) >= config.applicability_max_features,
        "memory_recorded": bool(
            (raw_rows["tracemalloc_peak_bytes"] > 0).all()
            and (raw_rows["process_peak_rss_growth_bytes"] >= 0).all()
            and any(row["process_peak_rss_growth_bytes"] > 0 for row in p_complexity)
        ),
        "stability_reported": bool(coordinate_stability and any(item["pair_count"] > 0 for item in rank_stability)),
        "response_curves_reported": bool(response_summary),
        "structure_diagnostics_reported": bool(
            structure_diagnostics["block_summary"] and structure_diagnostics["coordinate_redundancy"]
        ),
        "representative_real_task_repeats": False,
    }
    status, blocking_reasons = derive_study_status(config.mode.name, checks)
    decision = {
        "status": status,
        "ridge_lambda": selected_lambda,
        "row_budget_policy": selected_policy,
        "selection_rule": "valid/null-nonpositive, then response-to-repeat-SD; D2 by median repeat SD with preregistered simplicity tie-break",
        "repeats": config.mode.repeats,
        "owner": config.decision_owner,
        "date": config.decision_date,
        "gate_checks": checks,
        "blocking_reasons": blocking_reasons,
        "applicability": {
            "max_rows": config.applicability_max_rows,
            "max_features": config.applicability_max_features,
            "full_domain_feasibility": "validated only within these construction-study bounds",
        },
    }
    evidence = {
        "lambda_scores": [
            {"null_nonpositive": valid, "response_noise_score": score, "lambda": lambda_}
            for valid, score, lambda_ in lambda_scores
        ],
        "policy_summary": policy_summary.to_dicts(),
        "p_complexity": p_complexity,
        "coordinate_stability": coordinate_stability,
        "rank_stability": rank_stability,
        "response_summary": response_summary,
        "structure_diagnostics": structure_diagnostics,
    }
    return table, decision, evidence


def schema_payload(result: TaskCharacterization) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "coordinates": [asdict(coordinate) for coordinate in result.coordinates],
        "metadata": result.metadata,
    }


def write_study_artifacts(
    config: CharacterizationStudyConfig,
    project_root: Path,
    *,
    output: Path | None = None,
) -> dict[str, Any]:
    destination = output or (project_root / config.mode.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table, decision, evidence = run_study(config)
    table.write_parquet(destination / "coordinates.parquet")
    (destination / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
    (destination / "decision_log.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    (destination / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))
    (destination / "environment.json").write_text(
        json.dumps(environment_provenance(project_root), indent=2, sort_keys=True)
    )
    for representation in ("raw", "contrast"):
        characterizer = config.characterization.model_copy(update={"representation": representation})
        result = characterize_multiresolution(make_task("mixed", config, 0), characterizer)
        (destination / f"schema_{representation}.json").write_text(
            json.dumps(schema_payload(result), indent=2, sort_keys=True)
        )
    return decision
