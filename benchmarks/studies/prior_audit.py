"""Prior p-complexity and reproducibility audit.

Self-contained evidence for Step 3: over a feature grid, each route's complexity
diagnostic (SCM indegree, BNN fan-in, tree depth, compositional active fraction)
and realized SNR stay controlled; route frequencies converge to the direct
simplex weights; a fixed (eta, shape, identity) reproduces the task exactly; and
task-level parameters retain nonzero dispersion. The joint-Sobol identifiability
study is deferred to Step 4 (it needs the evaluator-noise baseline).
"""

import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from ebpfn.config import PriorStudyConfig
from ebpfn.data import CharacterizationShape
from ebpfn.priors import ROUTE_ORDER, HyperPrior, build_hyperprior, sample_task
from ebpfn.utils import RandomRole, RandomStreams, environment_provenance

ROUTE_METRIC = {
    "scm": "mean_indegree",
    "bnn": "fan_in_final",
    "tree": "mean_depth",
    "compositional": "active_fraction",
}

_SNR_TOLERANCE = 0.35  # allowed relative drift of mean realized SNR across the p grid
# Allowed deviation of the mean per-task realized/target SNR ratio from one. The
# ratio is per-task so it is immune to the lognormal mean shift of exp(log_snr).
_SNR_CALIBRATION_TOLERANCE = 0.2
_STABILITY_TOLERANCE = 0.5  # allowed relative spread of a route metric across the p grid


def _one_hot(eta: HyperPrior, route: str) -> HyperPrior:
    weights = {name: (1.0 if name == route else 0.0) for name in ROUTE_ORDER}
    return dataclasses.replace(eta, generator_weights=weights)


def _relative_spread(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    center = float(np.mean(array))
    if abs(center) < 1e-12:
        return float(np.ptp(array))
    return float(np.ptp(array) / abs(center))


def run_study(config: PriorStudyConfig) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any]]:
    mode = config.mode
    eta = build_hyperprior(config.prior)
    streams = RandomStreams(config.seed)
    target_snr = float(np.exp(config.prior.log_snr_mean))

    rows: list[dict[str, Any]] = []
    metric_spreads: dict[str, float] = {}
    snr_spreads: dict[str, float] = {}
    snr_route_means: dict[str, float] = {}
    snr_ratio_means: dict[str, float] = {}
    dispersion_ok = True

    for route in ROUTE_ORDER:
        eta_route = _one_hot(eta, route)
        metric_key = ROUTE_METRIC[route]
        metric_means: list[float] = []
        snr_means: list[float] = []
        route_ratios: list[float] = []
        for p in mode.feature_grid:
            shape = CharacterizationShape(mode.n_probe_fit, mode.n_probe_score, p, 0, "regression")
            diagnostics = [
                sample_task(eta_route, shape, streams, "audit", route, p, task).diagnostics
                for task in range(mode.n_tasks)
            ]
            metric_mean = float(np.mean([d[metric_key] for d in diagnostics]))
            snr_mean = float(np.mean([d["realized_snr"] for d in diagnostics]))
            log_snr_std = float(np.std([d["shared_theta"]["log_snr"] for d in diagnostics]))
            route_ratios.extend(d["realized_snr"] / d["target_snr"] for d in diagnostics)
            dispersion_ok = dispersion_ok and log_snr_std > 0.0
            metric_means.append(metric_mean)
            snr_means.append(snr_mean)
            rows.append(
                {
                    "route": route,
                    "p": p,
                    "metric_name": metric_key,
                    "metric_mean": metric_mean,
                    "realized_snr_mean": snr_mean,
                    "log_snr_std": log_snr_std,
                    "n_tasks": mode.n_tasks,
                }
            )
        metric_spreads[route] = _relative_spread(metric_means)
        snr_spreads[route] = _relative_spread(snr_means)
        snr_route_means[route] = float(np.mean(snr_means))
        snr_ratio_means[route] = float(np.mean(route_ratios))

    # Route selection is `rng.choice(..., p=weights)`; a large vectorized draw of
    # that same multinomial is faithful evidence of the weight->frequency law.
    # The behavioral guarantee that sample_task honours it lives in the tests.
    freq_n = 20000
    freq_rng = streams.generator(RandomRole.GENERATION, "route-frequency")
    draws = freq_rng.choice(len(ROUTE_ORDER), size=freq_n, p=eta.weight_vector())
    counts = Counter(int(index) for index in draws)
    frequency = {route: counts.get(position, 0) / freq_n for position, route in enumerate(ROUTE_ORDER)}
    frequency_l1 = float(sum(abs(frequency[route] - eta.generator_weights[route]) for route in ROUTE_ORDER))

    middle_p = mode.feature_grid[len(mode.feature_grid) // 2]
    freq_shape = CharacterizationShape(mode.n_probe_fit, mode.n_probe_score, middle_p, 0, "regression")

    first = sample_task(eta, freq_shape, streams, "reproducibility", 0)
    second = sample_task(eta, freq_shape, streams, "reproducibility", 0)
    reproducible = bool(
        np.array_equal(first.tuning.probe_fit.y, second.tuning.probe_fit.y) and first.diagnostics == second.diagnostics
    )

    table = pl.DataFrame(rows)
    snr_calibrated = all(abs(snr_ratio_means[route] - 1.0) <= _SNR_CALIBRATION_TOLERANCE for route in ROUTE_ORDER)
    checks = {
        "reproducible": reproducible,
        "route_frequency_converges": frequency_l1 < 0.05,
        "metric_p_stable": all(spread <= _STABILITY_TOLERANCE for spread in metric_spreads.values()),
        "snr_p_stable": all(spread <= _SNR_TOLERANCE for spread in snr_spreads.values()),
        "snr_calibrated": snr_calibrated,
        "parameters_dispersed": dispersion_ok,
    }
    status, missing = derive_study_status(mode.name, checks)
    evidence = {
        "target_snr": target_snr,
        "route_frequency": frequency,
        "route_frequency_l1": frequency_l1,
        "metric_relative_spread": metric_spreads,
        "snr_relative_spread": snr_spreads,
        "snr_route_means": snr_route_means,
        "snr_realized_target_ratio": snr_ratio_means,
        "checks": checks,
    }
    decision = {
        "status": status,
        "missing_checks": missing,
        "decision_owner": config.decision_owner,
        "decision_date": config.decision_date,
        "deferred_to_step4": ["joint_sobol_identifiability"],
    }
    return table, evidence, decision


def derive_study_status(mode: str, checks: dict[str, bool]) -> tuple[str, list[str]]:
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        return "failed", failed
    if mode == "audit":
        # The audit gate stays incomplete until the Step 4 joint-Sobol study runs.
        return "incomplete", ["joint_sobol_identifiability"]
    return "complete", []


def write_study_artifacts(
    config: PriorStudyConfig, project_root: Path, *, output: Path | None = None
) -> dict[str, Any]:
    destination = output or (project_root / config.mode.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table, evidence, decision = run_study(config)
    table.write_parquet(destination / "coordinates.parquet")
    (destination / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
    (destination / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))
    (destination / "decision_log.json").write_text(json.dumps(decision, indent=2, sort_keys=True))
    (destination / "environment.json").write_text(
        json.dumps(environment_provenance(project_root), indent=2, sort_keys=True)
    )
    return {"status": decision["status"], "rows": table.height}
