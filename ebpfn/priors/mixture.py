"""Route mixture orchestration: draw a route, realize a task, assemble output.

Each task uses one complete route drawn IID from the direct simplex weights. The
dispatch is registry-based so callers (and the Step 4 evaluator) never branch on
the route. Realized route, parameters, and SNR are recorded on the
``GeneratedTask.diagnostics`` and never on the ``TuningTask``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from ebpfn.data import CharacterizationShape, FeatureSchema, TaskPartition, TuningTask, content_hash
from ebpfn.priors import bnn, compositional, scm, tree
from ebpfn.priors.contracts import ROUTE_ORDER, GeneratedTask, HyperPrior, RouteName, RouteRealization, SharedTheta
from ebpfn.priors.features import apply_observation
from ebpfn.priors.targets import realize_target

_Route = Callable[[Any, int, int, SharedTheta, np.random.Generator], RouteRealization]

ROUTES: dict[RouteName, _Route] = {
    "scm": scm.realize,
    "bnn": bnn.realize,
    "tree": tree.realize,
    "compositional": compositional.realize,
}


def draw_shared_theta(eta: HyperPrior, route: RouteName, rng: np.random.Generator) -> SharedTheta:
    return SharedTheta(
        route=route,
        log_snr=float(rng.normal(eta.log_snr_mean, eta.snr_dispersion)),
        corr_strength=float(np.clip(rng.normal(eta.corr_strength_mean, eta.corr_dispersion), 0.0, 1.0)),
        heteroskedastic=bool(rng.random() < eta.heteroskedastic_rate),
        heavy_tail=bool(rng.random() < eta.heavy_tail_rate),
    )


def sample_task(
    eta: HyperPrior, shape: CharacterizationShape, rng: np.random.Generator, identity: tuple[Any, ...]
) -> GeneratedTask:
    if shape.p_categorical != 0:
        raise ValueError("primary V1 shapes must have no categorical features")
    if shape.task_type != "regression":
        raise NotImplementedError("only regression tasks are generated in V1")
    n = shape.n_probe_fit + shape.n_probe_score
    p = shape.p_numeric

    route = ROUTE_ORDER[int(rng.choice(len(ROUTE_ORDER), p=eta.weight_vector()))]
    shared = draw_shared_theta(eta, route, rng)
    realization = ROUTES[route](getattr(eta, route), n, p, shared, rng)
    target, target_diagnostics = realize_target(realization.signal, realization.x_raw, shared, rng)
    x_obs, missing_rates, observation_state = apply_observation(realization.x_raw, rng)

    names = tuple(f"x{index}" for index in range(p))
    schema = FeatureSchema(names, ("numeric",) * p)
    n_fit = shape.n_probe_fit
    row_ids = np.arange(n)
    probe_fit = TaskPartition(pl.DataFrame(x_obs[:n_fit], schema=names), target[:n_fit], row_ids[:n_fit])
    probe_score = TaskPartition(pl.DataFrame(x_obs[n_fit:], schema=names), target[n_fit:], row_ids[n_fit:])

    task_key = content_hash(route, n, p, identity, namespace="prior-task-1")
    tuning = TuningTask(
        task_id=f"prior-{task_key[:16]}",
        source_id="synthetic",
        task_type="regression",
        outer_split_id="synthetic-outer",
        characterization_split_id=f"synthetic-{task_key[:16]}",
        probe_fit=probe_fit,
        probe_score=probe_score,
        schema=schema,
        preprocessing_id="synthetic-raw",
        probe_fit_missing_rates=tuple(missing_rates.tolist()),
    )
    diagnostics: dict[str, Any] = {
        "route": route,
        "shape": {"n": n, "p": p, "n_probe_fit": n_fit, "n_probe_score": shape.n_probe_score},
        "shared_theta": {
            "log_snr": shared.log_snr,
            "corr_strength": shared.corr_strength,
            "heteroskedastic": shared.heteroskedastic,
            "heavy_tail": shared.heavy_tail,
        },
        "observation_state": observation_state,
        **realization.diagnostics,
        **target_diagnostics,
    }
    return GeneratedTask(tuning=tuning, diagnostics=diagnostics)
