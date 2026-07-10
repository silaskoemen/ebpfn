"""Population optimizer over the feasible unit hypercube.

Wraps SciPy differential evolution (a population method, no new dependency). The
objective is the simulator-only evaluator; infeasible vectors (outside the direct
simplex) receive a large penalty rather than being repaired, matching the
vectorizer's rejection semantics.
"""

from collections.abc import Callable

import numpy as np

# Penalty returned for infeasible candidate vectors so the population avoids them.
_INFEASIBLE_PENALTY = 1.0e6


def optimize_population(
    objective: Callable[[np.ndarray], float],
    feasible: Callable[[np.ndarray], bool],
    dimension: int,
    rng: np.random.Generator,
    *,
    maxiter: int,
    popsize: int,
) -> np.ndarray:
    """Return the best feasible unit vector found by differential evolution."""
    from scipy.optimize import differential_evolution  # lazy: keep `import ebpfn` scipy-free

    def penalized(vector: np.ndarray) -> float:
        if not feasible(vector):
            return _INFEASIBLE_PENALTY
        return objective(vector)

    result = differential_evolution(
        penalized,
        bounds=[(0.0, 1.0)] * dimension,
        maxiter=maxiter,
        popsize=popsize,
        seed=rng,
        polish=False,
        init="sobol",
        tol=0.0,
    )
    return np.asarray(result.x, dtype=float)
