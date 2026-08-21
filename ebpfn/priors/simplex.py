"""Bijective unit-cube coordinates for simplex-valued (mixture-weight) hyperparameters.

Stick-breaking with ``Beta(1, n-1-k)`` breaks: coordinate ``k`` consumes the fraction
``1 - u_k**(1/(n-1-k))`` of what is left and the last weight takes the remainder. The
exponent is what makes a uniform box draw uniform on the simplex rather than
front-loaded. Unlike the direct parameterization (``n-1`` free weights, reject when
they sum above one) every box point is valid, so the optimizer needs no feasibility
predicate — which matters because the direct feasible fraction is ``1/(n-1)!``.
"""

from collections.abc import Sequence

_TOL = 1e-9


def _validate_length(count: int) -> None:
    if count < 2:
        raise ValueError("a simplex group needs at least two members")


def sticks_to_weights(sticks: Sequence[float]) -> tuple[float, ...]:
    """Map ``n-1`` unit-interval coordinates onto a point on the (n-1)-simplex.

    Values marginally outside ``[0, 1]`` (floating-point slop from an optimizer) are
    clamped rather than rejected; anything further out is a caller bug and raises.
    """
    count = len(sticks) + 1
    _validate_length(count)

    weights: list[float] = []
    remaining = 1.0
    for index, raw in enumerate(sticks):
        if not -_TOL <= raw <= 1.0 + _TOL:
            raise ValueError(f"stick coordinate {index} is {raw}, outside [0, 1]")
        unit = min(max(float(raw), 0.0), 1.0)
        share = 1.0 - unit ** (1.0 / (count - 1 - index))
        weight = remaining * share
        weights.append(weight)
        remaining -= weight
    weights.append(max(remaining, 0.0))
    return tuple(weights)


def weights_to_sticks(weights: Sequence[float]) -> tuple[float, ...]:
    """Inverse of :func:`sticks_to_weights`.

    Once the stick is exhausted every later weight is zero and the coordinate producing
    it is ``1.0``; that branch avoids dividing by a spent remainder.
    """
    count = len(weights)
    _validate_length(count)
    if any(weight < -_TOL for weight in weights):
        raise ValueError("simplex weights must be nonnegative")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"simplex weights must sum to one, got {sum(weights)}")

    sticks: list[float] = []
    remaining = 1.0
    for index, raw in enumerate(weights[:-1]):
        weight = min(max(float(raw), 0.0), remaining)
        if remaining <= _TOL:
            sticks.append(1.0)
            continue
        share = weight / remaining
        sticks.append(min(max((1.0 - share) ** (count - 1 - index), 0.0), 1.0))
        remaining -= weight
    return tuple(sticks)
