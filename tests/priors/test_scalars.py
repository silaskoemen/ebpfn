"""The lifted scalar sites: exact at the origin, total across the cube, checked against upstream."""

import math

import numpy as np
import pytest
from ebpfn.data.types import CharacterizationShape
from ebpfn.priors.nanoprior import ORIGIN_ETA, REACHABLE_SITES, SITE_NAMES, PriorEta, sample_task
from ebpfn.priors.nanoprior.eta import ETA_VERSION
from ebpfn.priors.nanoprior.scalars import (
    BY_NAME,
    ORIGIN_COORD,
    SCALARS_VERSION,
    SITES,
    active_scalars,
    resolve,
)
from ebpfn.utils.random import RandomStreams

# Concrete intervals for the sites whose ends are computed at the call site.
DYNAMIC = {"z.discretization.n_centers": (2, 256), "z.em.n_inducing": (2, 17), "z.weights.decay": (0.05, 6.0)}


def _interval(name: str) -> tuple[float, float]:
    site = BY_NAME[name]
    low, high = DYNAMIC.get(name, (site.low, site.high))
    assert low is not None
    assert high is not None
    return low, high


def _shape() -> CharacterizationShape:
    return CharacterizationShape(192, 64, 8, 0, "regression")


def test_origin_coordinates_return_the_identical_constants() -> None:
    """Not merely equal: the same floats, so the same uniform draw gives the same value."""
    origin = dict.fromkeys(SITE_NAMES, (ORIGIN_COORD, ORIGIN_COORD))
    with active_scalars(origin):
        for name in SITE_NAMES:
            low, high = _interval(name)
            resolved = resolve(name, low, high)
            assert resolved == (low, high)
            assert all(a is b or a == b for a, b in zip(resolved, (low, high), strict=True))


def test_an_inactive_site_is_left_at_upstreams_interval() -> None:
    """A partially lifted eta is meaningful: unnamed sites keep upstream's constants."""
    with active_scalars({"z.gp.decay": (0.9, 0.1)}):
        assert resolve("z.gp.lengthscale", 0.5, 10.0) == (0.5, 10.0)
        assert resolve("z.gp.decay", 2.0, 20.0) != (2.0, 20.0)


def test_the_registry_is_checked_against_the_call_site() -> None:
    """Re-vendoring a release that changed a constant must raise, not retune silently."""
    with pytest.raises(ValueError, match="registered at low"):
        resolve("z.gp.decay", 3.0, 20.0)
    with pytest.raises(ValueError, match="registered at high"):
        resolve("z.gp.decay", 2.0, 25.0)


def test_a_site_with_a_computed_end_only_checks_the_fixed_one() -> None:
    assert resolve("z.em.n_inducing", 2, 33) == (2, 33)
    with pytest.raises(ValueError, match="registered at low"):
        resolve("z.em.n_inducing", 3, 33)


def test_unknown_sites_raise() -> None:
    with pytest.raises(ValueError, match="unknown sampling site"):
        resolve("z.nope", 1.0, 2.0)


@pytest.mark.parametrize("shift", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("spread", [0.0, 0.5, 1.0])
def test_the_map_is_total_over_the_cube(shift: float, spread: float) -> None:
    """No corner of the cube produces an inverted, zero, or non-finite interval."""
    with active_scalars(dict.fromkeys(SITE_NAMES, (shift, spread))):
        for name in SITE_NAMES:
            low, high = resolve(name, *_interval(name))
            assert 0.0 < low < high < math.inf


def test_active_scalars_restores_the_previous_table() -> None:
    with active_scalars({"z.gp.decay": (1.0, 0.5)}):
        with active_scalars({"z.gp.lengthscale": (0.0, 0.5)}):
            assert resolve("z.gp.decay", 2.0, 20.0) == (2.0, 20.0)
        assert resolve("z.gp.decay", 2.0, 20.0) != (2.0, 20.0)
    assert resolve("z.gp.decay", 2.0, 20.0) == (2.0, 20.0)


def test_every_registered_site_is_reached_by_the_generator() -> None:
    """A site nothing calls is a registry typo. Reachable ones must fire on the numeric path."""
    seen: set[str] = set()
    real_resolve = resolve

    import ebpfn.priors.nanoprior.prior as prior_module

    def spy(name: str, low: float, high: float) -> tuple[float, float]:
        seen.add(name)
        return real_resolve(name, low, high)

    prior_module.resolve = spy
    try:
        streams = RandomStreams(base_seed=0)
        for index in range(8):
            sample_task(_shape(), streams, "reach", index, eta=ORIGIN_ETA, max_attempts=50)
    finally:
        prior_module.resolve = real_resolve

    unreached = {site.name for site in REACHABLE_SITES} - seen
    assert not unreached, f"registered but never called: {sorted(unreached)}"
    gated = {site.name for site in SITES if not site.reachable}
    assert not (gated & seen), "a categorical-gated site fired on the numeric path"


def test_a_fully_lifted_origin_eta_is_bit_identical_to_an_unlifted_one() -> None:
    """The exactness the whole design rests on, measured end to end on real tasks."""
    lifted = PriorEta(scalars=dict.fromkeys(SITE_NAMES, (ORIGIN_COORD, ORIGIN_COORD)))
    streams = RandomStreams(base_seed=0)
    for index in range(3):
        plain = sample_task(_shape(), streams, "t", index, eta=ORIGIN_ETA, max_attempts=50).tuning
        same = sample_task(_shape(), streams, "t", index, eta=lifted, max_attempts=50).tuning
        np.testing.assert_array_equal(plain.probe_fit.X.to_numpy(), same.probe_fit.X.to_numpy())
        np.testing.assert_array_equal(plain.probe_fit.y, same.probe_fit.y)
        np.testing.assert_array_equal(plain.probe_score.y, same.probe_score.y)


@pytest.mark.parametrize("site", ["z.weights.decay", "z.graph.n_nodes", "g.column.kumaraswamy_shape"])
def test_moving_a_site_off_the_origin_moves_the_tasks(site: str) -> None:
    """A coordinate that changes nothing is dead weight in the search space."""
    streams = RandomStreams(base_seed=0)
    moved = 0
    for index in range(4):
        plain = sample_task(_shape(), streams, "t", index, eta=ORIGIN_ETA, max_attempts=50).tuning
        tuned = sample_task(
            _shape(), streams, "t", index, eta=PriorEta(scalars={site: (0.9, 0.5)}), max_attempts=50
        ).tuning
        if not np.array_equal(plain.probe_fit.y, tuned.probe_fit.y):
            moved += 1
    assert moved > 0


def test_eta_rejects_sites_the_generator_does_not_have() -> None:
    with pytest.raises(ValueError, match="unknown sampling sites"):
        PriorEta(scalars={"z.gp.nope": (0.5, 0.5)})


def test_eta_rejects_coordinates_outside_the_cube() -> None:
    with pytest.raises(ValueError, match="unit interval"):
        PriorEta(scalars={"z.gp.decay": (1.5, 0.5)})
    with pytest.raises(ValueError, match="shift, spread"):
        PriorEta(scalars={"z.gp.decay": (0.5,)})


def test_an_unset_site_reads_back_at_the_origin() -> None:
    assert ORIGIN_ETA.coordinates_for("z.gp.decay") == (ORIGIN_COORD, ORIGIN_COORD)
    assert PriorEta(scalars={"z.gp.decay": (0.2, 0.8)}).coordinates_for("z.gp.decay") == (0.2, 0.8)


def test_scalars_survive_a_payload_round_trip() -> None:
    eta = PriorEta(mechanism=(0.1, 0.2, 0.3, 0.4), scalars={"z.gp.decay": (0.2, 0.8)})
    payload = eta.to_dict()
    assert PriorEta.from_dict(payload).scalars["z.gp.decay"] == (0.2, 0.8)
    with pytest.raises(ValueError, match="unsupported eta version"):
        PriorEta.from_dict({**payload, "version": "nope"})
    with pytest.raises(ValueError, match="named under"):
        PriorEta.from_dict({**payload, "sites": "nanoprior-scalar-sites-0"})


def test_the_payload_names_the_versions_it_was_written_under() -> None:
    payload = ORIGIN_ETA.to_dict()
    assert payload["version"] == ETA_VERSION
    assert payload["sites"] == SCALARS_VERSION
    assert payload["scalars"] == {}
