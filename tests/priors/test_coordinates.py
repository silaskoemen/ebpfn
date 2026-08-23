"""The unit-cube coordinate space over the nanoprior hyperprior."""

import numpy as np
import pytest
from ebpfn.priors.nanoprior import ALL_GROUPS, EtaSpace, PriorEta, groups_in_block
from ebpfn.priors.nanoprior.coordinates import SPACE_VERSION
from ebpfn.priors.nanoprior.scalars import ORIGIN_COORD, REACHABLE_SITES

# The two mechanism simplices on their own: six coordinates, no lifted sites.
SIMPLEX = ("z.mechanism", "z.root")


def test_dimension_and_names_follow_the_groups() -> None:
    simplex = EtaSpace(active=SIMPLEX)
    assert simplex.dimension == 6
    assert simplex.coordinate_names[0] == "z.mechanism.stick.0"
    assert simplex.coordinate_names[-1] == "z.root.stick.2"

    full = EtaSpace()
    assert full.dimension == 6 + 2 * len(REACHABLE_SITES)
    assert "z.gp.decay.shift" in full.coordinate_names
    assert "z.gp.decay.spread" in full.coordinate_names


def test_scalar_sites_sit_at_the_origin_coordinate_and_round_trip() -> None:
    """Unlike the simplices, a lifted site's null *is* the middle of its own interval."""
    space = EtaSpace(active=("z.gp.decay",))
    assert list(space.origin_vector) == [ORIGIN_COORD, ORIGIN_COORD]
    restored = space.decode(np.array([0.25, 0.75]))
    assert restored.scalars["z.gp.decay"] == (0.25, 0.75)
    assert np.allclose(space.encode(restored), [0.25, 0.75])


def test_groups_can_be_taken_one_causal_block_at_a_time() -> None:
    z_groups = groups_in_block("z")
    assert "z.mechanism" in z_groups
    assert "z.gp.decay" in z_groups
    assert all(not name.startswith("g.") for name in z_groups)
    assert groups_in_block("p_y_z") == (), "no p(y|z) coordinates exist yet"
    # Categorical-gated sites are registered but stay out of the default search surface.
    assert "g.converter.neighbor_metric" in groups_in_block("g", reachable_only=False)
    assert "g.converter.neighbor_metric" not in groups_in_block("g")


def test_round_trip_through_the_cube() -> None:
    space = EtaSpace()
    eta = PriorEta(mechanism=(0.1, 0.2, 0.3, 0.4), root=(0.4, 0.3, 0.2, 0.1))
    restored = space.decode(space.encode(eta))
    assert restored.mechanism == pytest.approx(eta.mechanism)
    assert restored.root == pytest.approx(eta.root)


def test_origin_decodes_to_upstreams_mixture_and_is_not_the_centre() -> None:
    """Nothing may infer the null arm from 0.5; it has to be asked for."""
    space = EtaSpace()
    assert space.decode(space.origin_vector).mechanism == pytest.approx(PriorEta().mechanism)
    assert not np.allclose(space.origin_vector, 0.5)


def test_every_corner_of_the_cube_decodes() -> None:
    """The all-multiplicative vertex has no only_cheap support, so the box stops short of it."""
    space = EtaSpace(active=SIMPLEX)
    for corner in ((0.0, 6), (1.0, 6)):
        value, size = corner
        eta = space.decode(np.full(size, value))
        assert sum(eta.mechanism) == pytest.approx(1.0)
    extreme = space.decode(np.ones(6))
    assert extreme.mechanism[-1] > 0.999, "the vertex must still be approachable"


def test_inactive_blocks_stay_at_the_base() -> None:
    pinned = PriorEta(mechanism=(1.0, 0.0, 0.0, 0.0))
    space = EtaSpace(active=("z.root",), base=pinned)
    assert space.dimension == 3
    decoded = space.decode(np.full(3, 0.7))
    assert decoded.mechanism == pinned.mechanism
    assert decoded.root != pinned.root


def test_rejects_vectors_outside_the_contract() -> None:
    space = EtaSpace(active=SIMPLEX)
    with pytest.raises(ValueError, match="expected 6 coordinates"):
        space.decode(np.zeros(5))
    with pytest.raises(ValueError, match="unit cube"):
        space.decode(np.full(6, 1.5))
    with pytest.raises(ValueError, match="unit cube"):
        space.decode(np.full(6, np.nan))


def test_rejects_malformed_active_sets() -> None:
    with pytest.raises(ValueError, match="unknown groups"):
        EtaSpace(active=("z.nonsense",))
    with pytest.raises(ValueError, match="unique"):
        EtaSpace(active=("z.root", "z.root"))
    with pytest.raises(ValueError, match="at least one"):
        EtaSpace(active=())


def test_schema_identifies_the_search_space() -> None:
    """Provenance for the run, kept separate from the identity of the decoded eta."""
    full = EtaSpace().schema()
    assert full["version"] == SPACE_VERSION
    assert full["active"] == list(ALL_GROUPS)
    kinds = {entry["kind"] for entry in full["coordinates"]}
    assert kinds == {"simplex", "scalar"}
    groupings = {entry["grouping"] for entry in full["coordinates"] if entry["kind"] == "simplex"}
    assert groupings == {"mechanism-groups-4-v1"}
    assert full["hash"] == EtaSpace().schema()["hash"]
    assert full["hash"] != EtaSpace(active=("z.root",)).schema()["hash"]
