"""The unit-cube coordinate space over the nanoprior hyperprior."""

import numpy as np
import pytest
from ebpfn.priors.nanoprior import ALL_BLOCKS, EtaSpace, MechanismEta
from ebpfn.priors.nanoprior.coordinates import SPACE_VERSION


def test_dimension_and_names_follow_the_blocks() -> None:
    space = EtaSpace()
    assert space.dimension == 6
    assert space.coordinate_names[0] == "z.mechanism.stick.0"
    assert space.coordinate_names[-1] == "z.root.stick.2"


def test_round_trip_through_the_cube() -> None:
    space = EtaSpace()
    eta = MechanismEta(mechanism=(0.1, 0.2, 0.3, 0.4), root=(0.4, 0.3, 0.2, 0.1))
    restored = space.decode(space.encode(eta))
    assert restored.mechanism == pytest.approx(eta.mechanism)
    assert restored.root == pytest.approx(eta.root)


def test_origin_decodes_to_upstreams_mixture_and_is_not_the_centre() -> None:
    """Nothing may infer the null arm from 0.5; it has to be asked for."""
    space = EtaSpace()
    assert space.decode(space.origin_vector).mechanism == pytest.approx(MechanismEta().mechanism)
    assert not np.allclose(space.origin_vector, 0.5)


def test_every_corner_of_the_cube_decodes() -> None:
    """The all-multiplicative vertex has no only_cheap support, so the box stops short of it."""
    space = EtaSpace()
    for corner in ((0.0, 6), (1.0, 6)):
        value, size = corner
        eta = space.decode(np.full(size, value))
        assert sum(eta.mechanism) == pytest.approx(1.0)
    extreme = space.decode(np.ones(6))
    assert extreme.mechanism[-1] > 0.999, "the vertex must still be approachable"


def test_inactive_blocks_stay_at_the_base() -> None:
    pinned = MechanismEta(mechanism=(1.0, 0.0, 0.0, 0.0))
    space = EtaSpace(active=("z.root",), base=pinned)
    assert space.dimension == 3
    decoded = space.decode(np.full(3, 0.7))
    assert decoded.mechanism == pinned.mechanism
    assert decoded.root != pinned.root


def test_rejects_vectors_outside_the_contract() -> None:
    space = EtaSpace()
    with pytest.raises(ValueError, match="expected 6 coordinates"):
        space.decode(np.zeros(5))
    with pytest.raises(ValueError, match="unit cube"):
        space.decode(np.full(6, 1.5))
    with pytest.raises(ValueError, match="unit cube"):
        space.decode(np.full(6, np.nan))


def test_rejects_malformed_active_sets() -> None:
    with pytest.raises(ValueError, match="unknown blocks"):
        EtaSpace(active=("z.nonsense",))
    with pytest.raises(ValueError, match="unique"):
        EtaSpace(active=("z.root", "z.root"))
    with pytest.raises(ValueError, match="at least one"):
        EtaSpace(active=())


def test_schema_identifies_the_search_space() -> None:
    """Provenance for the run, kept separate from the identity of the decoded eta."""
    full = EtaSpace().schema()
    assert full["version"] == SPACE_VERSION
    assert full["active"] == list(ALL_BLOCKS)
    assert {entry["grouping"] for entry in full["coordinates"]} == {"mechanism-groups-4-v1"}
    assert full["hash"] == EtaSpace().schema()["hash"]
    assert full["hash"] != EtaSpace(active=("z.root",)).schema()["hash"]
