"""Mechanism/root simplices over the vendored TabICLv2 family library."""

import collections

import numpy as np
import pytest
import torch
from ebpfn.priors.nanoprior import mechanisms, prior
from ebpfn.priors.nanoprior.mechanisms import GROUP_ORDER, GROUPS, N_FAMILIES, N_STICKS, ORIGIN, MechanismEta

CHEAP = [
    prior.rand_lin_func,
    prior.rand_quad_func,
    prior.rand_gp_func,
    prior.rand_tree_func,
    prior.rand_discretization_func,
]
ALL = [*CHEAP, prior.rand_mlp_func, prior.rand_em_func, prior.rand_prod_func]


def _frequencies(candidates: list, eta: MechanismEta, role: str, draws: int = 40_000) -> dict[str, float]:
    np.random.seed(0)
    counts = collections.Counter(mechanisms.select(candidates, eta, role).__name__ for _ in range(draws))
    return {name: count / draws for name, count in counts.items()}


def test_groups_partition_the_family_library() -> None:
    """Every family upstream can draw belongs to exactly one group, and only those."""
    covered = [name for group in GROUP_ORDER for name in GROUPS[group]]
    assert sorted(covered) == sorted({func.__name__ for func in ALL})
    assert len(covered) == len(set(covered)) == N_FAMILIES == 8


def test_origin_reproduces_the_uniform_draw() -> None:
    """The exact null: origin weights induce 1/8 per family, not merely something close."""
    assert sum(ORIGIN) == pytest.approx(1.0)
    frequencies = _frequencies(ALL, MechanismEta(), "mechanism")
    assert len(frequencies) == N_FAMILIES
    for name, share in frequencies.items():
        assert share == pytest.approx(1.0 / N_FAMILIES, abs=0.01), name


def test_origin_survives_the_only_cheap_restriction() -> None:
    """Masking per family and renormalizing gives 1/5; dropping whole groups would not."""
    frequencies = _frequencies(CHEAP, MechanismEta(), "mechanism")
    assert len(frequencies) == len(CHEAP)
    for name, share in frequencies.items():
        assert share == pytest.approx(1.0 / len(CHEAP), abs=0.01), name


def test_group_mass_is_split_evenly_across_members() -> None:
    piecewise = MechanismEta(mechanism=(0.0, 0.0, 1.0, 0.0))
    frequencies = _frequencies(ALL, piecewise, "mechanism")
    assert set(frequencies) == set(GROUPS["piecewise"])
    for share in frequencies.values():
        assert share == pytest.approx(1.0 / len(GROUPS["piecewise"]), abs=0.01)


def test_roles_are_independent() -> None:
    eta = MechanismEta(mechanism=(1.0, 0.0, 0.0, 0.0), root=(0.0, 1.0, 0.0, 0.0))
    assert set(_frequencies(ALL, eta, "mechanism")) == set(GROUPS["smooth"])
    assert set(_frequencies(ALL, eta, "root")) == set(GROUPS["polynomial"])


def test_all_multiplicative_mass_is_rejected_at_construction() -> None:
    """The one excluded vertex: only_cheap has no multiplicative member to draw."""
    with pytest.raises(ValueError, match="only_cheap"):
        MechanismEta(mechanism=(0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="only_cheap"):
        MechanismEta(root=(0.0, 0.0, 0.0, 1.0))


def test_rejects_malformed_weights() -> None:
    with pytest.raises(ValueError, match="one entry per group"):
        MechanismEta(mechanism=(0.5, 0.5))
    with pytest.raises(ValueError, match="sum to one"):
        MechanismEta(mechanism=(0.5, 0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="nonnegative"):
        MechanismEta(mechanism=(-0.5, 0.5, 0.5, 0.5))


def test_stick_coordinates_round_trip() -> None:
    eta = MechanismEta(mechanism=(0.1, 0.2, 0.3, 0.4), root=(0.4, 0.3, 0.2, 0.1))
    sticks = eta.to_sticks()
    assert len(sticks) == N_STICKS
    restored = MechanismEta.from_sticks(sticks)
    assert restored.mechanism == pytest.approx(eta.mechanism)
    assert restored.root == pytest.approx(eta.root)


def test_sticks_carry_the_mechanism_role_first() -> None:
    eta = MechanismEta.from_sticks([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert eta.mechanism == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert eta.root == pytest.approx((0.0, 1.0, 0.0, 0.0))


def test_serialization_round_trips() -> None:
    eta = MechanismEta(mechanism=(0.1, 0.2, 0.3, 0.4))
    assert MechanismEta.from_dict(eta.to_dict()).mechanism == pytest.approx(eta.mechanism)
    with pytest.raises(ValueError, match="unsupported"):
        MechanismEta.from_dict({"version": "nope", "mechanism": list(ORIGIN), "root": list(ORIGIN)})


def test_rand_func_without_eta_leaves_upstream_untouched() -> None:
    """No eta means upstream's own randchoice runs, so draws stay bit-identical."""
    x = torch.randn(64, 4)
    np.random.seed(3)
    torch.manual_seed(3)
    without = prior.rand_func(x, 3)
    np.random.seed(3)
    torch.manual_seed(3)
    candidates = [*CHEAP, prior.rand_mlp_func, prior.rand_em_func, prior.rand_prod_func]
    reference = prior.randchoice(candidates)(x, 3)
    assert torch.equal(without, reference)


def test_rand_points_draws_from_the_root_role() -> None:
    """Root marginals ignore the mechanism simplex; only the root one shapes them."""
    eta = MechanismEta(mechanism=(1.0, 0.0, 0.0, 0.0), root=(0.0, 1.0, 0.0, 0.0))
    seen = []
    original = prior.rand_func

    def spy(x, d_out, only_cheap=False, eta=None, role="mechanism"):
        seen.append(role)
        return original(x, d_out, only_cheap=only_cheap, eta=eta, role=role)

    prior.rand_func = spy
    try:
        np.random.seed(0)
        torch.manual_seed(0)
        prior.rand_points(32, 3, eta)
    finally:
        prior.rand_func = original
    assert seen[0] == "root"
