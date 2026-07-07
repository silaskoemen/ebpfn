import numpy as np
import pytest
from ebpfn.config import HyperPriorConfig
from ebpfn.priors import DEFAULT_ACTIVE
from ebpfn.priors import EtaVectorizer
from ebpfn.priors import build_hyperprior


def _vectorizer() -> EtaVectorizer:
    return EtaVectorizer(build_hyperprior(HyperPriorConfig()))


def test_encode_decode_round_trips_a_valid_prior():
    eta = build_hyperprior(HyperPriorConfig())
    vectorizer = _vectorizer()
    vector = vectorizer.encode(eta)
    assert vector.shape == (vectorizer.dimension,)
    assert np.all((vector >= -1e-9) & (vector <= 1.0 + 1e-9))
    decoded = vectorizer.decode(vector)
    assert np.allclose(vectorizer.encode(decoded), vector, atol=1e-9)
    assert abs(sum(decoded.generator_weights.values()) - 1.0) < 1e-9


def test_reference_weight_is_one_minus_the_nonreference_weights():
    eta = build_hyperprior(HyperPriorConfig(generator_weights=[0.5, 0.2, 0.1, 0.2]))
    vectorizer = _vectorizer()
    decoded = vectorizer.decode(vectorizer.encode(eta))
    assert decoded.generator_weights["compositional"] == pytest.approx(0.2)


def test_supersimplex_vector_is_infeasible_and_undecodable():
    vectorizer = _vectorizer()
    vector = vectorizer.encode(build_hyperprior(HyperPriorConfig()))
    for index, name in enumerate(vectorizer.active):
        if name in ("w_scm", "w_bnn", "w_tree"):
            vector[index] = 0.9
    assert not vectorizer.is_feasible(vector)
    with pytest.raises(ValueError, match="infeasible"):
        vectorizer.decode(vector)


def test_out_of_unit_cube_vector_is_infeasible():
    vectorizer = _vectorizer()
    vector = vectorizer.encode(build_hyperprior(HyperPriorConfig()))
    vector[-1] = 1.5
    assert not vectorizer.is_feasible(vector)


def test_sobol_returns_only_feasible_points():
    vectorizer = _vectorizer()
    design = vectorizer.sobol(48, np.random.default_rng(0))
    assert design.shape[1] == vectorizer.dimension
    assert len(design) > 0
    assert all(vectorizer.is_feasible(point) for point in design)


def test_schema_reports_route_order_reference_and_active_transforms():
    schema = _vectorizer().schema()
    assert schema["route_order"] == ["scm", "bnn", "tree", "compositional"]
    assert schema["reference_route"] == "compositional"
    assert tuple(schema["active"]) == DEFAULT_ACTIVE
    assert schema["coordinates"]["scm_target_indegree_mean"]["transform"] == "log"


def test_unknown_active_coordinate_is_rejected():
    with pytest.raises(ValueError, match="unknown active"):
        EtaVectorizer(build_hyperprior(HyperPriorConfig()), active=("w_scm", "not_a_coordinate"))
