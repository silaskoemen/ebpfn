"""BNN route: independent/correlated Gaussian features through a random MLP.

Ported from the Gate-era ``BnnDgp``. Features come from the shared latent-factor
process (so correlation is a task-level knob), and the conditional mean is one
random fan-in-scaled network, giving a smooth, globally coupled ``f(x)``.
"""

import numpy as np

from ebpfn.priors.contracts import BnnHyperPrior
from ebpfn.priors.contracts import RouteRealization
from ebpfn.priors.contracts import SharedTheta
from ebpfn.priors.features import activation
from ebpfn.priors.features import sample_features


def realize(hp: BnnHyperPrior, n: int, p: int, shared: SharedTheta, rng: np.random.Generator) -> RouteRealization:
    if p < 1:
        raise ValueError(f"p must be at least one, got {p}")
    x_raw = sample_features(n, p, shared.corr_strength, rng)
    hidden = x_raw
    fan_in = p
    nonlinear_layers = 0
    for _ in range(hp.n_layers):
        weights = rng.standard_normal((fan_in, hp.hidden)) * (hp.weight_scale / np.sqrt(fan_in))
        act_name = "tanh" if rng.random() < hp.nonlinear_prob else "linear"
        nonlinear_layers += act_name == "tanh"
        hidden = activation(act_name)(hidden @ weights)
        fan_in = hp.hidden
    weights_out = rng.standard_normal(fan_in) * (hp.weight_scale / np.sqrt(fan_in))
    signal = hidden @ weights_out
    diagnostics = {
        "route": "bnn",
        "n_layers": int(hp.n_layers),
        "nonlinear_layers": int(nonlinear_layers),
        "fan_in_final": int(fan_in),
    }
    return RouteRealization(x_raw=x_raw, signal=signal, diagnostics=diagnostics)
