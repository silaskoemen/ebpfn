"""Top-level generator entrypoints.

``build_hyperprior`` assembles the runtime ``HyperPrior`` from validated config;
``sample_task``/``sample_cloud`` draw reproducible tasks through the named
GENERATION random stream so a given (eta, shape, identity) always reproduces the
same task and diagnostics.
"""

from typing import Any

from ebpfn.config.prior import HyperPriorConfig
from ebpfn.data import CharacterizationShape
from ebpfn.data import content_hash
from ebpfn.priors.contracts import ROUTE_ORDER
from ebpfn.priors.contracts import BnnHyperPrior
from ebpfn.priors.contracts import CompositionalHyperPrior
from ebpfn.priors.contracts import GeneratedTask
from ebpfn.priors.contracts import HyperPrior
from ebpfn.priors.contracts import ScmHyperPrior
from ebpfn.priors.contracts import TreeHyperPrior
from ebpfn.priors.mixture import sample_task as _mixture_sample_task
from ebpfn.utils import RandomRole
from ebpfn.utils import RandomStreams


def build_hyperprior(config: HyperPriorConfig) -> HyperPrior:
    weights: dict[str, float] = {
        str(name): float(weight) for name, weight in zip(ROUTE_ORDER, config.generator_weights, strict=True)
    }
    return HyperPrior(
        generator_weights=weights,
        corr_strength_mean=config.corr_strength_mean,
        log_snr_mean=config.log_snr_mean,
        heteroskedastic_rate=config.heteroskedastic_rate,
        heavy_tail_rate=config.heavy_tail_rate,
        snr_dispersion=config.snr_dispersion,
        corr_dispersion=config.corr_dispersion,
        scm=ScmHyperPrior(**config.scm.model_dump()),
        bnn=BnnHyperPrior(**config.bnn.model_dump()),
        tree=TreeHyperPrior(**config.tree.model_dump()),
        compositional=CompositionalHyperPrior(**config.compositional.model_dump()),
    )


def sample_task(
    eta: HyperPrior,
    shape: CharacterizationShape,
    streams: RandomStreams,
    *identity: str | int,
    common_random_numbers: bool = False,
) -> GeneratedTask:
    """Draw one task, optionally coupling the underlying uniforms across ``eta``."""
    eta_key = content_hash(eta, namespace="eta-1")
    # Default path keeps eta_key in its original leading position so generated
    # seeds are stable. Common random numbers drop eta_key alone to couple the
    # underlying uniforms across candidate hyperpriors.
    eta_tokens: tuple[str | int, ...] = () if common_random_numbers else (eta_key,)
    rng = streams.generator(
        RandomRole.GENERATION,
        *eta_tokens,
        shape.n_probe_fit,
        shape.n_probe_score,
        shape.p_numeric,
        *identity,
    )
    # Include the base seed so distinct streams never share a synthetic task id.
    tokens: tuple[Any, ...] = (streams.base_seed, eta_key, *identity)
    return _mixture_sample_task(eta, shape, rng, tokens)


def sample_cloud(
    eta: HyperPrior,
    shape: CharacterizationShape,
    n_members: int,
    streams: RandomStreams,
    *identity: str | int,
    common_random_numbers: bool = False,
) -> list[GeneratedTask]:
    """Draw fixed member slots, optionally coupled across candidate hyperpriors."""
    if n_members < 1:
        raise ValueError("n_members must be at least one")
    return [
        sample_task(
            eta,
            shape,
            streams,
            *identity,
            member,
            common_random_numbers=common_random_numbers,
        )
        for member in range(n_members)
    ]
