"""Exact content-addressed keys for simulator-only evaluations.

A key changes for every semantic input to an evaluation and does not change for
final-test-only edits (final-test contents are never part of a ``TuningTask``).
Environment, package, platform, and Git versions are provenance, not key inputs.
"""

from collections.abc import Sequence

from ebpfn.config import TuningConfig
from ebpfn.data import TuningTask, content_hash
from ebpfn.priors import HyperPrior

# Manual namespace/version for the whole cache identity scheme. Bump to
# invalidate every cached evaluation regardless of config contents.
CACHE_VERSION = "tuning-cache-3"


def evaluation_cache_key(
    config: TuningConfig,
    eta: HyperPrior,
    real_tasks: Sequence[TuningTask],
    base_seed: int,
    stage: str,
    fidelity: str,
    panel_identity: tuple[str | int, ...],
    energy_pair_ids: Sequence[tuple[int, int]] | None = None,
) -> str:
    """Return the exact evaluation cache key for one candidate at one stage.

    ``base_seed`` is the generation stream's seed (``RandomStreams.base_seed``),
    which drives cloud sampling; it is a distinct input from ``config.seed`` and
    must be part of identity or two runs with different streams collide.
    """
    schema_versions = (
        config.characterization.version,
        config.characterization.representation,
        config.compare.version,
    )
    key_config = config.model_copy(
        update={
            "search": config.search.model_copy(
                update={
                    "single_task_regularization": "none",
                    "trust_region_radius": None,
                    "competitive_tolerance": None,
                }
            )
        }
    )
    payload = (
        CACHE_VERSION,
        config.cache.cache_version,
        # Resolved run state minus the storage-only cache subconfig: the store
        # location/enabled flag and post-hoc single-task regularization are not
        # raw simulator evaluation identity.
        key_config.model_dump(mode="json", exclude={"cache"}),
        schema_versions,
        # Full tuning contents and split manifests (a TuningTask structurally
        # excludes any final-test partition), so changed data changes the key.
        tuple(real_tasks),
        base_seed,
        stage,
        fidelity,
        config.objective,
        eta,  # exact serialization; canonical hashing stores floats without rounding
        panel_identity,
        tuple((int(i), int(j)) for i, j in energy_pair_ids) if energy_pair_ids is not None else None,
    )
    return content_hash(*payload, namespace=CACHE_VERSION)
