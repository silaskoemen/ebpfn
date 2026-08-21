"""Vendored TabICLv2 prior (see NOTICE) plus ebpfn's adapter to it."""

from ebpfn.priors.nanoprior.tasks import sample_task, seeded_globals

__all__ = ["sample_task", "seeded_globals"]
