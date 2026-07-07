"""Order-independent named random streams."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class RandomRole(StrEnum):
    GENERATION = "generation"
    SEARCH = "search"
    SELECTION = "selection"
    FINAL_AUDIT = "final_audit"
    PFN_TRAINING = "pfn_training"
    REPORTING_BOOTSTRAP = "reporting_bootstrap"


@dataclass(frozen=True)
class RandomStreams:
    base_seed: int

    def __post_init__(self) -> None:
        if self.base_seed < 0:
            raise ValueError("base_seed must be nonnegative")

    def seed_sequence(self, role: RandomRole, *identity: str | int) -> np.random.SeedSequence:
        digest = hashlib.sha256()
        digest.update(str(self.base_seed).encode())
        digest.update(b"\0")
        digest.update(role.value.encode())
        for token in identity:
            digest.update(b"\0")
            digest.update(type(token).__name__.encode())
            digest.update(b":")
            digest.update(str(token).encode())
        entropy = np.frombuffer(digest.digest(), dtype=np.uint32).tolist()
        return np.random.SeedSequence(entropy)

    def generator(self, role: RandomRole, *identity: str | int) -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence(role, *identity))
