"""Shared validation policy for nested application configuration."""

from pydantic import BaseModel, ConfigDict


class StrictConfigModel(BaseModel):
    """Immutable strict configuration with recursive unknown-key rejection."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
