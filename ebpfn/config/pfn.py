"""Strict configuration for PFN architecture, training, and the feasibility study.

Unlike the tuning configs, these do carry model settings: they are the run state for
training a regression PFN on ebpfn's prior. The architecture config is deliberately
separable from training so the same architecture can be profiled or trained under
different schedules.
"""

from typing import Literal

from pydantic import field_validator, model_validator

from ebpfn.config.base import StrictConfigModel
from ebpfn.config.prior import HyperPriorConfig, ShapeJitterConfig


class PfnArchConfig(StrictConfigModel):
    """Backbone architecture plus the fixed standardized-target output grid.

    ``embed_dim`` must divide evenly by the column/row head counts, and the ICL block
    width ``embed_dim * n_cls_cols`` by ``icl_nhead`` (each attention head splits the
    embedding into equal head dimensions). Two bins are full-support half-normal tails
    beginning at ``+-target_inner_bound`` with scale ``target_tail_scale``; all other
    bins uniformly partition the shared interior. ``max_context`` is the declared
    maximum training context in rows; it is provenance for the feasibility profile,
    not a runtime cap.
    """

    n_bins: int = 256
    target_inner_bound: float = 5.0
    target_tail_scale: float = 1.0
    embed_dim: int = 128
    col_num_blocks: int = 3
    row_num_blocks: int = 3
    icl_num_blocks: int = 12
    col_nhead: int = 8
    row_nhead: int = 8
    icl_nhead: int = 8
    feature_group_size: int = 3
    n_cls_cols: int = 4
    n_cls_rows: int = 128
    max_context: int = 4096

    @model_validator(mode="after")
    def validate_values(self) -> "PfnArchConfig":
        if self.n_bins < 3:
            raise ValueError("n_bins must provide two tail bins and at least one interior bin")
        if self.target_inner_bound <= 0.0 or self.target_tail_scale <= 0.0:
            raise ValueError("target inner bound and tail scale must be positive")
        positive = (
            self.embed_dim,
            self.col_num_blocks,
            self.row_num_blocks,
            self.icl_num_blocks,
            self.col_nhead,
            self.row_nhead,
            self.icl_nhead,
            self.feature_group_size,
            self.n_cls_cols,
            self.n_cls_rows,
            self.max_context,
        )
        if any(value < 1 for value in positive):
            raise ValueError("architecture sizes and block/head counts must be positive")
        if self.embed_dim % self.col_nhead or self.embed_dim % self.row_nhead:
            raise ValueError("embed_dim must be divisible by col_nhead and row_nhead")
        if (self.embed_dim * self.n_cls_cols) % self.icl_nhead:
            raise ValueError("embed_dim * n_cls_cols must be divisible by icl_nhead")
        if self.max_context < 2:
            raise ValueError("max_context must be at least two")
        return self


class PfnTrainConfig(StrictConfigModel):
    """Optimizer schedule, shape policy, and the prior the PFN trains against.

    Each step samples ``tasks_per_step`` tasks at one jittered shape (the shape is
    homogeneous within a step so the backbone sees one ``(n_rows, n_cols)``). The
    anchor is the mean of the jittered shape distribution; ``jitter`` widens it.
    """

    seed: int = 0
    steps: int = 1000
    tasks_per_step: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    grad_clip: float = 1.0
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    checkpoint_interval: int = 200
    anchor_probe_fit: int = 512
    anchor_probe_score: int = 128
    anchor_features: int = 100
    jitter: ShapeJitterConfig = ShapeJitterConfig()
    prior: HyperPriorConfig = HyperPriorConfig()

    @model_validator(mode="after")
    def validate_values(self) -> "PfnTrainConfig":
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if min(self.steps, self.tasks_per_step, self.checkpoint_interval) < 1:
            raise ValueError("steps, tasks_per_step, and checkpoint_interval must be positive")
        if self.warmup_steps < 0 or self.warmup_steps > self.steps:
            raise ValueError("warmup_steps must be in [0, steps]")
        if self.lr <= 0.0 or self.weight_decay < 0.0 or self.grad_clip <= 0.0:
            raise ValueError("lr and grad_clip must be positive and weight_decay nonnegative")
        if min(self.anchor_probe_fit, self.anchor_probe_score) < 1 or self.anchor_features < 1:
            raise ValueError("anchor partition and feature counts must be positive")
        return self


class PfnStudyModeConfig(StrictConfigModel):
    """Scale, feasibility grid, and artifact location for the feasibility study."""

    name: Literal["fast", "audit"]
    output_dir: str
    smoke_steps: int
    profile_rows: tuple[int, ...]
    profile_features: tuple[int, ...]
    profile_tasks: int

    @field_validator("profile_rows", "profile_features", mode="before")
    @classmethod
    def freeze_grid(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "PfnStudyModeConfig":
        if not self.output_dir or self.smoke_steps < 1 or self.profile_tasks < 1:
            raise ValueError("study output, smoke_steps, and profile_tasks must be positive")
        if not self.profile_rows or not self.profile_features:
            raise ValueError("feasibility grid must be nonempty")
        if any(rows < 4 for rows in self.profile_rows) or any(feat < 1 for feat in self.profile_features):
            raise ValueError("feasibility rows must be at least four and features positive")
        return self


class PfnStudyConfig(StrictConfigModel):
    """Feasibility study: profile training/inference cost and run a smoke train."""

    mode: PfnStudyModeConfig
    arch: PfnArchConfig = PfnArchConfig()
    train: PfnTrainConfig = PfnTrainConfig()
    decision_owner: str
    decision_date: str

    @model_validator(mode="after")
    def validate_values(self) -> "PfnStudyConfig":
        if not self.decision_owner or not self.decision_date:
            raise ValueError("decision metadata must be nonempty")
        return self
