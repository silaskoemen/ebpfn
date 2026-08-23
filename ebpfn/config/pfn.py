"""Strict configuration for PFN architecture, training, and the feasibility study.

These carry model settings: they are the run state for training a regression PFN on
the nanoprior. The architecture config is deliberately separable from training so the
same architecture can be profiled or trained under different schedules.
"""

from typing import Literal

from pydantic import field_validator, model_validator

from ebpfn.config.base import StrictConfigModel


class NanopriorConfig(StrictConfigModel):
    """The prior the PFN trains against: a mechanism mixture plus lifted sampling sites.

    Anything unset stays at upstream's own distribution, which is what the released
    checkpoint was pretrained on, so a run left at the defaults is the null arm. The origin
    is not written out here: ``PriorEta`` owns it, and the config layer must not import the
    generator. This checks only shape -- that a mixture is a probability vector and that
    site coordinates lie in the unit cube. Whether a mixture is valid for the current
    grouping, and whether a site name exists, are ``PriorEta``'s to enforce.
    ``max_attempts`` bounds the learnability rejection loop, which corner etas can drive
    long.
    """

    mechanism: tuple[float, ...] | None = None
    root: tuple[float, ...] | None = None
    scalars: dict[str, tuple[float, float]] | None = None
    max_attempts: int | None = None

    @field_validator("mechanism", "root", mode="before")
    @classmethod
    def freeze_weights(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_values(self) -> "NanopriorConfig":
        for role, weights in (("mechanism", self.mechanism), ("root", self.root)):
            if weights is None:
                continue
            if any(weight < 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError(f"{role} weights must be nonnegative and sum to one")
        for site, coordinates in (self.scalars or {}).items():
            if any(not 0.0 <= value <= 1.0 for value in coordinates):
                raise ValueError(f"{site} coordinates must lie in the unit interval, got {coordinates}")
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts must be positive when set")
        return self


class ShapeJitterConfig(StrictConfigModel):
    sigma_n: float = 0.4
    sigma_p: float = 0.2
    n_min: int = 32
    n_max: int = 1024
    p_min: int = 1
    p_max: int = 100

    @model_validator(mode="after")
    def validate_values(self) -> "ShapeJitterConfig":
        if self.sigma_n <= 0.0 or self.sigma_p <= 0.0:
            raise ValueError("shape jitter widths must be positive")
        if self.sigma_p >= self.sigma_n:
            raise ValueError("sigma_p must be smaller than sigma_n")
        if not 2 <= self.n_min <= self.n_max:
            raise ValueError("row bounds must satisfy 2 <= n_min <= n_max")
        if not 1 <= self.p_min <= self.p_max <= 100:
            raise ValueError("feature bounds must satisfy 1 <= p_min <= p_max <= 100")
        return self


class PfnArchConfig(StrictConfigModel):
    """Backbone architecture plus the predictive head it emits into.

    ``embed_dim`` must divide evenly by the column/row head counts, and the ICL block
    width ``embed_dim * n_cls_cols`` by ``icl_nhead`` (each attention head splits the
    embedding into equal head dimensions). ``max_context`` is the declared maximum
    training context in rows; it is provenance for the feasibility profile, not a
    runtime cap.

    The head is the released TabICLv2 regression head: ``n_quantiles`` values on
    upstream's interior level grid. The defaults here match the released checkpoint, so a
    fine-tune from it needs no overrides. A fixed-border bar head was the alternative and
    is retired: the nanoprior's collapse-induced targets reach past any fixed interior
    (32% of batches exceed +-5 standardized), which a pinball loss absorbs and a binned
    density does not.
    """

    n_quantiles: int = 999
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
    max_context: int = 1024

    @property
    def n_outputs(self) -> int:
        """Width the backbone must emit for the head."""
        return self.n_quantiles

    @model_validator(mode="after")
    def validate_values(self) -> "PfnArchConfig":
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be at least two")
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

    Each optimizer step samples microbatches of ``tasks_per_step`` tasks at one
    jittered shape. ``gradient_accumulation_steps`` controls how many microbatches
    form the effective batch. The anchor is the mean of the jittered shape
    distribution; ``jitter`` widens it.

    ``finetune_scope`` restricts which stage receives gradient, splitting the backbone
    the way the released checkpoint does: ``"icl"`` trains the ICL predictor and head
    only (frozen column and row stages), ``"head"`` the output head alone. A mixture
    over mechanisms is a claim about *functions*, which the ICL predictor consumes, so
    the narrower scopes are the cheap end of the transfer-learning objective.
    """

    seed: int = 0
    steps: int = 1000
    tasks_per_step: int = 1
    gradient_accumulation_steps: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    grad_clip: float = 1.0
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    finetune_scope: Literal["all", "icl", "head"] = "all"
    checkpoint_interval: int = 200
    anchor_probe_fit: int = 512
    anchor_probe_score: int = 128
    anchor_features: int = 100
    jitter: ShapeJitterConfig = ShapeJitterConfig()
    prior: NanopriorConfig = NanopriorConfig()

    @model_validator(mode="after")
    def validate_values(self) -> "PfnTrainConfig":
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if min(self.steps, self.tasks_per_step, self.gradient_accumulation_steps, self.checkpoint_interval) < 1:
            raise ValueError(
                "steps, tasks_per_step, gradient_accumulation_steps, and checkpoint_interval must be positive"
            )
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
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
