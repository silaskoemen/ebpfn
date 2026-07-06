"""Probe-fit-only feature preprocessing shared by downstream consumers."""

from dataclasses import dataclass
import numpy as np
import polars as pl

from ebpfn.config import PreprocessingConfig
from ebpfn.data.hashing import content_hash
from ebpfn.data.types import FeatureSchema


@dataclass(frozen=True)
class FeatureTransform:
    input_schema: FeatureSchema
    output_schema: FeatureSchema
    impute: tuple[float, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    probe_fit_missing_rates: tuple[float, ...]
    excluded_categorical: tuple[str, ...]
    removed_constant: tuple[str, ...]
    clip: float
    version: str
    transform_id: str

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        missing = sorted(set(self.output_schema.names) - set(frame.columns))
        if missing:
            raise ValueError(f"frame is missing transformed features: {missing}")
        columns: list[pl.Series] = []
        for index, name in enumerate(self.output_schema.names):
            values = frame.get_column(name).cast(pl.Float64, strict=False).to_numpy().astype(float, copy=True)
            values[~np.isfinite(values)] = self.impute[index]
            values = np.clip((values - self.center[index]) / self.scale[index], -self.clip, self.clip)
            columns.append(pl.Series(name, values, dtype=pl.Float64))
        return pl.DataFrame(columns)


def _is_constant(values: np.ndarray, config: PreprocessingConfig) -> bool:
    spread = float(np.max(values) - np.min(values))
    magnitude = max(1.0, float(np.max(np.abs(values))))
    return spread <= config.constant_atol + config.constant_rtol * magnitude


def fit_feature_transform(frame: pl.DataFrame, schema: FeatureSchema, config: PreprocessingConfig) -> FeatureTransform:
    if tuple(frame.columns) != schema.names:
        raise ValueError("probe-fit frame must exactly match the input schema")
    names: list[str] = []
    impute: list[float] = []
    center: list[float] = []
    scale: list[float] = []
    missing_rates: list[float] = []
    categorical: list[str] = []
    constant: list[str] = []
    for name, kind in zip(schema.names, schema.kinds, strict=True):
        if kind == "categorical":
            categorical.append(name)
            continue
        values = frame.get_column(name).cast(pl.Float64, strict=False).to_numpy().astype(float, copy=True)
        missing_rate = float(np.mean(~np.isfinite(values)))
        finite = values[np.isfinite(values)]
        if not len(finite):
            constant.append(name)
            continue
        if kind == "binary":
            unique, counts = np.unique(finite, return_counts=True)
            if not set(unique) <= {0.0, 1.0}:
                raise ValueError(f"binary feature {name!r} must use 0/1 encoding")
            fill = float(unique[np.argmax(counts)])
            clean = np.where(np.isfinite(values), values, fill)
            prevalence = float(np.mean(clean))
            feature_scale = float(np.sqrt(prevalence * (1.0 - prevalence)))
            feature_center = prevalence
        else:
            fill = float(np.median(finite))
            clean = np.where(np.isfinite(values), values, fill)
            feature_center = float(np.median(clean))
            q25, q75 = np.quantile(clean, (0.25, 0.75))
            feature_scale = float((q75 - q25) / 1.349)
            if feature_scale <= config.scale_epsilon:
                feature_scale = float(np.std(clean))
        if _is_constant(clean, config) or feature_scale <= config.scale_epsilon:
            constant.append(name)
            continue
        names.append(name)
        impute.append(fill)
        center.append(feature_center)
        scale.append(feature_scale)
        missing_rates.append(missing_rate)
    if not names:
        raise ValueError("no usable numeric or binary predictors remain")
    if len(names) > config.max_features:
        raise ValueError(f"task has {len(names)} usable predictors; maximum is {config.max_features}")
    output_schema = schema.select(tuple(names))
    identity_inputs = (schema, output_schema, impute, center, scale, missing_rates, categorical, constant, config)
    return FeatureTransform(
        schema,
        output_schema,
        tuple(impute),
        tuple(center),
        tuple(scale),
        tuple(missing_rates),
        tuple(categorical),
        tuple(constant),
        config.clip,
        config.version,
        content_hash(identity_inputs, namespace="feature-transform-1"),
    )
