"""Fixed-map regression characterization."""

from ebpfn.characterize.budgets import build_row_budget_manifests
from ebpfn.characterize.contracts import (
    BudgetCharacterizationError,
    CharacterizationDiagnostics,
    CharacterizationSchema,
    Coordinate,
    RowBudgetManifest,
    TaskCharacterization,
)
from ebpfn.characterize.evaluate import characterize, characterize_multiresolution
from ebpfn.characterize.maps import FeatureMap, build_feature_maps
from ebpfn.characterize.ridge import RidgeResult, fit_ridge_probe, solve_ridge_coefficients
from ebpfn.characterize.targets import TARGET_NAMES, TargetFunctionals, target_functionals

__all__ = [
    "TARGET_NAMES",
    "BudgetCharacterizationError",
    "CharacterizationDiagnostics",
    "CharacterizationSchema",
    "Coordinate",
    "FeatureMap",
    "RidgeResult",
    "RowBudgetManifest",
    "TargetFunctionals",
    "TaskCharacterization",
    "build_feature_maps",
    "build_row_budget_manifests",
    "characterize",
    "characterize_multiresolution",
    "fit_ridge_probe",
    "solve_ridge_coefficients",
    "target_functionals",
]
