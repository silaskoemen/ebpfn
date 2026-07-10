"""Fixed-map regression characterization."""

from ebpfn.characterize.budgets import build_row_budget_manifests
from ebpfn.characterize.contracts import BudgetCharacterizationError
from ebpfn.characterize.contracts import CharacterizationDiagnostics
from ebpfn.characterize.contracts import CharacterizationSchema
from ebpfn.characterize.contracts import Coordinate
from ebpfn.characterize.contracts import RowBudgetManifest
from ebpfn.characterize.contracts import TaskCharacterization
from ebpfn.characterize.evaluate import characterize
from ebpfn.characterize.evaluate import characterize_multiresolution
from ebpfn.characterize.maps import FeatureMap
from ebpfn.characterize.maps import build_feature_maps
from ebpfn.characterize.ridge import RidgeResult
from ebpfn.characterize.ridge import fit_ridge_probe
from ebpfn.characterize.ridge import solve_ridge_coefficients
from ebpfn.characterize.targets import TARGET_NAMES
from ebpfn.characterize.targets import TargetFunctionals
from ebpfn.characterize.targets import target_functionals

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
