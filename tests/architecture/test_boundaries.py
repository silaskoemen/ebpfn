import ast
from pathlib import Path

from ebpfn.config import CharacterizationStudyConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _resolved_config(config_dir: Path) -> dict:
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="prepare_data", overrides=["mode=audit"])
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def test_application_dependencies_do_not_enter_reusable_modules():
    forbidden = ("hydra", "omegaconf", "openml")
    violations: list[str] = []
    for path in Path("ebpfn").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            if any(module.startswith(forbidden) for module in modules):
                violations.append(str(path))
    assert not violations, f"application dependency found in reusable modules: {sorted(set(violations))}"


def test_config_layer_does_not_import_the_domain():
    """``ebpfn.config`` must stay a leaf.

    ``ebpfn.data`` imports it, so any config module importing back into ``ebpfn.priors``
    or ``ebpfn.pfn`` closes a cycle that only shows up on some import orders. Defaults
    that need a domain constant are expressed as ``None`` and resolved where the domain
    object is built.
    """
    violations: list[str] = []
    for path in Path("ebpfn/config").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            if any(module.startswith(("ebpfn.priors", "ebpfn.pfn", "ebpfn.data")) for module in modules):
                violations.append(str(path))
    assert not violations, f"config module imports the domain: {sorted(set(violations))}"


def test_evaluation_never_imports_the_pfn():
    """The corpus/characterization path must stay free of the model it is used to judge."""
    violations: list[str] = []
    for path in Path("ebpfn/characterize").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(module.startswith("ebpfn.pfn") for module in modules):
                violations.append(str(path))
    assert not violations, f"characterization imports the PFN: {sorted(set(violations))}"


def test_operational_scripts_are_separated_from_scientific_benchmarks():
    # scripts/ holds operational entry points (prepare, tune, train); benchmarks/
    # holds paper-facing studies, diagnostics, comparisons, and ablations.
    assert Path("scripts/prepare_data.py").is_file()
    assert Path("benchmarks/data/openml.py").is_file()
    assert Path("benchmarks/studies").is_dir()
    assert Path("benchmarks/diagnostics").is_dir()
    assert not Path("scripts/studies").exists()
    assert not Path("scripts/diagnostics").exists()


def test_hydra_configuration_is_independent_of_working_directory(tmp_path, monkeypatch):
    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    expected = _resolved_config(config_dir)
    monkeypatch.chdir(tmp_path)
    actual = _resolved_config(config_dir)
    assert actual == expected
    assert actual["mode"]["name"] == "audit"


def test_characterization_study_configuration_is_strict_and_resolved():
    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        raw = compose(config_name="characterization", overrides=["characterization_mode=fast"])
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(resolved, dict)
    config = CharacterizationStudyConfig.model_validate(resolved)
    assert config.mode.name == "fast"
    assert config.characterization.maps.max_rff == 256
