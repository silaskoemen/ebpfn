import ast
from pathlib import Path

from ebpfn.config import CharacterizationStudyConfig
from hydra import compose
from hydra import initialize_config_dir
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


def test_executables_are_separated_from_benchmark_data_adapters():
    assert Path("benchmarks/data/openml.py").is_file()
    assert Path("scripts/data/prepare_data.py").is_file()
    assert Path("scripts/studies").is_dir()


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
