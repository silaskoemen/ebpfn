import json
from pathlib import Path

from benchmarks.studies import apparent_snr_calibration
from ebpfn.config import TuningStudyConfig
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _config() -> TuningStudyConfig:
    config_dir = (Path(__file__).parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        raw = compose(config_name="tuning", overrides=["tuning_mode=fast"])
    resolved = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(resolved, dict)
    return TuningStudyConfig.model_validate(resolved)


def _write_roles(root: Path) -> None:
    path = root / "configs/source_roles.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "policy_version": "test-roles-1",
                "pilot_source_ids": ["pilot"],
                "confirmatory_source_ids": ["confirmatory"],
            }
        )
    )


def test_calibration_writer_freezes_eta_only_when_gate_passes(tmp_path, monkeypatch) -> None:
    config = _config()
    _write_roles(tmp_path)
    monkeypatch.setattr(
        apparent_snr_calibration,
        "apparent_snr_report",
        lambda *_, **__: {
            "source_split_id": "split-1",
            "mean_gap_real_minus_synthetic": 0.05,
            "source_balanced_energy_score": 0.1,
        },
    )

    decision = apparent_snr_calibration.write_calibration_artifacts(config, tmp_path, output=tmp_path / "out")

    assert decision["status"] == "frozen_baseline"
    assert (tmp_path / "out/eta_0.json").exists()
    assert json.loads((tmp_path / "out/source_split.json").read_text())["split_id"]


def test_calibration_writer_requires_structural_check_before_freezing_recalibrated_eta(tmp_path, monkeypatch) -> None:
    config = _config()
    _write_roles(tmp_path)

    def report(*_, **kwargs):
        eta = kwargs.get("eta")
        moved = eta is not None and eta.log_snr_mean == 1.2
        preferred = moved and eta.snr_dispersion == 0.5
        return {
            "source_split_id": "split-1",
            "mean_gap_real_minus_synthetic": 0.05 if moved else 0.25,
            "source_balanced_energy_score": 0.1 if preferred else 0.2,
        }

    monkeypatch.setattr(apparent_snr_calibration, "apparent_snr_report", report)
    monkeypatch.setattr(
        apparent_snr_calibration,
        "_structural_sanity",
        lambda *_, **__: {"passed": True, "n_pairs": 4, "checks": {}},
    )

    decision = apparent_snr_calibration.write_calibration_artifacts(config, tmp_path, output=tmp_path / "out")

    assert decision["status"] == "frozen_recalibrated"
    assert decision["selected_log_snr_mean"] == 1.2
    assert decision["selected_snr_dispersion"] == 0.5
    assert (tmp_path / "out/eta_0_candidate.json").exists()
    assert (tmp_path / "out/eta_0.json").exists()
