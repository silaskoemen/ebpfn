import json

from benchmarks.studies.prior_audit import derive_study_status
from benchmarks.studies.prior_audit import write_study_artifacts
from ebpfn.config import HyperPriorConfig
from ebpfn.config import PriorStudyConfig
from ebpfn.config import PriorStudyModeConfig


def _config(name: str, output_dir: str) -> PriorStudyConfig:
    mode = PriorStudyModeConfig(
        name=name, output_dir=output_dir, feature_grid=(8, 32), n_probe_fit=120, n_probe_score=60, n_tasks=20
    )
    # Disable heavy-tail/heteroskedastic noise so the SNR-stability check is not
    # dominated by sampling noise at this small task count.
    prior = HyperPriorConfig(heteroskedastic_rate=0.0, heavy_tail_rate=0.0)
    return PriorStudyConfig(mode=mode, prior=prior, seed=0, decision_owner="tester", decision_date="2026-07-07")


def test_write_study_artifacts_emits_expected_files(tmp_path):
    config = _config("fast", "unused")
    summary = write_study_artifacts(config, tmp_path, output=tmp_path)
    for name in ("coordinates.parquet", "config.json", "evidence.json", "decision_log.json", "environment.json"):
        assert (tmp_path / name).is_file()
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert evidence["checks"]["reproducible"] is True
    assert evidence["checks"]["snr_calibrated"] is True
    assert summary["status"] == "complete"


def test_audit_status_stays_incomplete_pending_step4():
    passing = dict.fromkeys(
        (
            "reproducible",
            "route_frequency_converges",
            "metric_p_stable",
            "snr_p_stable",
            "snr_calibrated",
            "parameters_dispersed",
        ),
        True,
    )
    status, missing = derive_study_status("audit", passing)
    assert status == "incomplete"
    assert missing == ["joint_sobol_identifiability"]


def test_failed_checks_dominate_status():
    checks = {"reproducible": False, "route_frequency_converges": True}
    status, missing = derive_study_status("fast", checks)
    assert status == "failed"
    assert "reproducible" in missing
