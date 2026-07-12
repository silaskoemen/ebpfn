from pathlib import Path

from benchmarks.studies.pfn_feasibility import write_study_artifacts
from ebpfn.config.pfn import PfnArchConfig, PfnStudyConfig, PfnStudyModeConfig, PfnTrainConfig


def _config() -> PfnStudyConfig:
    return PfnStudyConfig(
        mode=PfnStudyModeConfig(
            name="fast", output_dir="unused", smoke_steps=6, profile_rows=[32], profile_features=[4], profile_tasks=2
        ),
        arch=PfnArchConfig(
            n_bins=64,
            embed_dim=16,
            col_num_blocks=1,
            row_num_blocks=1,
            icl_num_blocks=1,
            col_nhead=2,
            row_nhead=2,
            icl_nhead=2,
            n_cls_rows=8,
        ),
        train=PfnTrainConfig(
            seed=0,
            steps=6,
            tasks_per_step=2,
            warmup_steps=1,
            anchor_probe_fit=32,
            anchor_probe_score=16,
            anchor_features=4,
            device="cpu",
        ),
        decision_owner="test",
        decision_date="2026-01-01",
    )


def test_write_study_artifacts_emits_expected_files(tmp_path: Path) -> None:
    result = write_study_artifacts(_config(), project_root=tmp_path, output=tmp_path / "pfn")
    assert result["steps"] == 6
    destination = tmp_path / "pfn"
    expected = {
        "training.parquet",
        "feasibility.json",
        "loss_summary.json",
        "config.json",
        "summary.md",
        "environment.json",
        "run.log",
    }
    assert expected <= {path.name for path in destination.iterdir()}
    assert "PFN Feasibility Study Summary" in (destination / "summary.md").read_text()
