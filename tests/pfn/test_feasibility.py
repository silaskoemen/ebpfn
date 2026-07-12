import torch
from ebpfn.config.pfn import PfnArchConfig, PfnStudyModeConfig, PfnTrainConfig
from ebpfn.pfn.feasibility import profile


def test_profile_returns_well_formed_report() -> None:
    arch = PfnArchConfig(
        n_bins=32,
        embed_dim=16,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        n_cls_rows=8,
    )
    train = PfnTrainConfig(seed=0, anchor_probe_fit=32, anchor_probe_score=16, anchor_features=4, device="cpu")
    mode = PfnStudyModeConfig(
        name="fast", output_dir="unused", smoke_steps=1, profile_rows=[32], profile_features=[4, 8], profile_tasks=1
    )
    report = profile(arch, train, mode, reps=1)

    assert report["device"] == "cpu"
    assert report["n_parameters"] > 0
    assert len(report["cells"]) == 2
    for cell in report["cells"]:
        assert cell["train_ms"] >= 0.0
        assert cell["infer_ms"] >= 0.0
        assert {"rows", "features", "n_train", "n_test", "peak_memory_mb"} <= set(cell)
    assert report["in_regime"] is True
    assert set(report["realized_shapes"]) == {"n_rows", "n_features"}


def test_profile_executes_optimizer_steps(monkeypatch) -> None:
    calls = 0
    original_step = torch.optim.AdamW.step

    def counting_step(self, closure=None):
        nonlocal calls
        calls += 1
        return original_step(self, closure)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)
    arch = PfnArchConfig(
        n_bins=16,
        embed_dim=8,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        n_cls_rows=4,
    )
    train = PfnTrainConfig(seed=0, anchor_probe_fit=24, anchor_probe_score=8, anchor_features=4, device="cpu")
    mode = PfnStudyModeConfig(
        name="fast", output_dir="unused", smoke_steps=1, profile_rows=[16], profile_features=[2], profile_tasks=1
    )

    profile(arch, train, mode, reps=1)
    assert calls >= 2  # one allocation warmup and one measured update


def test_profile_rejects_anchor_beyond_max_context() -> None:
    arch = PfnArchConfig(
        n_bins=16,
        embed_dim=8,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        n_cls_rows=4,
        max_context=48,
    )
    train = PfnTrainConfig(
        anchor_probe_fit=48,
        anchor_probe_score=16,
        anchor_features=6,
        jitter={"sigma_n": 0.4, "sigma_p": 0.2, "n_min": 32, "n_max": 48, "p_min": 1, "p_max": 100},
        device="cpu",
    )
    mode = PfnStudyModeConfig(
        name="fast", output_dir="unused", smoke_steps=1, profile_rows=[16], profile_features=[2], profile_tasks=1
    )

    assert profile(arch, train, mode, reps=1)["in_regime"] is False


def test_profile_rejects_jitter_range_beyond_max_context() -> None:
    arch = PfnArchConfig(
        n_bins=16,
        embed_dim=8,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=1,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        n_cls_rows=4,
        max_context=128,
    )
    train = PfnTrainConfig(
        anchor_probe_fit=48,
        anchor_probe_score=16,
        anchor_features=6,
        jitter={"sigma_n": 0.4, "sigma_p": 0.2, "n_min": 32, "n_max": 256, "p_min": 1, "p_max": 100},
        device="cpu",
    )
    mode = PfnStudyModeConfig(
        name="fast", output_dir="unused", smoke_steps=1, profile_rows=[16], profile_features=[2], profile_tasks=1
    )

    assert profile(arch, train, mode, reps=1)["in_regime"] is False
