import pytest
import torch
from ebpfn.config.pfn import PfnArchConfig
from ebpfn.config.prior import HyperPriorConfig
from ebpfn.data.types import CharacterizationShape
from ebpfn.pfn import BarDistribution, EBPFNModel, fixed_borders
from ebpfn.pfn.data import PriorTaskSource
from ebpfn.pfn.quantile import QuantileDistribution, default_levels
from ebpfn.pfn.train import build_backbone, build_model
from ebpfn.priors import build_hyperprior
from ebpfn.utils import RandomStreams

SMALL = {
    "embed_dim": 32,
    "col_num_blocks": 1,
    "row_num_blocks": 1,
    "icl_num_blocks": 2,
    "col_nhead": 2,
    "row_nhead": 2,
    "icl_nhead": 2,
    "n_cls_rows": 16,
}


def _arch(**overrides) -> PfnArchConfig:
    return PfnArchConfig(**SMALL, **overrides)


def _model(n_bins: int = 64) -> tuple[EBPFNModel, PriorTaskSource]:
    torch.manual_seed(0)
    source = PriorTaskSource(build_hyperprior(HyperPriorConfig()), RandomStreams(0))
    return build_model(_arch(n_bins=n_bins)), source


def test_forward_output_shape() -> None:
    model, source = _model(64)
    batch = source.tensor_batch(3, CharacterizationShape(48, 24, 5, 0, "regression"), "test")
    logits = model(batch.x, batch.y_train_std)
    assert logits.shape == (3, 24, 64)


def test_forward_is_deterministic() -> None:
    model, source = _model(64)
    batch = source.tensor_batch(2, CharacterizationShape(48, 24, 5, 0, "regression"), "test")
    model.eval()
    first = model.predict_logits(batch.x, batch.y_train_std)
    second = model.predict_logits(batch.x, batch.y_train_std)
    assert torch.equal(first, second)


def test_loss_is_finite_and_differentiable() -> None:
    model, source = _model(64)
    batch = source.tensor_batch(3, CharacterizationShape(48, 24, 5, 0, "regression"), "test")
    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.x_embed.weight.grad is not None


def test_head_width_must_match_what_the_backbone_emits() -> None:
    """The mismatch this guards is real: a released 999-quantile backbone with a bar head."""
    model, _ = _model(64)
    assert model.n_outputs == 64
    assert isinstance(model.distribution, BarDistribution)

    with pytest.raises(ValueError, match="emits 64 outputs but the head expects 999"):
        EBPFNModel(build_backbone(_arch(n_bins=64)), QuantileDistribution(default_levels(999)))


def test_quantile_head_trains_on_pinball_not_nll() -> None:
    """Each head owns its native loss; NLL through a reconstructed density is not it."""
    torch.manual_seed(0)
    source = PriorTaskSource(build_hyperprior(HyperPriorConfig()), RandomStreams(0))
    model = build_model(_arch(head="quantile", n_quantiles=32))
    batch = source.tensor_batch(2, CharacterizationShape(48, 24, 5, 0, "regression"), "test")

    outputs = model(batch.x, batch.y_train_std)
    assert outputs.shape == (2, 24, 32)
    loss = model.loss(batch)
    assert torch.isfinite(loss)
    assert torch.equal(loss, model.distribution.pinball(outputs, batch.y_test_std).mean())
    loss.backward()
    assert model.backbone.x_embed.weight.grad is not None


def test_stored_borders_override_the_derived_grid() -> None:
    """Pre-v3 checkpoints carry their own grid; reloading must honour it exactly."""
    borders = fixed_borders(64, inner_bound=3.0, tail_scale=2.0)
    model = build_model(_arch(n_bins=64), borders)
    assert torch.equal(model.distribution.borders, borders)


def test_predict_logits_restores_training_mode() -> None:
    model, source = _model(64)
    batch = source.tensor_batch(1, CharacterizationShape(48, 24, 5, 0, "regression"), "test")
    model.train()
    model.predict_logits(batch.x, batch.y_train_std)
    assert model.training is True
