import torch
from ebpfn.config.prior import HyperPriorConfig
from ebpfn.data.types import CharacterizationShape
from ebpfn.pfn import EBPFNModel, fixed_borders
from ebpfn.pfn.data import PriorTaskSource
from ebpfn.priors import build_hyperprior
from ebpfn.utils import RandomStreams


def _model(n_bins: int = 64) -> tuple[EBPFNModel, PriorTaskSource]:
    torch.manual_seed(0)
    source = PriorTaskSource(build_hyperprior(HyperPriorConfig()), RandomStreams(0))
    borders = fixed_borders(n_bins)
    model = EBPFNModel(
        borders,
        embed_dim=32,
        col_num_blocks=1,
        row_num_blocks=1,
        icl_num_blocks=2,
        col_nhead=2,
        row_nhead=2,
        icl_nhead=2,
        n_cls_rows=16,
    )
    return model, source


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


def test_borders_bins_must_match_n_bins() -> None:
    model, _ = _model(64)
    assert model.n_bins == 64


def test_predict_logits_restores_training_mode() -> None:
    model, source = _model(64)
    batch = source.tensor_batch(1, CharacterizationShape(48, 24, 5, 0, "regression"), "test")
    model.train()
    model.predict_logits(batch.x, batch.y_train_std)
    assert model.training is True
