"""Released-checkpoint conversion onto the vendored NanoTabICLv2 backbone.

The key-mapping tests build a synthetic released-layout state dict so they run without
the 109 MB download; the tests that need the real weights are skipped unless it is
already cached (``EBPFN_CACHE_DIR`` / ``~/.cache/ebpfn/checkpoints``).
"""

import pytest
import torch
from ebpfn.pfn.checkpoints import (
    TABICL_V2_REGRESSOR,
    cache_dir,
    convert_state_dict,
    fetch_checkpoint,
    load_tabicl_v2,
)
from ebpfn.pfn.nanotabicl import NanoTabICLv2
from ebpfn.pfn.quantile import QuantileDistribution, default_levels

CONFIG = {
    "max_classes": 0,
    "num_quantiles": 999,
    "embed_dim": 128,
    "col_num_blocks": 3,
    "row_num_blocks": 3,
    "icl_num_blocks": 12,
    "col_nhead": 8,
    "row_nhead": 8,
    "icl_nhead": 8,
    "col_feature_group_size": 3,
    "row_num_cls": 4,
    "col_num_inds": 128,
    "bias_free_ln": True,
}


@pytest.fixture
def model() -> NanoTabICLv2:
    torch.manual_seed(0)
    return NanoTabICLv2(max_classes=0, out_dim=999)


def released_layout(model: NanoTabICLv2) -> dict[str, torch.Tensor]:
    """A released-style state dict: vendored keys renamed back, LayerNorm biases dropped."""
    forward = {
        "x_embed": "col_embedder.in_linear",
        "y_embed_in": "col_embedder.y_encoder",
        "y_embed_icl": "icl_predictor.y_encoder",
        "row_ln": "row_interactor.out_ln",
        "out_ln": "icl_predictor.ln",
        "out_mlp": "icl_predictor.decoder",
    }
    inner = {"mlp.0": "linear1", "mlp.2": "linear2", "ln_attn": "norm1", "ln_mlp": "norm2"}

    def rename_inner(rest: str) -> str:
        for ours, theirs in inner.items():
            if rest.startswith(f"{ours}."):
                return rest.replace(f"{ours}.", f"{theirs}.", 1)
        return rest if rest.startswith("ssmax_layer.") else f"attn.{rest}"

    out: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if key.endswith((".ln_attn.bias", ".ln_mlp.bias", "row_ln.bias", "out_ln.bias")):
            continue  # bias_free_ln
        if key == "row_cls_tokens":
            out["row_interactor.cls_tokens"] = value.reshape(1, value.shape[-2], value.shape[-1])
            continue
        head, _, rest = key.partition(".")
        if head in forward:
            out[f"{forward[head]}.{rest}"] = value
        elif head == "col_blocks":
            index, _, tail = rest.partition(".")
            if tail == "inducing_vectors":
                out[f"col_embedder.tf_col.blocks.{index}.ind_vectors"] = value.squeeze(0)
            else:
                which, _, sub = tail.partition(".")
                out[f"col_embedder.tf_col.blocks.{index}.multihead_attn{which[-1]}.{rename_inner(sub)}"] = value
        elif head == "row_blocks":
            index, _, tail = rest.partition(".")
            out[f"row_interactor.tf_row.blocks.{index}.{rename_inner(tail)}"] = value
        elif head == "icl_blocks":
            index, _, tail = rest.partition(".")
            out[f"icl_predictor.tf_icl.blocks.{index}.{rename_inner(tail)}"] = value
        else:
            raise AssertionError(f"unhandled vendored key in fixture: {key}")
    out["row_interactor.tf_row.rope.freqs"] = model.row_blocks[0].rope.inv_freq.clone()
    return out


def test_conversion_covers_every_parameter(model):
    mapped, rope = convert_state_dict(released_layout(model))
    reference = model.state_dict()
    absent = set(reference) - set(mapped)
    assert absent, "conversion produced no keys at all"
    assert all(key.endswith(".bias") for key in absent), "only LayerNorm biases may be absent"
    assert not set(mapped) - set(reference), "conversion produced keys the model does not have"
    for key, value in mapped.items():
        assert reference[key].shape == value.shape, key
    torch.testing.assert_close(rope, model.row_blocks[0].rope.inv_freq)


def test_conversion_rejects_unknown_keys(model):
    payload = released_layout(model)
    payload["icl_predictor.mystery.weight"] = torch.zeros(2)
    with pytest.raises(KeyError, match="unrecognized checkpoint key"):
        convert_state_dict(payload)


def test_conversion_requires_rope_buffer(model):
    payload = released_layout(model)
    del payload["row_interactor.tf_row.rope.freqs"]
    with pytest.raises(KeyError, match="RoPE"):
        convert_state_dict(payload)


def test_load_from_synthetic_checkpoint_roundtrips(tmp_path, model):
    path = tmp_path / "synthetic.ckpt"
    torch.save({"state_dict": released_layout(model), "config": CONFIG}, path)

    loaded = load_tabicl_v2(path=path)
    reference = model.state_dict()
    for key, value in loaded.state_dict().items():
        if key.endswith((".ln_attn.bias", ".ln_mlp.bias", "row_ln.bias", "out_ln.bias")):
            torch.testing.assert_close(value, torch.zeros_like(value))
        else:
            torch.testing.assert_close(value, reference[key])


def test_load_rejects_quantile_count_mismatch(tmp_path, model):
    path = tmp_path / "wrong.ckpt"
    torch.save({"state_dict": released_layout(model), "config": {**CONFIG, "num_quantiles": 512}}, path)
    with pytest.raises(ValueError, match="quantiles"):
        load_tabicl_v2(path=path)


def test_fetch_refuses_to_download_when_disallowed(tmp_path, monkeypatch):
    monkeypatch.setenv("EBPFN_CACHE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        fetch_checkpoint(TABICL_V2_REGRESSOR, allow_download=False)


@pytest.mark.skipif(
    not (cache_dir() / TABICL_V2_REGRESSOR.filename).exists(),
    reason="released checkpoint not cached locally",
)
def test_released_checkpoint_loads_and_predicts():
    """End-to-end: the real weights load strictly and produce ordered, sane quantiles."""
    model = load_tabicl_v2(allow_download=False).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 160, 6)
    y_train = x[0, :120, 0] * 0.8 + 0.2 * torch.randn(120)
    with torch.no_grad():
        quantiles = model(x, y_train.unsqueeze(0))
    assert quantiles.shape == (1, 40, 999)
    assert torch.isfinite(quantiles).all()

    dist = QuantileDistribution(default_levels(999))
    predicted = dist.mean(quantiles)
    assert predicted.shape == (1, 40)
    assert float(predicted.abs().max()) < 20.0
