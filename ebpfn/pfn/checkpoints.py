"""Load released TabICLv2 weights into the vendored NanoTabICLv2 backbone.

The published TabICLv2 checkpoints and ``ebpfn/pfn/nanotabicl`` implement the *same*
architecture with different module names, so the weights transfer exactly once the
keys are renamed. Two upstream conventions have to be reconciled:

* **key naming** — the release nests blocks under ``col_embedder``/``row_interactor``/
  ``icl_predictor`` and names the feed-forward/norm submodules ``linear1``/``linear2``/
  ``norm1``/``norm2``; the vendored module uses flat ``col_blocks``/``row_blocks``/
  ``icl_blocks`` with ``mlp``/``ln_attn``/``ln_mlp``.
* **bias-free LayerNorm** — the regression checkpoint is trained with
  ``bias_free_ln=True`` and therefore ships no LayerNorm biases at all, while
  ``nn.LayerNorm`` always allocates them. They are filled with zeros, which is exactly
  the function upstream computes.

The released regressor head emits ``num_quantiles`` quantiles rather than bar logits;
pair it with :class:`ebpfn.pfn.quantile.QuantileDistribution`, not ``BarDistribution``.
"""

import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ebpfn.pfn.nanotabicl import NanoTabICLv2

_HF_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"
_DOWNLOAD_CHUNK = 1 << 20


@dataclass(frozen=True)
class CheckpointSpec:
    """A pinned released checkpoint and the architecture it was trained with."""

    repo: str
    filename: str
    sha256: str
    num_quantiles: int

    @property
    def url(self) -> str:
        return _HF_URL.format(repo=self.repo, filename=self.filename)


# The repo `tabicl`'s own loader downloads from; BSD-3-Clause, ungated.
TABICL_V2_REGRESSOR = CheckpointSpec(
    repo="jingang/TabICL",
    filename="tabicl-regressor-v2-20260212.ckpt",
    sha256="0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a",
    num_quantiles=999,
)


def cache_dir() -> Path:
    """Directory holding downloaded checkpoints (``EBPFN_CACHE_DIR`` overrides)."""
    override = os.environ.get("EBPFN_CACHE_DIR")
    return Path(override) / "checkpoints" if override else Path.home() / ".cache" / "ebpfn" / "checkpoints"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_checkpoint(spec: CheckpointSpec = TABICL_V2_REGRESSOR, *, allow_download: bool = True) -> Path:
    """Return the local path to ``spec``, downloading it once and verifying its digest.

    Raises if a cached or freshly downloaded file does not match ``spec.sha256`` — a
    checkpoint that is not bit-identical to the pinned release invalidates every
    number measured against it.
    """
    path = cache_dir() / spec.filename
    if path.exists():
        if (found := _sha256(path)) != spec.sha256:
            raise ValueError(f"cached {path} has sha256 {found}, expected {spec.sha256}")
        return path
    if not allow_download:
        raise FileNotFoundError(f"{path} is absent and allow_download is False")

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(spec.url) as response, partial.open("wb") as handle:
        while chunk := response.read(_DOWNLOAD_CHUNK):
            handle.write(chunk)
    if (found := _sha256(partial)) != spec.sha256:
        partial.unlink()
        raise ValueError(f"downloaded {spec.filename} has sha256 {found}, expected {spec.sha256}")
    partial.rename(path)
    return path


def _rename_within_block(key: str) -> str:
    """Rename a block-internal key from the release layout to the vendored layout."""
    key = re.sub(r"^attn\.", "", key)  # attn.{in_proj_*, out_proj.*, ssmax_layer.*}
    key = re.sub(r"^linear1\.", "mlp.0.", key)
    key = re.sub(r"^linear2\.", "mlp.2.", key)
    key = re.sub(r"^norm1\.", "ln_attn.", key)  # norm_first=True, so norm1 precedes attention
    return re.sub(r"^norm2\.", "ln_mlp.", key)


def convert_state_dict(released: dict[str, Tensor]) -> tuple[dict[str, Tensor], Tensor]:
    """Rename released TabICLv2 keys to vendored ones.

    Returns the renamed tensors and the checkpoint's RoPE ``inv_freq`` buffer, which
    the vendored module recomputes rather than stores; :func:`load_tabicl_v2` compares
    the two as an independent check that the positional-encoding convention matches.
    """
    out: dict[str, Tensor] = {}
    rope_freqs: Tensor | None = None
    for key, value in released.items():
        if key == "col_embedder.in_linear.weight":
            out["x_embed.weight"] = value
        elif key == "col_embedder.in_linear.bias":
            out["x_embed.bias"] = value
        elif key.startswith("col_embedder.y_encoder."):
            out[f"y_embed_in.{key.rsplit('.', 1)[1]}"] = value
        elif key.startswith("icl_predictor.y_encoder."):
            out[f"y_embed_icl.{key.rsplit('.', 1)[1]}"] = value
        elif key == "row_interactor.cls_tokens":
            out["row_cls_tokens"] = value.reshape(1, 1, value.shape[-2], value.shape[-1])
        elif key == "row_interactor.out_ln.weight":
            out["row_ln.weight"] = value
        elif key == "icl_predictor.ln.weight":
            out["out_ln.weight"] = value
        elif key == "row_interactor.tf_row.rope.freqs":
            rope_freqs = value
        elif key.startswith("icl_predictor.decoder."):
            out[f"out_mlp.{key.split('.', 2)[2]}"] = value
        elif match := re.fullmatch(r"col_embedder\.tf_col\.blocks\.(\d+)\.(.*)", key):
            index, rest = match.groups()
            if rest == "ind_vectors":
                out[f"col_blocks.{index}.inducing_vectors"] = value.unsqueeze(0)
            elif inner := re.fullmatch(r"multihead_attn([12])\.(.*)", rest):
                out[f"col_blocks.{index}.tfm{inner.group(1)}.{_rename_within_block(inner.group(2))}"] = value
            else:
                raise KeyError(f"unrecognized column-block key: {key}")
        elif match := re.fullmatch(r"row_interactor\.tf_row\.blocks\.(\d+)\.(.*)", key):
            out[f"row_blocks.{match.group(1)}.{_rename_within_block(match.group(2))}"] = value
        elif match := re.fullmatch(r"icl_predictor\.tf_icl\.blocks\.(\d+)\.(.*)", key):
            out[f"icl_blocks.{match.group(1)}.{_rename_within_block(match.group(2))}"] = value
        else:
            raise KeyError(f"unrecognized checkpoint key: {key}")
    if rope_freqs is None:
        raise KeyError("checkpoint has no RoPE frequency buffer to verify against")
    return out, rope_freqs


def load_tabicl_v2(
    spec: CheckpointSpec = TABICL_V2_REGRESSOR,
    *,
    path: Path | None = None,
    allow_download: bool = True,
) -> NanoTabICLv2:
    """Build the vendored backbone and load released TabICLv2 weights into it.

    The load is strict: every released tensor must map onto a parameter of matching
    shape, the only tolerated gaps are the bias-free LayerNorm biases, and the RoPE
    frequencies must agree with the ones the vendored module derives.
    """
    checkpoint = torch.load(
        path if path is not None else fetch_checkpoint(spec, allow_download=allow_download),
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["config"]
    if config["num_quantiles"] != spec.num_quantiles:
        raise ValueError(f"checkpoint emits {config['num_quantiles']} quantiles, spec pins {spec.num_quantiles}")

    model = NanoTabICLv2(
        max_classes=config["max_classes"],
        out_dim=config["num_quantiles"],
        embed_dim=config["embed_dim"],
        col_num_blocks=config["col_num_blocks"],
        row_num_blocks=config["row_num_blocks"],
        icl_num_blocks=config["icl_num_blocks"],
        col_nhead=config["col_nhead"],
        row_nhead=config["row_nhead"],
        icl_nhead=config["icl_nhead"],
        feature_group_size=config["col_feature_group_size"],
        n_cls_cols=config["row_num_cls"],
        n_cls_rows=config["col_num_inds"],
    )
    mapped, rope_freqs = convert_state_dict(checkpoint["state_dict"])

    reference = model.state_dict()
    absent = sorted(set(reference) - set(mapped))
    if unexpected := [key for key in absent if not re.search(r"(^|\.)(ln_attn|ln_mlp|row_ln|out_ln)\.bias$", key)]:
        raise KeyError(f"checkpoint is missing non-LayerNorm-bias parameters: {unexpected}")
    if not config["bias_free_ln"]:
        raise ValueError("checkpoint was trained with LayerNorm biases but ships none")
    for key in absent:
        mapped[key] = torch.zeros_like(reference[key])
    for key, value in mapped.items():
        if reference[key].shape != value.shape:
            raise ValueError(f"shape mismatch at {key}: model {tuple(reference[key].shape)} vs {tuple(value.shape)}")

    inv_freq = model.get_buffer("row_blocks.0.rope.inv_freq")  # non-persistent: derived, never stored
    if inv_freq.shape != rope_freqs.shape or not torch.allclose(inv_freq, rope_freqs, atol=1e-6):
        raise ValueError("RoPE frequency convention differs between checkpoint and vendored module")

    model.load_state_dict(mapped, strict=True)
    return model
