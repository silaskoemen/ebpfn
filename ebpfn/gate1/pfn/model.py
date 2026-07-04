"""TabPFN-style 2D-attention transformer (plans/gate1_revised.md §3.2).

Vendored and lightly adapted from nanoTabPFN (https://github.com/automl/nanoTabPFN),
Apache-2.0, (c) the nanoTabPFN authors. Changes from upstream:
  - dropped the classification-only `NanoTabPFNClassifier` wrapper (we use a
    bar-distribution regression head + `PFNRegressor` instead);
  - `num_outputs` is the number of bar-distribution buckets.
The architecture is unchanged: per-cell feature/target embeddings, then a stack
of blocks alternating attention between features and between datapoints (with the
train/test causal split), then an MLP decoder over the target column.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import LayerNorm
from torch.nn.modules.transformer import Linear
from torch.nn.modules.transformer import MultiheadAttention


class PFNTransformer(nn.Module):
    def __init__(
        self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int
    ):
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList(
            TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size) for _ in range(num_layers)
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, train_test_split_index)  # (B,R,C,E)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)  # (B,R,1,E)
        src = torch.cat([x_src, y_src], 2)  # (B,R,C+1,E)
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)
        output = src[:, train_test_split_index:, -1, :]  # target column, test rows
        return self.decoder(output)  # (B, num_test, num_outputs)


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True) + 1e-20
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1).unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)
        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints (test attends to train; train attends to itself)
        src = src.transpose(1, 2).reshape(batch_size * col_size, rows_size, embedding_size)
        tr = src[:, :train_test_split_index]
        src_left = self.self_attention_between_datapoints(tr, tr, tr)[0]
        src_right = self.self_attention_between_datapoints(src[:, train_test_split_index:], tr, tr)[0]
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size).transpose(2, 1)
        src = self.norm2(src)
        # MLP
        src = self.linear2(F.gelu(self.linear1(src))) + src
        return self.norm3(src)


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))
