"""``EBPFNModel``: the vendored TabICLv2 backbone plus a bar-distribution head.

The backbone (``max_classes=0``) emits ``n_bins`` logits per test row; the
:class:`BarDistribution` interprets them as a full-support predictive density. The
model is config-agnostic (borders + explicit architecture kwargs) so it stays pure
torch; config bridging lives in ``train.py``.
"""

import torch
from torch import Tensor, nn

from ebpfn.pfn.data import TaskBatch
from ebpfn.pfn.distribution import BarDistribution
from ebpfn.pfn.nanotabicl import NanoTabICLv2


class EBPFNModel(nn.Module):
    def __init__(
        self,
        borders: Tensor,
        *,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        row_num_blocks: int = 3,
        icl_num_blocks: int = 12,
        col_nhead: int = 8,
        row_nhead: int = 8,
        icl_nhead: int = 8,
        feature_group_size: int = 3,
        n_cls_cols: int = 4,
        n_cls_rows: int = 128,
    ) -> None:
        super().__init__()
        self.distribution = BarDistribution(borders)
        self.backbone = NanoTabICLv2(
            max_classes=0,
            out_dim=self.distribution.n_bins,
            embed_dim=embed_dim,
            col_num_blocks=col_num_blocks,
            row_num_blocks=row_num_blocks,
            icl_num_blocks=icl_num_blocks,
            col_nhead=col_nhead,
            row_nhead=row_nhead,
            icl_nhead=icl_nhead,
            feature_group_size=feature_group_size,
            n_cls_cols=n_cls_cols,
            n_cls_rows=n_cls_rows,
        )

    @property
    def n_bins(self) -> int:
        return self.distribution.n_bins

    def forward(self, x: Tensor, y_train: Tensor) -> Tensor:
        """Bin logits for the test rows: ``(B, n_test, n_bins)``.

        ``x`` is ``(B, n_rows, n_cols)`` with the ``n_train`` context rows first;
        ``y_train`` is ``(B, n_train)`` standardized targets for those rows.
        """
        return self.backbone(x, y_train)

    def loss(self, batch: TaskBatch) -> Tensor:
        """Mean bar-distribution NLL of the query targets — the training objective."""
        logits = self(batch.x, batch.y_train_std)
        return self.distribution.nll(logits, batch.y_test_std).mean()

    @torch.no_grad()
    def predict_logits(self, x: Tensor, y_train: Tensor) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            return self(x, y_train)
        finally:
            self.train(was_training)
