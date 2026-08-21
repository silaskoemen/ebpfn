"""``EBPFNModel``: the vendored TabICLv2 backbone plus a pluggable predictive head.

Both parts are injected rather than constructed here, because the two bases we train
from differ in both: a fresh backbone with a :class:`BarDistribution` over ``n_bins``,
or released TabICLv2 weights with a :class:`QuantileDistribution` over 999 levels. The
model is config-agnostic; config bridging lives in ``train.py``.
"""

import torch
from torch import Tensor, nn

from ebpfn.pfn.data import TaskBatch
from ebpfn.pfn.metrics import PredictiveDistribution
from ebpfn.pfn.nanotabicl import NanoTabICLv2


def backbone_out_dim(backbone: NanoTabICLv2) -> int:
    """Head width the backbone emits, read off its output MLP."""
    final = backbone.out_mlp[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("backbone output MLP does not end in a Linear layer")
    return final.out_features


class EBPFNModel(nn.Module):
    def __init__(self, backbone: NanoTabICLv2, distribution: PredictiveDistribution) -> None:
        super().__init__()
        if not isinstance(distribution, nn.Module):
            raise TypeError("distribution must be an nn.Module so its buffers follow the model across devices")
        width = backbone_out_dim(backbone)
        if width != distribution.n_outputs:
            raise ValueError(f"backbone emits {width} outputs but the head expects {distribution.n_outputs}")
        self.backbone = backbone
        self.distribution = distribution

    @property
    def n_outputs(self) -> int:
        return self.distribution.n_outputs

    def forward(self, x: Tensor, y_train: Tensor, return_embedding: bool = False) -> Tensor:
        """Head outputs for the test rows: ``(B, n_test, n_outputs)``.

        ``x`` is ``(B, n_rows, n_cols)`` with the ``n_train`` context rows first;
        ``y_train`` is ``(B, n_train)`` standardized targets for those rows.

        With ``return_embedding=True`` returns the in-context embedding instead —
        the normalized pre-head representation ``(B, n_test, icl_dim)`` used as the
        characterization ``z`` for prior tuning (see ``PLAN.md``).
        """
        return self.backbone(x, y_train, return_embedding=return_embedding)

    def loss(self, batch: TaskBatch) -> Tensor:
        """Mean of the head's native training loss on the query targets."""
        outputs = self(batch.x, batch.y_train_std)
        return self.distribution.training_loss(outputs, batch.y_test_std).mean()

    @torch.no_grad()
    def predict_logits(self, x: Tensor, y_train: Tensor) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            return self(x, y_train)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def embed(self, x: Tensor, y_train: Tensor) -> Tensor:
        """Frozen in-context embedding of the test rows: ``(B, n_test, icl_dim)``.

        The characterization ``z`` for prior tuning — a plain frozen forward pass,
        no head. See ``PLAN.md`` for how it is used.
        """
        was_training = self.training
        self.eval()
        try:
            return self(x, y_train, return_embedding=True)
        finally:
            self.train(was_training)
