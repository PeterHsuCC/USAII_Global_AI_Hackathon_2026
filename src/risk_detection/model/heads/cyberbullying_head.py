import torch
from torch import nn

from ..encoder.aggregation import max_mean_top3

DEFAULT_NUM_CLASSES = 2
NON_BULLYING_INDEX = 1


class CyberbullyingHead(nn.Module):
    """Stage 1: p_cb_i^(1) = Softmax(W_cb^(1) h_i + b_cb^(1)).
    Stage 2: p_cb_i^(2) = Softmax(W_cb^(2) [h_i ; z_t] + b_cb^(2)).

    W_cb^(2) is initialized as [W_cb^(1) | 0] (zero-initialized context
    columns) so Stage 2 reproduces Stage 1's single-message predictions
    exactly before any context-aware training.
    """

    def __init__(
        self,
        d: int,
        d_z: int | None = None,
        num_classes: int = DEFAULT_NUM_CLASSES,
        non_bullying_index: int = NON_BULLYING_INDEX,
    ):
        super().__init__()
        self.d = d
        self.d_z = d if d_z is None else d_z
        self.num_classes = num_classes
        self.non_bullying_index = non_bullying_index

        self.stage1 = nn.Linear(d, num_classes)
        self.stage2 = nn.Linear(d + self.d_z, num_classes)
        self.sync_stage2_from_stage1()

    def sync_stage2_from_stage1(self) -> None:
        """W_cb^(2) <- [W_cb^(1) | 0]; b_cb^(2) <- b_cb^(1).

        Call once Stage 1 training has converged and before Stage 2 training
        begins, so Stage 2 starts from Stage 1's learned single-message
        behavior.
        """
        with torch.no_grad():
            self.stage2.weight[:, : self.d] = self.stage1.weight
            self.stage2.weight[:, self.d :] = 0.0
            self.stage2.bias.copy_(self.stage1.bias)

    def forward_stage1(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d) -> class probabilities (..., num_classes)."""
        return torch.softmax(self.stage1(h), dim=-1)

    def forward_stage2(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """h: (T, d) per-message encodings (or (B, T, d) batched); z: the
        matching conversation representation, (d_z,) or (B, d_z), shared
        across every message in its window. Returns (..., num_classes)."""
        if z.dim() == h.dim() - 1:
            z = z.unsqueeze(-2).expand(*h.shape[:-1], z.shape[-1])
        context = torch.cat([h, z], dim=-1)
        return torch.softmax(self.stage2(context), dim=-1)

    def risk(self, p: torch.Tensor) -> torch.Tensor:
        """p_cb,i^risk = 1 - p_cb,i^nonbullying."""
        return 1.0 - p[..., self.non_bullying_index]

    def window_score(self, risk: torch.Tensor) -> torch.Tensor:
        """S_cb(t) = 0.6 * max_i(p_cb,i^risk) + 0.4 * MeanTop3(p_cb,i^risk)."""
        return max_mean_top3(risk)
