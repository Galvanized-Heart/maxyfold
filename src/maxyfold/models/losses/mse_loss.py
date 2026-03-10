import torch
from torch import nn

class MSELoss(nn.Module):
    """
    A simple masked mean squared error loss.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred_coords: torch.Tensor, true_coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_coords: Predicted coordinates [B, L, 27, 3]
            true_coords: Ground truth coordinates [B, L, 27, 3]
            mask: Atom mask [B, L, 27]
        Returns:
            A scalar loss value.
        """
        diff = (true_coords - pred_coords) ** 2
        loss = (diff.sum(dim=-1) * mask).sum() / (mask.sum() + 1e-6)
        return loss * self.weight