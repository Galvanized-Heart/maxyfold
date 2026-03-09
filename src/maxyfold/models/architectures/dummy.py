import torch
from torch import nn
from maxyfold.data.constants import MAX_ATOM_COUNT

class DummyNet(nn.Module):
    """
    A minimal baseline model for testing the PDB training pipeline.
    It has a single learnable parameter, a tensor representing the 
    predicted structure, which it broadcasts to the batch size.
    This is useful for verifying that the loss function and training
    loop are working correctly.
    """
    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size
        # A learnable parameter representing the "mean" structure
        self.output_structure = nn.Parameter(
            torch.randn(1, self.crop_size, MAX_ATOM_COUNT, 3)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Ignores the input and returns its learnable output structure,
        repeated for the batch size.
        """
        batch_size = batch["res_type"].shape[0]
        return self.output_structure.repeat(batch_size, 1, 1, 1)