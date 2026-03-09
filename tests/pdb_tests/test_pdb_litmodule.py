import pytest
import torch
import torch.nn as nn
from maxyfold.models.litmodules.pdb_module import PDBLitModule
from maxyfold.data.constants import MAX_ATOM_COUNT

# A simple model that just returns a tensor of the correct output shape
class DummyModel(nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size
        # A learnable parameter to ensure gradients can flow
        self.output_tensor = nn.Parameter(torch.randn(1, self.crop_size, MAX_ATOM_COUNT, 3))

    def forward(self, batch):
        batch_size = batch["res_type"].shape[0]
        # Broadcast the parameter to the batch size
        return self.output_tensor.repeat(batch_size, 1, 1, 1)

@pytest.fixture
def synthetic_pdb_batch():
    """Creates a fake batch of PDB data for testing."""
    batch_size = 4
    crop_size = 64
    return {
        "pdb_id": ["1ABC", "1DEF", "1GHI", "1JKL"],
        "res_type": torch.randint(0, 30, (batch_size, crop_size), dtype=torch.long),
        "coords": torch.randn(batch_size, crop_size, MAX_ATOM_COUNT, 3, dtype=torch.float),
        "mask": torch.randint(0, 2, (batch_size, crop_size, MAX_ATOM_COUNT), dtype=torch.float),
        "atom_elements": torch.randint(0, 100, (batch_size, crop_size, MAX_ATOM_COUNT), dtype=torch.long),
        "chain_ids": torch.randint(0, 2, (batch_size, crop_size), dtype=torch.long)
    }

def test_pdb_litmodule_training_step(synthetic_pdb_batch):
    """Tests a single training step of the PDBLitModule."""
    # --- Arrange ---
    crop_size = synthetic_pdb_batch["res_type"].shape[1]
    dummy_model = DummyModel(crop_size)
    
    # Use a dummy optimizer and scheduler config for instantiation
    optimizer_cfg = {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3}
    
    lit_module = PDBLitModule(model=dummy_model, optimizer=optimizer_cfg, scheduler=None)

    # --- Act ---
    loss = lit_module.training_step(synthetic_pdb_batch, batch_idx=0)

    # --- Assert ---
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad, "Loss must have requires_grad=True to allow backpropagation"
    
    # Test that backpropagation does not raise an error
    try:
        loss.backward()
    except Exception as e:
        pytest.fail(f"loss.backward() failed with an exception: {e}")

    print("\nPDBLitModule training step and backward pass test successful!")