import pytest
import numpy as np
import torch
from maxyfold.data.cropping.croppers import (
    ContiguousCropper, 
    SpatialCropper, 
    InterfaceBiasedCropper,
    EntityStratifiedCropper
)
from maxyfold.data.constants import LIGAND_IDX, MAX_ATOM_COUNT

@pytest.fixture
def synthetic_complex():
    """
    Creates a fake complex with:
    - 100 Residue Protein (Chain 0)
    - 50 Atom Ligand (Chain 1)
    - Total L = 150
    """
    L = 150
    data = {
        "pdb_id": "test_150",
        "res_type": np.zeros((L,), dtype=np.int32),
        "coords": np.random.rand(L, MAX_ATOM_COUNT, 3).astype(np.float32) * 100, # Spread out coords
        "mask": np.zeros((L, MAX_ATOM_COUNT), dtype=np.float32),
        "atom_elements": np.zeros((L, MAX_ATOM_COUNT), dtype=np.int32),
        "chain_ids": np.zeros((L,), dtype=np.int32)
    }
    
    # Setup Protein (Indices 0-99)
    data["mask"][:100, :4] = 1.0 # Backbone
    data["chain_ids"][:100] = 0
    
    # Setup Ligand (Indices 100-149)
    data["res_type"][100:] = LIGAND_IDX
    data["chain_ids"][100:] = 1
    data["mask"][100:, 0] = 1.0 # Ligands only use index 0
    
    # Place ligand far away from start of protein to test spatial
    data["coords"][100:, 0] += 500.0 
    
    return data

@pytest.mark.parametrize("CropperClass", [
    ContiguousCropper,
    SpatialCropper,
    InterfaceBiasedCropper,
    EntityStratifiedCropper
])
def test_cropper_shapes(CropperClass, synthetic_complex):
    """Verify output shapes are always exactly crop_size."""
    crop_size = 64
    cropper = CropperClass(crop_size=crop_size)
    
    cropped = cropper(synthetic_complex)
    
    assert len(cropped["res_type"]) == crop_size
    assert cropped["coords"].shape == (crop_size, MAX_ATOM_COUNT, 3)
    assert cropped["mask"].shape == (crop_size, MAX_ATOM_COUNT)

def test_padding_logic(synthetic_complex):
    """Verify padding works when L < crop_size."""
    crop_size = 200 # Original is 150
    cropper = ContiguousCropper(crop_size=crop_size)
    
    padded = cropper(synthetic_complex)
    
    assert len(padded["res_type"]) == 200
    # Check that the last 50 entries are masked out (Padding)
    assert np.sum(padded["mask"][150:]) == 0.0

def test_spatial_locality(synthetic_complex):
    """
    Spatial cropper should pick tokens that are close.
    We put protein at ~0,0,0 and ligand at ~500,500,500.
    A crop of size 10 should essentially never contain BOTH.
    """
    cropper = SpatialCropper(crop_size=10)
    cropped = cropper(synthetic_complex)
    
    has_protein = np.any(cropped["chain_ids"] == 0)
    has_ligand = np.any(cropped["chain_ids"] == 1)
    
    # With highly separated chains, a small spatial crop shouldn't bridge the gap
    # (Unless random chance picked the one token in between, but our data is bipartite)
    assert not (has_protein and has_ligand), "Spatial crop failed to maintain locality"

def test_ligand_integrity(synthetic_complex):
    """
    Spatial cropper must include ALL atoms of a ligand if it picks one.
    """
    # Force crop size to be larger than ligand (50) to allow it to fit
    cropper = SpatialCropper(crop_size=60)
    
    # Mock data: Token 100 is center. Tokens 100-149 are the ligand.
    # We want to ensure if we pick token 100, we get 101..149 too.
    
    # We monkeypatch the random center to force it to pick the ligand
    # This relies on implementation detail of SpatialCropper calling rand int
    # A cleaner way is to inspect output.
    
    # Run multiple times to try and hit the ligand
    for _ in range(10):
        cropped = cropper(synthetic_complex)
        
        # Count ligand tokens in crop
        lig_tokens_in_crop = np.sum(cropped["res_type"] == LIGAND_IDX)
        
        if lig_tokens_in_crop > 0:
            # If we picked ANY ligand token, we should have picked ALL 50
            # (Since crop_size 60 > ligand size 50)
            assert lig_tokens_in_crop == 50
            break