import pytest
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch

@pytest.mark.parametrize("cropper_name", ["contiguous", "spatial"])
def test_pdb_datamodule(cropper_name: str) -> None:
    """Tests PDBDataModule instantiation and batching."""
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        # We override the cropper to test both types
        cfg = hydra.compose(config_name="train_pdb", overrides=[f"cropper={cropper_name}"])

    # 1. Test Instantiation
    datamodule = hydra.utils.instantiate(cfg.data)
    assert datamodule is not None
    
    # 2. Test Setup (Requires data/pdb/processed files to exist)
    try:
        datamodule.setup()
    except FileNotFoundError:
        pytest.skip("PDB split files not found. Skipping data-dependent tests.")

    # 3. Test Dataloader
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))

    # 4. Verify Shapes
    batch_size = cfg.data.batch_size
    crop_size = cfg.cropper.crop_size
    
    assert batch["res_type"].shape == (batch_size, crop_size)
    assert batch["coords"].shape == (batch_size, crop_size, 27, 3)
    assert batch["mask"].shape == (batch_size, crop_size, 27)
    assert batch["chain_ids"].shape == (batch_size, crop_size)
    
    # 5. Verify Dtypes
    assert batch["coords"].dtype == torch.float32
    assert batch["res_type"].dtype == torch.int64