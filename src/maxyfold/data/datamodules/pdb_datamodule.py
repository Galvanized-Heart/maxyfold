from typing import Optional
import hydra
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from maxyfold.data.components.lmdb_backend import LMDBBackend
from maxyfold.data.datasets.pdb_dataset import PDBDataset
from maxyfold.data.cropping.croppers import BaseCropper

class PDBDataModule(LightningDataModule):
    def __init__(
        self,
        lmdb_path: str,
        processed_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        cropper: BaseCropper = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cropper = cropper

        # Instantiate datasets
        self.data_train: Optional[PDBDataset] = None
        self.data_val: Optional[PDBDataset] = None
        self.data_test: Optional[PDBDataset] = None

    def _load_keys(self, split: str) -> list[bytes]:
        """Loads the PDB ID keys from a .txt file for a given split."""
        keys_path = Path(self.hparams.processed_dir) / f"{split}_keys.txt"

        if not keys_path.exists():
            raise FileNotFoundError(f"Split file not found: {keys_path}. Run `create_splits.py` first.")
        
        with open(keys_path, 'r') as f:
            # Encode LMDB keys
            keys = [line.strip().encode('ascii') for line in f if line.strip()]
        return keys

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if not self.data_train and not self.data_val and not self.data_test:
            # Load keys for each split
            train_keys = self._load_keys("train")
            val_keys = self._load_keys("val")
            test_keys = self._load_keys("test")

            # Create dedicated backend for each split
            train_backend = LMDBBackend(path=self.hparams.lmdb_path, keys=train_keys)
            val_backend = LMDBBackend(path=self.hparams.lmdb_path, keys=val_keys)
            test_backend = LMDBBackend(path=self.hparams.lmdb_path, keys=test_keys)
            
            # Instantiate datasets
            self.data_train = PDBDataset(backend=train_backend, cropper=self.cropper)
            self.data_val = PDBDataset(backend=val_backend, cropper=self.cropper)
            self.data_test = PDBDataset(backend=test_backend, cropper=self.cropper)
            
            print(f"DataModule setup complete. Train: {len(self.data_train)}, Val: {len(self.data_val)}, Test: {len(self.data_test)}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )