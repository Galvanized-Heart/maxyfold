from typing import Optional
import hydra
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from maxyfold.data import DataBackendBackend, BaseCropper, PDBDataset



class PDBDataModule(LightningDataModule):
    def __init__(
        self,
        bakcend: DataBackend,
        processed_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        cropper: BaseCropper = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["backend", "cropper"])
        self.backend = backend
        self.cropper = cropper

        # Instantiate datasets
        self.data_train: Optional[PDBDataset] = None
        self.data_val: Optional[PDBDataset] = None
        self.data_test: Optional[PDBDataset] = None

    def _read_keys(self, file_path: str) -> list[str]:
        """Helper to read and clean keys from a text file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Key file not found at {path}. Have you run 'maxyfold split'?")
        with open(path, 'r') as f:
            # Enforce uppercase to match how keys are stored in backend
            return [line.strip().upper() for line in f if line.strip()]

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if not self.data_train and not self.data_val and not self.data_test:
            
            print(f"Loading data splits for {self.backend.__class__.__name__}...")
            
            # Read PDB ID keys for each split
            train_keys = self._read_keys(self.hparams.train_set_path)
            val_keys = self._read_keys(self.hparams.val_set_path)
            test_keys = self._read_keys(self.hparams.test_set_path)

            # Create dedicated backend for each split
            train_backend = self.backend.__class__(path=self.backend.path, keys=train_keys)
            val_backend = self.backend.__class__(path=self.backend.path, keys=val_keys)
            test_backend = self.backend.__class__(path=self.backend.path, keys=test_keys)
            
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
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False,
        )