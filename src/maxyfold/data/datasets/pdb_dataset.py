import torch
from torch.utils.data import Dataset

from maxyfold.data.components.backend import DataBackend
from maxyfold.data.cropping.croppers import BaseCropper



class PDBDataset(Dataset):
    def __init__(self, backend: DataBackend, cropper: BaseCropper = None):
        """
        Args:
            backend: Handles reading raw data from backend.
            cropper: A callable that crops/pads the sequence to a fixed size.
        """
        self.backend = backend
        self.cropper = cropper

    def __len__(self):
        return len(self.backend)

    def __getitem__(self, idx):
        raw_data = self.backend.get_raw_data(idx)
        
        # Apply crop/pad
        if self.cropper is not None:
            processed_data = self.cropper(raw_data)
        else:
            processed_data = raw_data

        item = {
            "pdb_id": processed_data["pdb_id"],
            "res_type": torch.from_numpy(processed_data["res_type"]).long(),
            "coords": torch.from_numpy(processed_data["coords"]).float(),
            "mask": torch.from_numpy(processed_data["mask"]).float(),
            "atom_elements": torch.from_numpy(processed_data["atom_elements"]).long(),
            "chain_ids": torch.from_numpy(processed_data["chain_ids"]).long()
        }

        return item