import torch
from torch.utils.data import Dataset

from maxyfold.data.components.backend import DataBackend

class PDBDataset(Dataset):
    def __init__(self, backend: DataBackend):
        self.backend = backend

    def __len__(self):
        return len(self.backend)

    def __getitem__(self, idx):
        record = self.backend.get_raw_data(idx)
        
        item = {
            "coords": torch.from_numpy(record['coords']),
            "mask": torch.from_numpy(record['mask']),
            "sequence": record['sequence']
        }

        return item