import lmdb
from safetensors.numpy import load
import numpy as np

from .backend import DataBackend

class LMDBBackend(DataBackend):
    def __init__(self, path: str):
        self.path = path
        self.env = None
        self.keys = None

        # Preload keys
        with lmdb.open(path, subdir=False, readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.keys = list(txn.cursor().iternext(values=False))

    def _connect(self):
        # Keep env open as long as the object is in use
        if self.env is None:
            self.env = lmdb.open(self.path, subdir=False, readonly=True, 
                                lock=False, readahead=False, meminit=False)
            self.txn = self.env.begin(write=False)

    def __len__(self):
        return len(self.keys)

    def get_raw_data(self, idx):
        self._connect()
        
        key = self.keys[idx]

        data_bytes = self.txn.get(key)
        if data_bytes is None:
            return None

        # Load safetensors directly into a dict
        data = load(data_bytes)
        
        return {
            "pdb_id": key.decode('ascii'),
            "coords": data["coords"],
            "mask": data["mask"].astype(bool),
            "res_type": data["res_type"],
            "atom_elements": data["atom_elements"],
            "chain_ids": data["chain_ids"]
        }