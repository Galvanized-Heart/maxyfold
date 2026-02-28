import lmdb
import numpy as np
from safetensors.numpy import load, save
from contextlib import contextmanager
from typing import List, ContextManager

from .base import DataBackend, DataWriter



class LMDBWriter(DataWriter):
    def __init__(self, db):
        self.db = db
        self.txn = db.begin(write=True)

    def write(self, key: str, data: dict):
        tensors = {
            "coords": data["coords"],
            "mask": data["mask"].astype(np.uint8),
            "res_type": data["res_type"],
            "atom_elements": data["atom_elements"],
            "chain_ids": data["chain_ids"]
        }
        metadata = {"pdb_id": data["pdb_id"]}
        data_bytes = save(tensors, metadata=metadata)
        self.txn.put(key.encode('ascii'), data_bytes)

    def commit(self):
        if self.txn:
            self.txn.commit()
            self.txn = self.db.begin(write=True)

    def close(self):
        if self.txn:
            self.txn.commit()
            self.txn = None

# Helper function to turn factory into context manager
@contextmanager
def _writer_context_manager(factory):
    writer = factory()
    try:
        yield writer
    finally:
        writer.close()



class LMDBBackend(DataBackend):
    def __init__(self, path: str, keys: List[str] = None, map_size: int = 100 * 1024**3):
        self.path = str(path)
        self.map_size = map_size
        self.env = None
        self._preloaded_keys = keys
        self.keys = keys

    def _connect(self):
        if self.env is None:
            self.env = lmdb.open(self.path, subdir=False, readonly=True, lock=False)
            self.txn = self.env.begin(write=False)
            # If keys were not preloaded, load them on first connection
            if self._preloaded_keys is None:
                self.keys = self.get_keys()

    def __len__(self) -> int:
        if self.keys is None: self._connect()
        return len(self.keys)

    def get_keys(self, limit: int = 0) -> List[str]:
        keys = []
        with lmdb.open(self.path, subdir=False, readonly=True, lock=False) as env:
            with env.begin() as txn:
                cursor = txn.cursor()
                for i, key in enumerate(cursor.iternext(values=False)):
                    keys.append(key.decode('ascii').upper())
                    if limit > 0 and (i + 1) >= limit:
                        break
        return keys

    def get_raw_data(self, idx: int) -> dict:
        if self.keys is None: 
            self._connect()

        key = self.keys[idx]
        
        if self.env is None: 
            self._connect()

        data_bytes = self.txn.get(key.encode('ascii'))
        if data_bytes is None: return None

        data = load(data_bytes)
        return {
            "pdb_id": key, "coords": data["coords"], "mask": data["mask"].astype(bool),
            "res_type": data["res_type"], "atom_elements": data["atom_elements"],
            "chain_ids": data["chain_ids"]
        }

    def get_writer(self) -> ContextManager[DataWriter]:
        @_writer_context_manager
        def writer_factory():
            db = lmdb.open(self.path, map_size=self.map_size, subdir=False, readonly=False, map_async=True)
            return LMDBWriter(db)
        return writer_factory