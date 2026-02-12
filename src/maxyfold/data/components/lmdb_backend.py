import lmdb
import pickle

from .backend import DataBackend

class LMDBBackend(DataBackend):
    def __init__(self, path: str):
        self.path = path
        self.env = None
        self.keys = None
        # Pre-load keys
        with lmdb.open(path, subdir=False, readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.keys = list(txn.cursor().iternext(values=False))

    def _connect(self):
        if self.env is None:
            self.env = lmdb.open(self.path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return len(self.keys)

    def get_raw_data(self, idx):
        self._connect()
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            data_bytes = txn.get(key)
        return pickle.loads(data_bytes)