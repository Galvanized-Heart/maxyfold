import lmdb
import pickle
from pathlib import Path

class LMDBWriter:
    def __init__(self, output_path: str, map_size: int = 100 * 1024**3):
        self.output_path = str(output_path)
        self.map_size = map_size
        self.db = None
        self.txn = None

    def __enter__(self):
        self.db = lmdb.open(
            self.output_path,
            map_size=self.map_size,
            subdir=False,
            readonly=False,
            meminit=False,
            map_async=True,
        )
        self.txn = self.db.begin(write=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.txn: self.txn.commit()
        if self.db: self.db.close()

    def write(self, key: str, data: dict):
        self.txn.put(key.encode('ascii'), pickle.dumps(data, protocol=-1))

    def commit(self):
        self.txn.commit()
        self.txn = self.db.begin(write=True)