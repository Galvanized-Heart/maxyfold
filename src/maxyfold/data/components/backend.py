from abc import ABC, abstractmethod
import lmdb
import h5py
import pickle

class DataBackend(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_raw_data(self, idx: int) -> dict:
        """Returns the dictionary (seq, coords, mask) for a given index."""
        pass