from abc import ABC, abstractmethod
from typing import List, ContextManager

class DataWriter(ABC):
    @abstractmethod
    def write(self, key: str, data: dict):
        pass



class DataBackend(ABC):
    @abstractmethod
    def get_keys(self, limit: int = 0) -> List[str]:
        """Returns a list of keys available in the backend."""
        pass
    
    @abstractmethod
    def get_writer(self) -> ContextManager[DataWriter]:
        """Returns a context manager that yields a writer object."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_raw_data(self, idx: int) -> dict:
        """Returns the dictionary for a given index."""
        pass