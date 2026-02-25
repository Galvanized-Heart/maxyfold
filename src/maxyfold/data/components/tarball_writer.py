import tarfile
from pathlib import Path

class TarballWriter:
    """
    A context manager class that writes files to a .tar.gz archive.
    Adheres to the Single Responsibility Principle by abstracting compression logic.
    """
    def __init__(self, tar_path: Path | str):
        self.tar_path = Path(tar_path)
        self.tar = None

    def __enter__(self):
        # Open in write-gzip mode
        self.tar = tarfile.open(self.tar_path, "w:gz")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tar:
            self.tar.close()

    def add_file(self, filepath: Path | str, arcname: str = None, delete_original: bool = False):
        """
        Adds a file to the archive.
        
        :param filepath: Path to the file to add.
        :param arcname: Name of the file inside the archive (defaults to file's actual name).
        :param delete_original: If True, deletes the original file after adding.
        """
        filepath = Path(filepath)
        if filepath.exists():
            # Use provided arcname, otherwise use the file's base name
            self.tar.add(filepath, arcname=arcname or filepath.name)
            
            if delete_original:
                filepath.unlink()