import io
import gzip
import tarfile

class TarballReader:
    """
    An iterable class that flattens nested .tar.gz archives.
    Yields (pdb_id, cif_string) one at a time.
    Stops cleanly when `file_limit` is reached.
    """
    def __init__(self, tar_paths: list, file_limit: int = 0):
        self.tar_paths = tar_paths
        self.file_limit = file_limit

    def __iter__(self):
        count = 0
        for tar_path in self.tar_paths:
            try:
                with tarfile.open(tar_path, "r:gz") as tar:
                    for member in tar:
                        # Skip directories and non-gz files
                        if not member.isfile() or not member.name.endswith(".gz"):
                            continue
                        
                        # Check our file limit before doing heavy extraction
                        if self.file_limit > 0 and count >= self.file_limit:
                            return # Exits the iterator completely
                            
                        pdb_id = member.name.split("-")[0]
                        
                        try:
                            # Extract and decompress in memory
                            f_obj = tar.extractfile(member)
                            if f_obj is None: continue
                            
                            gz_content = f_obj.read()
                            with gzip.open(io.BytesIO(gz_content), 'rt') as f_str:
                                cif_string = f_str.read()
                            
                            count += 1
                            yield pdb_id, cif_string
                            
                        except Exception as e:
                            print(f"Error extracting {pdb_id} from {tar_path.name}: {e}")
                            
            except Exception as e:
                print(f"CRITICAL: Failed to open {tar_path.name}: {e}")