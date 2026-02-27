import os
import math
from pathlib import Path
from typing import Dict, Any



class DataPipelineManager:
    """Orchestrates the downloading, compressing, and processing of datasets."""
    def __init__(self, paths_cfg, query_cfg=None):
        self.paths = paths_cfg
        self.query_cfg = query_cfg
        
        # Resolve config paths
        self.raw_dir = Path(self.paths.pdb_raw_dir)
        self.processed_dir = Path(self.paths.pdb_processed_dir)
        self.assemblies_dir = self.raw_dir / "assemblies"
        self.ccd_dir = self.raw_dir / "ccd"
        
        # Ensure directories exist
        self.assemblies_dir.mkdir(parents=True, exist_ok=True)
        self.ccd_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, ids=True, ccd=True, assemblies=True, batch_size=20000, limit=0):
        """Runs the requested download steps."""
        from maxyfold.data.download.pdb_downloader import PDBDownloader
        
        downloader = PDBDownloader(query_cfg=self.query_cfg)
        
        if ids:
            downloader.fetch_filtered_ids(output_file=self.raw_dir / "pdb_ids.txt")
            
        if ccd:
            downloader.download_ccd(output_dir=str(self.ccd_dir))
            
        if assemblies:
            self._download_and_batch_assemblies(downloader, batch_size, limit)

    def _download_and_batch_assemblies(self, downloader, batch_size, limit):
        """Handles the batching, async downloading, and tarballing."""
        import asyncio
        from maxyfold.data.components.tarball_writer import TarballWriter

        id_file_path = self.raw_dir / "pdb_ids.txt"
        if not id_file_path.exists():
            raise FileNotFoundError(f"PDB ID list not found at {id_file_path}. Run with '--ids' first.")

        with open(id_file_path, "r") as f:
            all_pdb_ids = [line.strip().upper() for line in f if line.strip()]

        if limit > 0:
            all_pdb_ids = all_pdb_ids[:limit]
            print(f"\nLimiting download to the first {limit} structures.")
        
        total_files = len(all_pdb_ids)
        num_batches = math.ceil(total_files / batch_size)
        
        print(f"\nProcessing {total_files} structures in {num_batches} batches of {batch_size}...")

        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        for i in range(num_batches):
            tar_path = self.assemblies_dir / f"assemblies_batch_{i}.tar.gz"
            if tar_path.exists():
                print(f"Batch {i+1}/{num_batches} already completed ({tar_path.name}). Skipping.")
                continue

            start = i * batch_size
            end = min(start + batch_size, total_files)
            batch_ids = all_pdb_ids[start:end]
            
            print(f"\nBatch {i+1}/{num_batches} (IDs {start}-{end})")

            # Download
            asyncio.run(downloader.download_assemblies(
                pdb_ids=batch_ids,
                output_dir=self.assemblies_dir,
                log_file_name=f"download_log_batch_{i}.txt"
            ))

            # Compress
            print(f"Archiving batch to {tar_path.name}...")
            removed_count = 0
            
            with TarballWriter(tar_path) as writer:
                for pdb_id in batch_ids:
                    filename = f"{pdb_id.lower()}-assembly1.cif.gz"
                    filepath = self.assemblies_dir / filename
                    
                    if filepath.exists():
                        writer.add_file(filepath, delete_original=True)
                        removed_count += 1
            
            print(f"Batch {i+1} complete. Archived and removed {removed_count} uncompressed files.")

    def process_to_lmdb(self, file_limit=0):
        """Converts raw tarballs to ML-ready LMDB."""
        from tqdm import tqdm
        from maxyfold.data.processing.pdb_processor import PDBProcessor
        from maxyfold.data.storage.lmdb_io import LMDBWriter
        from maxyfold.data.components.tarball_reader import TarballReader

        lmdb_path = Path(self.paths.lmdb_path)
        ccd_atoms_path = self.paths.ccd_atoms_map_path
        
        tar_files = sorted(list(self.assemblies_dir.glob("assemblies_batch_*.tar.gz")))
        if not tar_files:
            raise FileNotFoundError("No tarballs found in raw directory!")

        print("Initializing PDB Processor...")
        processor = PDBProcessor(ligand_map_path=str(ccd_atoms_path))
        cif_stream = TarballReader(tar_paths=tar_files, file_limit=file_limit)

        total_complexes = 0
        total_errors = 0
        pbar_total = file_limit if file_limit > 0 else None
        
        print(f"Writing ALL-ATOM dataset to {lmdb_path}...")
        with LMDBWriter(str(lmdb_path)) as writer:
            for pdb_id, cif_string in tqdm(cif_stream, total=pbar_total, desc="Processing PDBs"):
                result = processor.parse_cif_string(cif_string, pdb_id)
                
                if result:
                    writer.write(pdb_id, result)
                    total_complexes += 1
                    if total_complexes % 1000 == 0:
                        writer.commit()
                else:
                    total_errors += 1

        return total_complexes, total_errors

    def create_splits(self, mmseqs_config: Dict, splitting_config: Dict):
        """Orchestrates the data splitting process."""
        from maxyfold.data.splits.splitter import PDBDataSplitter

        splitter = PDBDataSplitter(
            lmdb_path=self.paths.lmdb_path,
            raw_assemblies_dir=self.assemblies_dir,
            output_dir=self.processed_dir,
            mmseqs_config=mmseqs_config,
            splitting_config=splitting_config
        )
        splitter.create()