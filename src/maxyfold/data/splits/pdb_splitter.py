# In new file: src/maxyfold/data/splits/splitter.py

import lmdb
import tempfile
import subprocess
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil
import numpy as np

from maxyfold.data.components.tarball_reader import TarballReader

try:
    import gemmi
except ImportError:
    # This should not happen if the environment is set up correctly
    raise ImportError("Gemmi is required for splitting. Please install it.")

class PDBDataSplitter:
    def __init__(self, lmdb_path, raw_assemblies_dir, output_dir, **kwargs):
        self.lmdb_path = Path(lmdb_path)
        self.raw_assemblies_dir = Path(raw_assemblies_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Using your configurable parameters ---
        self.config = {
            "seq_id": kwargs.get("seq_id", 0.3),
            "coverage": kwargs.get("coverage", 0.8),
            "cov_mode": kwargs.get("cov_mode", 0),
            "cluster_mode": kwargs.get("cluster_mode", 0),
            "split_ratios": kwargs.get("split_ratios", (0.9, 0.05, 0.05)),
            "seed": kwargs.get("seed", 42),
            "threads": kwargs.get("threads", 8)
        }
        
        # Mapping from 3-letter to 1-letter AA code
        self.res_map = gemmi.standard_res_map()

    def _extract_protein_sequences(self, keys_to_process: set) -> dict:
        # This part is the same as my original sketch
        print("Extracting protein sequences from raw files...")
        sequences = {}
        tar_files = sorted(list(self.raw_assemblies_dir.glob("assemblies_batch_*.tar.gz")))
        
        # We need a way to map PDB IDs back to their tarball for quick access
        # For now, TarballReader is fine, but could be optimized if slow.
        cif_stream = TarballReader(tar_paths=tar_files)
        
        for pdb_id, cif_string in tqdm(cif_stream, desc="Extracting Seqs"):
            if pdb_id.upper() not in keys_to_process:
                continue
            
            try:
                doc = gemmi.cif.read_string(cif_string)
                block = doc.sole_block()
                st = gemmi.make_structure_from_block(block)
                
                for chain in st[0]: # model 0
                    seq = gemmi.one_letter_sequence(chain.get_polymer(), self.res_map)
                    if seq and len(seq) > 20: # Filter out short peptides
                        chain_id = f"{pdb_id.upper()}_{chain.name}"
                        sequences[chain_id] = seq
            except Exception:
                continue
        return sequences

    def run_mmseqs2_clustering(self, sequences: dict, work_dir: Path) -> pd.DataFrame:
        """Runs MMseqs2 and returns a pandas DataFrame of the clusters."""
        fasta_path = work_dir / "sequences.fasta"
        
        print(f"Writing {len(sequences)} sequences to FASTA file...")
        with open(fasta_path, "w") as f:
            for chain_id, seq in sequences.items():
                f.write(f">{chain_id}\n{seq}\n")
        
        print("Running MMseqs2 clustering...")
        # --- Using your more robust, multi-step command invocation ---
        db_path = work_dir / "DB"
        cluster_path = work_dir / "clu"
        tmp_path = work_dir / "tmp"
        tsv_path = work_dir / "clusters.tsv"

        cmds = [
            ["mmseqs", "createdb", str(fasta_path), str(db_path), "-v", "0"],
            ["mmseqs", "cluster", str(db_path), str(cluster_path), str(tmp_path),
             "--min-seq-id", str(self.config['seq_id']),
             "-c", str(self.config['coverage']),
             "--cov-mode", str(self.config['cov_mode']),
             "--threads", str(self.config['threads']),
             "--cluster-mode", str(self.config['cluster_mode']),
             "-v", "0"],
            ["mmseqs", "createtsv", str(db_path), str(db_path), str(cluster_path), str(tsv_path), "-v", "0"]
        ]
        
        for cmd in cmds:
            try:
                # Use capture_output=True to hide MMseqs2's verbose output unless there's an error
                subprocess.run(cmd, check=True, text=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"MMseqs2 command failed: {' '.join(cmd)}")
                print("STDERR:", e.stderr)
                raise
        
        return pd.read_csv(tsv_path, sep='\t', header=None, names=['representative', 'member'])

    def assign_splits(self, cluster_df: pd.DataFrame):
        """Uses your more robust splitting logic."""
        # Get unique PDB IDs from member chain IDs
        cluster_df['pdb_id'] = cluster_df['member'].apply(lambda x: x.split('_')[0])
        
        # Group members by their representative (i.e., by cluster)
        clusters = cluster_df.groupby('representative')['pdb_id'].apply(set).tolist()
        
        print(f"Found {len(clusters)} clusters. Assigning to splits with seed {self.config['seed']}...")
        
        # --- Using your reproducible shuffling ---
        rng = random.Random(self.config['seed'])
        rng.shuffle(clusters)
        
        n_clusters = len(clusters)
        ratios = self.config['split_ratios']
        
        train_end = int(ratios[0] * n_clusters)
        val_end = train_end + int(ratios[1] * n_clusters)
        
        train_reps = clusters[:train_end]
        val_reps = clusters[train_end:val_end]
        test_reps = clusters[val_end:]

        train_set = set().union(*train_reps)
        val_set = set().union(*val_reps)
        test_set = set().union(*test_reps)

        # Final cleanup to ensure strict separation
        val_set -= train_set
        test_set -= (train_set | val_set)

        print(f"Final split sizes (PDB IDs): Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        
        for name, s in [("train", train_set), ("val", val_set), ("test", test_set)]:
            output_file = self.output_dir / f"{name}_keys.txt"
            with open(output_file, "w") as f:
                for pdb_id in sorted(list(s)):
                    f.write(f"{pdb_id}\n")
            print(f"Wrote {len(s)} keys to {output_file}")


    def create(self):
        """Main entrypoint to create the splits."""
        print("Starting data splitting process...")
        env = lmdb.open(str(self.lmdb_path), readonly=True, subdir=False, lock=False)
        with env.begin() as txn:
            processed_keys = {key.decode('ascii').upper() for key in txn.cursor().iternext(values=False)}
        print(f"Found {len(processed_keys)} successfully processed entries in LMDB.")
        
        sequences = self._extract_protein_sequences(processed_keys)
        if not sequences:
            print("No protein sequences could be extracted. Cannot create splits.")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            cluster_df = self.run_mmseqs2_clustering(sequences, Path(tmpdir))
            self.assign_splits(cluster_df)
        
        print(f"Splits created successfully in {self.output_dir}")