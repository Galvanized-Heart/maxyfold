import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    raise ImportError("RDKit is required for ligand splitting. Please run 'pip install rdkit'")

class PDBDataSplitter:
    def __init__(self, manifest_path: Path, output_dir: Path, mmseqs_config: Dict, splitting_config: Dict):
        self.output_dir = output_dir
        self.mmseqs_config = mmseqs_config
        self.splitting_config = splitting_config
        
        print(f"Loading manifest from {manifest_path}...")
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

    def _cluster_sequences(self, sequences: Dict[str, str], work_dir: Path) -> Dict[str, str]:
        if not sequences:
            return {}
        
        fasta_path = work_dir / "sequences.fasta"
        with open(fasta_path, "w") as f:
            for seq_id, seq in sequences.items():
                f.write(f">{seq_id}\n{seq}\n")
        
        db_path = work_dir / "DB"
        cluster_path = work_dir / "clu"
        tmp_path = work_dir / "tmp"
        tsv_path = work_dir / "clusters.tsv"

        cmds = [
            ["mmseqs", "createdb", str(fasta_path), str(db_path), "-v", "0"],
            ["mmseqs", "cluster", str(db_path), str(cluster_path), str(tmp_path),
             "--min-seq-id", str(self.mmseqs_config['seq_id']),
             "-c", str(self.mmseqs_config['coverage']),
             "--cov-mode", str(self.mmseqs_config['cov_mode']),
             "--threads", str(self.mmseqs_config.get('threads', 8)),
             "--cluster-mode", str(self.mmseqs_config['cluster_mode']),
             "-v", "0"],
            ["mmseqs", "createtsv", str(db_path), str(db_path), str(cluster_path), str(tsv_path), "-v", "0"]
        ]
        
        for cmd in cmds:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        df = pd.read_csv(tsv_path, sep='\t', header=None, names=['representative', 'member'])
        return pd.Series(df.representative.values, index=df.member).to_dict()

    def _cluster_ligands(self, ligands: Dict[str, str]) -> Dict[str, str]:
        if not ligands:
            return {}
        
        scaffolds = {}
        for ligand_id, smiles in tqdm(ligands.items(), desc="Generating Ligand Scaffolds"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None: continue
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                scaffolds[ligand_id] = scaffold_smiles if scaffold_smiles else smiles
            except Exception:
                continue
        return scaffolds

    def create(self):
        print("Starting advanced data splitting...")
        protein_seqs, nucleic_seqs, ligand_smiles = {}, {}, {}
        for entry in self.manifest.values():
            protein_seqs.update(entry.get("protein_sequences", {}))
            nucleic_seqs.update(entry.get("nucleic_sequences", {}))
            ligand_smiles.update(entry.get("ligand_smiles", {}))

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            print("\n--- Clustering Proteins ---")
            protein_map = self._cluster_sequences(protein_seqs, work_dir / "prot")
            print(f"Found {len(set(protein_map.values()))} protein clusters.")

            print("\n--- Clustering Nucleic Acids ---")
            nucleic_map = self._cluster_sequences(nucleic_seqs, work_dir / "nuc")
            print(f"Found {len(set(nucleic_map.values()))} nucleic acid clusters.")
        
        print("\n--- Clustering Ligands by Scaffold ---")
        ligand_map = self._cluster_ligands(ligand_smiles)
        print(f"Found {len(set(ligand_map.values()))} unique ligand scaffolds.")

        # Build a map from each PDB ID to all its associated cluster IDs
        pdb_to_clusters: Dict[str, Set[str]] = {pdb_id: set() for pdb_id in self.manifest}
        for pdb_id, entry in self.manifest.items():
            for chain_id in entry["protein_sequences"]:
                if chain_id in protein_map:
                    pdb_to_clusters[pdb_id].add(f"p_{protein_map[chain_id]}")
            for chain_id in entry["nucleic_sequences"]:
                if chain_id in nucleic_map:
                    pdb_to_clusters[pdb_id].add(f"n_{nucleic_map[chain_id]}")
            for ligand_id in entry["ligand_smiles"]:
                if ligand_id in ligand_map:
                    pdb_to_clusters[pdb_id].add(f"l_{ligand_map[ligand_id]}")
        
        # Get all unique cluster IDs from the entire dataset
        all_clusters = sorted(list(set.union(*pdb_to_clusters.values())))
        rng = random.Random(self.splitting_config['seed'])
        rng.shuffle(all_clusters)
        
        # Split the list of CLUSTER IDs
        ratios = self.splitting_config['ratios']
        n = len(all_clusters)
        train_end = int(ratios[0] * n)
        val_end = train_end + int(ratios[1] * n)
        
        train_clusters = set(all_clusters[:train_end])
        val_clusters = set(all_clusters[train_end:val_end])
        test_clusters = set(all_clusters[val_end:])
        
        # Assign PDBs to splits with strict hierarchy: Test > Val > Train
        train_pdbs, val_pdbs, test_pdbs = set(), set(), set()
        for pdb_id, clusters in pdb_to_clusters.items():
            if not clusters.isdisjoint(test_clusters):
                test_pdbs.add(pdb_id)
            elif not clusters.isdisjoint(val_clusters):
                val_pdbs.add(pdb_id)
            else:
                train_pdbs.add(pdb_id)
        
        print("\n--- Final Split Sizes ---")
        print(f"Train: {len(train_pdbs)}, Val: {len(val_pdbs)}, Test: {len(test_pdbs)}")
        
        for name, s in [("train", train_pdbs), ("val", val_pdbs), ("test", test_pdbs)]:
            path = self.output_dir / f"{name}_keys.txt"
            with open(path, "w") as f:
                for pdb_id in sorted(list(s)):
                    f.write(f"{pdb_id}\n")
            print(f"Wrote {len(s)} keys to {path}")