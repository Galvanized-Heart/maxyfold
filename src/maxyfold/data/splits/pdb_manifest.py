import json
from pathlib import Path
from tqdm import tqdm

from maxyfold.data.components.tarball_reader import TarballReader
from maxyfold.data.constants import AA_3_TO_1

try:
    import gemmi
except ImportError:
    raise ImportError("Gemmi is required for this script.")

class PDBManifest:
    def __init__(self, raw_assemblies_dir: Path, ccd_smiles_path: Path, limit: int = 0):
        self.raw_assemblies_dir = raw_assemblies_dir
        with open(ccd_smiles_path, 'r') as f:
            self.smiles_map = json.load(f)
        
        self.protein_res = set(AA_3_TO_1.keys())
        self.nucleic_res = {'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U'}

        self.limit=limit

    def create(self) -> dict:
        manifest = {}
        tar_files = sorted(list(self.raw_assemblies_dir.glob("assemblies_batch_*.tar.gz")))
        cif_stream = TarballReader(tar_paths=tar_files, file_limit=self.limit)
        
        pbar = tqdm(cif_stream, desc="Creating Manifest")

        for pdb_id, cif_string in pbar:
            pdb_id_upper = pdb_id.upper()
            
            entry = {
                "protein_sequences": {},
                "nucleic_sequences": {},
                "ligand_smiles": {}
            }

            try:
                doc = gemmi.cif.read_string(cif_string)
                block = doc.sole_block()
                st = gemmi.make_structure_from_block(block)

                for chain in st[0]:
                    polymer = chain.get_polymer()
                    if not polymer:
                        for res in chain:
                            if res.name in self.smiles_map:
                                entry["ligand_smiles"][f"{pdb_id_upper}_{chain.name}_{res.seqid.num}"] = self.smiles_map[res.name]
                        continue

                    # Heuristic to classify polymer type
                    res_names = {res.name for res in polymer}
                    is_protein = len(res_names & self.protein_res) > len(res_names & self.nucleic_res)

                    seq_list = [AA_3_TO_1.get(res.name, 'X') for res in polymer]
                    seq = "".join(seq_list)
                    
                    if len(seq) > 10: # Lower threshold for manifest
                        chain_id = f"{pdb_id_upper}_{chain.name}"
                        if is_protein:
                            entry["protein_sequences"][chain_id] = seq
                        else:
                            entry["nucleic_sequences"][chain_id] = seq
                
                manifest[pdb_id_upper] = entry
                
            except Exception:
                continue
        
        return manifest