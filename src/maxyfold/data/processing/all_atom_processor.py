import numpy as np
import warnings
import io
import gzip
import json
from pathlib import Path
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from maxyfold.data.residue_constants import restype_to_idx, ATOM_MAPS, MAX_ATOM_COUNT

warnings.simplefilter('ignore', PDBConstructionWarning)

class AllAtomProcessor:
    def __init__(self, ccd_json_path: str = "data/pdb/processed/ccd_smiles.json"):
        # Load SMILES mapping
        self.smiles_map = {}
        path = Path(ccd_json_path)
        if path.exists():
            with open(path, 'r') as f:
                self.smiles_map = json.load(f)
        else:
            print("Warning: CCD JSON not found. Ligands will lack SMILES.")

    def process_cif(self, file_bytes: bytes, pdb_id: str):
        parser = MMCIFParser(QUIET=True)
        with gzip.open(io.BytesIO(file_bytes), 'rt') as f:
            structure = parser.get_structure(pdb_id, f)
        
        model = list(structure.get_models())[0]
        
        # Containers for the final flat list of tokens
        all_tokens_coords = []  # (N, 27, 3)
        all_tokens_mask = []    # (N, 27)
        all_tokens_types = []   # (N,) int
        all_tokens_chain = []   # (N,) int (Chain ID)
        
        ligand_smiles = {}      # {ChainID: SMILES_String}

        chain_counter = 0

        for chain in model:
            # 1. Determine Type (Protein/DNA/RNA vs Ligand)
            # Simple heuristic: If it has standard residues, it's a polymer
            residues = list(chain.get_residues())
            if not residues: continue

            first_res = residues[0].get_resname().strip()
            is_polymer = (first_res in ATOM_MAPS) # Checks if it's in our known map
            
            if is_polymer:
                # --- PROCESS POLYMER ---
                for res in residues:
                    resname = res.get_resname().strip()
                    
                    # 1. Get Token Type ID
                    # Map 3-letter to our internal ID. Need a lookup or 'UNK'
                    # (Simplified: assume 'UNK' for now if not found)
                    token_type_id = restype_to_idx.get(resname, restype_to_idx['UNK'])
                    
                    # 2. Get Atom Map
                    atom_map = ATOM_MAPS.get(resname, {})
                    
                    # 3. Fill Tensor
                    coords = np.zeros((MAX_ATOM_COUNT, 3), dtype=np.float32)
                    mask = np.zeros((MAX_ATOM_COUNT,), dtype=np.float32)
                    
                    for atom in res.get_atoms():
                        if atom.name in atom_map:
                            idx = atom_map[atom.name]
                            coords[idx] = atom.get_coord()
                            mask[idx] = 1.0
                    
                    all_tokens_coords.append(coords)
                    all_tokens_mask.append(mask)
                    all_tokens_types.append(token_type_id)
                    all_tokens_chain.append(chain_counter)
            
            else:
                # --- PROCESS LIGAND ---
                # A ligand "Chain" might contain multiple distinct molecules (e.g. 5 waters)
                # We usually treat each distinct HETATM group as a "Molecule"
                
                for res in residues:
                    resname = res.get_resname().strip()
                    if resname in ['HOH', 'DOD']: continue # Skip water for now?
                    
                    # Get SMILES
                    smiles = self.smiles_map.get(resname, "")
                    # Store unique SMILES for this chain/residue combo
                    # Note: In a real pipeline, handle multiple ligands properly
                    
                    # Iterate ATOMS (1 Token per Atom)
                    for atom in res.get_atoms():
                        if atom.element == 'H': continue # Skip Hydrogens
                        
                        coords = np.zeros((MAX_ATOM_COUNT, 3), dtype=np.float32)
                        mask = np.zeros((MAX_ATOM_COUNT,), dtype=np.float32)
                        
                        # Ligand atoms go to index 0
                        coords[0] = atom.get_coord()
                        mask[0] = 1.0
                        
                        all_tokens_coords.append(coords)
                        all_tokens_mask.append(mask)
                        all_tokens_types.append(restype_to_idx['LIGAND'])
                        all_tokens_chain.append(chain_counter)
                        
                        # Store atomic number as feature? 
                        # Ideally, 'all_tokens_types' should be expanded to include element info for ligands.
            
            chain_counter += 1

        if not all_tokens_coords:
            return None

        return {
            "pdb_id": pdb_id,
            "token_coords": np.array(all_tokens_coords, dtype=np.float32), # (N, 27, 3)
            "token_mask": np.array(all_tokens_mask, dtype=np.float32),     # (N, 27)
            "token_types": np.array(all_tokens_types, dtype=np.int32),     # (N,)
            "chain_ids": np.array(all_tokens_chain, dtype=np.int32),       # (N,)
            "ligand_smiles": ligand_smiles # Dict
        }