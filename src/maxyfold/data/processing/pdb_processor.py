import numpy as np
import json
from pathlib import Path
try:
    import gemmi
except ImportError:
    pass

from maxyfold.data.constants import (
    ATOM_MAPS, res_to_idx, MAX_ATOM_COUNT, 
    UNK_IDX, LIGAND_IDX, get_element_id
)

class PDBProcessor:
    def __init__(self, ligand_map_path="data/pdb/processed/ccd_atoms.json"):
        self.res_lookups = {}
        for res, atoms in ATOM_MAPS.items():
            self.res_lookups[res] = {name: i for i, name in enumerate(atoms)}

        # Protein vocab
        self.protein_res = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
        
        # Nucleic acid vocab
        self.nucleic_res = {'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U'}

        # CCD vocab
        self.ligand_ref_atoms = {}
        path = Path(ligand_map_path)
        if path.exists():
            with open(path, 'r') as f:
                self.ligand_ref_atoms = json.load(f)
        else:
            print(f"Warning: Ligand atom map not found at {path}. Ligands will be ignored!")

    def parse_cif_string(self, cif_string: str, pdb_id: str):
        try:
            doc = gemmi.cif.read_string(cif_string)
            if not doc: return None
            block = doc[0]
            st = gemmi.make_structure_from_block(block)
            model = st[0]
        except Exception as e:
            return None

        # Pass the CIF block to get polymer
        entity_types, chain_to_entity = self._get_entity_info(block)

        data = {"res_type": [], "coords": [], "mask": [], "atom_elements": [], "chain_ids": []}
        
        chain_counter = 0
        
        for chain in model:
            # Get entity label
            subchains = chain.subchains()
            label_asym_id = subchains[0].subchain_id() if len(subchains) > 0 else chain.name
            entity_id = chain_to_entity.get(label_asym_id)
            
            if not entity_id: 
                continue
                
            base_type = entity_types.get(entity_id)
            
            # Skip water entities
            if base_type == "water":
                continue

            residues = list(chain)
            if not residues: 
                continue

            # Assign polymer type for appropriate tokenization
            if base_type == "polymer":
                polymer_type = self._get_chain_polymer_type(block, entity_id)
                if polymer_type in ["PROTEIN", "NUCLEIC"]:
                    self._process_polymer(residues, data, chain_counter)
                else:
                    self._process_ligand(residues, data, chain_counter)
            
            elif base_type == "non-polymer":
                self._process_ligand(residues, data, chain_counter)
            
            chain_counter += 1

        if len(data["res_type"]) == 0: 
            return None
        
        return {
            "pdb_id": pdb_id,
            "res_type": np.array(data["res_type"], dtype=np.int32),
            "coords": np.array(data["coords"], dtype=np.float32),
            "mask": np.array(data["mask"], dtype=np.float32),
            "atom_elements": np.array(data["atom_elements"], dtype=np.int32),
            "chain_ids": np.array(data["chain_ids"], dtype=np.int32)
        }

    def _get_entity_info(self, block: gemmi.cif.Block):
        """Extracts entity classifications and chain-to-entity mapping from a CIF Block."""
        entity_types = {}
        entity_loop = block.find_loop("_entity.id")
        type_loop = block.find_loop("_entity.type")
        if entity_loop and type_loop:
            for entity_id, entity_type in zip(entity_loop, type_loop):
                entity_types[entity_id] = entity_type

        chain_to_entity = {}
        asym_loop = block.find_loop("_struct_asym.id")
        ent_id_loop = block.find_loop("_struct_asym.entity_id")
        if asym_loop and ent_id_loop:
            for chain_id, entity_id in zip(asym_loop, ent_id_loop):
                chain_to_entity[chain_id] = entity_id

        return entity_types, chain_to_entity

    def _get_chain_polymer_type(self, block: gemmi.cif.Block, entity_id):
        """
        Uses a hierarchy of checks to robustly determine if a polymer
        is a PROTEIN or a NUCLEIC acid.
        """        
        # Check 1 for polymer type
        poly_type_loop = block.find_loop("_entity_poly.type")
        poly_ent_loop = block.find_loop("_entity_poly.entity_id")
        if poly_type_loop and poly_ent_loop:
            for ent_id, poly_type in zip(poly_ent_loop, poly_type_loop):

                # Entity type matches polymer type
                if ent_id == entity_id:
                    poly_type = poly_type.upper()
                    if "PEPTIDE" in poly_type: 
                        return "PROTEIN"
                    if "NUCLEOTIDE" in poly_type: 
                        return "NUCLEIC"
        
        # Check 2 for polymer type
        poly_seq_ent_loop = block.find_loop("_entity_poly_seq.entity_id")
        poly_seq_mon_loop = block.find_loop("_entity_poly_seq.mon_id")
        if poly_seq_ent_loop and poly_seq_mon_loop:
            res_list = [mon for ent, mon in zip(poly_seq_ent_loop, poly_seq_mon_loop) if ent == entity_id]
            
            n_prot = sum(1 for res in res_list if res in self.protein_res)
            n_nuc = sum(1 for res in res_list if res in self.nucleic_res)

            # >80% of residues are of a specific polymer type
            if len(res_list) > 0:
                if n_prot / len(res_list) > 0.8: 
                    return "PROTEIN"
                if n_nuc / len(res_list) > 0.8: 
                    return "NUCLEIC"

        return "UNKNOWN"

    def _process_polymer(self, residues, data, chain_id):
        for res in residues:
            rname = res.name.strip().upper()
            if rname in ['HOH', 'DOD', 'WAT']: 
                continue
            
            token_id = res_to_idx.get(rname, UNK_IDX)
            atom_map = self.res_lookups.get(rname, self.res_lookups.get('UNK', {}))
            
            token_coords = np.zeros((MAX_ATOM_COUNT, 3), dtype=np.float32)
            token_mask = np.zeros(MAX_ATOM_COUNT, dtype=np.float32)
            token_elems = np.zeros(MAX_ATOM_COUNT, dtype=np.int32)
            
            for atom in res:
                aname = atom.name
                if "'" in aname: 
                    aname = aname.replace("'", "'")
                if aname == "O1P": 
                    aname = "OP1"
                if aname == "O2P": 
                    aname = "OP2"
                
                idx = atom_map.get(aname, -1)
                if idx != -1:
                    token_coords[idx] = [atom.pos.x, atom.pos.y, atom.pos.z]
                    token_mask[idx] = 1.0
                    token_elems[idx] = get_element_id(atom.element.name)
            
            data["res_type"].append(token_id)
            data["coords"].append(token_coords)
            data["mask"].append(token_mask)
            data["atom_elements"].append(token_elems)
            data["chain_ids"].append(chain_id)

    def _process_ligand(self, residues, data, chain_id):
        for res in residues:
            rname = res.name.strip().upper()
            if rname in ['HOH', 'DOD', 'WAT', 'SOL']: 
                continue

            # Get full atom list from CCD
            reference_atoms = self.ligand_ref_atoms.get(rname)
            
            # Skip if there's no CCD entry
            if not reference_atoms:
                continue 

            # Map atoms present in cif file
            pdb_atoms = {atom.name.strip(): atom for atom in res}

            for ref_atom_name, ref_element in reference_atoms:
                token_coords = np.zeros((MAX_ATOM_COUNT, 3), dtype=np.float32)
                token_mask = np.zeros(MAX_ATOM_COUNT, dtype=np.float32)
                token_elems = np.zeros(MAX_ATOM_COUNT, dtype=np.int32)
                
                token_elems[0] = get_element_id(ref_element)
                
                # Create mask
                if ref_atom_name in pdb_atoms:
                    atom = pdb_atoms[ref_atom_name]
                    token_coords[0] = [atom.pos.x, atom.pos.y, atom.pos.z]
                    token_mask[0] = 1.0
                else:
                    token_mask[0] = 0.0
                
                data["res_type"].append(LIGAND_IDX)
                data["coords"].append(token_coords)
                data["mask"].append(token_mask)
                data["atom_elements"].append(token_elems)
                data["chain_ids"].append(chain_id)