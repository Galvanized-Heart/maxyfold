import numpy as np
import warnings
try:
    import gemmi
except ImportError:
    pass

from maxyfold.data.constants import (
    ATOM_MAPS, res_to_idx, MAX_ATOM_COUNT, 
    UNK_IDX, LIGAND_IDX, get_element_id
)

class AllAtomProcessor:
    def __init__(self):
        # Pre-compute reverse lookups for speed
        self.res_lookups = {}
        for res, atoms in ATOM_MAPS.items():
            self.res_lookups[res] = {name: i for i, name in enumerate(atoms)}

    def parse_cif_string(self, cif_string: str, pdb_id: str):
        try:
            st = gemmi.read_structure(cif_string)
            model = st[0]
        except Exception as e:
            return None

        # Containers
        res_types = []
        coords = []
        masks = []
        elements = []
        chain_ids = []
        
        chain_counter = 0

        for chain in model:
            # We want to ignore pure water chains
            residues = list(chain)
            if not residues: continue

            for res in residues:
                rname = res.name.strip().upper()
                if rname in ['HOH', 'DOD', 'WAT', 'SOL']:
                    continue # Skip water

                # POLYMER LOGIC (Dense Packing)
                if rname in self.res_lookups:
                    token_coords = np.zeros((MAX_ATOM_COUNT, 3), dtype=np.float32)
                    token_mask = np.zeros(MAX_ATOM_COUNT, dtype=np.float32)
                    token_elems = np.zeros(MAX_ATOM_COUNT, dtype=np.int32)
                    
                    atom_map = self.res_lookups[rname]
                    
                    for atom in res:
                        aname = atom.name
                        # Handle PDB naming quirks
                        if '*' in aname: aname = aname.replace('*', "'")
                        if aname == "O1P": aname = "OP1"
                        if aname == "O2P": aname = "OP2"
                        
                        idx = atom_map.get(aname, -1)
                        if idx != -1:
                            token_coords[idx] = [atom.pos.x, atom.pos.y, atom.pos.z]
                            token_mask[idx] = 1.0
                            token_elems[idx] = get_element_id(atom.element.name)

                    res_types.append(res_to_idx.get(rname, UNK_IDX))
                    coords.append(token_coords)
                    masks.append(token_mask)
                    elements.append(token_elems)
                    chain_ids.append(chain_counter)

                # LIGAND LOGIC (Exploded Packing: 1 Token = 1 Atom)
                else:
                    for atom in res:
                        el_name = atom.element.name.upper()
                        if el_name in ['H', 'D', '']: continue # Ignore Hydrogens
                        
                        token_coords = np.zeros((MAX_ATOM_COUNT, 3), dtype=np.float32)
                        token_mask = np.zeros(MAX_ATOM_COUNT, dtype=np.float32)
                        token_elems = np.zeros(MAX_ATOM_COUNT, dtype=np.int32)
                        
                        # Always Index 0
                        token_coords[0] = [atom.pos.x, atom.pos.y, atom.pos.z]
                        token_mask[0] = 1.0
                        token_elems[0] = get_element_id(el_name)
                        
                        res_types.append(LIGAND_IDX)
                        coords.append(token_coords)
                        masks.append(token_mask)
                        elements.append(token_elems)
                        chain_ids.append(chain_counter)
                        
            chain_counter += 1

        if len(res_types) == 0:
            return None

        # Return single complex entry
        return {
            "pdb_id": pdb_id,
            "res_type": np.array(res_types, dtype=np.int32),
            "coords": np.array(coords, dtype=np.float32),
            "mask": np.array(masks, dtype=np.float32),
            "atom_elements": np.array(elements, dtype=np.int32),
            "chain_ids": np.array(chain_ids, dtype=np.int32)
        }