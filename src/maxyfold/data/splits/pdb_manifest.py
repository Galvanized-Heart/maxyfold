import json
from pathlib import Path
from tqdm import tqdm
import io
import numpy as np

try:
    import gemmi
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
except ImportError:
    pass

from maxyfold.data.components.tarball_reader import TarballReader
from maxyfold.data.constants.atom_constants import SOLVENT_RESIDUES

class GemmiPDBManifest:
    def __init__(self, raw_assemblies_dir: Path, ccd_smiles_path: Path, invalid_ids: set, limit: int = 0):
        self.raw_assemblies_dir = raw_assemblies_dir
        self.invalid_ids = invalid_ids
        self.limit = limit

        with open(ccd_smiles_path, 'r') as f:
            self.smiles_map = json.load(f)

    def _get_assembly_chains(self, block: gemmi.cif.Block) -> set:
        """Returns set of label_asym_ids in Assembly 1."""
        valid_chains = set()
        table = block.find("_pdbx_struct_assembly_gen.", ["assembly_id", "asym_id_list"])
        for row in table:
            asm_id, chain_list_str = row[0], row[1]
            if asm_id == "1":
                chains = [c.strip() for c in chain_list_str.split(',')]
                valid_chains.update(chains)
        return valid_chains

    def _map_chains_to_entities(self, block: gemmi.cif.Block) -> dict:
        """Returns {label_asym_id: entity_id}."""
        chain_to_ent = {}
        table = block.find("_struct_asym.", ["id", "entity_id"])
        for row in table:
            chain_to_ent[row[0]] = row[1]
        return chain_to_ent

    def _get_entity_data(self, block: gemmi.cif.Block) -> dict:
        """
        Returns {entity_id: {'type': ..., 'seq': ..., 'ligand_id': ...}}
        """
        entities = {}
        
        # Initialize types
        table = block.find("_entity.", ["id", "type"])
        for row in table:
            entities[row[0]] = {'type': row[1], 'seq': None, 'ligand_id': None}

        # Get polymer sequences
        table_poly = block.find("_entity_poly.", ["entity_id", "pdbx_seq_one_letter_code_can"])
        for row in table_poly:
            ent_id, seq_can = row[0], row[1]
            if ent_id in entities:
                if seq_can and seq_can not in ['.', '?']:
                    clean_seq = seq_can.replace('\n', '').replace(';', '').strip()
                    entities[ent_id]['seq'] = clean_seq

        # Get non-polymer CCD IDs
        table_non = block.find("_pdbx_entity_nonpoly.", ["entity_id", "comp_id"])
        for row in table_non:
            ent_id, comp_id = row[0], row[1]
            if ent_id in entities:
                entities[ent_id]['ligand_id'] = comp_id

        return entities

    def create(self) -> dict:
        manifest = {}
        tar_files = sorted(list(self.raw_assemblies_dir.glob("assemblies_batch_*.tar.gz")))
        cif_stream = TarballReader(tar_paths=tar_files, file_limit=self.limit)
        
        pbar = tqdm(cif_stream, desc="Creating Manifest")

        for pdb_id, cif_string in pbar:
            pdb_id_upper = pdb_id.upper()
            
            if pdb_id_upper in self.invalid_ids:
                continue
            
            try:
                doc = gemmi.cif.read_string(cif_string)
                block = doc.sole_block()

                # Get chains in assembly
                asm_chains = self._get_assembly_chains(block)
                if not asm_chains: 
                    asm_chains = set(self._map_chains_to_entities(block).keys())

                # Get chain entity type
                chain_to_ent = self._map_chains_to_entities(block)
                entity_data = self._get_entity_data(block)

                entry = {
                    "chains": {},
                    "protein_sequences": {},
                    "nucleic_sequences": {},
                    "ligands": {}
                }

                total_residues = 0

                # Process chains to sequence/SMILES
                for chain_id in asm_chains:
                    ent_id = chain_to_ent.get(chain_id)
                    if not ent_id or ent_id not in entity_data: 
                        continue
                    
                    data = entity_data[ent_id]
                    ent_type = data['type']
                    
                    # Skip water
                    if ent_type == 'water':
                        entry['chains'][chain_id] = 'water'
                        continue

                    # Polymers (Protein/Nucleic)
                    if ent_type == 'polymer':
                        seq = data['seq']

                        # Filter polymers that are unknown
                        if not seq:
                            entry['chains'][chain_id] = 'polymer_unknown'
                            continue

                        seq_len = len(seq)
                        total_residues += seq_len

                        # Filter sequences that are <4 residues
                        if seq_len < 4:
                            entry['chains'][chain_id] = 'polymer_too_short'
                            continue

                        # Classify protein/nucleic acid polymers
                        is_protein = any(c in 'DEFHIKLMNPQRSVWY' for c in seq)
                        chain_key = f"{pdb_id_upper}_{chain_id}"
                        if is_protein:
                            entry['chains'][chain_id] = 'protein'
                            entry['protein_sequences'][chain_key] = seq
                        else:
                            entry['chains'][chain_id] = 'nucleic_acid'
                            entry['nucleic_sequences'][chain_key] = seq
                    
                    # Ligands
                    elif ent_type == 'non-polymer':
                        comp_id = data['ligand_id']
                        
                        # Filter solvent chains
                        if comp_id in SOLVENT_RESIDUES: 
                            entry['chains'][chain_id] = 'solvent'
                            continue

                        entry['chains'][chain_id] = 'ligand'

                        if comp_id and comp_id in self.smiles_map:
                            smiles = self.smiles_map[comp_id]
                            if smiles != "O":
                                chain_key = f"{pdb_id_upper}_{chain_id}"
                                entry['ligands'][chain_key] = {
                                    "ccd_id": comp_id,
                                    "smiles": smiles
                                }
                                total_residues += 1 

                # Filter large complexes
                if total_residues > 5000:
                    continue
                
                # Cleanup empty dicts
                entry = {k: v for k, v in entry.items() if v}

                if entry.get('protein_sequences') or entry.get('nucleic_sequences') or entry.get('ligands'):
                    manifest[pdb_id_upper] = entry

            except Exception:
                continue
        
        return manifest



class BiotitePDBManifest:
    def __init__(self, raw_assemblies_dir: Path, ccd_smiles_path: Path, invalid_ids: set, limit: int = 0):
        self.raw_assemblies_dir = raw_assemblies_dir
        self.invalid_ids = invalid_ids
        self.limit = limit

        with open(ccd_smiles_path, 'r') as f:
            self.smiles_map = json.load(f)

    def _get_assembly_chains(self, block: pdbx.CIFBlock) -> set:
        """Returns set of label_asym_ids in Assembly 1."""
        valid_chains = set()
        if "pdbx_struct_assembly_gen" in block:
            try:
                asm_ids = block["pdbx_struct_assembly_gen"]["assembly_id"].as_array(str)
                asym_lists = block["pdbx_struct_assembly_gen"]["asym_id_list"].as_array(str)
                for a_id, a_list in zip(asm_ids, asym_lists):
                    if a_id == "1":
                        chains = [c.strip() for c in a_list.split(',')]
                        valid_chains.update(chains)
            except KeyError:
                pass
        return valid_chains

    def _map_chains_to_entities(self, block: pdbx.CIFBlock) -> dict:
        """Returns {label_asym_id: entity_id}."""
        chain_to_ent = {}
        if "struct_asym" in block:
            try:
                asym_ids = block["struct_asym"]["id"].as_array(str)
                ent_ids = block["struct_asym"]["entity_id"].as_array(str)
                for a_id, e_id in zip(asym_ids, ent_ids):
                    chain_to_ent[a_id] = e_id
            except KeyError:
                pass
        return chain_to_ent

    def _get_entity_data(self, block: pdbx.CIFBlock) -> dict:
        """
        Returns {entity_id: {'type': ..., 'seq': ..., 'ligand_id': ...}}
        """
        entities = {}
        
        # Initialize types
        if "entity" in block:
            try:
                ent_ids = block["entity"]["id"].as_array(str)
                ent_types = block["entity"]["type"].as_array(str)
                for e_id, e_type in zip(ent_ids, ent_types):
                    entities[e_id] = {'type': e_type, 'seq': None, 'ligand_id': None}
            except KeyError:
                pass

        # Get polymer sequences
        if "entity_poly" in block:
            try:
                ent_ids = block["entity_poly"]["entity_id"].as_array(str)
                # Not all structures have pdbx_seq_one_letter_code_can
                if "pdbx_seq_one_letter_code_can" in block["entity_poly"]:
                    seqs = block["entity_poly"]["pdbx_seq_one_letter_code_can"].as_array(str)
                    for e_id, seq in zip(ent_ids, seqs):
                        if e_id in entities:
                            if seq and seq not in ['.', '?']:
                                clean_seq = seq.replace('\n', '').replace(';', '').strip()
                                entities[e_id]['seq'] = clean_seq
            except KeyError:
                pass

        # Get non-polymer CCD IDs
        if "pdbx_entity_nonpoly" in block:
            try:
                ent_ids = block["pdbx_entity_nonpoly"]["entity_id"].as_array(str)
                comp_ids = block["pdbx_entity_nonpoly"]["comp_id"].as_array(str)
                for e_id, c_id in zip(ent_ids, comp_ids):
                    if e_id in entities:
                        entities[e_id]['ligand_id'] = c_id
            except KeyError:
                pass

        return entities

    def create(self) -> dict:
        manifest = {}
        tar_files = sorted(list(self.raw_assemblies_dir.glob("assemblies_batch_*.tar.gz")))
        cif_stream = TarballReader(tar_paths=tar_files, file_limit=self.limit)
        
        pbar = tqdm(cif_stream, desc="Creating Manifest")

        for pdb_id, cif_string in pbar:
            pdb_id_upper = pdb_id.upper()
            
            if pdb_id_upper in self.invalid_ids:
                continue
            
            try:
                # Parse CIF
                cif_file = pdbx.CIFFile.read(io.StringIO(cif_string))
                block = cif_file.block  # Main data block
                
                # Extract metadata
                asm_chains = self._get_assembly_chains(block)
                chain_to_ent = self._map_chains_to_entities(block)
                if not asm_chains: 
                    asm_chains = set(chain_to_ent.keys())

                entity_data = self._get_entity_data(block)

                entry = {
                    "chains": {},
                    "protein_sequences": {},
                    "nucleic_sequences": {},
                    "ligands": {},
                    "interfaces": [] # New feature
                }

                total_residues = 0

                # Process chains to sequence/SMILES
                for chain_id in asm_chains:
                    ent_id = chain_to_ent.get(chain_id)
                    if not ent_id or ent_id not in entity_data: 
                        continue
                    
                    data = entity_data[ent_id]
                    ent_type = data['type']
                    
                    if ent_type == 'water':
                        entry['chains'][chain_id] = 'water'
                        continue

                    # Polymers
                    if ent_type == 'polymer':
                        seq = data['seq']
                        if not seq:
                            entry['chains'][chain_id] = 'polymer_unknown'
                            continue

                        seq_len = len(seq)
                        total_residues += seq_len
                        if seq_len < 4:
                            entry['chains'][chain_id] = 'polymer_too_short'
                            continue

                        is_protein = any(c in 'DEFHIKLMNPQRSVWY' for c in seq)
                        chain_key = f"{pdb_id_upper}_{chain_id}"
                        if is_protein:
                            entry['chains'][chain_id] = 'protein'
                            entry['protein_sequences'][chain_key] = seq
                        else:
                            entry['chains'][chain_id] = 'nucleic_acid'
                            entry['nucleic_sequences'][chain_key] = seq
                    
                    # Ligands
                    elif ent_type == 'non-polymer':
                        comp_id = data['ligand_id']
                        if comp_id in SOLVENT_RESIDUES: 
                            entry['chains'][chain_id] = 'solvent'
                            continue

                        entry['chains'][chain_id] = 'ligand'

                        if comp_id and comp_id in self.smiles_map:
                            smiles = self.smiles_map[comp_id]
                            if smiles != "O":
                                chain_key = f"{pdb_id_upper}_{chain_id}"
                                entry['ligands'][chain_key] = {
                                    "ccd_id": comp_id,
                                    "smiles": smiles
                                }
                                total_residues += 1 

                if total_residues > 5000:
                    continue

                # Interface detection
                try:
                    atom_array = pdbx.get_assembly(cif_file, assembly_id="1", model=1, use_author_fields=False)
                except Exception:
                    atom_array = pdbx.get_structure(cif_file, model=1, use_author_fields=False)
                
                unique_generated_chains = np.unique(atom_array.chain_id)
                asm_to_base = {}
                for c in unique_generated_chains:
                    if c in asm_chains:
                        asm_to_base[c] = c
                    else:
                        base = c.split('_')[0]
                        asm_to_base[c] = base if base in asm_chains else c
                
                # Filter heavy atoms
                heavy_array = atom_array[atom_array.element != "H"]
                
                if len(heavy_array) > 0:
                    # 5.0 Angstroms cutoff for heavy atom contacts
                    cell_list = struc.CellList(heavy_array, cell_size=5.0)
                    interfaces = set()
                    
                    # Calculate vectors by chain
                    chain_starts = struc.get_chain_starts(heavy_array)
                    chain_ends = np.append(chain_starts[1:], len(heavy_array))
                    
                    for start, end in zip(chain_starts, chain_ends):
                        chain_a = heavy_array.chain_id[start]
                        base_a = asm_to_base.get(chain_a, chain_a)
                        
                        if base_a not in entry['chains']:
                            continue
                            
                        # Get neighbors indices for all atoms in the current chain
                        neighbors = cell_list.get_atoms(heavy_array.coord[start:end], radius=5.0)
                        
                        # Flatten and remove paddings
                        valid_neighbors = np.unique(neighbors)
                        valid_neighbors = valid_neighbors[valid_neighbors != -1]
                        
                        interacting_chains = np.unique(heavy_array.chain_id[valid_neighbors])
                        
                        for chain_b in interacting_chains:
                            if chain_a != chain_b:
                                base_b = asm_to_base.get(chain_b, chain_b)
                                if base_b in entry['chains']:
                                    pair = tuple(sorted([base_a, base_b]))
                                    interfaces.add(pair)
                                    
                    entry["interfaces"] = [list(p) for p in sorted(interfaces)]

                # Cleanup empty dicts/lists
                entry = {k: v for k, v in entry.items() if v}

                if entry.get('protein_sequences') or entry.get('nucleic_sequences') or entry.get('ligands'):
                    manifest[pdb_id_upper] = entry

            except Exception as e:
                continue
        
        return manifest