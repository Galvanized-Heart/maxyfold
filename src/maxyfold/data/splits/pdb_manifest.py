import json
from pathlib import Path
from tqdm import tqdm
import io
import numpy as np

try:
    import gemmi
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx_io
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

        self.protein_res = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                           'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
        self.nucleic_res = {'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U'}

    def _get_block(self, cif_file, pdb_id_upper: str):
        """Robust block lookup — most files have one block named after the PDB ID"""
        if pdb_id_upper in cif_file:
            return cif_file[pdb_id_upper]
        # Fallback: first (and usually only) block
        if len(cif_file) > 0:
            return list(cif_file.values())[0]
        raise ValueError("No block found in CIF file")

    def _get_assembly_chains(self, block) -> set:
        valid_chains = set()
        category = block.get("_pdbx_struct_assembly_gen")   # ← correct: on block, not cif_file
        if category:
            assembly_ids = category.get("assembly_id", [])
            asym_id_lists = category.get("asym_id_list", [])
            for asm_id, chain_list_str in zip(assembly_ids, asym_id_lists):
                if asm_id == "1":
                    chains = [c.strip() for c in str(chain_list_str).split(',')]
                    valid_chains.update(chains)
        return valid_chains

    def _map_chains_to_entities(self, block) -> dict:
        chain_to_ent = {}
        category = block.get("_struct_asym")
        if category:
            ids = category.get("id", [])
            ent_ids = category.get("entity_id", [])
            for cid, eid in zip(ids, ent_ids):
                chain_to_ent[cid] = eid
        return chain_to_ent

    def _get_entity_data(self, block) -> dict:
        entities = {}
        cat = block.get("_entity")
        if cat:
            for eid, etype in zip(cat.get("id", []), cat.get("type", [])):
                entities[eid] = {'type': etype, 'seq': None, 'ligand_id': None}

        cat_poly = block.get("_entity_poly")
        if cat_poly:
            for eid, seq_can in zip(cat_poly.get("entity_id", []), cat_poly.get("pdbx_seq_one_letter_code_can", [])):
                if eid in entities and seq_can and seq_can not in ['.', '?']:
                    clean_seq = str(seq_can).replace('\n', '').replace(';', '').strip()
                    entities[eid]['seq'] = clean_seq

        cat_non = block.get("_pdbx_entity_nonpoly")
        if cat_non:
            for eid, comp_id in zip(cat_non.get("entity_id", []), cat_non.get("comp_id", [])):
                if eid in entities:
                    entities[eid]['ligand_id'] = comp_id
        return entities

    def _compute_interfaces(self, structure: struc.AtomArray, cutoff: float = 5.0) -> list:
        if struc.get_chain_count(structure) < 2:
            return []
        interfaces = []
        chain_ids = np.unique(structure.chain_id)
        cell_list = struc.CellList(structure, cell_size=cutoff)
        for i, chain1 in enumerate(chain_ids):
            for chain2 in chain_ids[i + 1:]:
                mask1 = structure.chain_id == chain1
                mask2 = structure.chain_id == chain2
                contacts = cell_list.get_atoms(structure.coord[mask1], radius=cutoff)
                contact_count = int(np.sum((contacts != -1).any(axis=1)))
                if contact_count > 0:
                    interfaces.append({
                        "chain1": chain1,
                        "chain2": chain2,
                        "contacts": contact_count
                    })
        return interfaces

    def create(self) -> dict:

        manifest = {}
        tar_files = sorted(list(self.raw_assemblies_dir.glob("assemblies_batch_*.tar.gz")))
        cif_stream = TarballReader(tar_paths=tar_files, file_limit=self.limit)

        pbar = tqdm(cif_stream, desc="Creating Manifest")

        for pdb_id, cif_string in pbar:
            pdb_id_upper = pdb_id.upper()
            if pdb_id_upper in self.invalid_ids:
                continue
            

            fileobj = io.StringIO(cif_string)
            cif_file = pdbx_io.CIFFile.read(fileobj)
            atoms = pdbx_io.get_structure(cif_file, model=1, use_author_fields=False)
            #['_abc_impl', '_annot', '_array_length', '_bonds', '_box', '_coord', '_copy_annotations', '_del_element',
            # '_max_atoms_printed', '_max_models_printed', '_set_element', '_subarray', 'add_annotation', 'array_length',
            # 'atom_name', 'bonds', 'box', 'chain_id', 'coord', 'copy', 'del_annotation', 'element',
            # 'equal_annotation_categories', 'equal_annotations', 'get_annotation', 'get_annotation_categories',
            # 'get_atom', 'hetero', 'ins_code', 'res_id', 'res_name', 'set_annotation', 'shape']
            chains = struc.get_chains(atoms)
            seq = pdbx_io.get_sequence(cif_file)
            comp = pdbx_io.list_assemblies(cif_file)
            
            print("")
            print(chains)
            #print(*zip(atoms.res_name, atoms.chain_id))
            #print(dir(atoms))
            #print(atoms.equal_annotation_categories())
            #print(atoms.get_annotation_categories())
            #print(atoms.shape)
            print(comp)
            print(seq)
            print(atoms.chain_id)
            print(atoms.res_name)


        return manifest

        """    # Create biotite AtomArray
            fileobj = io.StringIO(cif_string)
            cif_file = pdbx_io.CIFFile.read(fileobj)
            block = self._get_block(cif_file, pdb_id_upper)
            structure = pdbx_io.get_structure(cif_file, model=1)
            
            
            # Assembly chains with robust fallback
            asm_chains = self._get_assembly_chains(block)
            print(f"\n***{asm_chains}***\n")
            if not asm_chains:
                asm_chains = set(self._map_chains_to_entities(block).keys())
                if not asm_chains:
                    asm_chains = set(np.unique(structure.chain_id))

        return manifest"""










        manifest = {}
        tar_files = sorted(list(self.raw_assemblies_dir.glob("assemblies_batch_*.tar.gz")))
        cif_stream = TarballReader(tar_paths=tar_files, file_limit=self.limit)
        pbar = tqdm(cif_stream, desc="Creating Manifest (Biotite + interfaces)")

        for pdb_id, cif_string in pbar:
            pdb_id_upper = pdb_id.upper()
            if pdb_id_upper in self.invalid_ids:
                continue

            try:
                fileobj = io.StringIO(cif_string)
                cif_file = pdbx_io.CIFFile.read(fileobj)
                block = self._get_block(cif_file, pdb_id_upper)          # ← robust block
                structure = pdbx_io.get_structure(cif_file, model=1)     # still works

                asm_chains = self._get_assembly_chains(block)
                if not asm_chains:
                    asm_chains = set(self._map_chains_to_entities(block).keys())
                    if not asm_chains:
                        asm_chains = set(np.unique(structure.chain_id))  # ultimate fallback

                chain_to_ent = self._map_chains_to_entities(block)
                entity_data = self._get_entity_data(block)

                # Extra fields
                method = ""
                res_high = None
                cat_exptl = block.get("exptl")
                if cat_exptl:
                    method = str(cat_exptl.get("method", [""])[0])
                cat_reflns = block.get("reflns")
                if cat_reflns and "d_resolution_high" in cat_reflns:
                    res_high = float(cat_reflns["d_resolution_high"][0])
                cat_refine = block.get("refine")
                if cat_refine and "ls_d_res_high" in cat_refine:
                    res_high = float(cat_refine["ls_d_res_high"][0])

                interfaces = self._compute_interfaces(structure)

                entry = {
                    "chains": {},
                    "protein_sequences": {},
                    "nucleic_sequences": {},
                    "ligands": {},
                    "resolution": res_high,
                    "method": method,
                    "interfaces": interfaces
                }
                total_residues = 0

                for chain_id in asm_chains:
                    ent_id = chain_to_ent.get(chain_id)
                    data = entity_data.get(ent_id) if ent_id else None

                    # Fallback classification using structure (robust for old PDBs)
                    chain_mask = structure.chain_id == chain_id
                    chain_atoms = structure[chain_mask]
                    res_names = np.unique(chain_atoms.res_name) if len(chain_atoms) > 0 else []

                    is_protein = any(r in self.protein_res for r in res_names)
                    is_nucleic = any(r in self.nucleic_res for r in res_names)

                    if data and data['type'] == 'water':
                        entry['chains'][chain_id] = 'water'
                        continue

                    if data and data['type'] == 'polymer':
                        seq = data['seq']
                        if not seq:
                            entry['chains'][chain_id] = 'polymer_unknown'
                            continue
                        seq_len = len(seq)
                        total_residues += seq_len
                        if seq_len < 4:
                            entry['chains'][chain_id] = 'polymer_too_short'
                            continue

                        chain_key = f"{pdb_id_upper}_{chain_id}"
                        if is_protein:
                            entry['chains'][chain_id] = 'protein'
                            entry['protein_sequences'][chain_key] = seq
                        else:
                            entry['chains'][chain_id] = 'nucleic_acid'
                            entry['nucleic_sequences'][chain_key] = seq

                    elif data and data['type'] == 'non-polymer':
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

                    else:
                        # Pure fallback when no entity data
                        if is_protein or is_nucleic:
                            entry['chains'][chain_id] = 'protein' if is_protein else 'nucleic_acid'
                        elif len(chain_atoms) > 0:
                            entry['chains'][chain_id] = 'ligand'

                if total_residues > 5000:
                    continue

                entry = {k: v for k, v in entry.items() if v or k in ("resolution", "method", "interfaces")}
                if entry.get('protein_sequences') or entry.get('nucleic_sequences') or entry.get('ligands'):
                    manifest[pdb_id_upper] = entry

            except Exception:
                continue

        return manifest