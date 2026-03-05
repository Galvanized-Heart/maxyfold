import json
from pathlib import Path
from tqdm import tqdm

from maxyfold.data.components.tarball_reader import TarballReader
from maxyfold.data.constants.atom_constants import SOLVENT_RESIDUES

try:
    import gemmi
except ImportError:
    raise ImportError("Gemmi is required for this script.")

class PDBManifest:
    def __init__(self, raw_assemblies_dir: Path, ccd_smiles_path: Path, limit: int = 0):
        self.raw_assemblies_dir = raw_assemblies_dir
        with open(ccd_smiles_path, 'r') as f:
            self.smiles_map = json.load(f)
        self.limit = limit

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