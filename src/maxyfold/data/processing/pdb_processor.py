import gzip
import io
import os
import tempfile
import numpy as np
import warnings
from typing import Dict, Any, Optional, List
from Bio.PDB.PDBExceptions import PDBConstructionWarning

try:
    from pdbfixer import PDBFixer
    from openmm import app as openmm_app
    from openmm import unit
except ImportError:
    raise ImportError("PDBFixer/OpenMM not installed.")

warnings.simplefilter('ignore', PDBConstructionWarning)

class PDBProcessor:
    def __init__(self):
        # 3-letter to 1-letter mapping
        self.aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        # Set of valid 3-letter codes for filtering
        self.standard_residues = set(self.aa_map.keys())

    def parse_cif_content(self, file_content: bytes, pdb_id: str) -> Optional[Dict[str, Any]]:
        # Create temp pdb file
        with tempfile.NamedTemporaryFile(suffix=".cif", mode='wb', delete=False) as tmp_file:
            with gzip.open(io.BytesIO(file_content), 'rb') as gz_f:
                tmp_file.write(gz_f.read())
            tmp_path = tmp_file.name

        try:
            fixer = PDBFixer(filename=tmp_path)

            # Remove hydrogens
            fixer.removeHeterogens(keepWater=False)
            
            # Find gaps for masking
            fixer.findMissingResidues()
            fixer.missingResidues = {} 

            # Fix non-standard residues
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            
            # Fill in missing atoms
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            
            # Remove hydrogens (just in case openmm added some during fixing)
            modeller = openmm_app.Modeller(fixer.topology, fixer.positions)
            modeller.deleteWater()
            hydrogens = [atom for atom in modeller.topology.atoms() if atom.element.symbol == 'H']
            modeller.delete(hydrogens)

            return self._extract_data(modeller.topology, modeller.positions, pdb_id)

        except Exception:
            return None

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _extract_data(self, topology: Any, positions: List[Any], pdb_id: str) -> Optional[Dict[str, Any]]:
        # Iterator for efficient access
        pos_iter = iter(positions)
        
        chains_data = []
        
        for chain in topology.chains():
            chain_coords = []
            chain_seq_list = []
            chain_mask = []
            
            for residue in chain.residues():
                # Skip non-standard residues
                if residue.name not in self.standard_residues:
                    for _ in residue.atoms(): next(pos_iter)
                    continue

                # Map atoms
                res_atoms = {atom.name: next(pos_iter).value_in_unit(unit.angstroms) for atom in residue.atoms()}
                
                # Check Backbone
                try:
                    bb = [res_atoms[n] for n in ['N', 'CA', 'C', 'O']]
                    chain_coords.append(bb)
                    chain_mask.append(1)
                    # Convert 3-letter to 1-letter immediately
                    chain_seq_list.append(self.aa_map[residue.name])
                except KeyError:
                    # Masking Case: Missing atoms
                    chain_coords.append([[0,0,0]] * 4)
                    chain_mask.append(0)
                    chain_seq_list.append(self.aa_map[residue.name])

            if not chain_coords:
                continue
            
            # Convert sequence list to single str
            sequence_str = "".join(chain_seq_list)
            
            # Convert mask to bool
            mask_np = np.array(chain_mask, dtype=np.bool_)
            
            # Remove structures with <10 aa or <20% of total aa in sequence
            if len(sequence_str) < 10 or mask_np.mean() < 0.2:
                continue

            chains_data.append({
                "chain_id": chain.id,
                "sequence": sequence_str,
                "coords": np.array(chain_coords, dtype=np.float32), 
                "mask": mask_np
            })

        if not chains_data:
            return None

        return {"pdb_id": pdb_id, "chains": chains_data}