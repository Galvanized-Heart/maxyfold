import pytest
import numpy as np
import json
from pathlib import Path
import rootutils

# Setup paths
ROOT = rootutils.find_root(indicator=".project-root")
REAL_DATA_DIR = ROOT / "data/pdb/analysis"
CCD_MAP_PATH = ROOT / "data/pdb/processed/ccd_atoms.json"

from maxyfold.data.processing.pdb_processor import PDBProcessor
from maxyfold.data.constants import MAX_ATOM_COUNT, LIGAND_IDX, UNK_IDX



# --- INTEGRATION TEST ---
@pytest.mark.skipif(
    not (REAL_DATA_DIR / "100D.cif").exists() or not CCD_MAP_PATH.exists(),
    reason="Real data files (100D.cif or ccd_atoms.json) not found."
)
def test_real_structure_100D():
    """
    Verifies that the processor correctly handles the DNA-RNA-Spermine hybrid
    we spent so much time debugging.
    """
    # 1. Initialize with the CCD map
    processor = PDBProcessor(ligand_map_path=str(CCD_MAP_PATH))
    
    # 2. Load the file content
    with open(REAL_DATA_DIR / "100D.cif", "r") as f:
        cif_string = f.read()

    # 3. Process
    result = processor.parse_cif_string(cif_string, "100D")
    
    assert result is not None, "Failed to parse 100D.cif"
    
    # 4. Verify Dimensions (Based on your inspection script results)
    # L should be 34 (20 Polymer Residues + 14 Spermine Atoms)
    assert len(result["res_type"]) == 34
    
    # 5. Verify Ligand Count
    num_ligands = np.sum(result["res_type"] == LIGAND_IDX)
    assert num_ligands == 14, f"Expected 14 Spermine atoms, found {num_ligands}"
    
    # 6. Verify Polymer Integrity
    # Token 0 is 'C' (Cytosine). Index 0,1,2 (P, OP1, OP2) should be masked (0.0)
    # Index 3 (O5') should be present (1.0)
    token_0_mask = result["mask"][0]
    assert token_0_mask[0] == 0.0 # P missing at 5' end
    assert token_0_mask[3] == 1.0 # O5' present
    
    print("\n100D Integration Test Passed: Hybrid Polymer + Ligand detected correctly.")