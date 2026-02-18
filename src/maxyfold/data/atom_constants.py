

# A data point should have shape (NUM_ATOMS, ATOMS_CONTEXT, 3).

# There are 20 standard AAs that will each have 1 token and get mapped to ATOMS_CONTEXT.
# There are 8 standard nucleotides (DNA/RNA) that will each have 1 token and get mapped to ATOMS_CONTEXT.
# For each atom in a ligand, modified AA/NA mapped to ATOMS_CONTEXT, where each atom is 1 token.

# The data should also have a mask tensor to let us know which residues 
# are missing and which are present (as well as for NUM_ATOMS < MAX_ATOMS).

import numpy as np

# The Token Vocabulary
# 0-19: Amino Acids
# 20-27: DNA/RNA
# 28: Unknown/Mask
# 29: Ligand (Generic container for single-atom tokens)
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
    'DA', 'DC', 'DG', 'DT', 'RA', 'RC', 'RG', 'RU',
    'UNK', 'LIGAND'
]
restype_to_idx = {res: i for i, res in enumerate(restypes)}

# 2. Dimensions
MAX_ATOM_COUNT = 27

# 3. Backbone Atoms (Standardized Indices)
# Proteins: N=0, CA=1, C=2, O=3
# Nucleic:  P=0, OP1=1, OP2=2, O5'=3, C5'=4...
protein_inner_atoms = ['N', 'CA', 'C', 'O']
nucleic_inner_atoms = ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]

# 4. Full Atom Maps
# (This is a subset. You should expand this list for all 20 AAs and 8 Nucleotides)
# You can copy the full dictionaries from OpenFold/AlphaFold repos.
ATOM_MAPS = {
    'ALA': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4},
    'GLY': {'N': 0, 'CA': 1, 'C': 2, 'O': 3},
    # ... add all AAs ...
    'DA': {k: i for i, k in enumerate(nucleic_inner_atoms + ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'])},
    # ... add all Nucleotides ...
}

def get_atom_map(res_name):
    """Returns {atom_name: index}."""
    # 1. Check Standard Polymer Maps
    if res_name in ATOM_MAPS:
        return ATOM_MAPS[res_name]
    
    # 2. Check for DNA/RNA synonyms (e.g., 'A' in PDB might be Adenosine RNA)
    # This logic depends on PDB cleaning. Usually we map 'A' -> 'RA' or 'DA' based on context.
    
    # 3. Fallback for Ligands
    # Ligands are 1-atom-per-token. The atom always goes to index 0.
    return {'*': 0}