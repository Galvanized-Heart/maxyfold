import numpy as np

# ----------------------------------------------------------------------
#  TENSOR DIMENSIONS
# ----------------------------------------------------------------------
MAX_ATOM_COUNT = 27

# ----------------------------------------------------------------------
#  TOKEN VOCABULARY
# ----------------------------------------------------------------------
restypes = [
    # 0-19: Standard Amino Acids
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # 20-27: Standard Nucleic Acids (DNA & RNA)
    'DA', 'DC', 'DG', 'DT', 'A', 'C', 'G', 'U',
    # 28: Unknowns
    'UNK',
    # 29: Ligand
    'LIGAND'
]
res_to_idx = {res: i for i, res in enumerate(restypes)}
UNK_IDX = res_to_idx['UNK']
LIGAND_IDX = res_to_idx['LIGAND']

# ----------------------------------------------------------------------
# AMINO ACID MAPPING (3-letter to 1-letter)
# ----------------------------------------------------------------------
AA_3_TO_1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'UNK': 'X',
}

# ----------------------------------------------------------------------
# ATOM MAPPINGS
# ----------------------------------------------------------------------
ATOM_MAPS = {}

# PROTEIN BACKBONE
prot_backbone = ['N', 'CA', 'C', 'O']

# PROTEIN SIDECHAINS
ATOM_MAPS.update({
    'ALA': prot_backbone + ['CB'],
    'ARG': prot_backbone + ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': prot_backbone + ['CB', 'CG', 'OD1', 'ND2'],
    'ASP': prot_backbone + ['CB', 'CG', 'OD1', 'OD2'],
    'CYS': prot_backbone + ['CB', 'SG'],
    'GLN': prot_backbone + ['CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': prot_backbone + ['CB', 'CG', 'CD', 'OE1', 'OE2'],
    'GLY': prot_backbone,
    'HIS': prot_backbone + ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': prot_backbone + ['CB', 'CG1', 'CG2', 'CD1'],
    'LEU': prot_backbone + ['CB', 'CG', 'CD1', 'CD2'],
    'LYS': prot_backbone + ['CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': prot_backbone + ['CB', 'CG', 'SD', 'CE'],
    'PHE': prot_backbone + ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': prot_backbone + ['CB', 'CG', 'CD'],
    'SER': prot_backbone + ['CB', 'OG'],
    'THR': prot_backbone + ['CB', 'OG1', 'CG2'],
    'TRP': prot_backbone + ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': prot_backbone + ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': prot_backbone + ['CB', 'CG1', 'CG2'],
    'UNK': prot_backbone + ['CB']
})

# NUCLEIC ACIDS
dna_backbone = ['P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
rna_backbone = dna_backbone + ["O2'"]

ATOM_MAPS.update({
    'A': rna_backbone + ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],
    'G': rna_backbone + ['N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],
    'C': rna_backbone + ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
    'U': rna_backbone + ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'],
    'DA': dna_backbone + ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],
    'DG': dna_backbone + ['N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],
    'DC': dna_backbone + ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
    'DT': dna_backbone + ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C7', 'C6'],
})

# ----------------------------------------------------------------------
# ELEMENT MAPPING
# ----------------------------------------------------------------------
elements = [
    "X", "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
    "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
    "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
    "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
    "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
    "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
    "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
    "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
    "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
    "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG"
]
el_to_idx = {e: i for i, e in enumerate(elements)}

def get_element_id(symbol):
    if not isinstance(symbol, str): return 0
    symbol = symbol.upper().strip()
    if symbol == "D": return el_to_idx["H"]
    return el_to_idx.get(symbol, 0)