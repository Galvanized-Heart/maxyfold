import json
import click
import rootutils
from pathlib import Path
from tqdm import tqdm

try:
    import gemmi
except ImportError:
    print("Error: Gemmi is not installed. Run: pip install gemmi")
    exit(1)

@click.command()
@click.option("--ccd-path", default="data/pdb/raw/ccd/components.cif.gz", help="Path to components.cif.gz")
@click.option("--output-path", default="data/pdb/processed/ccd_atoms.json", help="Where to save the JSON mapping")
def main(ccd_path, output_path):
    """
    Parses the PDB CCD to extract the exact list of heavy atoms for every ligand.
    Output: { "HEM": [ ["CHA", "C"], ["FE", "FE"], ... ] }
    """
    root = rootutils.find_root(indicator=".project-root")
    inp_file = root / ccd_path
    out_file = root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not inp_file.exists():
        print(f"Error: CCD file not found at {inp_file}")
        return

    print(f"Loading CCD from {inp_file}...")
    doc = gemmi.cif.read(str(inp_file))
    print(f"Found {len(doc)} chemical components. Extracting atoms...")

    mapping = {}
    
    for block in tqdm(doc):
        res_code = block.name
        
        # We want the atom names and their element types
        atom_names = block.find_loop("_chem_comp_atom.atom_id")
        elements = block.find_loop("_chem_comp_atom.type_symbol")
        
        if not atom_names or not elements:
            continue

        heavy_atoms = []
        for atom_name, el in zip(atom_names, elements):
            atom_name = atom_name.strip('"\'')
            el = el.strip('"\'').upper()
            
            # Skip hydrogens. We only tokenize heavy atoms.
            if el in ["H", "D"]:
                continue
                
            heavy_atoms.append([atom_name, el])

        if heavy_atoms:
            mapping[res_code] = heavy_atoms

    print(f"Extracted atom lists for {len(mapping)} components.")
    print(f"Saving to {out_file}...")
    
    with open(out_file, "w") as f:
        # Save compact JSON to save space
        json.dump(mapping, f, separators=(',', ':'))

    print("Done!")

if __name__ == "__main__":
    main()