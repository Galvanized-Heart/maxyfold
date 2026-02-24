import json
import click
import rootutils
from pathlib import Path
from tqdm import tqdm

# Import Gemmi (The standard C++ library for Structural Biology)
try:
    import gemmi
except ImportError:
    print("Error: Gemmi is not installed.")
    print("Please run: pip install gemmi")
    exit(1)

@click.command()
@click.option("--ccd-path", default="data/pdb/raw/ccd/components.cif.gz", help="Path to the raw components.cif.gz")
@click.option("--output-path", default="data/pdb/processed/ccd_smiles.json", help="Where to save the JSON mapping")
def main(ccd_path, output_path):
    """
    Parses the PDB Chemical Component Dictionary (CCD) using Gemmi.
    Extracts a mapping of {ResidueCode: Isometric_SMILES}.
    """

    root = rootutils.find_root(indicator=".project-root")
    inp_file = root / ccd_path
    out_file = root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not inp_file.exists():
        print(f"Error: CCD file not found at {inp_file}")
        print("Run 'maxyfold download --ccd' first.")
        return

    print(f"Loading CCD from {inp_file}...")

    # Parse CIF with Gemmi
    try:
        doc = gemmi.cif.read(str(inp_file))
    except Exception as e:
        print(f"Failed to parse CIF file: {e}")
        return

    print(f"Found {len(doc)} chemical components. Extracting SMILES...")

    mapping = {}
    
    for block in tqdm(doc):
        res_code = block.name
        
        # gemmi.cif.Block.find_loop returns a Loop object or None
        loop = block.find_loop("_pdbx_chem_comp_descriptor.type")
        descriptors = block.find_loop("_pdbx_chem_comp_descriptor.descriptor")
        
        if not loop or not descriptors:
            continue

        best_smiles = None
        
        # Filter for the best SMILES string
        candidates = list(zip(loop, descriptors))
        
        # Canonical (Isomeric) SMILES
        for desc_type, desc_val in candidates:
            if "SMILES_CANONICAL" in desc_type:
                best_smiles = desc_val
                break
        
        # Generic SMILES
        if not best_smiles:
            for desc_type, desc_val in candidates:
                if "SMILES" in desc_type:
                    best_smiles = desc_val
                    break

        if best_smiles:
            best_smiles = best_smiles.strip('"\'')
            mapping[res_code] = best_smiles

    # Save to JSON
    print(f"Extracted SMILES for {len(mapping)} components.")
    print(f"Saving to {out_file}...")
    
    with open(out_file, "w") as f:
        json.dump(mapping, f, indent=None)

    print("Done!")

if __name__ == "__main__":
    main()