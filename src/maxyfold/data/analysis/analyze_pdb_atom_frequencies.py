import click
import rootutils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import tarfile
import io
import gzip

try:
    import gemmi
except ImportError:
    print("Please run: pip install gemmi")
    exit(1)

# Standard residues to ignore
STANDARD_RESIDUES = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
    'DA','DT','DC','DG','DI','A','U','C','G','I', 
    'UNK', 'HOH', 'DOD', 'WAT', 'SOL'
}

def scan_tar_atoms(tar_path):
    """
    Opens a .tar.gz archive and counts the atomic elements 
    present in LIGAND residues only.
    """
    element_counts = Counter()
    
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".gz"):
                    continue

                try:
                    f_obj = tar.extractfile(member)
                    if f_obj is None: continue
                    
                    # Decompress
                    gz_content = f_obj.read()
                    with gzip.open(io.BytesIO(gz_content), 'rt') as f_str:
                        cif_string = f_str.read()
                    
                    # Parse gemmi
                    doc = gemmi.cif.read_string(cif_string)
                    if not doc: 
                        continue
                    block = doc[0]
                    
                    # Column extraction
                    res_names = block.find_loop("_atom_site.label_comp_id")
                    if not res_names:
                        res_names = block.find_loop("_atom_site.auth_comp_id")
                        
                    elements = block.find_loop("_atom_site.type_symbol")
                    
                    if not res_names or not elements: 
                        continue
                        
                    for res_name, el in zip(res_names, elements):
                        res_name = res_name.strip('"\'')
                        el = el.strip('"\'')
                        
                        # Filter standard residues
                        if res_name in STANDARD_RESIDUES:
                            continue
                            
                        # Filter hydrogens
                        if el.upper() in ["H", "D"]:
                            continue
                            
                        # Count elements
                        element_counts[el.upper()] += 1
                        
                except Exception:
                    continue
                    
    except Exception as e:
        print(f"Error reading tar {tar_path}: {e}")
        return Counter()
        
    return element_counts

@click.command()
@click.option("--pdb-dir", default="data/pdb/raw/assemblies")
@click.option("--output-dir", default="data/pdb/analysis")
@click.option("--workers", default=4, help="Number of CPU cores")
@click.option("--force", is_flag=True, help="Force rescanning even if CSV exists.")
def main(pdb_dir, output_dir, workers, force):
    """
    Scans PDB files to count actual atomic elements in ligands.
    """
    root = rootutils.find_root(indicator=".project-root")
    out = root / output_dir
    out.mkdir(parents=True, exist_ok=True)
    
    csv_path = out / "pdb_ligand_atom_counts.csv"
    df = None

    # Verify cache for existing csv
    if csv_path.exists() and not force:
        print(f"Found existing data at {csv_path}. Loading...")
        try:
            df = pd.read_csv(csv_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading CSV ({e}). Rescanning...")
            df = None

    # Run atom scan through cif files
    if df is None:
        pdb_path_obj = root / pdb_dir
        tar_files = list(pdb_path_obj.glob("*.tar.gz"))
        
        if not tar_files:
            print(f"No tarballs found in {pdb_path_obj}")
            return

        print(f"Scanning {len(tar_files)} batches for ligand ATOMS...")
        
        total_atom_counts = Counter()
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(scan_tar_atoms, tar_files), total=len(tar_files)))
            
        for res in results:
            total_atom_counts.update(res)

        df = pd.DataFrame(list(total_atom_counts.items()), columns=['element', 'count'])
        
        if not df.empty:
            df.to_csv(csv_path, index=False)
        else:
            print("No ligand atoms found! (Check input data)")
            return

    # Plot results
    df = df.sort_values('count', ascending=False).reset_index(drop=True)
    
    print(f"\nTotal Ligand Heavy Atoms Counted: {df['count'].sum()}")
    print("\nTop 30 Elements in Dataset:")
    print(df.head(30))

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df.head(30), x='element', y='count', palette='viridis')
    plt.yscale('log')
    plt.title("Actual Element Frequency in Ligand Dataset")
    plt.xlabel("Element")
    plt.ylabel("Count (Log Scale)")
    plt.tight_layout()
    plot_path = out / "pdb_ligand_atom_frequencies.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    main()