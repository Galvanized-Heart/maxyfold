import click
import rootutils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tarfile
import io
import gzip
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

try:
    import gemmi
except ImportError:
    print("Please run: pip install gemmi")
    exit(1)

# Water is usually excluded from "Context Length" calculations in training
EXCLUDED_RESIDUES = {'HOH', 'DOD', 'WAT'}

def analyze_tar_contents(tar_path):
    """
    Returns a list of dicts:
    [
      {'pdb': '101m', 'n_poly': 154, 'n_lig': 45, 'elements': {'C', 'N', 'O', 'FE'}},
      ...
    ]
    """
    results = []
    
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".gz"):
                    continue
                
                pdb_id = member.name.split("-")[0]
                
                try:
                    f_obj = tar.extractfile(member)
                    if f_obj is None: continue
                    
                    gz_content = f_obj.read()
                    with gzip.open(io.BytesIO(gz_content), 'rt') as f_str:
                        cif_string = f_str.read()
                    
                    doc = gemmi.cif.read_string(cif_string)
                    if not doc: continue
                    block = doc[0]
                    
                    # 1. Count Polymer Tokens (Residues in chains)
                    # We look at _entity_poly_seq to get the exact sequence length
                    # This is faster/safer than counting atoms
                    n_poly = 0
                    poly_loop = block.find_loop("_entity_poly_seq.mon_id")
                    if poly_loop:
                        n_poly = len(poly_loop)
                    
                    # 2. Count Ligand Tokens (1 Token per Heavy Atom)
                    # We need to scan atoms for non-polymers
                    n_lig = 0
                    elements_present = set()
                    
                    atom_site = block.init_loop("_atom_site.", ["group_PDB", "label_comp_id", "type_symbol"])
                    
                    for row in atom_site:
                        group, res_name, el = row
                        el = el.strip().upper()
                        
                        # Track Elements (globally for the file)
                        if el not in ["H", "D"]: # Ignore Hydrogens
                            elements_present.add(el)
                        
                        # Count Ligand Tokens
                        # Logic: If it's HETATM and NOT Water -> It's a ligand token
                        # Note: In PDBx, even polymers use ATOM, non-polymers use HETATM usually.
                        # Ideally we check entity type, but this is a good heuristic for scanning.
                        if group == "HETATM":
                            if res_name not in EXCLUDED_RESIDUES and el not in ["H", "D"]:
                                n_lig += 1

                    results.append({
                        'pdb': pdb_id,
                        'n_poly': n_poly,
                        'n_lig': n_lig,
                        'total_tokens': n_poly + n_lig,
                        'elements': ",".join(sorted(list(elements_present)))
                    })
                        
                except Exception:
                    continue
                    
    except Exception as e:
        print(f"Error reading {tar_path}: {e}")
        
    return results

@click.command()
@click.option("--pdb-dir", default="data/pdb/raw/assemblies")
@click.option("--output-dir", default="data/pdb/analysis")
@click.option("--workers", default=4)
def main(pdb_dir, output_dir, workers):
    root = rootutils.find_root(indicator=".project-root")
    out = root / output_dir
    out.mkdir(parents=True, exist_ok=True)
    
    tar_files = list((root / pdb_dir).glob("*.tar.gz"))
    if not tar_files:
        print("No tarballs found.")
        return

    print(f"Analyzing dimensions of {len(tar_files)} batches...")
    
    all_data = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        batch_results = list(tqdm(executor.map(analyze_tar_contents, tar_files), total=len(tar_files)))
        
    for batch in batch_results:
        all_data.extend(batch)
        
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("No data extracted.")
        return
        
    print(f"Analyzed {len(df)} structures.")
    
    # --- ANALYSIS 1: Context Length (Total Tokens) ---
    print("\n--- Context Length Stats (Polymers + Ligands) ---")
    print(df['total_tokens'].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_tokens'], bins=100, log_scale=(True, False))
    plt.axvline(x=df['total_tokens'].quantile(0.95), color='r', linestyle='--', label='95th Percentile')
    plt.title("Distribution of Total Tokens per Structure")
    plt.xlabel("Number of Tokens (Residues + Ligand Atoms)")
    plt.legend()
    plt.savefig(out / "token_length_distribution.png")
    
    # --- ANALYSIS 2: Element Coverage ---
    # Convert element string back to sets
    df['el_set'] = df['elements'].apply(lambda x: set(x.split(',')))
    
    # Get master list of all elements sorted by frequency (from previous script logic)
    # We re-calculate frequency here quickly
    all_els = [e for row in df['el_set'] for e in row]
    el_counts = Counter(all_els)
    # Be careful with Sodium "NA" string being NaN in lists, though set strings handle it better
    sorted_elements = [x[0] for x in el_counts.most_common()]
    
    coverage_stats = []
    
    # Simulate: "If we support Top K elements, how many PDBs are fully valid?"
    # We check K from 5 to len(sorted_elements)
    for k in range(5, len(sorted_elements) + 1):
        allowed = set(sorted_elements[:k])
        
        # Check if a structure's elements are a SUBSET of allowed
        # Set issubset is fast
        valid_mask = df['el_set'].apply(lambda s: s.issubset(allowed))
        coverage = valid_mask.mean() * 100
        
        coverage_stats.append({
            'k': k,
            'last_added': sorted_elements[k-1],
            'coverage_pct': coverage,
            'structures_lost': len(df) - valid_mask.sum()
        })
        
        # Optimization: If coverage > 99.99%, we can probably stop for the chart
        if coverage > 99.999:
             # Continue to finish the list for the CSV, but maybe break loop if massive
             pass

    df_cov = pd.DataFrame(coverage_stats)
    df_cov.to_csv(out / "element_coverage_analysis.csv", index=False)
    
    print("\n--- Element Coverage Analysis ---")
    print(df_cov.iloc[::5].to_string(index=False)) # Print every 5th row
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_cov, x='k', y='coverage_pct', marker='o')
    plt.title("Dataset Coverage vs. Vocabulary Size")
    plt.xlabel("Number of Allowed Elements (Top K)")
    plt.ylabel("% of PDB Structures Fully Covered")
    plt.ylim(90, 100.1) # Zoom in on the top
    plt.grid(True)
    plt.savefig(out / "element_coverage_curve.png")
    
    print(f"\nAnalysis complete. Check {out}")

if __name__ == "__main__":
    main()