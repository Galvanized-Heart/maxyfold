import click
import rootutils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

try:
    import gemmi
except ImportError:
    print("Please run: pip install gemmi")
    exit(1)

@click.command()
@click.option("--ccd-path", default="data/pdb/raw/ccd/components.cif.gz", help="Path to CCD")
@click.option("--output-dir", default="data/pdb/analysis", help="Output directory for plots/csv")
def main(ccd_path, output_dir):
    """
    Analyzes the Chemical Component Dictionary to determine:
    1. Which elements exist and their frequency (Vocabulary Size).
    2. The distribution of ligand sizes (Context Length).
    """
    root = rootutils.find_root(indicator=".project-root")
    inp_file = root / ccd_path
    out_dir = root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading CCD from {inp_file}...")
    doc = gemmi.cif.read(str(inp_file))
    
    # Storage
    element_counts = Counter()
    ligand_sizes = [] # List of ints (num heavy atoms per ligand)
    modified_aa_elements = Counter()
    
    print(f"Analyzing {len(doc)} components...")
    
    for block in tqdm(doc):
        # 1. Determine Type (Ligand vs Modified AA)
        # _chem_comp.type
        comp_type_loop = block.find_loop("_chem_comp.type")
        comp_type = comp_type_loop[0] if comp_type_loop else "NON-POLYMER"
        
        is_modified_aa = "LINKING" in comp_type.upper() or "PEPTIDE" in comp_type.upper()
        
        # 2. Get Atoms
        # _chem_comp_atom.type_symbol
        atom_types = block.find_loop("_chem_comp_atom.type_symbol")
        
        if not atom_types:
            continue
            
        # Filter Hydrogens (AF3/Boltz usually ignore H for tokenization limits)
        heavy_atoms = [a for a in atom_types if a != "H"]
        
        # Update Stats
        ligand_sizes.append(len(heavy_atoms))
        
        for atom in heavy_atoms:
            # Standardize (Pt -> PT, etc)
            atom = atom.upper()
            element_counts[atom] += 1
            
            if is_modified_aa:
                modified_aa_elements[atom] += 1

    # --- PROCESS RESULTS ---

    # 1. Element Frequency DataFrame
    df_elements = pd.DataFrame.from_dict(element_counts, orient='index', columns=['count'])
    df_elements.index.name = 'element'
    df_elements = df_elements.sort_values('count', ascending=False).reset_index()
    
    # Calculate Cumulative %
    total_atoms = df_elements['count'].sum()
    df_elements['percent'] = (df_elements['count'] / total_atoms) * 100
    df_elements['cumulative_percent'] = df_elements['percent'].cumsum()
    
    # Save CSV
    csv_path = out_dir / "atom_counts.csv"
    df_elements.to_csv(csv_path, index=False)
    print(f"\nSaved Element Rankings to {csv_path}")
    
    # 2. Ligand Size DataFrame
    df_sizes = pd.DataFrame(ligand_sizes, columns=['num_atoms'])
    
    # --- VISUALIZATION ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot A: Element Frequency (Log Scale)
    sns.barplot(data=df_elements.head(30), x='element', y='count', ax=axes[0], palette="viridis")
    axes[0].set_yscale("log")
    axes[0].set_title("Top 30 Elements in CCD (Log Scale)")
    axes[0].set_ylabel("Total Occurrences (Log)")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot B: Ligand Size Distribution
    # Cutoff at 99th percentile for visualization clarity
    p99 = np.percentile(ligand_sizes, 99)
    sns.histplot(data=df_sizes[df_sizes['num_atoms'] < p99], x='num_atoms', bins=50, ax=axes[1], color="salmon")
    axes[1].set_title(f"Ligand Size Distribution (Heavy Atoms)\n99% of ligands have < {int(p99)} atoms")
    axes[1].set_xlabel("Number of Heavy Atoms (Tokens)")
    
    plt.tight_layout()
    plot_path = out_dir / "atom_analysis.png"
    plt.savefig(plot_path)
    print(f"Saved plots to {plot_path}")
    
    # Print summary
    print("\n" + "="*40)
    print("ANALYSIS SUMMARY")
    print("="*40)
    print(f"Total Unique Elements Found: {len(df_elements)}")
    print(f"Top 5 Elements: {df_elements['element'].head(5).tolist()}")
    print("-" * 40)
    print("Context Length Decisions:")
    print(f"Average Ligand Size: {np.mean(ligand_sizes):.1f} atoms")
    print(f"Max Ligand Size:     {np.max(ligand_sizes)} atoms")
    print(f"95th Percentile:     {np.percentile(ligand_sizes, 95):.1f} atoms")
    print(f"99th Percentile:     {np.percentile(ligand_sizes, 99):.1f} atoms")
    print("-" * 40)
    print("Modified Amino Acid Check:")
    print(f"Elements found in Modified AAs: {list(modified_aa_elements.keys())}")

if __name__ == "__main__":
    main()