import json
import click
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import rootutils
from pathlib import Path

@click.command()
@click.option("--json-path", default="data/pdb/analysis/gaps_report.json", help="Path to the JSON report.")
def main(json_path):
    """
    Analyzes the gaps_report.json to generate statistics and visualizations
    regarding missing residues in the PDB dataset.
    """
    root = rootutils.find_root(indicator=".project-root")
    json_full_path = root / json_path
    plots_dir = root / "data/pdb/analysis/plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not json_full_path.exists():
        print(f"Error: {json_full_path} not found. Run screen_gaps.py first.")
        return

    print("Loading JSON report...")
    with open(json_full_path, "r") as f:
        report = json.load(f)

    # --- 1. Flatten Data into DataFrames ---
    # We need two views: Gap-Level (one row per gap) and Chain-Level (one row per protein chain)
    
    gap_rows = []
    chain_rows = []

    for pdb_id, chains in report.items():
        for chain_id, data in chains.items():
            # Chain Level Data
            max_gap = 0
            if data['gap_details']:
                max_gap = max(g[2] for g in data['gap_details'])
            
            chain_rows.append({
                "pdb_id": pdb_id,
                "chain": chain_id,
                "num_gaps": data['num_gaps'],
                "total_missing": data['total_missing_residues'],
                "present_length": data['chain_length_present'],
                "max_gap_size": max_gap,
                "missing_percent": (data['total_missing_residues'] / (data['chain_length_present'] + data['total_missing_residues'])) * 100
            })

            # Gap Level Data
            for gap in data['gap_details']:
                # gap is [start, end, length]
                gap_rows.append({
                    "pdb_id": pdb_id,
                    "length": gap[2]
                })

    df_chains = pd.DataFrame(chain_rows)
    df_gaps = pd.DataFrame(gap_rows)

    print(f"Loaded data for {len(df_chains)} gapped chains containing {len(df_gaps)} individual gaps.")

    # --- 2. Statistical Summary ---
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY")
    print("="*50)
    
    print("\n--- Individual Gap Sizes ---")
    print(df_gaps['length'].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    
    print("\n--- Gaps Per Chain ---")
    print(df_chains['num_gaps'].describe(percentiles=[0.5, 0.75, 0.9, 0.99]).to_string())

    # --- 3. The "Recovery" Analysis ---
    # This answers: "If I fix gaps <= X size, how many proteins do I keep?"
    
    print("\n" + "="*50)
    print("RECOVERY ANALYSIS (Filtering Strategies)")
    print("="*50)
    print(f"{'Max Gap Threshold':<20} | {'Proteins Recoverable':<20} | {'% of Gapped Set':<15}")
    print("-" * 65)

    thresholds = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100]
    recovery_data = []

    for t in thresholds:
        # Filter: Keep chain if ALL its gaps are <= t
        # Equivalent to: max_gap_size <= t
        recoverable = df_chains[df_chains['max_gap_size'] <= t]
        count = len(recoverable)
        pct = (count / len(df_chains)) * 100
        print(f"<= {t:<17} | {count:<20} | {pct:.1f}%")
        recovery_data.append({"threshold": t, "pct": pct})

    # --- 4. Visualization ---
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Missing Residue Analysis', fontsize=16)

    # Plot A: Distribution of Gap Sizes (Log Scale)
    # Most gaps are small, log scale helps see the long tail
    sns.histplot(df_gaps['length'], bins=50, ax=axes[0, 0], log_scale=(True, False), color='skyblue')
    axes[0, 0].set_title('Distribution of Individual Gap Sizes (Log X)')
    axes[0, 0].set_xlabel('Gap Length (Residues)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(x=4, color='r', linestyle='--', label='Loop Fix Limit (4)')
    axes[0, 0].legend()

    # Plot B: Number of Gaps per Chain
    sns.histplot(df_chains['num_gaps'], bins=range(1, 20), ax=axes[0, 1], color='salmon', discrete=True)
    axes[0, 1].set_title('Fragmentation: Number of Gaps per Chain')
    axes[0, 1].set_xlabel('Number of Gaps')
    axes[0, 1].set_xlim(1, 15)

    # Plot C: Max Gap Size vs Missing Percentage
    # Helps identify "Swiss Cheese" (Low Max Gap, High Missing %) vs "Grand Canyon" (High Max Gap)
    sns.scatterplot(data=df_chains, x='max_gap_size', y='missing_percent', alpha=0.3, ax=axes[1, 0], color='purple')
    axes[1, 0].set_title('Max Gap Size vs. Total Missing %')
    axes[1, 0].set_xlabel('Largest Gap in Chain')
    axes[1, 0].set_ylabel('Total % of Chain Missing')
    axes[1, 0].set_xscale('log')

    # Plot D: Recovery Curve
    # Visualizing the table printed above
    rec_df = pd.DataFrame(recovery_data)
    sns.lineplot(data=rec_df, x='threshold', y='pct', marker='o', ax=axes[1, 1], color='green', linewidth=2)
    axes[1, 1].set_title('Recovery Curve: If we fix gaps <= X...')
    axes[1, 1].set_xlabel('Max Gap Size Threshold')
    axes[1, 1].set_ylabel('% of Gapped Proteins Recovered')
    axes[1, 1].set_xticks(thresholds)
    axes[1, 1].grid(True)

    plt.tight_layout()
    save_path = plots_dir / "gap_analysis.png"
    plt.savefig(save_path)
    print(f"\nPlots saved to {save_path}")

if __name__ == "__main__":
    main()