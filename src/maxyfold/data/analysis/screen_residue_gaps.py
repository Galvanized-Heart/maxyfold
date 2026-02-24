import click
import tarfile
import json
import gzip
import io
import warnings
import rootutils
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress biopython warnings
warnings.simplefilter('ignore', PDBConstructionWarning)

def get_gaps_for_structure(cif_content: bytes, pdb_id: str) -> dict:
    """
    Parses a CIF stream and finds residue numbering gaps in polypeptide chains.
    """
    parser = MMCIFParser(QUIET=True)
    try:
        # Parse in memory
        with gzip.open(io.BytesIO(cif_content), 'rt') as f:
            structure = parser.get_structure(pdb_id, f)
        
        gaps_report = {}
        model = list(structure.get_models())[0] # Assume model 0

        for chain in model:
            # 1. Extract protein residues (must have CA atom)
            # Use tuple (res_seq, insertion_code) for sorting
            residues = []
            for res in chain:
                if 'CA' in res:
                    # id[1] is sequence number, id[2] is insertion code
                    residues.append(res.id[1])
            
            if not residues:
                continue
            
            # Sort residues by number to ensure correct order
            residues.sort()
            
            chain_gaps = []
            total_missing = 0
            
            # 2. Scan for discontinuities
            for i in range(len(residues) - 1):
                current_id = residues[i]
                next_id = residues[i+1]
                
                # If they are not consecutive (e.g. 50 then 52)
                if next_id > current_id + 1:
                    gap_len = next_id - current_id - 1
                    # Record the gap: (Start Res, End Res, Length)
                    chain_gaps.append([current_id, next_id, gap_len])
                    total_missing += gap_len

            if chain_gaps:
                gaps_report[chain.get_id()] = {
                    "num_gaps": len(chain_gaps),
                    "total_missing_residues": total_missing,
                    "gap_details": chain_gaps, # List of [start, end, length]
                    "chain_length_present": len(residues)
                }

        return gaps_report

    except Exception as e:
        return None

@click.command()
@click.option("--max-files", default=0, help="Limit number of files to scan (0 for all). Useful for testing.")
@click.option("--batch-limit", default=0, help="Limit number of tarballs to scan.")
def main(max_files, batch_limit):
    """
    Scans raw PDB tarballs for missing residues (sequence gaps).
    Outputs a JSON report to data/pdb/analysis/gaps.json.
    """
    root = rootutils.find_root(indicator=".project-root")
    raw_dir = root / "data/pdb/raw/assemblies"
    out_dir = root / "data/pdb/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tar_files = sorted(list(raw_dir.glob("assemblies_batch_*.tar.gz")))
    
    if batch_limit > 0:
        tar_files = tar_files[:batch_limit]

    print(f"Found {len(tar_files)} tarballs. Starting gap analysis...")

    # Data structure to hold results
    # { "1abc": { "A": { ... } } }
    full_report = {}
    
    scanned_count = 0
    gapped_proteins_count = 0
    total_gaps_found = 0

    for tar_path in tqdm(tar_files, desc="Processing Tarballs"):
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                
                for member in members:
                    if max_files > 0 and scanned_count >= max_files:
                        break
                        
                    pdb_id = member.name.split("-")[0]
                    f_obj = tar.extractfile(member)
                    
                    if f_obj:
                        content = f_obj.read()
                        gaps = get_gaps_for_structure(content, pdb_id)
                        
                        scanned_count += 1
                        
                        if gaps:
                            full_report[pdb_id] = gaps
                            gapped_proteins_count += 1
                            total_gaps_found += sum(c['num_gaps'] for c in gaps.values())
        except Exception as e:
            print(f"Error reading tarball {tar_path.name}: {e}")

        if max_files > 0 and scanned_count >= max_files:
            break

    # --- Save JSON ---
    out_json = out_dir / "gaps_report.json"
    with open(out_json, "w") as f:
        json.dump(full_report, f, indent=2)

    # --- Print Summary ---
    print("\n" + "="*40)
    print(f"Gap Analysis Complete")
    print("="*40)
    print(f"Total Structures Scanned: {scanned_count}")
    print(f"Structures with Gaps:     {gapped_proteins_count} ({gapped_proteins_count/scanned_count*100:.1f}%)")
    print(f"Total Gaps Found:         {total_gaps_found}")
    print(f"Detailed report saved to: {out_json}")
    
    # --- Quick Pandas Analysis (Optional) ---
    if full_report:
        print("\nTop 5 Worst Offenders (Most missing residues):")
        flat_data = []
        for pid, chains in full_report.items():
            for cid, data in chains.items():
                flat_data.append({
                    "pdb": pid, 
                    "chain": cid, 
                    "missing": data['total_missing_residues'],
                    "num_gaps": data['num_gaps']
                })
        
        df = pd.DataFrame(flat_data)
        if not df.empty:
            print(df.sort_values("missing", ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    main()