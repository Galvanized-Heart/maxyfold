import tarfile
import io
import gzip
import click
import rootutils
from pathlib import Path

try:
    import gemmi
except ImportError:
    print("Gemmi not installed.")
    exit(1)

# A smaller filter list to ensure we don't accidentally filter everything
STANDARD_RESIDUES = {'ALA', 'GLY', 'HOH', 'WAT'} 

@click.command()
@click.option("--tar-path", default="data/pdb/raw/assemblies/assemblies_batch_0.tar.gz", required=True, help="Path to ONE .tar.gz file (e.g. data/pdb/raw/assemblies/assemblies_batch_0.tar.gz)")
def main(tar_path):
    root = rootutils.find_root(indicator=".project-root")
    full_path = root / tar_path
    
    print(f"DEBUGGING: {full_path}")
    
    if not full_path.exists():
        print("File not found!")
        return

    try:
        with tarfile.open(full_path, "r:gz") as tar:
            print("1. Tarball opened successfully.")
            
            # Inspect first 5 files
            count = 0
            for member in tar:
                if count >= 5: break
                
                print(f"\n--- Checking Member: {member.name} ---")
                
                if not member.isfile():
                    print("   -> Not a file, skipping.")
                    continue
                
                if not member.name.endswith(".gz"):
                    print("   -> Not a .gz file, skipping.")
                    continue

                count += 1
                
                # 1. Extract
                try:
                    f_obj = tar.extractfile(member)
                    gz_content = f_obj.read()
                    print(f"   -> Extracted {len(gz_content)} bytes of compressed data.")
                except Exception as e:
                    print(f"   -> ERROR extracting: {e}")
                    continue

                # 2. Decompress
                try:
                    with gzip.open(io.BytesIO(gz_content), 'rt') as f_str:
                        cif_string = f_str.read()
                    print(f"   -> Decompressed to {len(cif_string)} characters of CIF text.")
                    print(f"   -> First 50 chars: {cif_string[:50]}...")
                except Exception as e:
                    print(f"   -> ERROR decompressing: {e}")
                    continue

                # 3. Parse Gemmi
                try:
                    doc = gemmi.cif.read_string(cif_string)
                    block = doc[0]
                    print(f"   -> Gemmi Parsed. Block name: {block.name}")
                    
                    # 4. Check Loop
                    loop = block.init_loop("_atom_site.", ["label_comp_id", "type_symbol"])
                    if not loop:
                        print("   -> ERROR: No _atom_site loop found in this CIF!")
                        continue
                    
                    print(f"   -> Found atom loop with {loop.length()} rows.")
                    
                    # 5. Check Content
                    ligands_found = []
                    for i, row in enumerate(loop):
                        res_name = row[0]
                        element = row[1]
                        
                        # Just print the first non-standard thing we see
                        if res_name not in STANDARD_RESIDUES:
                            ligands_found.append(f"{res_name}({element})")
                        
                        if len(ligands_found) >= 5: 
                            break
                    
                    if ligands_found:
                        print(f"   -> SUCCESS! Found potential ligands: {ligands_found}")
                    else:
                        print("   -> Only standard residues (ALA, GLY, HOH) found in first check.")

                except Exception as e:
                    print(f"   -> ERROR parsing CIF with Gemmi: {e}")
                    continue

    except Exception as e:
        print(f"CRITICAL ERROR opening tarball: {e}")

if __name__ == "__main__":
    main()