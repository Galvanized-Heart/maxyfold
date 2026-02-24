import lmdb
import rootutils
import numpy as np
import click
from pathlib import Path
from safetensors.numpy import load

from maxyfold.data.constants import restypes, elements, LIGAND_IDX

@click.command()
@click.option("--db-name", default="pdb_dataset.lmdb", help="Name of the LMDB file.")
@click.option("--num-entries", default=1, help="How many entries to inspect.")
@click.option("--verbose-atoms", default=False, help="Bool to see exhaustive list of mask, atom type, and coords.")
def main(db_name, num_entries, verbose_atoms):
    root = rootutils.find_root(indicator=".project-root")
    db_path = root / "data" / "pdb" / "processed" / db_name
    
    if not db_path.exists():
        print(f"Dataset not found at {db_path}!")
        print("Did you run `maxyfold process` yet?")
        return

    print(f"Opening LMDB at {db_path}...\n")
    
    env = lmdb.open(str(db_path), readonly=True, subdir=False, lock=False)

    print(f"Total unique entries: {env.stat()['entries']}")
    
    with env.begin() as txn:
        cursor = txn.cursor()
        
        count = 0
        for key, value in cursor:
            if count >= num_entries:
                break
                
            pdb_id = key.decode('ascii')
            
            data = load(value) 
            
            print(f"Chain IDs for {pdb_id}: {data['chain_ids'].shape}\n{data['chain_ids']}\n\n")

            mapped_res = np.array(restypes)[data['res_type']]
            print(f"Residue Type: {data['res_type'].shape}\n{mapped_res}\n\n")

            if verbose_atoms:
                for idx, elems in enumerate(data['atom_elements']):
                    mapped_elems = np.array(elements)[elems]
                    mask_elem_coords = np.hstack((data['mask'][idx].reshape(-1, 1), mapped_elems.reshape(-1, 1), data['coords'][idx])) 
                    print(f"Mask, Atom Element, Coords {idx}: {mask_elem_coords.shape}\n{mask_elem_coords}\n\n")

    return

if __name__ == "__main__":
    main()