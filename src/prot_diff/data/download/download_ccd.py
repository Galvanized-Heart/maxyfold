import requests
import os
from pathlib import Path
import rootutils

def download_ccd():
    """Downloads the full Chemical Component Dictionary from the PDB archive."""
    root_path = rootutils.find_root(indicator=".project-root")
    output_dir = root_path / "data" / "pdb" / "raw" / "ccd"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://files.rcsb.org/pub/pdb/data/monomers/components.cif.gz"
    file_path = output_dir / "components.cif.gz"

    print(f"Downloading Chemical Component Dictionary from:\n  {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, "wb") as f:
            print(f"Saving to: {file_path}")
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded_size += len(chunk)
                print(f"\r  -> Progress: {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB", end="")

        print("\n\nSUCCESS! CCD downloaded successfully.")

    except requests.exceptions.RequestException as e:
        print(f"\n\nFAILED! An error occurred during download: {e}")

if __name__ == "__main__":
    download_ccd()