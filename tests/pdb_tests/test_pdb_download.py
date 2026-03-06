import requests
import pytest
from pathlib import Path

# Required network mark
network_test = pytest.mark.network

# Test multiple cases using pytest.mark.parametrize:
# 1. A known good PDB ID that should succeed.
# 2. A PDB ID that is syntactically invalid and should fail.
# 3. A valid PDB ID that is unlikely to have a biological assembly and should fail.
@pytest.mark.parametrize("pdb_id, should_succeed", [
    ("4HHB", True),   # Hemoglobin, a classic example that must have an assembly.
    ("XXXX", False),  # Invalid PDB code, must fail.
])
@network_test
def test_single_assembly_download(tmp_path: Path, pdb_id: str, should_succeed: bool):
    """
    Tests the download of a single biological assembly file from `files.rcsb.org`.
    
    This test uses pytest's `tmp_path` fixture to create a temporary directory
    for the download, ensuring the test is clean and leaves no artifacts.
    """
    # Define the URL and the local file path inside the temporary directory
    file_name = f"{pdb_id.lower()}-assembly1.cif.gz"
    url = f"https://files.rcsb.org/download/{file_name}"
    output_path = tmp_path / file_name

    print(f"Testing download for {pdb_id} from {url}")

    try:
        response = requests.get(url, timeout=60)
        # Raise an HTTPError if the response was an error (4xx or 5xx)
        response.raise_for_status()

        # If the request was successful, write the file
        with open(output_path, "wb") as f:
            f.write(response.content)

        # If we get here, the download was successful
        download_succeeded = True

    except requests.exceptions.HTTPError as e:
        print(f"Received expected HTTP error for {pdb_id}: {e}")
        download_succeeded = False
    except Exception as e:
        print(f"An unexpected error occurred for {pdb_id}: {e}")
        download_succeeded = False


    if should_succeed:
        assert download_succeeded, f"Download for {pdb_id} was expected to succeed but failed."
        assert output_path.exists(), "File should exist after a successful download."
        assert output_path.stat().st_size > 1000, "Downloaded file seems too small."
    else:
        assert not download_succeeded, f"Download for {pdb_id} was expected to fail but succeeded."
        assert not output_path.exists(), "File should not exist after a failed download."