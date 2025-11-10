import requests
import json
from tqdm import tqdm
import rootutils

# Define the search query based on Boltz-1 criteria
SEARCH_REQUEST = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "less_or_equal",
                    "value": "2021-09-30T23:59:59Z" # Entries from before Sep 30th, 2021 (same cutoff for training data as AlphaFold2)
                }
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": 9.0 # Maximum resolution tolerance of 9 Angstroms
                }
            },
            #{ # NOTE: Not sure if this filter is necessary
            #    "type": "terminal",
            #    "service": "text",
            #    "parameters": {
            #        "attribute": "exptl.method",
            #        "operator": "exact_match",
            #        "value": "X-RAY DIFFRACTION" # Only XRD structures
            #     }
            #}
        ]
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {
            "start": 0,
            "rows": 10_000 # Max rows per request
        },
        "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}]
    }
}

def get_pdb_ids():
    """
    Queries the RCSB Search API to get PDB IDs based on release date and resolution.
    """
    pdb_ids = set()
    start = 0
    total_count = -1

    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    with tqdm(desc="Fetching PDB IDs") as pbar:
        while total_count == -1 or start < total_count:
            SEARCH_REQUEST["request_options"]["paginate"]["start"] = start
            
            response = requests.post(search_url, json=SEARCH_REQUEST)
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                print(response.text)
                break

            data = response.json()
            if total_count == -1:
                total_count = data["total_count"]
                pbar.total = total_count

            current_ids = {result["identifier"] for result in data["result_set"]}
            pdb_ids.update(current_ids)
            
            start += len(current_ids)
            pbar.update(len(current_ids))

            if not data["result_set"]:
                break
                
    return sorted(list(pdb_ids))

if __name__ == "__main__":
    print("Starting PDB ID fetch...")
    ids = get_pdb_ids()
    
    if ids:
        root_path = rootutils.find_root(indicator=".project-root")
        output_file = root_path / "data" / "pdb" / "pdb_ids.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            for pdb_id in ids:
                f.write(f"{pdb_id}\n")
        print(f"\nSuccessfully fetched {len(ids)} PDB IDs and saved to {output_file}")
    else:
        print("No PDB IDs were fetched. Please check the script and your connection.")