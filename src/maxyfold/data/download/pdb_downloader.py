import asyncio
import aiohttp
import aiofiles
import requests
from pathlib import Path
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Tuple, Dict, Any

class PDBDownloader:
    """
    Handles finding PDB IDs, downloading biological assemblies,
    and fetching the Chemical Component Dictionary.
    """

    def __init__(self, query_cfg: Dict[str, Any] = None):
        """
        Initializes the downloader with a query configuration.
        """
        self.query_cfg = query_cfg or {}
        self.search_request = self._build_search_request()

    def _build_search_request(self) -> dict:
        """Dynamically builds the RCSB search JSON payload from the config."""
        # Default fallback if no config is provided
        filters = self.query_cfg.get("filters", {})
        nodes = []

        if "max_release_date" in filters:
            nodes.append({
                "type": "terminal", "service": "text", 
                "parameters": {"attribute": "rcsb_accession_info.initial_release_date", "operator": "less_or_equal", "value": filters["max_release_date"]}
            })
        if "max_resolution" in filters:
            nodes.append({
                "type": "terminal", "service": "text", 
                "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": filters["max_resolution"]}
            })
        if "method" in filters:
            nodes.append({
                "type": "terminal", "service": "text", 
                "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": filters["method"]}
            })

        # If no filters were provided, do a basic query to prevent API failure
        if not nodes:
            nodes.append({"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 9.0}})

        paginate = self.query_cfg.get("paginate", {"start": 0, "rows": 10000})
        sort = self.query_cfg.get("sort", [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}])

        return {
            "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
            "return_type": "entry",
            "request_options": {
                "paginate": dict(paginate),
                "sort": [dict(s) for s in sort]
            }
        }

    def fetch_filtered_ids(self, output_file: Path):
        """Queries the RCSB Search API to get PDB IDs and saves them to a file."""
        print("Starting PDB ID fetch...")
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists():
            print(f"\nList of PDB IDs already exists at {output_file}. Skipping fetch.")
            return

        pdb_ids = set()
        start = 0
        total_count = -1
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

        try:
            with tqdm(desc="Fetching PDB IDs") as pbar:
                while total_count == -1 or start < total_count:
                    self.search_request["request_options"]["paginate"]["start"] = start
                    response = requests.post(search_url, json=self.search_request)
                    response.raise_for_status()

                    data = response.json()
                    if total_count == -1:
                        total_count = data.get("total_count", 0)
                        pbar.total = total_count

                    result_set = data.get("result_set", [])
                    if not result_set: break

                    current_ids = {result["identifier"] for result in result_set}
                    pdb_ids.update(current_ids)
                    
                    start += len(current_ids)
                    pbar.update(len(current_ids))
            
            sorted_ids = sorted(list(pdb_ids))
            with open(output_file, "w") as f:
                for pdb_id in sorted_ids:
                    f.write(f"{pdb_id}\n")
            
            print(f"\nSuccessfully fetched {len(sorted_ids)} PDB IDs to {output_file}")

        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred during API request: {e}")

    async def download_assemblies(self, pdb_ids: List[str], output_dir: Path, log_file_name: str = "download_log.txt", limit: int = 100):
        """Asynchronously downloads biological assemblies."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / log_file_name
        
        base_url = "https://files.rcsb.org/download/"
        tasks_to_run = []
        
        for pdb_id in pdb_ids:
            filename = f"{pdb_id.lower()}-assembly1.cif.gz"
            filepath = output_dir / filename
            if not filepath.exists():
                url = f"{base_url}{filename}"
                tasks_to_run.append((url, filepath))

        if not tasks_to_run:
            print("All requested files in this batch already exist.")
            return

        print(f"Downloading {len(tasks_to_run)} files to {output_dir}...")
        
        connector = aiohttp.TCPConnector(limit=limit)
        async with aiohttp.ClientSession(connector=connector) as session:
            pbar = async_tqdm(total=len(tasks_to_run), desc="Downloading")
            coroutines = [self._download_single_file(session, url, path, pbar) for url, path in tasks_to_run]
            results = await asyncio.gather(*coroutines)
            pbar.close()

        failed = [res for res in results if res[1] != "Success"]
        
        if failed:
            print(f"    Found {len(failed)} failed downloads. Logging to {log_path}")
            with open(log_path, "a") as f:
                f.write(f"Batch Report:\n")
                for url, status in failed:
                    f.write(f"{url}: {status}\n")

    async def _download_single_file(self, session: aiohttp.ClientSession, url: str, path: Path, pbar: async_tqdm) -> Tuple[str, str]:
        """Coroutine to download a single file."""
        try:
            timeout = aiohttp.ClientTimeout(total=300)
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.read()
                    async with aiofiles.open(path, 'wb') as f:
                        await f.write(content)
                    pbar.update(1)
                    return url, "Success"
                else:
                    pbar.update(1)
                    return url, f"Failed (HTTP {response.status})"
        except Exception as e:
            pbar.update(1)
            return url, f"Failed ({type(e).__name__})"

    def download_ccd(self, output_dir: Path):
        """Downloads the Chemical Component Dictionary."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://files.rcsb.org/pub/pdb/data/monomers/components.cif.gz"
        file_path = output_dir / "components.cif.gz"

        print(f"Starting CCD download from: {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(file_path, "wb") as f, tqdm(
                desc="Downloading CCD", total=total_size, unit='iB', unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    pbar.update(len(chunk))
                    f.write(chunk)
            
            print(f"\nSuccessfully downloaded CCD to {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"\nFAILED! An error occurred during CCD download: {e}")