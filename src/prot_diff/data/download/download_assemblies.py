import asyncio
import aiohttp
import aiofiles
import os
import argparse
from tqdm.asyncio import tqdm
from pathlib import Path
import rootutils

async def download_file(session, url: str, path: Path):
    """
    Coroutine to download a single file.
    Returns a tuple of (url, status_string).
    """
    try:
        timeout = aiohttp.ClientTimeout(total=300) # 5 minutes
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                content = await response.read()
                async with aiofiles.open(path, 'wb') as f:
                    await f.write(content)
                return url, "Success"
            else:
                return url, f"Failed (HTTP {response.status})"
    except asyncio.TimeoutError:
        return url, "Failed (TimeoutError)"
    except Exception as e:
        return url, f"Failed ({type(e).__name__})"

async def main(input_file: str, limit: int):
    """Main function to orchestrate the download process."""
    root_path = rootutils.find_root(indicator=".project-root")
    output_path = root_path / "data" / "pdb" / "raw" / "assemblies"
    log_path = output_path / "download_log.txt"
    input_file_path = root_path / input_file
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read PDB IDs from the input file
    with open(input_file, "r") as f:
        pdb_ids = [line.strip().upper() for line in f if line.strip()]

    base_url = "https://files.rcsb.org/download/"
    
    tasks_to_run = []
    for pdb_id in pdb_ids:
        file_name = f"{pdb_id.lower()}-assembly1.cif.gz"
        url = f"{base_url}{file_name}"
        file_path = output_path / file_name
        
        # Only add the task if the file doesn't already exist
        if not file_path.exists():
            tasks_to_run.append((url, file_path))

    if not tasks_to_run:
        print("All required assembly files already exist. Nothing to download.")
        return

    print(f"Starting download of {len(tasks_to_run)} biological assemblies.")
    print(f"Concurrency limit set to: {limit}")
    
    failed_downloads = []
    
    connector = aiohttp.TCPConnector(limit=limit)
    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=len(tasks_to_run), desc="Downloading")
        
        results = []
        for url, path in tasks_to_run:
            task = asyncio.create_task(download_file(session, url, path))
            task.add_done_callback(lambda p: pbar.update(1))
            results.append(task)
            
        final_results = await asyncio.gather(*results)
        pbar.close()

    for url, status in final_results:
        if status != "Success":
            failed_downloads.append(f"{url}: {status}")

    success_count = len(tasks_to_run) - len(failed_downloads)
    print(f"\nDownload complete. {success_count}/{len(tasks_to_run)} files downloaded successfully.")

    if failed_downloads:
        print(f"Found {len(failed_downloads)} failed downloads. See {log_path} for details.")
        with open(log_path, "w") as f:
            for line in failed_downloads:
                f.write(f"{line}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch download biological assemblies from the RCSB PDB.")
    parser.add_argument("-f", "--input-file", type=str, required=True, help="Path to the text file containing PDB IDs, one per line.")
    parser.add_argument("-l", "--limit", type=int, default=100, help="Maximum number of concurrent downloads.")
    
    args = parser.parse_args()

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main(args.input_file, args.limit))