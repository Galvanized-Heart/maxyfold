import asyncio

import math
import os
from pathlib import Path
import tarfile

import click

from maxyfold.data import PDBDownloader

@click.group()
def cli():
    pass


# Test example of click cli
@cli.command()
@click.option('--name', default='User', help='The name to greet.')
def hello(name):
    """A simple hello command to test the CLI setup."""
    click.echo(f"Hello, {name}! Welcome to MaxyFold! :D")


# Download structural data from PDB
@cli.command()
@click.option("--ids", "-i", is_flag=True, help="Download the filtered list of PDB IDs.")
@click.option("--assemblies", "-a", is_flag=True, help="Download the biological assembly files.")
@click.option("--ccd", "-c", is_flag=True, help="Download the Chemical Component Dictionary.")
@click.option("--batch-size", type=int, default=20000, help="Number of biological assembly files to download before tarring/cleaning.")
def download(ids, assemblies, ccd, batch_size):
    """Command for downloading raw PDB ids, ccd, and assemblies to `/data/pdb/raw/`."""
    if not (ids or assemblies or ccd):
        click.echo("No specific components requested. Defaulting to download all.")
        ids = True
        assemblies = True
        ccd = True

    # Initialize downloader 
    downloader = PDBDownloader()

    if ids:
        click.echo("Fetching PDB IDs...")
        downloader.fetch_filtered_ids()
    
    if ccd:
        click.echo("\nDownloading CCD...")
        downloader.download_ccd()
        
    if assemblies:
        id_file_path = downloader.root_path / "data/pdb/raw/pdb_ids.txt"
        if not id_file_path.exists():
            click.echo(click.style("\nERROR: PDB ID list not found. Run with '--ids' first.", fg="red"))
            exit(1)

        # Read PDB ids
        with open(id_file_path, "r") as f:
            all_pdb_ids = [line.strip().upper() for line in f if line.strip()]
        
        total_files = len(all_pdb_ids)
        num_batches = math.ceil(total_files / batch_size)
        
        raw_dir_rel = "data/pdb/raw/assemblies"
        raw_dir_abs = downloader.root_path / raw_dir_rel
        
        click.echo(f"\nProcessing {total_files} structures in {num_batches} batches of {batch_size}...")

        # Batch process assemblies
        for i in range(num_batches):
            # Check for existing batches
            tar_name = f"assemblies_batch_{i}.tar.gz"
            tar_path = raw_dir_abs / tar_name
            if tar_path.exists():
                click.echo(f"Batch {i+1}/{num_batches} already completed ({tar_name} exists). Skipping.")
                continue

            start = i * batch_size
            end = start + batch_size
            batch_ids = all_pdb_ids[start:end]
            
            click.echo(click.style(f"\n--- Batch {i+1}/{num_batches} (IDs {start}-{min(end, total_files)}) ---", bold=True, fg="green"))

            # Download batch
            if os.name == 'nt':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            asyncio.run(downloader.download_assemblies(
                pdb_ids=batch_ids,
                output_dir_rel=raw_dir_rel,
                log_file_name=f"download_log_batch_{i}.txt"
            ))

            # Compress batch with tar
            tar_name = f"assemblies_batch_{i}.tar.gz"
            tar_path = raw_dir_abs / tar_name
            click.echo(f"Archiving batch to {tar_name}...")
            
            # Append batch to tarfile
            with tarfile.open(tar_path, "w:gz") as tar:
                for pdb_id in batch_ids:
                    filename = f"{pdb_id.lower()}-assembly1.cif.gz"
                    filepath = raw_dir_abs / filename
                    if filepath.exists():
                        tar.add(filepath, arcname=filename)
            
            # Delete last batch of uncompressed files
            click.echo("Cleaning up raw files...")
            removed_count = 0
            for pdb_id in batch_ids:
                filename = f"{pdb_id.lower()}-assembly1.cif.gz"
                filepath = raw_dir_abs / filename
                if filepath.exists():
                    filepath.unlink()
                    removed_count += 1
            
            click.echo(f"Batch {i+1} complete. Archived and removed {removed_count} files.")

    click.echo("\nAll download steps completed!")


if __name__ == '__main__':
    cli()