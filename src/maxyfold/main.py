import click
import rootutils
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from maxyfold.data.pipeline import DataPipelineManager

# Load root path
root = rootutils.find_root(indicator=".project-root")

# Load config
GlobalHydra.instance().clear()
with initialize(version_base="1.3", config_path="../../configs"):
    cfg = compose(config_name="pipeline")



@click.group()
def cli():
    """MaxyFold Data Management CLI"""
    pass



@cli.command()
@click.option('--name', default='User', help='The name to greet.')
def hello(name):
    """A simple hello command to test the CLI setup."""
    click.echo(f"Hello, {name}! Welcome to MaxyFold! :D")



@cli.command()
@click.option("--ids", "-i", is_flag=True, help="Download the filtered list of PDB IDs.")
@click.option("--assemblies", "-a", is_flag=True, help="Download the biological assembly files.")
@click.option("--ccd", "-c", is_flag=True, help="Download the Chemical Component Dictionary.")
@click.option("--batch-size", type=int, default=20000, help="Number of files to download before tarring.")
@click.option("--file-limit", type=int, default=0, help="Limit the number of PDBs to download (0 for all). Useful for testing.")
def download(ids, assemblies, ccd, batch_size, file_limit):
    """Command for downloading raw PDB ids, ccd, and assemblies."""
    click.echo("Downloading raw PDB files...")

    if not (ids or assemblies or ccd):
        click.echo("No specific components requested. Defaulting to download all.")
        ids = assemblies = ccd = True

    manager = DataPipelineManager(paths_cfg=cfg.paths, query_cfg=cfg.data)

    try:
        manager.download_dataset(
            ids=ids, 
            ccd=ccd, 
            assemblies=assemblies, 
            batch_size=batch_size, 
            limit=file_limit
        )
        click.echo(click.style("\nAll download steps completed successfully!", fg="green", bold=True))
    except Exception as e:
        click.echo(click.style(f"\nPipeline failed: {str(e)}", fg="red", bold=True))



@cli.command()
@click.option("--file-limit", default=0, help="Limit exact number of PDB files to process.")
def process(file_limit):
    """Processes raw .tar.gz archives into a clean, ML-ready LMDB dataset."""
    click.echo("Processing raw PDB files...")

    manager = DataPipelineManager(paths_cfg=cfg.paths)

    try:
        total_complexes, total_errors = manager.process_to_lmdb(file_limit=file_limit)
        
        click.echo(click.style(f"\nProcessing Complete!", fg="green", bold=True))
        click.echo(f"Total Complexes Saved: {total_complexes}")
        click.echo(f"Skipped/Errors:        {total_errors}")
    except Exception as e:
        click.echo(click.style(f"\nPipeline failed: {str(e)}", fg="red", bold=True))