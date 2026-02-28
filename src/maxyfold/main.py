import click
import rootutils
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict
import shutil

# Load root path
root = rootutils.find_root(indicator=".project-root")
# This root variable must exist for config/paths/default.yaml to work correctly.
# Run `export PROJECT_ROOT=$(pwd)` to set to current dir.
# Verify PROJECT_ROOT was updated by running `echo $PROJECT_ROOT`.

# Load config
GlobalHydra.instance().clear()
with initialize(version_base="1.3", config_path="../../configs"):
    cfg = compose(config_name="pipeline")



@click.group()
def cli():
    """MaxyFold data management CLI"""
    pass



@cli.command()
@click.option('--name', default='User', help='The name to greet.')
def hello(name):
    """A simple hello command to test the CLI setup."""
    click.echo(f"Hello, {name}! Welcome to MaxyFold! :D")



@cli.command()
@click.option("--config-name", "-n", default="pipeline", help="The name of the config file (without .yaml).")
@click.argument("overrides", nargs=-1)
def show_config(config_name, overrides):
    """Helper function to display the resolved Hydra configuration."""
    try:
        # Reinitialize to allow dynamic config selection and overrides
        GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path="../../configs"):
            resolved_cfg = compose(config_name=config_name, overrides=list(overrides))
            
            click.echo(click.style(f"Resolved Config: {config_name}", fg="cyan", bold=True))
            click.echo(OmegaConf.to_yaml(resolved_cfg))
            click.echo(click.style("-" * (len(config_name) + 25), fg="cyan"))
            
    except Exception as e:
        click.echo(click.style(f"Error loading config: {e}", fg="red"))



@cli.command()
@click.option("--ids", "-i", is_flag=True, help="Download the filtered list of PDB IDs.")
@click.option("--assemblies", "-a", is_flag=True, help="Download the biological assembly files.")
@click.option("--ccd", "-c", is_flag=True, help="Download the Chemical Component Dictionary.")
@click.option("--batch-size", type=int, default=20000, help="Number of files to download before tarring.")
@click.option("--file-limit", default=0, help="Limit exact number of PDB files to process.")
def download(ids, assemblies, ccd, batch_size, file_limit):
    """Command for downloading raw PDB ids, ccd, and assemblies."""
    click.echo("Downloading raw PDB files...")
    from maxyfold.data.pipeline import DataPipelineManager

    if not (ids or assemblies or ccd):
        click.echo("No specific components requested. Defaulting to download all.")
        ids = assemblies = ccd = True

    manager = DataPipelineManager(paths_cfg=cfg.paths, query_cfg=cfg.query)

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
    """Processes raw tar archives into clean LMDB dataset."""
    click.echo("Processing raw PDB files...")
    from maxyfold.data.pipeline import DataPipelineManager

    manager = DataPipelineManager(paths_cfg=cfg.paths, storage_cfg=cfg.storage)

    try:
        total_complexes, total_errors = manager.process(file_limit=file_limit)
        click.echo(click.style(f"\nProcessing Complete!", fg="green", bold=True))
        click.echo(f"Total Complexes Saved: {total_complexes}")
        click.echo(f"Skipped/Errors:        {total_errors}")
    except Exception as e:
        click.echo(click.style(f"\nPipeline failed: {str(e)}", fg="red", bold=True))


@cli.command()
@click.option("--file-limit", default=0, help="Limit number of PDBs to include in the manifest (for testing).")
def manifest(file_limit):
    """Scans the dataset to create a manifest of its contents (sequences, ligands)."""
    click.echo("Creating dataset manifest...")
    from maxyfold.data.pipeline import DataPipelineManager
    
    manager = DataPipelineManager(paths_cfg=cfg.paths, storage_cfg=cfg.storage)
    try:
        manager.create_manifest(limit=file_limit)
        click.echo(click.style("\nManifest created successfully!", fg="green", bold=True))
    except Exception as e:
        click.echo(click.style(f"\nManifest creation failed: {str(e)}", fg="red", bold=True))




@cli.command()
@click.option("--seq-id", default=cfg.split.mmseqs.seq_id, type=float, help="MMseqs2: Min sequence identity.")
@click.option("--coverage", default=cfg.split.mmseqs.coverage, type=float, help="MMseqs2: Min sequence coverage.")
@click.option("--cov-mode", default=cfg.split.mmseqs.cov_mode, type=int, help="MMseqs2: Coverage mode.")
@click.option("--cluster-mode", default=cfg.split.mmseqs.cluster_mode, type=int, help="MMseqs2: Clustering algorithm.")
@click.option("--seed", default=cfg.split.splitting.seed, type=int, help="Random seed for splitting.")
def split(seq_id, coverage, cov_mode, cluster_mode, seed):
    """Clusters processed PDBs by sequence identity and creates train/val/test splits."""
    click.echo("Starting up splitting process...")
    from maxyfold.data.pipeline import DataPipelineManager

    # Input validation
    try:
        ratios = list(cfg.split.splitting.ratios)
        total_sum = sum(ratios)
        if len(ratios) != 3 or total_sum != 1.0:
            click.echo(click.style(f"Config Error: Split ratios in pipeline.yaml must sum to 1.0", fg="red"))
    except ValueError as e:
        click.echo(click.style(f"Invalid --split-ratios: {e}", fg="red"))
        return
    
    # Check for mmseqs2
    if not shutil.which("mmseqs"):
        click.echo(click.style("""
        Error: 'mmseqs' command not found in PATH.

        Try running:
            wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
            tar xvfz mmseqs-linux-avx2.tar.gz
            export PATH=$(pwd)/mmseqs/bin/:$PATH
            mv mmseqs/bin/mmseqs ~/.local/bin/
            rm -rf mmseqs mmseqs-linux-avx2.tar.gz
        """, fg="red"))
        return

    # Copy configs
    mmseqs_config = cfg.split.mmseqs.copy()
    splitting_config = cfg.split.splitting.copy()

    # Update configs with args
    with open_dict(mmseqs_config):
        mmseqs_config.seq_id = seq_id
        mmseqs_config.coverage = coverage
        mmseqs_config.cov_mode = cov_mode
        mmseqs_config.cluster_mode = cluster_mode
    with open_dict(splitting_config):
        splitting_config.seed = seed

    manager = DataPipelineManager(paths_cfg=cfg.paths)

    try:
        manager.create_splits(mmseqs_config, splitting_config)
        click.echo(click.style("\nData splits created successfully!", fg="green", bold=True))
    except Exception as e:
        click.echo(click.style(f"\nSplitting failed: {str(e)}", fg="red", bold=True))



if __name__ == "__main__":
    cli()