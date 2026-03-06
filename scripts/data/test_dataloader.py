import rootutils
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Load project root
root = rootutils.find_root(indicator=".project-root")

def main():
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        # Load the pdb data config
        cfg = hydra.compose(config_name="train_pdb")

    print(click.style("Instantiating PDBDataModule...", fg="cyan"))
    
    # Instantiate the datamodule exactly as the trainer will
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Run the setup (loads keys, creates datasets)
    datamodule.setup()
    
    print(click.style("\nFetching a single batch from train_dataloader...", fg="cyan"))
    train_loader = datamodule.train_dataloader()
    
    # Get the first batch
    batch = next(iter(train_loader))
    
    print("\n--- BATCH TENSOR SHAPES ---")
    for key, tensor in batch.items():
        if hasattr(tensor, 'shape'):
            print(f"{key:<15}: {tensor.shape} (dtype: {tensor.dtype})")
        else:
            # For things like a list of pdb_ids
            print(f"{key:<15}: {type(tensor)} of length {len(tensor)}")
            
    print("\nSanity Check Complete! If all tensors have a uniform Batch dimension (e.g. 16), you are ready for modeling.")

if __name__ == "__main__":
    import click
    main()