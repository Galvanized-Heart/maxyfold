from typing import Any, Dict, Tuple
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

#############################################################################
# TODO: Refactor this to be able to run inference from just the .ckpt file! #
#############################################################################
# - Pass dicts wherever possible and instantiate within LitModule.
# - Consider if using torch.jit.script() is a good option for this.

class PDBLitModule(LightningModule):
    """LightningModule for Protein Structure tasks."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the network."""
        return self.model(batch)

    def model_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """A single pass + loss calculation."""
        gt_coords = batch["coords"] # [B, L, 27, 3]
        mask = batch["mask"]        # [B, L, 27]

        pred_coords = self.forward(batch)

        loss = self.loss_fn(
            pred_coords=pred_coords,
            true_coords=gt_coords,
            mask=mask
        )
        
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        loss = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        loss = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}