from typing import Any, Dict, Tuple
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

class PDBLitModule(LightningModule):
    """LightningModule for Protein Structure tasks."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model"])
        self.model = model
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the network."""
        return self.model(batch)

    def model_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """A single pass + loss calculation."""
        # Ground truth coordinates
        gt_coords = batch["coords"] # [B, L, 27, 3]
        mask = batch["mask"]        # [B, L, 27]

        # Predicted coordinates (Output of your future network)
        # For now, we assume the network returns predicted coords in the same shape
        pred_coords = self.forward(batch)

        # Simple MSE Loss, only calculated for atoms that exist (mask == 1)
        # In the future, we will replace this with structural losses like FAPE
        diff = (gt_coords - pred_coords) ** 2
        loss = (diff.sum(dim=-1) * mask).sum() / (mask.sum() + 1e-6)
        
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