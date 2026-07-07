from typing import Any, Dict

import torch
from lightning import LightningModule

from maxyfold.models.metrics import make_structure_metric_collection

#############################################################################
# TODO: Refactor this to be able to run inference from just the .ckpt file! #
#############################################################################
# - Pass dicts wherever possible and instantiate within LitModule.
# - Consider if using torch.jit.script() is a good option for this.

STRUCTURE_METRIC_KEYS = (
    "coord_mse",
    "coord_rmse",
    "coord_mae",
    "pairwise_distance_mae",
    "valid_atom_fraction",
    "padding_fraction",
    "coord_norm_mean",
    "coord_norm_max",
)

class PDBLitModule(LightningModule):
    """LightningModule for Protein Structure tasks."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        sync_dist: bool = False,
        log_structure_metrics: bool = True,
        max_pairwise_tokens: int | None = 512,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn

        self.train_structure_metrics = make_structure_metric_collection(
            prefix="train/",
            max_pairwise_tokens=max_pairwise_tokens,
        )
        self.val_structure_metrics = make_structure_metric_collection(
            prefix="val/",
            max_pairwise_tokens=max_pairwise_tokens,
        )
        self.test_structure_metrics = make_structure_metric_collection(
            prefix="test/",
            max_pairwise_tokens=max_pairwise_tokens,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the network."""
        return self.model(batch)

    def model_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run model forward pass and compute loss.

        Returns a dict so callbacks/logging can reuse the
        same prediction tensors without another forward pass.
        """
        true_coords = batch["coords"]  # [B, L, A, 3]
        mask = batch["mask"]  # [B, L, A]

        pred_coords = self.forward(batch)

        loss = self.loss_fn(
            pred_coords=pred_coords,
            true_coords=true_coords,
            mask=mask,
        )

        return {
            "loss": loss,
            "pred_coords": pred_coords,
            "true_coords": true_coords,
            "mask": mask,
        }
    
    def _log_loss(
        self,
        stage: str,
        loss: torch.Tensor,
        batch: Dict[str, Any],
        *,
        prog_bar: bool,
    ) -> None:
        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
            sync_dist=self.hparams.sync_dist,
        )

    def _update_and_log_structure_metrics(
        self,
        stage: str,
        outputs: Dict[str, Any],
    ) -> None:
        """Update and log stage-specific TorchMetrics."""
        if not self.hparams.log_structure_metrics:
            return

        if stage == "train":
            metrics = self.train_structure_metrics
        elif stage == "val":
            metrics = self.val_structure_metrics
        elif stage == "test":
            metrics = self.test_structure_metrics
        else:
            raise ValueError(f"Unknown stage: {stage}")

        metrics.update(
            outputs["pred_coords"],
            outputs["true_coords"],
            outputs["mask"],
        )

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=self.hparams.sync_dist,
        )

    def _detach_for_callbacks(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return detached outputs for future callbacks.

        This lets StructureLoggerCallback save local prediction bundles later
        without rerunning the model.
        """
        callback_outputs: Dict[str, Any] = {
            "loss": outputs["loss"].detach(),
            "pred_coords": outputs["pred_coords"].detach(),
            "true_coords": outputs["true_coords"].detach(),
            "mask": outputs["mask"].detach(),
        }

        passthrough_keys = (
            "pdb_id",
            "res_type",
            "atom_elements",
            "chain_ids",
        )

        for key in passthrough_keys:
            if key not in batch:
                continue

            value = batch[key]
            if isinstance(value, torch.Tensor):
                callback_outputs[key] = value.detach()
            else:
                callback_outputs[key] = value

        return callback_outputs





















    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.model_step(batch)

        self._log_loss("train", outputs["loss"], batch, prog_bar=True)
        self._update_and_log_structure_metrics("train", outputs)

        return outputs["loss"]

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, Any]:
        outputs = self.model_step(batch)

        self._log_loss("val", outputs["loss"], batch, prog_bar=True)
        self._update_and_log_structure_metrics("val", outputs)

        return self._detach_for_callbacks(outputs, batch)

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, Any]:
        outputs = self.model_step(batch)

        self._log_loss("test", outputs["loss"], batch, prog_bar=True)
        self._update_and_log_structure_metrics("test", outputs)

        return self._detach_for_callbacks(outputs, batch)


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