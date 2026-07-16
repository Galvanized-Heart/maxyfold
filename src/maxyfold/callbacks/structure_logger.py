from __future__ import annotations

from typing import Any

import torch
from lightning import Callback, LightningModule, Trainer

from maxyfold.data.structure import LocalStructureWriter, StructureRecord
from maxyfold.models.metrics.structure_functional import (
    coord_norm_max,
    coord_norm_mean,
    empty_token_count,
    masked_coord_mae,
    masked_coord_mse,
    masked_coord_rmse,
    rep_token_distance_mae,
    safe_divide,
    total_atom_count,
    total_token_count,
    valid_atom_count,
)


def _to_float(value: torch.Tensor) -> float:
    """Convert a scalar tensor to a regular Python float."""
    return float(value.detach().cpu().item())


class StructureLoggerCallback(Callback):
    """Save selected validation predictions as local structure bundles.

    This callback does not interact with the Lightning logger or W&B. It writes
    only through LocalStructureWriter inside the current Hydra run directory.
    """

    def __init__(
        self,
        run_dir: str,
        every_n_epochs: int = 5,
        max_samples: int = 8,
        save_epoch_zero: bool = True,
        save_final_epoch: bool = True,
        max_pairwise_tokens: int | None = 512,
    ) -> None:
        super().__init__()

        if every_n_epochs < 0:
            raise ValueError("every_n_epochs must be non-negative")

        if max_samples < 0:
            raise ValueError("max_samples must be non-negative")

        self.writer = LocalStructureWriter(run_dir=run_dir)
        self.every_n_epochs = every_n_epochs
        self.max_samples = max_samples
        self.save_epoch_zero = save_epoch_zero
        self.save_final_epoch = save_final_epoch
        self.max_pairwise_tokens = max_pairwise_tokens

        self._saved_this_epoch = 0

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del trainer, pl_module
        self._saved_this_epoch = 0

    def _should_save_epoch(self, trainer: Trainer) -> bool:
        """Return whether predictions should be saved this validation epoch."""
        if trainer.sanity_checking:
            return False

        if not trainer.is_global_zero:
            return False

        if self.max_samples == 0:
            return False

        epoch = trainer.current_epoch

        is_epoch_zero = epoch == 0 and self.save_epoch_zero

        is_periodic = (
            epoch > 0
            and self.every_n_epochs > 0
            and epoch % self.every_n_epochs == 0
        )

        is_final = (
            self.save_final_epoch
            and trainer.max_epochs > 0
            and epoch == trainer.max_epochs - 1
        )

        return is_epoch_zero or is_periodic or is_final

    @staticmethod
    def _validate_outputs(outputs: Any) -> dict[str, Any]:
        """Validate the callback output contract from PDBLitModule."""
        if not isinstance(outputs, dict):
            raise TypeError(
                "StructureLoggerCallback expects validation_step() to return "
                "a dictionary."
            )

        required = {
            "pred_coords",
            "true_coords",
            "mask",
            "pdb_id",
            "res_type",
            "atom_elements",
            "chain_ids",
        }

        missing = required.difference(outputs)

        if missing:
            raise KeyError(
                "Validation outputs are missing fields required for structure "
                f"logging: {sorted(missing)}"
            )

        return outputs

    @staticmethod
    def _get_pdb_id(
        pdb_ids: str | list[str] | tuple[str, ...],
        sample_idx: int,
    ) -> str:
        """Extract one PDB ID from a collated validation batch."""
        if isinstance(pdb_ids, str):
            return pdb_ids

        return str(pdb_ids[sample_idx])

    def _compute_sample_metrics(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        """Compute metrics for one saved prediction.

        Inputs retain a batch dimension of one because the shared functional
        metrics expect [B, L, A, 3] coordinates and [B, L, A] masks.
        """
        with torch.no_grad():
            valid_fraction = safe_divide(
                valid_atom_count(mask),
                total_atom_count(mask),
            )
            padding_fraction = safe_divide(
                empty_token_count(mask),
                total_token_count(mask),
            )

            return {
                "coord_mse": _to_float(
                    masked_coord_mse(pred_coords, true_coords, mask)
                ),
                "coord_rmse": _to_float(
                    masked_coord_rmse(pred_coords, true_coords, mask)
                ),
                "coord_mae": _to_float(
                    masked_coord_mae(pred_coords, true_coords, mask)
                ),
                "rep_token_distance_mae": _to_float(
                    rep_token_distance_mae(
                        pred_coords,
                        true_coords,
                        mask,
                        max_tokens=self.max_pairwise_tokens,
                    )
                ),
                "valid_atom_fraction": _to_float(valid_fraction),
                "padding_fraction": _to_float(padding_fraction),
                "coord_norm_mean": _to_float(
                    coord_norm_mean(pred_coords, mask)
                ),
                "coord_norm_max": _to_float(
                    coord_norm_max(pred_coords, mask)
                ),
            }

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module, batch

        if dataloader_idx != 0:
            return

        if not self._should_save_epoch(trainer):
            return

        if self._saved_this_epoch >= self.max_samples:
            return

        outputs = self._validate_outputs(outputs)

        pred_coords = outputs["pred_coords"]
        true_coords = outputs["true_coords"]
        mask = outputs["mask"]

        if pred_coords.ndim != 4:
            raise ValueError(
                "pred_coords must have shape [B, L, A, 3], "
                f"got {tuple(pred_coords.shape)}"
            )

        batch_size = pred_coords.shape[0]

        for sample_idx in range(batch_size):
            if self._saved_this_epoch >= self.max_samples:
                break

            sample_mask = mask[sample_idx]

            # Do not attempt to serialize an entirely empty sample.
            if not torch.any(sample_mask > 0):
                continue

            sample_pred = pred_coords[sample_idx : sample_idx + 1]
            sample_true = true_coords[sample_idx : sample_idx + 1]
            sample_mask_batched = mask[sample_idx : sample_idx + 1]

            metrics = self._compute_sample_metrics(
                pred_coords=sample_pred,
                true_coords=sample_true,
                mask=sample_mask_batched,
            )

            record = StructureRecord(
                pdb_id=self._get_pdb_id(
                    outputs["pdb_id"],
                    sample_idx,
                ),
                coords=pred_coords[sample_idx],
                mask=sample_mask,
                res_type=outputs["res_type"][sample_idx],
                atom_elements=outputs["atom_elements"][sample_idx],
                chain_ids=outputs["chain_ids"][sample_idx],
            )

            self.writer.write(
                record=record,
                metrics=metrics,
                split="val",
                epoch=trainer.current_epoch,
                global_step=trainer.global_step,
                batch_idx=batch_idx,
                sample_idx=sample_idx,
            )

            self._saved_this_epoch += 1