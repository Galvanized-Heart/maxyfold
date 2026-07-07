from __future__ import annotations

import torch
from torchmetrics import Metric, MetricCollection

from maxyfold.models.metrics.structure_functional import (
    coord_norm_max,
    coord_norm_sum_and_count,
    empty_token_count,
    masked_l1_coordinate_sum,
    masked_squared_distance_sum,
    rep_token_distance_error_sum_and_count,
    safe_divide,
    total_atom_count,
    total_token_count,
    valid_atom_count,
    valid_coordinate_count,
)


class MaskedCoordMSE(Metric):
    """Mean squared Euclidean atom displacement over valid atoms."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.sse += masked_squared_distance_sum(pred_coords, true_coords, mask)
        self.count += valid_atom_count(mask)

    def compute(self) -> torch.Tensor:
        return safe_divide(self.sse, self.count)


class MaskedCoordRMSE(Metric):
    """Root mean squared Euclidean atom displacement over valid atoms."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.sse += masked_squared_distance_sum(pred_coords, true_coords, mask)
        self.count += valid_atom_count(mask)

    def compute(self) -> torch.Tensor:
        return torch.sqrt(safe_divide(self.sse, self.count).clamp_min(0.0))


class MaskedCoordMAE(Metric):
    """Mean absolute coordinate-component error over valid xyz components."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("sae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.sae += masked_l1_coordinate_sum(pred_coords, true_coords, mask)
        self.count += valid_coordinate_count(mask)

    def compute(self) -> torch.Tensor:
        return safe_divide(self.sae, self.count)


class RepTokenDistanceMAE(Metric):
    """Mean absolute pairwise representative-token distance error."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, max_tokens: int | None = 512) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.add_state("error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        error_sum, count = rep_token_distance_error_sum_and_count(
            pred_coords=pred_coords,
            true_coords=true_coords,
            mask=mask,
            max_tokens=self.max_tokens,
        )
        self.error_sum += error_sum
        self.count += count

    def compute(self) -> torch.Tensor:
        return safe_divide(self.error_sum, self.count)


class ValidAtomFraction(Metric):
    """Fraction of valid atom slots."""

    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("valid_atoms", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_atoms", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        del pred_coords, true_coords
        self.valid_atoms += valid_atom_count(mask)
        self.total_atoms += total_atom_count(mask)

    def compute(self) -> torch.Tensor:
        return safe_divide(self.valid_atoms, self.total_atoms)


class PaddingFraction(Metric):
    """Fraction of empty token slots."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("empty_tokens", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tokens", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        del pred_coords, true_coords
        self.empty_tokens += empty_token_count(mask)
        self.total_tokens += total_token_count(mask)

    def compute(self) -> torch.Tensor:
        return safe_divide(self.empty_tokens, self.total_tokens)


class CoordNormMean(Metric):
    """Mean predicted coordinate vector norm over valid atoms."""

    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("norm_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        del true_coords
        norm_sum, count = coord_norm_sum_and_count(pred_coords, mask)
        self.norm_sum += norm_sum
        self.count += count

    def compute(self) -> torch.Tensor:
        return safe_divide(self.norm_sum, self.count)


class CoordNormMax(Metric):
    """Maximum predicted coordinate vector norm over valid atoms."""

    is_differentiable = False
    higher_is_better = None
    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state("max_norm", default=torch.tensor(0.0), dist_reduce_fx="max")

    def update(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        del true_coords
        self.max_norm = torch.maximum(self.max_norm, coord_norm_max(pred_coords, mask))

    def compute(self) -> torch.Tensor:
        return self.max_norm


def make_structure_metric_collection(
    prefix: str,
    max_pairwise_tokens: int | None = 512,
) -> MetricCollection:
    """Create a stage-specific MetricCollection for structure prediction.

    TorchMetrics recommends keeping separate metric instances for train/val/test
    rather than reusing one stateful object across stages.
    """
    return MetricCollection(
        {
            "coord_mse": MaskedCoordMSE(),
            "coord_rmse": MaskedCoordRMSE(),
            "coord_mae": MaskedCoordMAE(),
            "rep_token_distance_mae": RepTokenDistanceMAE(
                max_tokens=max_pairwise_tokens
            ),
            "valid_atom_fraction": ValidAtomFraction(),
            "padding_fraction": PaddingFraction(),
            "coord_norm_mean": CoordNormMean(),
            "coord_norm_max": CoordNormMax(),
        },
        prefix=prefix,
    )