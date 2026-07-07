from __future__ import annotations

from typing import Tuple

import torch


EPS = 1e-8


def validate_structure_tensors(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Validate standard MaxyFold dense structure tensors.

    Expected:
        pred_coords: [B, L, A, 3]
        true_coords: [B, L, A, 3]
        mask:        [B, L, A]
    """
    if pred_coords.shape != true_coords.shape:
        raise ValueError(
            "pred_coords and true_coords must have the same shape. "
            f"Got {pred_coords.shape=} and {true_coords.shape=}."
        )

    if pred_coords.ndim != 4 or pred_coords.shape[-1] != 3:
        raise ValueError(
            "pred_coords and true_coords must have shape [B, L, A, 3]. "
            f"Got {pred_coords.shape=}."
        )

    if mask.shape != pred_coords.shape[:-1]:
        raise ValueError(
            "mask must have shape [B, L, A], matching coords without xyz dim. "
            f"Got {mask.shape=} for coords {pred_coords.shape=}."
        )


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide while avoiding NaN/Inf for empty masks."""
    return numerator / denominator.clamp_min(EPS)


def valid_atom_count(mask: torch.Tensor) -> torch.Tensor:
    """Number of valid atoms in a dense [B, L, A] atom mask."""
    return mask.float().sum()


def total_atom_count(mask: torch.Tensor) -> torch.Tensor:
    """Total number of atom slots in a dense [B, L, A] atom mask."""
    return torch.as_tensor(mask.numel(), dtype=mask.dtype, device=mask.device)


def valid_coordinate_count(mask: torch.Tensor) -> torch.Tensor:
    """Number of valid xyz coordinate components."""
    return valid_atom_count(mask) * 3.0


def valid_token_count(mask: torch.Tensor) -> torch.Tensor:
    """Number of tokens with at least one valid atom."""
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape [B, L, A]. Got {mask.shape=}.")
    return (mask.float().sum(dim=-1) > 0).float().sum()


def total_token_count(mask: torch.Tensor) -> torch.Tensor:
    """Total number of token slots in a dense [B, L, A] atom mask."""
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape [B, L, A]. Got {mask.shape=}.")
    return torch.as_tensor(
        mask.shape[0] * mask.shape[1],
        dtype=mask.dtype,
        device=mask.device,
    )


def empty_token_count(mask: torch.Tensor) -> torch.Tensor:
    """Number of tokens with no valid atoms."""
    return total_token_count(mask) - valid_token_count(mask)


def masked_squared_distance_sum(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sum squared Euclidean atom errors over valid atoms.

    This matches the current MSELoss semantics:
        ((true - pred) ** 2).sum(xyz) / valid_atom_count
    """
    validate_structure_tensors(pred_coords, true_coords, mask)
    atom_sq_dist = (pred_coords - true_coords).square().sum(dim=-1)
    return (atom_sq_dist * mask.float()).sum()


def masked_l1_coordinate_sum(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Sum absolute coordinate-component errors over valid xyz components."""
    validate_structure_tensors(pred_coords, true_coords, mask)
    return ((pred_coords - true_coords).abs() * mask.float().unsqueeze(-1)).sum()


def masked_coord_mse(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared Euclidean atom displacement over valid atoms."""
    return safe_divide(
        masked_squared_distance_sum(pred_coords, true_coords, mask),
        valid_atom_count(mask),
    )


def masked_coord_rmse(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Root mean squared Euclidean atom displacement over valid atoms."""
    return torch.sqrt(masked_coord_mse(pred_coords, true_coords, mask).clamp_min(0.0))


def masked_coord_mae(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute coordinate-component error over valid xyz components."""
    return safe_divide(
        masked_l1_coordinate_sum(pred_coords, true_coords, mask),
        valid_coordinate_count(mask),
    )


def representative_token_coords(
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get one representative coordinate per token.

    Uses the first valid atom for each token instead of hardcoding atom index 0.

    Returns:
        rep_coords: [B, L, 3]
        token_mask: [B, L], bool
    """
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [B, L, A, 3]. Got {coords.shape=}.")

    if mask.shape != coords.shape[:-1]:
        raise ValueError(f"mask shape {mask.shape} does not match coords {coords.shape}.")

    valid = mask > 0
    token_mask = valid.any(dim=-1)

    first_valid_idx = valid.float().argmax(dim=-1)  # [B, L]
    gather_idx = first_valid_idx[..., None, None].expand(-1, -1, 1, 3)

    rep_coords = coords.gather(dim=2, index=gather_idx).squeeze(2)

    return rep_coords, token_mask


def rep_token_distance_error_sum_and_count(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
    max_tokens: int | None = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise representative-token distance absolute error.

    This is a cheap structural metric based on token representative atoms,
    not a full all-atom pairwise metric.
    """
    validate_structure_tensors(pred_coords, true_coords, mask)

    pred_rep, pred_token_mask = representative_token_coords(pred_coords, mask)
    true_rep, true_token_mask = representative_token_coords(true_coords, mask)
    token_mask = pred_token_mask & true_token_mask

    if max_tokens is not None and pred_rep.shape[1] > max_tokens:
        pred_rep = pred_rep[:, :max_tokens]
        true_rep = true_rep[:, :max_tokens]
        token_mask = token_mask[:, :max_tokens]

    pred_dist = torch.cdist(pred_rep, pred_rep)
    true_dist = torch.cdist(true_rep, true_rep)

    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]

    n_tokens = pair_mask.shape[-1]
    eye = torch.eye(n_tokens, dtype=torch.bool, device=pair_mask.device)
    pair_mask = pair_mask & ~eye[None, :, :]

    pair_mask_f = pair_mask.float()

    error_sum = ((pred_dist - true_dist).abs() * pair_mask_f).sum()
    count = pair_mask_f.sum()

    return error_sum, count


def rep_token_distance_mae(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
    max_tokens: int | None = 512,
) -> torch.Tensor:
    """Mean absolute representative-token pairwise distance error."""
    error_sum, count = rep_token_distance_error_sum_and_count(
        pred_coords=pred_coords,
        true_coords=true_coords,
        mask=mask,
        max_tokens=max_tokens,
    )
    return safe_divide(error_sum, count)


def coord_norm_sum_and_count(
    coords: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum coordinate vector norms and valid atom count."""
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [B, L, A, 3]. Got {coords.shape=}.")

    if mask.shape != coords.shape[:-1]:
        raise ValueError(f"mask shape {mask.shape} does not match coords {coords.shape}.")

    atom_norms = torch.linalg.vector_norm(coords, dim=-1)
    count = valid_atom_count(mask)

    return (atom_norms * mask.float()).sum(), count


def coord_norm_mean(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean coordinate vector norm over valid atoms."""
    norm_sum, count = coord_norm_sum_and_count(coords, mask)
    return safe_divide(norm_sum, count)


def coord_norm_max(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max coordinate vector norm over valid atoms."""
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [B, L, A, 3]. Got {coords.shape=}.")

    if mask.shape != coords.shape[:-1]:
        raise ValueError(f"mask shape {mask.shape} does not match coords {coords.shape}.")

    valid = mask > 0
    if not torch.any(valid):
        return coords.new_tensor(0.0)

    atom_norms = torch.linalg.vector_norm(coords, dim=-1)
    return atom_norms[valid].max()