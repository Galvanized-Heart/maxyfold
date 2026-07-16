from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor


def _to_numpy(value: ArrayLike) -> np.ndarray:
    """Convert an array or tensor to a CPU NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


@dataclass(frozen=True, slots=True)
class StructureRecord:
    """Structure data required for serialization and visualization.

    Shapes:
        coords:         [L, A, 3]
        mask:           [L, A]
        res_type:       [L]
        atom_elements:  [L, A]
        chain_ids:      [L]
    """

    pdb_id: str
    coords: ArrayLike
    mask: ArrayLike
    res_type: ArrayLike
    atom_elements: ArrayLike
    chain_ids: ArrayLike
    metadata: dict[str, Any] = field(default_factory=dict)

    def numpy(self) -> StructureRecord:
        """Return the same record backed by normalized NumPy arrays."""
        return StructureRecord(
            pdb_id=str(self.pdb_id).upper(),
            coords=_to_numpy(self.coords).astype(np.float32, copy=False),
            mask=_to_numpy(self.mask).astype(bool, copy=False),
            res_type=_to_numpy(self.res_type).astype(np.int64, copy=False),
            atom_elements=_to_numpy(self.atom_elements).astype(
                np.int64,
                copy=False,
            ),
            chain_ids=_to_numpy(self.chain_ids).astype(np.int64, copy=False),
            metadata=self.metadata,
        )

    def validate(self) -> None:
        """Validate only the invariants required by structure writers."""
        coords = np.asarray(self.coords)
        mask = np.asarray(self.mask)
        res_type = np.asarray(self.res_type)
        atom_elements = np.asarray(self.atom_elements)
        chain_ids = np.asarray(self.chain_ids)

        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"coords must have shape [L, A, 3], got {coords.shape}"
            )

        expected_atom_shape = coords.shape[:2]
        expected_token_shape = (coords.shape[0],)

        if mask.shape != expected_atom_shape:
            raise ValueError(
                f"mask must have shape {expected_atom_shape}, got {mask.shape}"
            )

        if atom_elements.shape != expected_atom_shape:
            raise ValueError(
                "atom_elements must have shape "
                f"{expected_atom_shape}, got {atom_elements.shape}"
            )

        if res_type.shape != expected_token_shape:
            raise ValueError(
                f"res_type must have shape {expected_token_shape}, "
                f"got {res_type.shape}"
            )

        if chain_ids.shape != expected_token_shape:
            raise ValueError(
                f"chain_ids must have shape {expected_token_shape}, "
                f"got {chain_ids.shape}"
            )

        if not np.isfinite(coords[mask]).all():
            raise ValueError("Valid atom coordinates must all be finite")