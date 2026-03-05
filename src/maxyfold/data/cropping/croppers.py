import numpy as np

from maxyfold.data.constants import LIGAND_IDX



class BaseCropper:
    """
    Abstract base class containing shared logic for padding and spatial selection.
    """
    def __init__(self, crop_size: int = 384, pad_token_id: int = 0):
        self.crop_size = crop_size
        self.pad_token_id = pad_token_id

    def __call__(self, data: dict) -> dict:
        raise NotImplementedError

    def _pad_to_size(self, data: dict) -> dict:
        """Pads sequence to crop_size with zeros/defaults."""
        L = len(data["res_type"])
        pad_len = self.crop_size - L
        
        # Helper lambda for padding
        pad = lambda arr, val: np.pad(arr, (0, pad_len), constant_values=val)
        pad2d = lambda arr, val: np.pad(arr, ((0, pad_len), (0, 0)), constant_values=val)
        pad3d = lambda arr, val: np.pad(arr, ((0, pad_len), (0, 0), (0, 0)), constant_values=val)

        return {
            "pdb_id": data["pdb_id"],
            "res_type": pad(data["res_type"], self.pad_token_id),
            "chain_ids": pad(data["chain_ids"], -1),
            "mask": pad2d(data["mask"], 0.0),
            "atom_elements": pad2d(data["atom_elements"], 0),
            "coords": pad3d(data["coords"], 0.0),
        }

    def _get_representative_coords(self, data: dict) -> np.ndarray:
        """
        Extracts a single (x,y,z) coordinate for every token to calculate distances.
        Uses the first resolved atom (Index 0 for Ligands/RNA, Index 0 (N) for Proteins).
        """
        L = len(data["res_type"])
        rep_coords = np.zeros((L, 3), dtype=np.float32)
        
        # Vectorized check for first non-zero mask is hard due to raggedness.
        # This loop is O(L) and fast enough (L < 20k).
        for i in range(L):
            # argmax returns index of first True. 
            # We check if max is > 0 to ensure it's not a fully masked token.
            if np.max(data["mask"][i]) > 0:
                first_valid_idx = np.argmax(data["mask"][i])
                rep_coords[i] = data["coords"][i, first_valid_idx]
            else:
                rep_coords[i] = [0.0, 0.0, 0.0]
        return rep_coords

    def _spatial_crop_from_center(self, data: dict, center_idx: int, rep_coords: np.ndarray) -> dict:
        """
        The core engine. Given a center token index, select neighbors
        while preserving ligand integrity.
        """
        # Calculate distances
        center_coord = rep_coords[center_idx]
        dists = np.sum((rep_coords - center_coord)**2, axis=-1)
        sorted_indices = np.argsort(dists)

        # Select tokens
        selected_indices = set()
        
        for idx in sorted_indices:
            if len(selected_indices) >= self.crop_size:
                break
                
            # Ligand Integrity Logic
            if data["res_type"][idx] == LIGAND_IDX:
                ligand_chain_id = data["chain_ids"][idx]
                
                # Find siblings
                ligand_atoms = np.where(
                    (data["chain_ids"] == ligand_chain_id) & 
                    (data["res_type"] == LIGAND_IDX)
                )[0]
                
                # Add all or nothing
                if len(selected_indices) + len(ligand_atoms) <= self.crop_size:
                    selected_indices.update(ligand_atoms)
                else:
                    continue # Skip if molecule doesn't fit
            else:
                selected_indices.add(idx)

        # Fill gaps with remaining closest tokens
        selected_list = sorted(list(selected_indices))
        if len(selected_list) < self.crop_size:
            shortfall = self.crop_size - len(selected_list)
            remaining = [i for i in sorted_indices if i not in selected_indices]
            selected_list.extend(remaining[:shortfall])
            selected_list.sort()

        crop_idx = np.array(selected_list)

        return {
            "pdb_id": data["pdb_id"],
            "res_type": data["res_type"][crop_idx],
            "coords": data["coords"][crop_idx],
            "mask": data["mask"][crop_idx],
            "atom_elements": data["atom_elements"][crop_idx],
            "chain_ids": data["chain_ids"][crop_idx],
        }


class ContiguousCropper(BaseCropper):
    """
    Slices a continuous chunk. Good for learning local polymer physics.
    """
    def __init__(self, crop_size: int = 384, pad_token_id: int = 0):
        super().__init__(crop_size, pad_token_id)

    def __call__(self, data: dict) -> dict:
        L = len(data["res_type"])
        
        if L <= self.crop_size:
            return self._pad_to_size(data)

        # Pick random start token
        max_start = L - self.crop_size
        start_idx = np.random.randint(0, max_start + 1)
        end_idx = start_idx + self.crop_size

        return {
            "pdb_id": data["pdb_id"],
            "res_type": data["res_type"][start_idx:end_idx],
            "coords": data["coords"][start_idx:end_idx],
            "mask": data["mask"][start_idx:end_idx],
            "atom_elements": data["atom_elements"][start_idx:end_idx],
            "chain_ids": data["chain_ids"][start_idx:end_idx],
        }



class SpatialCropper(BaseCropper):
    """
    Random center -> Closest neighbors. Good for general 3D structure.
    """
    def __init__(self, crop_size: int = 384, pad_token_id: int = 0):
        super().__init__(crop_size, pad_token_id)

    def __call__(self, data: dict) -> dict:
        L = len(data["res_type"])
        
        if L <= self.crop_size:
            return self._pad_to_size(data)

        rep_coords = self._get_representative_coords(data)
        
        # Pick random center token
        center_idx = np.random.randint(0, L)
        
        return self._spatial_crop_from_center(data, center_idx, rep_coords)



class InterfaceBiasedCropper(BaseCropper):
    """
    Centers crop on a token that is close to a different chain.
    """
    def __init__(self, crop_size: int = 384, pad_token_id: int = 0, interface_cutoff: float = 8.0):
        super().__init__(crop_size, pad_token_id)
        self.interface_cutoff = interface_cutoff

    def __call__(self, data: dict) -> dict:
        L = len(data["res_type"])
        
        if L <= self.crop_size:
            return self._pad_to_size(data)

        rep_coords = self._get_representative_coords(data)

        # Calculate interface tokens
        diffs = rep_coords[:, None, :] - rep_coords[None, :, :] 
        sq_dists = np.sum(diffs**2, axis=-1)
        
        chain_ids = data["chain_ids"]
        # Mask out self-chain distances
        same_chain_mask = chain_ids[:, None] == chain_ids[None, :]
        sq_dists[same_chain_mask] = np.inf 
        
        # Tokens with any neighbor < 8 Angstroms
        cutoff_sq = self.interface_cutoff ** 2
        interface_tokens = np.where(np.any(sq_dists < cutoff_sq, axis=1))[0]

        if len(interface_tokens) > 0:
            center_idx = np.random.choice(interface_tokens)
        else:
            # Fallback for single chains or far-apart complexes
            center_idx = np.random.randint(0, L)

        return self._spatial_crop_from_center(data, center_idx, rep_coords)



class EntityStratifiedCropper(BaseCropper):
    """
    50/50 chance to center on a Ligand or Polymer.
    """
    def __init__(self, crop_size: int = 384, pad_token_id: int = 0, ligand_prob: float = 0.5):
        super().__init__(crop_size, pad_token_id)
        self.ligand_prob = ligand_prob

    def __call__(self, data: dict) -> dict:
        L = len(data["res_type"])
        
        if L <= self.crop_size:
            return self._pad_to_size(data)

        rep_coords = self._get_representative_coords(data)

        ligand_indices = np.where(data["res_type"] == LIGAND_IDX)[0]
        polymer_indices = np.where(data["res_type"] != LIGAND_IDX)[0]

        # Weighted coin flip
        if len(ligand_indices) > 0 and np.random.rand() < self.ligand_prob:
            center_idx = np.random.choice(ligand_indices)
        elif len(polymer_indices) > 0:
            center_idx = np.random.choice(polymer_indices)
        else:
            center_idx = np.random.randint(0, L)

        return self._spatial_crop_from_center(data, center_idx, rep_coords)