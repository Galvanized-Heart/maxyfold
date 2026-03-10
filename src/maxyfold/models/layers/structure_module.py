from torch import nn
from maxyfold.data.constants import MAX_ATOM_COUNT

class StructureModule(nn.Module):
    def __init__(self, single_dim):
        super().__init__()
        # Each token's feature vector in 's' needs to be projected
        # to the coordinates of its constituent atoms.
        self.to_coords = nn.Linear(single_dim, MAX_ATOM_COUNT * 3)

    def forward(self, s):
        # s: [B, N, C_s]
        pred_coords_flat = self.to_coords(s) # [B, N, MAX_ATOM_COUNT * 3]
        
        b, n, _ = pred_coords_flat.shape
        pred_coords = pred_coords_flat.view(b, n, MAX_ATOM_COUNT, 3)
        
        return pred_coords