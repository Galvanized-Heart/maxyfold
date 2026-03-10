import torch
from torch import nn

class PDBInputEmbedder(nn.Module):
    def __init__(self, res_type_vocab_size, single_dim, pair_dim):
        super().__init__()
        self.res_type_embedding = nn.Embedding(res_type_vocab_size, single_dim)
        
        # Project single features to create initial pair features
        self.to_pair_embed = nn.Linear(single_dim * 2, pair_dim)

    def forward(self, batch):
        # Create single representation (s)
        s = self.res_type_embedding(batch['res_type'])
        
        # Create pair representation (z)
        s_i = s.unsqueeze(2).repeat(1, 1, s.shape[1], 1) # [B, N, 1, C_s] -> [B, N, N, C_s]
        s_j = s.unsqueeze(1).repeat(1, s.shape[1], 1, 1) # [B, 1, N, C_s] -> [B, N, N, C_s]
        
        pair_input = torch.cat([s_i, s_j], dim=-1) # [B, N, N, C_s * 2]
        z = self.to_pair_embed(pair_input)
        
        return s, z