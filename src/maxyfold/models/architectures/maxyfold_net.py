from torch import nn
from maxyfold.models.layers import PDBInputEmbedder, PairformerBlock, StructureModule
from maxyfold.data.constants import restypes

class MaxyFoldNet(nn.Module):
    def __init__(self, single_dim=128, pair_dim=64, num_blocks=4):
        super().__init__()
        self.embedder = PDBInputEmbedder(res_type_vocab_size=len(restypes), single_dim=single_dim, pair_dim=pair_dim)
        
        self.blocks = nn.ModuleList([
            PairformerBlock(single_dim, pair_dim) for _ in range(num_blocks)
        ])
        
        self.structure_module = StructureModule(single_dim)

    def forward(self, batch):
        s, z = self.embedder(batch)
        
        for block in self.blocks:
            s, z = block(s, z)
            
        pred_coords = self.structure_module(s)
        
        # The LitModule expects the predicted coords to be in the same
        # key as the ground truth. Let's return a dictionary.
        # **Correction:** The LitModule expects a tensor, not a dict.
        # It gets the batch and passes it to the model.
        return pred_coords