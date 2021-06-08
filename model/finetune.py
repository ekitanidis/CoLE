from torch import nn


class SumLayer(nn.Module):
    
    def forward(self, x):
        output = x.sum(dim=1)
        return output

    
class FinetuneModel(nn.Module):
    
    def __init__(self, encoder, agglayer):
        super().__init__()
        self.encoder = encoder
        self.agg = agglayer
        self.linear = nn.Linear(encoder.output_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, src, src_mask=None):
        out = self.encoder(src, src_mask=src_mask)
        out = self.sigmoid(self.linear(self.agg(out)))
        return out
