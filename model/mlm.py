import torch
from torch import nn, optim
import torch.nn.functional as F


class MaskEncoder(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        input_dim = backbone.d_model
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid())

    def forward(self, x, **backbone_args):
        out = self.backbone(x, **backbone_args)
        out = out.view(-1, out.size(-1))        # (B,T,E) --> (B*T,E)
        out = self.projector(out)               # (B*T,E) --> (B*T,1)
        out = out.view(x.size(0), x.size(-1))   # (B*T,1) --> (B,T)
        return out
