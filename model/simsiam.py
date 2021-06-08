import torch
from torch import nn, optim
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, backbone, input_dim, hidden_dim=None, output_dim=2048):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        backbone.output_dim = input_dim
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x, **backbone_args):
        out = self.backbone(x, **backbone_args)
        out = self.avgpool(out.transpose(1,2)).squeeze()
        out = self.projector(out)
        return out


class Predictor(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.fc2(out)
        return out


class SimSiam(nn.Module):

    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        
    def forward(self, x, **backbone_args):
        z = self.encoder(x, **backbone_args)
        p = self.predictor(z)
        return z, p
