import torch
from torch import nn
import torch.nn.functional as F


class SimSiamLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2, p1, p2):
        loss = 0.5 * self.cosine_with_stopgrad(p1, z2) + 0.5 * self.cosine_with_stopgrad(p2, z1)
        return loss

    def cosine_with_stopgrad(self, p, z):
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    

class BCELoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred, target):
        loss = self.criterion(pred, target.to(torch.float))
        return loss
