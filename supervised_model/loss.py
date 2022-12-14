import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from torchvision.transforms.functional import InterpolationMode
import torch.optim as optim


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha, gamma):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()