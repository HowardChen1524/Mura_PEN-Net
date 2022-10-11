import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
import pandas as pd
import random
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image


class AI9_Dataset(Dataset):
    def __init__(self, feature, target, name, transform=None):
        super(AI9_Dataset, self).__init__()
        self.split = split # 'test'
        self.w, self.h = data_args['w'], data_args['h']
        self.X = path
        self.Y = target
        self.N = name
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        return self.transform(img), self.Y[idx], self.N[idx]