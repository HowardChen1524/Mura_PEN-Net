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




class wei_augumentation(object):
    def __call__(self, img):
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.rgb_to_grayscale(img)
        img2 = tf.image.sobel_edges(img[None, ...])
        equal_img = tfa.image.equalize(img, bins=256)
        img = tf.concat([equal_img, img2[0, :, :, 0]], 2)
        image_array  = tf.keras.preprocessing.image.array_to_img(img)
        
        return image_array
    def __repr__(self):
        return self.__class__.__name__+'()'

class tjwei_augumentation(object):
    def __call__(self, img):
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.rgb_to_grayscale(img)
        img2 = tf.image.sobel_edges(img[None, ...])
        img = tf.concat([img, img2[0, :, :, 0]], 2)
        image_array = tf.keras.preprocessing.image.array_to_img(img)
        
        return image_array
    def __repr__(self):
        return self.__class__.__name__+'()'

data_transforms = {
    "train": transforms.Compose([
        # transforms.Resize([512, 512], interpolation=InterpolationMode.BILINEAR),
        transforms.Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
        # transforms.CenterCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        tjwei_augumentation(),
        transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        # transforms.Resize([512, 512], interpolation=InterpolationMode.BILINEAR),
        transforms.Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
        # transforms.CenterCrop(size=(224, 224)),
        # transforms.RandomHorizontalFlip(),
        tjwei_augumentation(),
        transforms.ToTensor()
    ])
}

class AI9_Dataset(Dataset):
    def __init__(self, feature, target, name, transform=None):
        self.X = feature # path
        self.Y = target # label
        self.N = name # name
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        return self.transform(img), self.Y[idx], self.N[idx]

def make_training_dataloader(ds):
    mura_ds = ds["train"]["mura"]
    normal_ds = ds["train"]["normal"]
    min_len = min(len(mura_ds), len(normal_ds))
    sample_num = int(4 * min_len)
    # sample_num = 32
    normal_ds = torch.utils.data.Subset(normal_ds,random.sample(list(range(len(normal_ds))), sample_num))
    train_ds = torch.utils.data.ConcatDataset([mura_ds, normal_ds])
    # train_ds = torch.utils.data.ConcatDataset([normal_ds])
    dataloader = DataLoader(train_ds, 
                            batch_size=16,
                            shuffle=True, 
                            num_workers=0,
                           )
    return dataloader

def make_test_dataloader(ds):
    m = ds["test"]["mura"]
    n = ds["test"]["normal"]
    test_ds = torch.utils.data.ConcatDataset([m, n])
    s_dataloader = DataLoader(m, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=0,
                           )
    n_dataloader = DataLoader(n, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=0,
                           )
    return [n_dataloader, s_dataloader]

def make_val_dataloader(ds):
    m = ds["val"]["mura"]
    n = ds["val"]["normal"]
    val_ds = torch.utils.data.ConcatDataset([m, n])
    dataloader = DataLoader(val_ds, 
                            batch_size=4,
                            shuffle=False, 
                            num_workers=0,
                           )
    return dataloader
