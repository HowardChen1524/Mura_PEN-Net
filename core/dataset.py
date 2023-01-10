import random
from random import shuffle
import os 
import math 
import numpy as np 
from PIL import Image, ImageFilter
from glob import glob

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from core.utils import ZipReader

# New import
from core.image_folder import make_dataset 
import cv2
from core.utils import tensor2im
# from torchvision.transforms.functional import InterpolationMode
import pandas as pd

class AUO_Dataset(torch.utils.data.Dataset):
  def __init__(self, data_args, split='train', level=None):
    super(AUO_Dataset, self).__init__()
    self.split = split # 'train' or 'test'
    self.level = level # None
    self.w, self.h = data_args['w'], data_args['h']

    # 依據現在是要train還是test去放不同參數
    if split == 'train':
      self.dir = data_args['train_data_root']
      self.edge_index_list = [0, 105, 210, 14, 119, 224]
    elif split == 'test':
      self.dir = data_args['test_data_root']

    self.rand_crop_num = data_args['rand_crop_num']
    self.slid_crop_stride = data_args['slid_crop_stride']
    self.data = make_dataset(self.dir) # return image path list (image_folder.py)

    self.color = data_args['color'] # RGB or gray
    self.crop_size = data_args['crop_size'] # 小圖尺寸

    self.mask_type = data_args.get('mask', 'pconv') # if 沒有 mask 這個 key，預設回傳 'pconv' 這個字串
    if self.mask_type == 'pconv':
      self.mask = [os.path.join(data_args['zip_root'], 'mask/{}.png'.format(str(i).zfill(5))) for i in range(2000, 12000)]
      if self.level is not None:
        self.mask = [os.path.join(data_args['zip_root'], 'mask/{}.png'.format(str(i).zfill(5))) for i in range(self.level*2000, (self.level+1)*2000)]
      self.mask = self.mask*(max(1, math.ceil(len(self.data)/len(self.mask))))
    else:
      self.mask = [0]*len(self.data)
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    try:
      item = self.load_item(index)
    except:
      print('loading error: ' + self.data[index])
      raise
    return item

  def load_item(self, index):
    img_path = self.data[index]
    img_name = img_path[len(self.dir):] # 注意路徑結尾要是'/'，才不會取錯
    img = cv2.imread(img_path)  
    img_size = img.shape[:2] # h, w, c
    if img_size != (self.h,self.w): # if not 512,512 -> resize
      img = cv2.resize(img, (self.h,self.w), interpolation=cv2.INTER_AREA)

    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
    img = img.convert(self.color)
    
    crop_imgs = []
    
    img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_tensor = img_transform(img)

    # sliding crop
    (c, w, h) = img_tensor.size()
    y_end_crop, x_end_crop = False, False
    for y in range(0, h, self.slid_crop_stride): # stride default 32
      # print(f"y {y}")
      crop_y = y
      if (y + self.crop_size) > h:
        break
      for x in range(0, w, self.slid_crop_stride):
        crop_x = x
        if (x + self.crop_size) > w:
          break
        crop_img = transforms.functional.crop(img_tensor, crop_y, crop_x, self.crop_size, self.crop_size)
        crop_imgs.append(crop_img)

    if self.split == 'train':
      crop_index_list = []
      for i in range(0,225):
        if i not in self.edge_index_list:
            crop_index_list.append(i)
      random.shuffle(crop_index_list)
      crop_index_list = crop_index_list[:self.rand_crop_num-len(self.edge_index_list)] + self.edge_index_list
      crop_imgs = [crop_imgs[crop_index] for crop_index in crop_index_list]

    crop_imgs = torch.stack(crop_imgs)
    crop_num = crop_imgs.shape[0]

    # Mask image
    mask = np.zeros((self.crop_size, self.crop_size)).astype(np.uint8) # black
    mask[int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2),
          int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2)] = 255 # white
    mask = Image.fromarray(mask).convert('L') # gray

    mask_transform = transforms.Compose([transforms.ToTensor()]) # 0 ~ 1
    masks = mask_transform(mask).repeat(crop_num,1,1,1)

    return crop_imgs, masks, img_name

  def create_iterator(self, batch_size):
    while True:
      sample_loader = DataLoader(dataset=self,batch_size=batch_size,drop_last=True)
      for item in sample_loader:
        yield item

class AI9_Dataset(torch.utils.data.Dataset):
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