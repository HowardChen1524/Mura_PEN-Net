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
      if not data_args['continue']:
        self.dir = data_args['train_data_root']
        self.rand_crop_num = data_args['rand_crop_num']
        self.data = make_dataset(self.dir) # return image path list (image_folder.py)
        
        shuffle(self.data)
        if data_args['train_data_max'] != -1: # -1 表示全部
          self.data = self.data[:data_args['train_data_max']]

        recover_list = []
        for i, path in enumerate(self.data):
            # print(i)
            recover_list.append(path[len(self.dir)+1:].replace('.png','.bmp'))
        recover_df = pd.DataFrame(recover_list, columns=['PIC_ID'])
        recover_df.to_csv('./training_imgs.csv', index=False, columns=['PIC_ID'])
        print(f"Record {len(self.data)} filename successful!")
      else:
        recover_list = []
        recover_df = pd.read_csv('./training_imgs.csv')
        data_df = pd.read_csv(r'/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv')
        recover_fn = pd.merge(recover_df, data_df, on='PIC_ID', how='inner')['PIC_ID'].tolist()
        for fn in recover_fn:
            recover_list.append(f"{self.dir_A}{fn.replace('bmp','png')}")
        self.data = recover_list
        print(f"Recover img num: {len(self.data)}")

    elif split == 'test':
      self.dir = data_args['test_data_root']
      self.slid_crop_stride = data_args['slid_crop_stride']
      self.data = make_dataset(self.dir)
      # shuffle(self.data)
      if data_args['test_data_max'] != -1:
        self.data = self.data[:data_args['test_data_max']]

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
    
    # self.data.sort()

  def __len__(self):
    return len(self.data)
  
  def set_subset(self, start, end):
    self.mask = self.mask[start:end]
    self.data = self.data[start:end] 

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
    
    # do crop prepro
    # # load mask 
    # if self.mask_type == 'pconv':
    #   m_index = random.randint(0, len(self.mask)-1) if self.split == 'train' else index
    #   mask_path = os.path.dirname(self.mask[m_index]) + '.zip'
    #   mask_name = os.path.basename(self.mask[m_index])
    #   mask = ZipReader.imread(mask_path, mask_name).convert('L')
    # else: # square
    #   m = np.zeros((self.h, self.w)).astype(np.uint8)
    #   if self.split == 'train':
    #     t, l = random.randint(0, self.h//2), random.randint(0, self.w//2)
    #     m[t:t+self.h//2, l:l+self.w//2] = 255
    #   else:
    #     m[self.h//4:self.h*3//4, self.w//4:self.w*3//4] = 255
    #   mask = Image.fromarray(m).convert('L')
    
    # augment 
    # if self.split == 'train': 
    #   img = transforms.RandomHorizontalFlip()(img)
    #   img = transforms.ColorJitter(0.05, 0.05, 0.05, 0.05)(img)
    #   mask = transforms.RandomHorizontalFlip()(mask)
    #   mask = mask.rotate(random.randint(0,45), expand=True)
    #   mask = mask.filter(ImageFilter.MaxFilter(3))
    # img = img.resize((self.w, self.h))
    # mask = mask.resize((self.w, self.h), Image.NEAREST)
    # return F.to_tensor(img)*2-1., F.to_tensor(mask), img_name
    
    crop_imgs = []
    if self.split == 'train':
      img_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      transforms.RandomCrop(self.crop_size)])
      for i in range(self.rand_crop_num):
          crop_imgs.append(img_transform(img))
    elif self.split == 'test':
      img_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      img_tensor = img_transform(img)

      # sliding crop
      (c, w, h) = img_tensor.size()
      y_end_crop, x_end_crop = False, False
      for y in range(0, h, self.slid_crop_stride): # stride default 32
          # print(f"y {y}")
          # y_end_crop = False
          crop_y = y
          if (y + self.crop_size) > h:
              # crop_y =  w - self.opt.fineSize
              # y_end_crop = True
              break
          for x in range(0, w, self.slid_crop_stride):
              # print(f"x {x}")
              # x_end_crop = False
              crop_x = x
              if (x + self.crop_size) > w:
                  # crop_x = h - self.opt.fineSize
                  # x_end_crop = True
                  break
              crop_img = transforms.functional.crop(img_tensor, crop_y, crop_x, self.crop_size, self.crop_size)
              crop_imgs.append(crop_img)
              
              # if x_end_crop:
              #    break
          # if x_end_crop and y_end_crop:
              # break

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