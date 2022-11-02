import os
import time
import math
import glob
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
import torch.distributed as dist

from core.dataset import AUO_Dataset
from core.utils import set_seed, set_device, Progbar, postprocess, tensor2im
from core.loss import AdversarialLoss, PerceptualLoss, StyleLoss, VGG19
from core import metric as module_metric
import cv2
from PIL import Image, ImageDraw

class Tester():
  def __init__(self, config):
    self.config = config
    self.crop_size = config['data_loader']['crop_size']
    self.crop_stride = config['data_loader']['slid_crop_stride']
    self.w = config['data_loader']['w']
    self.h = config['data_loader']['h']
    # setup data set and data loader
    self.test_dataset = AUO_Dataset(config['data_loader'], split='test')
    self.test_loader = DataLoader(self.test_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=config['trainer']['num_workers'], 
                            pin_memory=True)
   
    # create inpainting & position directory
    if self.config['test_type'] == 'position':
      if self.config['pos_normalized']:
        self.inpainting_path = os.path.join(self.config['result_path'], 'check_inpaint_pos_normalized')
        os.makedirs(self.inpainting_path, exist_ok=True)
        self.draw_path = os.path.join(self.config['result_path'], 'true_pred_postion_pos_normalized')
        os.makedirs(self.draw_path, exist_ok=True)
      else:
        self.inpainting_path = os.path.join(self.config['result_path'], 'check_inpaint')
        os.makedirs(self.inpainting_path, exist_ok=True)
        self.draw_path = os.path.join(self.config['result_path'], 'true_pred_postion')
        os.makedirs(self.draw_path, exist_ok=True)

    # Model    
    net = importlib.import_module('model.'+config['model_name']) # 根據不同項目的配置，動態導入對應的模型
    self.netG = set_device(net.InpaintGenerator())
    self.netD = set_device(net.Discriminator(in_channels=3, use_sigmoid=config['losses']['gan_type'] != 'hinge'))
    self.load()
    
    # anomaly score
    self.l1_loss = nn.L1Loss()
    self.l2_loss = nn.MSELoss()

  # load netG and netD
  def load(self):
    if self.config['model_epoch'] == -1: # latest
      ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(os.path.join(self.config['save_dir'], '*.pth'))]
      ckpts.sort()
      model_epoch = ckpts[-1] if len(ckpts)>0 else None
    else:
      model_epoch = self.config['model_epoch']
    
    model_path = self.config['save_dir']

    gen_path = os.path.join(model_path, 'gen_{}.pth'.format(str(model_epoch).zfill(5)))
    print('Loading G model from {}...'.format(gen_path)) 
    data = torch.load(gen_path, map_location = lambda storage, loc: set_device(storage)) 
    self.netG.load_state_dict(data['netG'])
    self.netG.eval()
    dis_path = os.path.join(model_path, 'dis_{}.pth'.format(str(model_epoch).zfill(5)))
    print('Loading D model from {}...'.format(dis_path))
    data = torch.load(dis_path, map_location = lambda storage, loc: set_device(storage)) 
    self.netD.load_state_dict(data['netD'])
    self.netD.eval()
    
  def compute_score(self, imgs, feats, re_imgs, anomaly_score):
    if anomaly_score == 'MSE':
      crop_scores = []
      for i in range(0,225):  
          crop_scores.append(self.l2_loss(imgs[i], re_imgs[i]).detach().cpu().numpy())
      crop_scores = np.array(crop_scores)
      return crop_scores    

    elif anomaly_score == 'Mask_MSE':
      mask_imgs = imgs[:, :, int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2),
           int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2)]
      mask_re_imgs = re_imgs[:, :, int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2),
           int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2)]
      crop_scores = []
      for i in range(0,225):  
          crop_scores.append(self.l2_loss(mask_imgs[i], mask_re_imgs[i]).detach().cpu().numpy())
      crop_scores = np.array(crop_scores)
      return crop_scores    

    elif anomaly_score == 'MAE': 
      crop_scores = []
      for i in range(0,225):  
          crop_scores.append(self.l1_loss(imgs[i], re_imgs[i]).detach().cpu().numpy())
      crop_scores = np.array(crop_scores)
      return crop_scores  

    elif anomaly_score == 'Mask_MAE': 
      mask_imgs = imgs[:, :, int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2),
           int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2)]
      mask_re_imgs = re_imgs[:, :, int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2),
           int(self.crop_size/4):int(self.crop_size/4)+int(self.crop_size/2)]
      crop_scores = []
      for i in range(0,225):  
          crop_scores.append(self.l1_loss(mask_imgs[i], mask_re_imgs[i]).detach().cpu().numpy())
      crop_scores = np.array(crop_scores)
      return crop_scores   

    elif anomaly_score == 'Discriminator': 
      ori_feat = self.netD(imgs)
      re_feat = self.netD(re_imgs)
      # MSE
      crop_scores = []
      for i in range(0,225):  
          crop_scores.append(self.l2_loss(ori_feat[i], re_feat[i]).detach().cpu().numpy())
      crop_scores = np.array(crop_scores)
      return crop_scores 

    elif anomaly_score == 'Pyramid_L1': 
      crop_scores = []
      for i in range(0,225):  
        pyramid_loss = 0 
        for _, f in enumerate(feats):
          pyramid_loss += self.l1_loss(f, F.interpolate(imgs, size=f.size()[2:4], mode='bilinear', align_corners=True)).detach().cpu().numpy()
        crop_scores.append(pyramid_loss)
      crop_scores = np.array(crop_scores)
      return crop_scores
    
    else:
      raise ValueError("Please choose one measure mode!")
  
  def draw_mura_position(self, w, h, fp, fn, fn_series_list, imgs_pos, stride):
    # 讀取大圖
    img = Image.open(fp)
    img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
    # =====actual mura===== 
    actual_pos_list = []
    for i in range(0, fn_series_list.shape[0]):
        fn_series = fn_series_list.iloc[i]
        # actual_pos_list.append((fn_series['x0'], fn_series['y0'], fn_series['x1'], fn_series['y1']))
        actual_pos_list.append((int(fn_series['x0']/3.75), int(fn_series['y0']/2.109375), int(fn_series['x1']/ 3.75), int(fn_series['y1']/2.109375))) # 1920*1080 -> 512*512
    
    for actual_pos in actual_pos_list:
        draw = ImageDraw.Draw(img)  
        draw.rectangle(actual_pos, outline ="yellow")
    
    # =====predict mura=====
    if 'Mask' in self.config['anomaly_score']:
      bounding_box = (self.crop_size//2,self.crop_size//2) # 32
    else:
      bounding_box = (self.crop_size,self.crop_size) # 64

    for crop_pos in imgs_pos:
        # max_crop_img_pos 0~225
        x = crop_pos % 15  # 0~14
        y = crop_pos // 15 # 0~14
        # 0,32,64,96,128,160,192,224,256,288,320,352,384,416,448
        crop_x = x * stride
        crop_y = y * stride
        
        # 如果是最後一塊
        if crop_x + self.crop_size > 512:
          crop_x = 512 - self.crop_size # 448
        if crop_y + self.crop_size > 512:
          crop_y = 512 - self.crop_size # 448

        if 'Mask' in self.config['anomaly_score']:
          pred_pos = [crop_x+(bounding_box[0]//2), crop_y+(bounding_box[1]//2), crop_x+(bounding_box[0]//2)+bounding_box[0]-1, crop_y+(bounding_box[1]//2)+bounding_box[1]-1]
        else:
          pred_pos = [crop_x, crop_y, crop_x+bounding_box[0]-1, crop_y+bounding_box[1]-1]
      
        # create rectangle image
        draw = ImageDraw.Draw(img)  
        draw.rectangle(pred_pos, outline ="red")

    img.save(os.path.join(self.draw_path, fn))

  def test(self, n_mean, n_std):
    big_imgs_scores = None
    big_imgs_scores_max = None
    big_imgs_scores_mean = None
    big_imgs_fn = []
    # iteration through datasets
    for idx, (images, masks, names) in enumerate(self.test_loader):
      print('[{}] {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        idx, len(self.test_loader), names[0]))
      big_imgs_fn.append(names[0])
      # delete first dim
      bs, ncrops, c, h, w = images.size()
      images = images.view(-1, c, h, w)
      bs, ncrops, c, h, w = masks.size()
      masks = masks.view(-1, c, h, w)
      
      images, masks = set_device([images, masks])
      images_masked = images*(1-masks) + masks # 中心變白的
      with torch.no_grad():
        feats, output = self.netG(torch.cat((images_masked, masks), dim=1), masks)
      
      # compute loss
      imgs_scores = self.compute_score(images, feats, output, self.config['anomaly_score'])

      # 如果是正常測試且需要做 normalized 才跑
      if self.config['pos_normalized'] and self.config['test_type'] == "normal": 
        if len(n_mean) == 0 or len(n_std) == 0:
          pass
        else:
          for pos in range(0,imgs_scores.shape[0]):
            imgs_scores[pos] = (imgs_scores[pos]-n_mean[pos])/n_std[pos]

      max_score = np.max(imgs_scores) # Anomaly max
      mean_score = np.mean(imgs_scores) # Anomaly mean
      print(f"{self.config['anomaly_score']} Max: {max_score}")
      print(f"{self.config['anomaly_score']} Mean: {mean_score}")

      if idx == 0:
        big_imgs_scores = imgs_scores.copy()
        big_imgs_scores_max = max_score.copy()
        big_imgs_scores_mean = mean_score.copy()
      else:
        big_imgs_scores = np.append(big_imgs_scores, imgs_scores)
        big_imgs_scores_max = np.append(big_imgs_scores_max, max_score)
        big_imgs_scores_mean = np.append(big_imgs_scores_mean, mean_score)

    return big_imgs_scores, big_imgs_scores_max, big_imgs_scores_mean, np.array(big_imgs_fn)

  def test_position(self, df, n_mean=None, n_std=None):
    big_imgs_scores = None
    # iteration through datasets
    for idx, (images, masks, names) in enumerate(self.test_loader):
      print('[{}] {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        idx, len(self.test_loader), names[0]))

      # delete first dim
      bs, ncrops, c, h, w = images.size()
      images = images.view(-1, c, h, w)
      bs, ncrops, c, h, w = masks.size()
      masks = masks.view(-1, c, h, w)

      images, masks = set_device([images, masks])
      images_masked = images*(1-masks) + masks
      with torch.no_grad():
        feats, output = self.netG(torch.cat((images_masked, masks), dim=1), masks)
      
      # compute loss
      imgs_scores = self.compute_score(images, feats, output, self.config['anomaly_score'])

      # 如果需要做 normalized 才跑
      if self.config['pos_normalized']:
        for pos in range(0,imgs_scores.shape[0]):
          imgs_scores[pos] = (imgs_scores[pos]-n_mean[pos])/n_std[pos]
        # if n_mean == None or n_std == None: # use a.any() or a.all()
        #   raise
        # else:
          # for pos in range(0,imgs_scores.shape[0]):
          #   imgs_scores[pos] = (imgs_scores[pos]-n_mean[pos])/n_std[pos]

      # find img info from df
      fn_series_list = df[df['fn']==names[0]]
      inpainting_img_path = os.path.join(self.inpainting_path, names[0][:-4])
      os.makedirs(inpainting_img_path, exist_ok=True)

      # the num of mura
      top_n = fn_series_list.shape[0]
      imgs_pos = np.argsort(-imgs_scores)[:top_n] # 取前 n 張

      # save pred pos
      imgs_pos_str = [f"{pos}\n" for pos in imgs_pos]
      with open(os.path.join(self.draw_path, f'{names[0]}_pred_pos.txt'), 'w') as f:
          f.writelines(imgs_pos_str)
      
      # draw pos
      fp = f"{self.config['data_loader']['test_data_root_smura']}/{names[0]}"
      self.draw_mura_position(self.w, self.h, fp, names[0], fn_series_list, imgs_pos, self.crop_stride)

      # save inpainting image
      for i in range(0, images.shape[0]): 
        inpainting_img_pos_path = os.path.join(inpainting_img_path, str(i))
        os.makedirs(inpainting_img_pos_path, exist_ok=True)
        fake_img = tensor2im(torch.unsqueeze(output[i],0))
        fake_img.save(os.path.join(inpainting_img_pos_path, 'inpaint.png'))
        real_img = tensor2im(torch.unsqueeze(images[i],0))
        real_img.save(os.path.join(inpainting_img_pos_path, 'origin.png'))

      if idx == 0:
        big_imgs_scores = imgs_scores.copy()
      else:
        big_imgs_scores = np.append(big_imgs_scores, imgs_scores)

    print(big_imgs_scores.mean())
    print(big_imgs_scores.std())
      
    return big_imgs_scores

  def position_normalize(self):
    n_all_crop_scores = []
    # iteration through datasets
    for idx, (images, masks, names) in enumerate(self.test_loader):
      print('[{}] {}/{}: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        idx, len(self.test_loader), names[0]))
      
      # delete first dim
      bs, ncrops, c, h, w = images.size()
      images = images.view(-1, c, h, w)
      bs, ncrops, c, h, w = masks.size()
      masks = masks.view(-1, c, h, w)
      
      images, masks = set_device([images, masks])
      images_masked = images*(1-masks) + masks # 中心變白的
      with torch.no_grad():
        feats, output = self.netG(torch.cat((images_masked, masks), dim=1), masks)
      
      # compute loss
      imgs_scores = self.compute_score(images, feats, output, self.config['anomaly_score'])

      n_all_crop_scores.append(imgs_scores)

    n_all_crop_scores = np.array(n_all_crop_scores)
    print(n_all_crop_scores.shape)
    n_pos_mean = np.mean(n_all_crop_scores, axis=0)
    n_pos_std = np.std(n_all_crop_scores, axis=0)

    return n_pos_mean, n_pos_std