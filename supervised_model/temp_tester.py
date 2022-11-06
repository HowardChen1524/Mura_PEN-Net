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

from core.dataset import Dataset
from core.utils import set_seed, set_device, Progbar, postprocess, tensor2im
from core.loss import AdversarialLoss, PerceptualLoss, StyleLoss, VGG19
from core import metric as module_metric
import cv2
from PIL import Image, ImageDraw

from supervised_model.loss import WeightedFocalLoss
from supervised_model.dataset import AI9_Dataset

class Tester_Supervised():
  def __init__(self, config, ds_type):
    self.config = config
    # setup data set and data loader
    if ds_type == 0: # normal
      self.dataset_type = 'normal'
    elif ds_type == 1:
      self.dataset_type = 'smura'
    self.test_dataset = AI9_Dataset(config['data_loader'], split='test')
    self.test_loader = DataLoader(self.test_dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=0, 
                            pin_memory=True)
      
    # self.loss = WeightedFocalLoss(config['supervised']['focalloss_alpha'],config['supervised']['focalloss_gamma'])

    # Model
    self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
    self.model.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid()
    )
    self.model.load_state_dict(torch.load('./supervised_model/model.pt'))
    
    self.model.eval().cuda()

  def test(self):
    preds_res = []
    labels_res = []
    files_res = []
    
    
    for inputs, labels, names in tqdm(self.test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
          preds = model(inputs)
        
        preds = torch.reshape(preds, (-1,)).cpu()
        labels = labels.cpu()
        
        names = list(names)

        files_res.extend(names)
        preds_res.extend(preds)
        labels_res.extend(labels)

    preds_res = np.array(preds_res)
    labels_res = np.array(labels_res)

    model_pred_result = predict_report(preds_res, labels_res, files_res)
    model_pred_result.to_csv(os.path.join(save_path, "model_pred_result.csv"), index=None)
    print("model predict record finished!")

    fig = plot_roc_curve(labels_res, preds_res)
    fig.savefig(os.path.join(save_path, "roc_curve.png"))
    print("roc curve saved!")

    model_report, curve_df = calc_matrix(labels_res, preds_res)
    model_report.to_csv(os.path.join(save_path, "model_report.csv"))
    curve_df.to_csv(os.path.join(save_path, "model_precision_recall_curve.csv"))
    print("model report record finished!")


