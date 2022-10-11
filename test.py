# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import argparse
import copy
import importlib
import datetime
import random
import sys
import json
import glob

### My libs
from core.utils import set_device, postprocess, ZipReader, set_seed
from core.utils import postprocess
from core.dataset import Dataset
from core.tester import Tester

from datetime import date
from collections import defaultdict
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="MGP")
parser.add_argument("-c", "--config", type=str, required=True)
# parser.add_argument("-l", "--level",  type=int, required=True)
parser.add_argument("-l", "--level",  type=int)
parser.add_argument("-mn", "--model_name", type=str, required=True)
parser.add_argument("-m", "--mask", default=None, type=str)
parser.add_argument("-s", "--size", default=None, type=int)
parser.add_argument("-p", "--port", type=str, default="23451")
# new add
parser.add_argument("-me", "--model_epoch", type=int, default=-1)
parser.add_argument("-as", "--anomaly_score", type=str, default='MSE', help='MSE | Mask_MSE')
parser.add_argument("-np", "--normal_path", type=str, default=None)
parser.add_argument("-sp", "--smura_path", type=str, default=None)
parser.add_argument("-t", "--test_type", type=str, default='normal', help='normal | position')
parser.add_argument("-n", "--normalized", action='store_true', help='Use for typecplus')

args = parser.parse_args()

# world_size : GPU num -> 1
# local_rank : GPU device -> 0
# global_rank : GPU device -> 0
def roc(labels, scores, path, name):


    fpr, tpr, th = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    
    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]

    plot_roc_curve(fpr, tpr, path, name)
    # optimal_th = 6.4e-05
    # optimal_th = 5.75e-05
    return roc_auc, optimal_th
def plot_roc_curve(fpr, tpr, path, name):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(f"{path}/{name}_roc.png")
    plt.clf()
def plot_distance_distribution(n_scores, s_scores, path, name):
    # bins = np.linspace(0.000008,0.00005) # Mask MSE
    plt.hist(s_scores, bins=50, alpha=0.5, density=True, label="smura")
    plt.hist(n_scores, bins=50, alpha=0.5, density=True, label="normal")
    plt.xlabel('Anomaly Score')
    plt.title('Score Distribution')
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_dist_mean.png")
    plt.clf()
def plot_distance_scatter(n_max, s_max, n_mean, s_mean, path, name):
    # normal
    x1 = n_max
    y1 = n_mean
    # smura
    x2 = s_max
    y2 = s_mean
    # 設定座標軸
    # normal
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.3, label="normal")
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_normal_scatter.png")
    plt.clf()
    # smura
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.3, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_smura_scatter.png")
    plt.clf()
    # all
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.3, label="normal")
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.3, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}__scatter.png")
    plt.clf()
def prediction(labels, scores, path, name):
    result_msg = ''
    pred_labels = [] 
    roc_auc, optimal_th = roc(labels, scores, path, name)
    for score in scores:
        if score >= optimal_th:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    
    cm = confusion_matrix(labels, pred_labels)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]
    DATA_NUM = TN + FP + FN + TP
    result_msg += f"Confusion Matrix (row1: TN,FP | row2: FN,TP):\n{cm}"
    result_msg += f"\nAUC: {roc_auc}\n"
    result_msg += f"Threshold (highest TPR-FPR): {optimal_th}\n"
    result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
    result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
    result_msg += f"TNR: {TN/(FP+TN)}\n"
    result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
    result_msg += f"NPV: {TN/(FN+TN)}\n"
    result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
    result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
    result_msg += f"F1-Score: {f1_score(labels, pred_labels)}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
    return result_msg
def max_meam_prediction(labels, max_scores, mean_scores, path, name):
    result_msg = ''
    # score = a*max + b*mean
    best_a, best_b = 0, 0
    best_auc = 0
    for ten_a in range(0, 10, 1):
        a = ten_a/10.0
        for ten_b in range(0, 10, 1):
            b = ten_b/10.0
            
            scores = a*max_scores + b*mean_scores
            fpr, tpr, th = roc_curve(labels, scores)
            current_auc = auc(fpr, tpr)
            if current_auc >= best_auc:
                best_auc = current_auc
                best_a = a
                best_b = b         

    result_msg += f"Param a: {best_a}, b: {best_b}\n"

    best_scores = best_a*max_scores + best_b*mean_scores
    pred_labels = [] 
    roc_auc, optimal_th = roc(labels, best_scores, path, name)
    for score in best_scores:
        if score >= optimal_th:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    
    cm = confusion_matrix(labels, pred_labels)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]
    DATA_NUM = TN + FP + FN + TP
    result_msg += f"Confusion Matrix (row1: TN,FP | row2: FN,TP):\n{cm}"
    result_msg += f"\nAUC: {roc_auc}\n"
    result_msg += f"Threshold (highest TPR-FPR): {optimal_th}\n"
    result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
    result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
    result_msg += f"TNR: {TN/(FP+TN)}\n"
    result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
    result_msg += f"NPV: {TN/(FN+TN)}\n"
    result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
    result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
    result_msg += f"F1-Score: {f1_score(labels, pred_labels)}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
    return result_msg
def show_and_save_result(result, path, name):
  all_max_anomaly_score = np.concatenate([result['max']['n'], result['max']['s']])
  all_mean_anomaly_score = np.concatenate([result['mean']['n'], result['mean']['s']])
  true_label = [0]*result['mean']['n'].shape[0]+[1]*result['mean']['s'].shape[0]
  
  plot_distance_distribution(result['mean']['n'], result['mean']['s'], path, name)
  plot_distance_scatter(result['max']['n'], result['max']['s'], 
                          result['mean']['n'], result['mean']['s'], path, name)

  log_name = os.path.join(path, 'result_log.txt')
  msg = ''
  with open(log_name, "w") as log_file:
    now = time.strftime("%c")
    msg += f"=============== Testing result {now} ===================\n"
    msg += f"=============== All small image mean & std =============\n"
    msg += f"Normal mean: {result['all']['n'].mean()}\n"
    msg += f"Normal std: {result['all']['n'].std()}\n"
    msg += f"Smura mean: {result['all']['s'].mean()}\n"
    msg += f"Smura std: {result['all']['s'].std()}\n"
    msg += f"=============== anomaly max prediction =================\n"    
    msg += prediction(true_label, all_max_anomaly_score, path, f"{name}_max")
    msg += f"=============== anomaly mean prediction ================\n"
    msg += prediction(true_label, all_mean_anomaly_score, path, f"{name}_mean")
    msg += f"=============== anomaly max & mean prediction ==========\n"
    msg += max_meam_prediction(true_label, all_max_anomaly_score, all_mean_anomaly_score, path, f"{name}_max_mean")
    
    log_file.write(msg)

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  set_seed(config['seed'])

  if config['test_type'] == 'normal':
    result_dict = defaultdict(list)
    result_dict['all'] = defaultdict(list)
    result_dict['max'] = defaultdict(list)
    result_dict['mean'] = defaultdict(list)

    dataset_type_list = [config['data_loader']['test_data_root_normal'], config['data_loader']['test_data_root_smura']]
    for idx, data_path in enumerate(dataset_type_list):
      config['data_loader']['test_data_root'] = data_path
      print(f"Now path: {data_path}")

      tester = Tester(config) # Default debug = False
      big_imgs_scores, big_imgs_scores_max, big_imgs_scores_mean = tester.test() # max mean 
      
      if idx == 0:
        result_dict['all']['n'] = big_imgs_scores.copy() # all 小圖
        result_dict['max']['n'] = big_imgs_scores_max.copy() # max
        result_dict['mean']['n'] = big_imgs_scores_mean.copy() # mean
      elif idx == 1:
        result_dict['all']['s'] = big_imgs_scores.copy() # all 小圖
        result_dict['max']['s'] = big_imgs_scores_max.copy() # max
        result_dict['mean']['s'] = big_imgs_scores_mean.copy() # mean
      else:
        raise
    result_name = f"{config['data_loader']['name']}_crop{config['data_loader']['crop_size']}_{config['anomaly_score']}_epoch{config['model_epoch']}"
    show_and_save_result(result_dict, config['result_path'], result_name)
  elif config['test_type'] == 'position':
    df = pd.read_csv(r'./Mura_type_c_plus.csv')
    print(f"Mura 最大 :\n{df.iloc[(df['h']+df['w']).argmax()][['fn','w','h']]}")
    print(f"Mura 最小 :\n{df.iloc[(df['h']+df['w']).argmin()][['fn','w','h']]}")

    result_dict = defaultdict(list)
    result_dict['all'] = defaultdict(list)

    if config['normalized']:
      # for idx, data_path in enumerate([config['data_loader']['test_data_root_normal']]):
      #   config['data_loader']['test_data_root'] = data_path
      #   print("Start to compute normal mean and std")
      #   print(f"Now path: {data_path}")

      #   tester = Tester(config) # Default debug = False
      #   big_imgs_scores, _, _ = tester.test() # max mean 
      #   normal_mean = big_imgs_scores.mean()
      #   normal_std = big_imgs_scores.std()
      # print(normal_mean)
      # print(normal_std)
      # Mask
      # normal_mean = 4.2993142e-05
      # normal_std = 1.3957943e-05
      normal_mean = 4.2993142e-05
      normal_std = 1.3958662e-05

    for idx, data_path in enumerate([config['data_loader']['test_data_root_smura']]):
      config['data_loader']['test_data_root'] = data_path
      print(f"Now path: {data_path}")

      tester = Tester(config) # Default debug = False

      if config['normalized']:
        big_imgs_scores = tester.test_position(df, normal_mean, normal_std)
        print(f"Mean: {big_imgs_scores.mean()}")
        print(f"std: {big_imgs_scores.std()}")
      else:
        big_imgs_scores = tester.test_position(df)

      result_dict['all']['s'] = big_imgs_scores.copy() # all 小圖
    
    print(f"Smura mean: {result_dict['all']['s'].mean()}")
    print(f"Smura std: {result_dict['all']['s'].std()}")
    

if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  config = json.load(open(args.config))
  if args.mask is not None:
    config['data_loader']['mask'] = args.mask
  if args.size is not None:
    config['data_loader']['w'] = config['data_loader']['h'] = args.size
  if args.normal_path is not None:
    config['data_loader']['test_data_root_normal'] = args.normal_path
  if args.smura_path is not None:
    config['data_loader']['test_data_root_smura'] = args.smura_path
  config['model_name'] = args.model_name
  # config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}_{}'.format(config['model_name'], 
  #   config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w'], date.today()))
  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}_{}'.format(config['model_name'], 
    config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w'], "2022-09-29"))
  config['model_epoch'] = args.model_epoch
  config['anomaly_score'] = args.anomaly_score
  config['test_type'] = args.test_type
  config['normalized'] = args.normalized


  gpu_device = 0
  # print('using {} GPUs for testing ... '.format(ngpus_per_node))
  print(f'using GPU device {gpu_device} for testing ... ')
  # setup distributed parallel training environments
  # ngpus_per_node = torch.cuda.device_count()
  # config['world_size'] = ngpus_per_node
  # config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  # config['distributed'] = True
  # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))

  # create result directory
  result_path = os.path.join(config['save_dir'], 'results_{}_{}'.format(str(config['model_epoch']).zfill(5), config['anomaly_score']))
  os.makedirs(result_path, exist_ok=True)

  config['result_path'] = result_path
  config['distributed'] = False
  main_worker(gpu_device, 1, config) # GPU device, GPU num, config