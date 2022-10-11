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

# supervised
from supervised_model.wei_dataloader import make_test_dataloader, make_training_dataloader, AI9_Dataset, data_transforms
from tqdm import tqdm
from sklearn import metrics
import numpy as np

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
parser.add_argument("-pn", "--pos_normalized", action='store_true', help='Use for typecplus')
parser.add_argument("-mm", "--minmax", action='store_true', help='Use for combine supervised')
args = parser.parse_args()

# supervised
def calc_metric(labels_res, pred_res, threshold):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=labels_res, y_pred=(pred_res >= threshold)).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return threshold, tpr, tnr, precision, recall 

def predict_report(preds, labels, names):
    df_res = pd.DataFrame(list(zip(names, preds)), columns=["Img", "Predict"])
    df_res["Label"] = labels
    return df_res

def get_curve_df(labels_res, preds_res):
    pr_list = []

    for i in tqdm(np.linspace(0, 1, num=10001)):
        pr_result = calc_metric(labels_res, preds_res, i)
        pr_list.append(pr_result)

    curve_df = pd.DataFrame(pr_list, columns=['threshold', 'tpr', 'tnr', 'precision', 'recall'])
    
    return curve_df

def calc_matrix(labels_res, preds_res):
    results = {'accuracy': [],
           'balance_accuracy': [],
           'tpr': [],
           'tnr': [],
           'tnr0.99_precision': [],
           'tnr0.99_recall': [],
           'tnr0.995_precision': [],
           'tnr0.995_recall': [],
           'tnr0.999_precision': [],
           'tnr0.999_recall': [],
           'tnr0.9996_precision': [],
           'tnr0.9996_recall': [],
           'precision': [],
           'recall': []
    }

    tn, fp, fn, tp = metrics.confusion_matrix(y_true=labels_res, y_pred=(preds_res >= 0.5)).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    fnr = fn / (tp + fn)
    fpr = fp / (fp + tn)

    results['accuracy'].append((tn + tp) / (tn + fp + fn + tp))
    results['tpr'].append(tpr)
    results['tnr'].append(tnr) 
    results['balance_accuracy'].append(((tp / (tp + fn) + tn / (tn + fp)) / 2))
    results['precision'].append(tp / (tp + fp))
    results['recall'].append(tp / (tp + fn))

    curve_df = get_curve_df(labels_res, preds_res)
    results['tnr0.99_recall'].append((((curve_df[curve_df['tnr'] > 0.99].iloc[0]) + (curve_df[curve_df['tnr'] < 0.99].iloc[-1])) / 2).recall)
    results['tnr0.995_recall'].append((((curve_df[curve_df['tnr'] > 0.995].iloc[0]) + (curve_df[curve_df['tnr'] < 0.995].iloc[-1])) / 2).recall)
    results['tnr0.99_precision'].append((((curve_df[curve_df['tnr'] > 0.99].iloc[0]) + (curve_df[curve_df['tnr'] < 0.99].iloc[-1])) / 2).precision)
    results['tnr0.995_precision'].append((((curve_df[curve_df['tnr'] > 0.995].iloc[0]) + (curve_df[curve_df['tnr'] < 0.995].iloc[-1])) / 2).precision)
    results['tnr0.999_recall'].append((((curve_df[curve_df['tnr'] > 0.999].iloc[0]) + (curve_df[curve_df['tnr'] < 0.999].iloc[-1])) / 2).recall)
    results['tnr0.999_precision'].append((((curve_df[curve_df['tnr'] > 0.999].iloc[0]) + (curve_df[curve_df['tnr'] < 0.999].iloc[-1])) / 2).precision)
    results['tnr0.9996_recall'].append((((curve_df[curve_df['tnr'] > 0.9996].iloc[0]) + (curve_df[curve_df['tnr'] < 0.9996].iloc[-1])) / 2).recall)
    results['tnr0.9996_precision'].append((((curve_df[curve_df['tnr'] > 0.9996].iloc[0]) + (curve_df[curve_df['tnr'] < 0.9996].iloc[-1])) / 2).precision)

def get_data_info(t, l, image_info, csv_path):
    res = []
    image_info = image_info[(image_info["train_type"] == t) & (image_info["label"] == l) & (image_info["PRODUCT_CODE"] == "T850MVR05")]
        
    for path, img, label, JND, t in zip(image_info["path"],image_info["name"],image_info["label"],image_info["MULTI_JND"],image_info["train_type"]):
        img_path = os.path.join(os.path.dirname(csv_path), path,img)
        res.append([img_path, label, JND, t, img])
    X = []
    Y = []
    N = []
    
    for d in res:
        # dereference ImageFile obj
        X.append(os.path.join(d[0]))
        Y.append(d[1])
        N.append(d[4])
    dataset = AI9_Dataset(feature=X,
                          target=Y,
                          name=N,
                          transform=data_transforms[t])
    # print(dataset.__len__())
    return dataset

def plot_roc_curve_supervised(labels_res, preds_res):
    fpr, tpr, threshold = metrics.roc_curve(y_true=labels_res, y_score=preds_res)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    return plt

def evaluate(model, testloaders, save_path='./supervised_model/'):
    model.eval().cuda()
    res = defaultdict(dict)
    for l in ['preds_res','labels_res','files_res']:
      for t in ['n', 's']:
        res[l][t] = []
    # preds_res = []
    # labels_res = []
    # files_res = []

    with torch.no_grad():
      for idx, loader in enumerate(testloaders):
        for inputs, labels, names in tqdm(loader):
          inputs = inputs.cuda()
          labels = labels.cuda()
          
          preds = model(inputs)
          
          preds = torch.reshape(preds, (-1,)).cpu()
          labels = labels.cpu()
          
          names = list(names)

          if idx == 0:
            res['files_res']['n'].extend(names)
            res['preds_res']['n'].extend(preds)
            res['labels_res']['n'].extend(labels)
          elif idx == 1:
            res['files_res']['s'].extend(names)
            res['preds_res']['s'].extend(preds)
            res['labels_res']['s'].extend(labels)
      
          # files_res.extend(names)
          # preds_res.extend(preds)
          # labels_res.extend(labels)
    res['files_res']['all'] = res['files_res']['n'] + res['files_res']['s']
    res['preds_res']['all'] = np.array(res['preds_res']['n'] + res['preds_res']['s'])
    res['labels_res']['all'] = np.array(res['labels_res']['n'] + res['labels_res']['s'])
    # preds_res = np.array(preds_res)
    # labels_res = np.array(labels_res)
    # print(preds_res)
    # print(labels_res)
    # raise
    model_pred_result = predict_report(res['preds_res']['all'], res['labels_res']['all'], res['files_res']['all'])
    model_pred_result.to_csv(os.path.join(save_path, "model_pred_result.csv"), index=None)
    print("model predict record finished!")

    fig = plot_roc_curve_supervised(res['labels_res']['all'], res['preds_res']['all'])
    fig.savefig(os.path.join(save_path, "roc_curve.png"))
    print("roc curve saved!")
  
    return res
# unsupervised
# world_size : GPU num -> 1
# local_rank : GPU device -> 0
# global_rank : GPU device -> 0
def roc(labels, scores, path, name):

    fpr, tpr, th = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    
    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]

    plot_roc_curve(fpr, tpr, path, name)
    
    return roc_auc, optimal_th

def plot_roc_curve(fpr, tpr, path, name):
    plt.clf()
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(f"{path}/{name}_roc.png")
    plt.clf()

def plot_distance_distribution(n_scores, s_scores, path, name):
    plt.clf()
    # bins = np.linspace(0.000008,0.00005) # Mask MSE
    plt.hist(s_scores, bins=50, alpha=0.5, density=True, label="smura")
    plt.hist(n_scores, bins=50, alpha=0.5, density=True, label="normal")
    if "_sup" in name:
      plt.xlabel('Confidence')
    else:
      plt.xlabel('Anomaly Score')
    plt.title('Score Distribution')
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_dist.png")
    plt.clf()

def sup_unsup_prediction(labels, all_conf_sup, all_score_unsup, path, name):
    # result_msg = ''
    # score = a*max + b*mean
    # best_a, best_b = 0, 0
    # best_auc = 0
    # for ten_a in range(0, 10, 1):
    #     a = ten_a/10.0
    #     for ten_b in range(0, 10, 1):
    #         b = ten_b/10.0
            
    #         scores = a*max_scores + b*mean_scores
    #         fpr, tpr, th = roc_curve(labels, scores)
    #         current_auc = auc(fpr, tpr)
    #         if current_auc >= best_auc:
    #             best_auc = current_auc
    #             best_a = a
    #             best_b = b         

    # result_msg += f"Param a: {best_a}, b: {best_b}\n"

    # best_scores = best_a*max_scores + best_b*mean_scores
    # pred_labels = [] 
    # roc_auc, optimal_th = roc(labels, best_scores, path, name)
    # for score in best_scores:
    #     if score >= optimal_th:
    #         pred_labels.append(1)
    #     else:
    #         pred_labels.append(0)
    
    # y = -1.333x + 0.8
    # 1.333x+y-0.8 = 0
    best_scores = 1.333*all_score_unsup + all_conf_sup - 0.8
    pred_labels = [] 

    for score in best_scores:
            if score >= 0:
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
    # result_msg += f"\nAUC: {roc_auc}\n"
    # result_msg += f"Threshold (highest TPR-FPR): {optimal_th}\n"
    result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
    result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
    result_msg += f"TNR: {TN/(FP+TN)}\n"
    result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
    result_msg += f"NPV: {TN/(FN+TN)}\n"
    result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
    result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
    result_msg += f"F1-Score: {f1_score(labels, pred_labels)}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
    return result_msg

def show_and_save_result(conf_sup, score_unsup, path, name):
  all_conf_sup = np.concatenate([conf_sup['preds_res']['n'], conf_sup['preds_res']['s']])
  all_score_unsup = np.concatenate([score_unsup['mean']['n'], score_unsup['mean']['s']])
  true_label = [0]*score_unsup['mean']['n'].shape[0]+[1]*score_unsup['mean']['s'].shape[0]
  
  plot_distance_distribution(conf_sup['preds_res']['n'], conf_sup['preds_res']['s'], path, f"{name}_sup")
  plot_distance_distribution(score_unsup['mean']['n'], score_unsup['mean']['s'], path, f"{name}_unsup")

  plot_sup_unsup_scatter(conf_sup, score_unsup, path, name)
  
  log_name = os.path.join(path, 'result_log.txt')
  msg = ''
  with open(log_name, "w") as log_file:
    now = time.strftime("%c")
    msg += f"=============== Testing result {now} ===================\n"
    msg += sup_unsup_prediction(true_label, all_conf_sup, all_score_unsup, path, f"{name}_sup_unsup")
    
    log_file.write(msg)

def plot_sup_unsup_scatter(conf_sup, score_unsup, path, name):
    # normal
    n_x = score_unsup['mean']['n']
    n_y = conf_sup['preds_res']['n']

    # smura
    s_x = score_unsup['mean']['s']
    s_y = conf_sup['preds_res']['s']

    # 設定座標軸
    # normal
    plt.clf()
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(n_x, n_y, s=5, c ="blue", alpha=0.5, label="normal")
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_normal_scatter.png")
    plt.clf()
    # smura
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(s_x, s_y, s=5, c ="red", alpha=0.5, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_smura_scatter.png")
    plt.clf()
    # all
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(n_x, n_y, s=5, c ="blue", alpha=0.5, label="normal")
    plt.scatter(s_x, s_y, s=5, c ="red", alpha=0.5, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_all_scatter.png")
    plt.clf()

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  set_seed(config['seed'])

  # supervised
  # create dataset dataloader
  csv_path = 'E:/CSE/AI/Mura/mura_data/d23/data_merged.csv'
  image_info = pd.read_csv(csv_path)
  ds_sup = defaultdict(dict)
  for x in ["test"]:
    for y in ["mura", "normal"]:
      if y == "mura":
        label = 1
      elif y == "normal":
        label = 0
      ds_sup[x][y] = get_data_info(x, label, image_info, csv_path)

  dataloaders = make_test_dataloader(ds_sup)
  # read model
  model_sup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
  model_sup.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid()
    )
  model_sup.load_state_dict(torch.load('./supervised_model/model.pt'))  
  
  res_sup = evaluate(model_sup, dataloaders)
  print(res_sup['preds_res']['all'].shape)
  
  # unsupervised
  res_unsup = defaultdict(dict)
  for l in ['all','max','meab']:
    for t in ['n','s']:
      res_unsup[l][t] = None
  # res_unsup['all'] = defaultdict(list)
  # res_unsup['max'] = defaultdict(list)
  # res_unsup['mean'] = defaultdict(list)
  normal_mean = 4.2993142e-05
  normal_std = 1.3958662e-05
  dataset_type_list = [config['data_loader']['test_data_root_normal'], config['data_loader']['test_data_root_smura']]
  for idx, data_path in enumerate(dataset_type_list):
      config['data_loader']['test_data_root'] = data_path
      print(f"Now path: {data_path}")

      tester = Tester(config) # Default debug = False
      _, _, big_imgs_scores_mean = tester.test(normal_mean,normal_std) # max mean 

      if idx == 0:
          res_unsup['mean']['n'] = big_imgs_scores_mean.copy() # mean
      elif idx == 1:
          res_unsup['mean']['s'] = big_imgs_scores_mean.copy() # mean
      else:
          raise

  result_name = f"{config['data_loader']['name']}_crop{config['data_loader']['crop_size']}_{config['anomaly_score']}_epoch{config['model_epoch']}_with_seresnext101"
  show_and_save_result(res_sup, res_unsup, config['result_path'], result_name)

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
  config['pos_normalized'] = args.pos_normalized
  config['minmax'] = args.minmax

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
  result_path = os.path.join(config['save_dir'], 'results_{}_{}_with_sup'.format(str(config['model_epoch']).zfill(5), config['anomaly_score']))
  os.makedirs(result_path, exist_ok=True)

  config['result_path'] = result_path
  config['distributed'] = False
  main_worker(gpu_device, 1, config) # GPU device, GPU num, config