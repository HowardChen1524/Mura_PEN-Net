# -*- coding: utf-8 -*-

import os
import argparse
import json

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn

from opt.option import get_test_parser
from core.utils import set_seed
from core.tester import Tester
from core.utils_howard import mkdir, get_data_info, make_test_dataloader, evaluate

args = get_test_parser()

def initail_setting(with_sup_model=False):
  
  config = json.load(open(args.config))

  # ===== dataset setting =====
  if args.mask is not None:
    config['data_loader']['mask'] = args.mask

  if args.size is not None:
    config['data_loader']['w'] = config['data_loader']['h'] = args.size

  if args.normal_path is not None:
    config['data_loader']['test_data_root_normal'] = args.normal_path

  if args.smura_path is not None:
    config['data_loader']['test_data_root_smura'] = args.smura_path

  if args.dataset_name is not None:
    config['data_loader']['name'] = args.dataset_name

  if args.dataset_path is not None:
    config['data_loader']['data_root'] = args.dataset_path

  if args.normal_num is not None:
    config['data_loader']['test_normal_num'] = args.normal_num

  if args.smura_num is not None:
    config['data_loader']['test_smura_num'] = args.smura_num

  # ===== model setting =====
  config['model_name'] = args.model_name

  if args.model_version is not None:
    config['model']['version'] = args.model_version

  # ===== Test setting =====
  config['model_epoch'] = args.model_epoch
  config['test_type'] = args.test_type
  config['anomaly_score'] = args.anomaly_score
  config['pos_normalized'] = args.pos_normalized
  config['minmax'] = args.minmax
  config['using_record'] = args.using_record
  config['using_threshold'] = args.using_threshold

  # ===== Path setting =====
  if args.csv_path is not None:
    config['data_loader']['csv_path'] = args.csv_path

  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
    config['model']['version'], config['data_loader']['mask'], config['data_loader']['w']))

  if with_sup_model:
    if config['pos_normalized']:
      result_path = os.path.join(config['save_dir'], '{}_results_{}_{}_with_sup_pn'.format(config['data_loader']['name'], str(config['model_epoch']).zfill(5), config['anomaly_score']))
    else:
      result_path = os.path.join(config['save_dir'], '{}_results_{}_{}_with_sup'.format(config['data_loader']['name'], str(config['model_epoch']).zfill(5), config['anomaly_score']))
  else:
    if config['pos_normalized']:
      result_path = os.path.join(config['save_dir'], '{}_results_{}_{}_pn'.format(config['data_loader']['name'], str(config['model_epoch']).zfill(5), config['anomaly_score']))
    else:
      result_path = os.path.join(config['save_dir'], '{}_results_{}_{}'.format(config['data_loader']['name'], str(config['model_epoch']).zfill(5), config['anomaly_score']))
  mkdir(result_path)
  
  config['result_path'] = result_path
  
  # ===== GPU setting =====
  gpu = args.gpu_id 
  torch.cuda.set_device(gpu)
  print(f'using GPU device {gpu} for testing ... ')

  # ===== Seed setting =====
  set_seed(config['seed'])

  return config, gpu
  
def export_conf_score(conf_sup, score_unsup, path):
  sup_name = conf_sup['files_res']['all']
  sup_conf = np.concatenate([conf_sup['preds_res']['n'], conf_sup['preds_res']['s']])
  sup_label = [0]*len(conf_sup['preds_res']['n'])+[1]*len(conf_sup['preds_res']['s'])
  df_sup = pd.DataFrame(list(zip(sup_name,sup_conf,sup_label)), columns=['name', 'conf', 'label'])
  df_sup.to_csv(os.path.join(path, 'sup_conf.csv'), index=False)

  unsup_name = score_unsup['fn']['n'] + score_unsup['fn']['s']
  unsup_label = [0]*score_unsup['mean']['n'].shape[0]+[1]*score_unsup['mean']['s'].shape[0]

  unsup_score_max = np.concatenate([score_unsup['max']['n'], score_unsup['max']['s']])
  df_unsup_max = pd.DataFrame(list(zip(unsup_name,unsup_score_max,unsup_label)), columns=['name', 'score_max', 'label'])
  df_unsup_max.to_csv(os.path.join(path, 'unsup_score_max.csv'), index=False)

  unsup_score_mean = np.concatenate([score_unsup['mean']['n'], score_unsup['mean']['s']])
  df_unsup_mean = pd.DataFrame(list(zip(unsup_name,unsup_score_mean,unsup_label)), columns=['name', 'score_mean', 'label'])
  df_unsup_mean.to_csv(os.path.join(path, 'unsup_score_mean.csv'), index=False)

  unsup_score_all = np.concatenate([score_unsup['all']['n'], score_unsup['all']['s']])
  unsup_label_all = [0]*score_unsup['all']['n'].shape[0]+[1]*score_unsup['all']['s'].shape[0]
  df_unsup_all = pd.DataFrame(list(zip(unsup_score_all,unsup_label_all)), columns=['score', 'label'])
  df_unsup_all.to_csv(os.path.join(path, 'unsup_score_all.csv'), index=False)
  print("save conf score finished!")

def supervised_model_prediction(config, gpu):
  csv_path = config['data_loader']['csv_path']
  image_info = pd.read_csv(csv_path)
  ds_sup = defaultdict(dict)
  for x in ["test"]:
    for y in ["mura", "normal"]:
      if y == "mura":
        label = 1
      elif y == "normal":
        label = 0
      ds_sup[x][y] = get_data_info(x, label, image_info, config['data_loader']['data_root'], csv_path)

  dataloaders = make_test_dataloader(ds_sup)
  model_sup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
  model_sup.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid()
    )
  print(config['supervised']['model_path'])
  model_sup.load_state_dict(torch.load(config['supervised']['model_path'], map_location=torch.device(f"cuda:{gpu}")))  
  
  return evaluate(model_sup, dataloaders, config['result_path']) 

def unsupervised_model_prediction(config):
  res_unsup = defaultdict(dict)
  for l in ['all','max','mean', 'fn']:
    for t in ['n','s']:
      res_unsup[l][t] = None

  if config['pos_normalized']:
    for idx, (data_path, data_num) in enumerate([config['data_loader']['test_data_root_normal'], config['data_loader']['test_normal_num']]):
      config['data_loader']['test_data_root'] = data_path
      print("Start to compute normal mean and std")
      print(f"Now path: {data_path}")

      tester = Tester(config)
      n_pos_mean, n_pos_std = tester.position_normalize()
  else:
    n_pos_mean = n_pos_std = []

  dataset_type_list = [[config['data_loader']['test_data_root_normal'], config['data_loader']['test_normal_num']], [config['data_loader']['test_data_root_smura'],config['data_loader']['test_smura_num']]]
  for idx, (data_path, data_num) in enumerate(dataset_type_list):
    config['data_loader']['test_num'] = data_num
    config['data_loader']['test_data_root'] = data_path
    print(f"Now path: {data_path}")

    tester = Tester(config)
    big_imgs_scores, big_imgs_scores_max, big_imgs_scores_mean, big_imgs_fn = tester.test(n_pos_mean, n_pos_std, False) 
    
    if idx == 0: # normal
      res_unsup['all']['n'] = big_imgs_scores.copy() # all ??????
      res_unsup['max']['n'] = big_imgs_scores_max.copy() # max
      res_unsup['mean']['n'] = big_imgs_scores_mean.copy() # mean
      res_unsup['fn']['n'] = big_imgs_fn
    elif idx == 1: # smura
      res_unsup['all']['s'] = big_imgs_scores.copy() # all ??????
      res_unsup['max']['s'] = big_imgs_scores_max.copy() # max
      res_unsup['mean']['s'] = big_imgs_scores_mean.copy() # mean
      res_unsup['fn']['s'] = big_imgs_fn

  return res_unsup

if __name__ == '__main__':
  
  with_sup_model = True
  config, gpu = initail_setting(with_sup_model)  
  
  # ===== supervised =====
  res_sup = supervised_model_prediction(config, gpu)
  
  # ===== unsupervised =====
  res_unsup = unsupervised_model_prediction(config)
  
  export_conf_score(res_sup, res_unsup, config['result_path']) # ???????????????????????????????????????


  