# -*- coding: utf-8 -*-
import os
import json

import numpy as np
import pandas as pd
from collections import defaultdict

import torch

from opt.option import get_test_parser
from core.utils import set_seed
from core.tester import Tester
from core.utils_howard import mkdir, minmax_scaling, \
                              plot_score_distribution, plot_score_scatter, \
                              unsup_calc_metric, unsup_find_param_max_mean

args = get_test_parser().parse_args()

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

def export_score(score_max, score_mean, path):
  log_name = os.path.join(path, 'score_max_log.txt')
  np.savetxt(log_name, score_max, delimiter=",")
  log_name = os.path.join(path, 'score_mean_log.txt')
  np.savetxt(log_name, score_mean, delimiter=",")

  print("save score finished!")

def show_and_save_result(score_unsup, minmax, path, name):
  all_max_anomaly_score = np.concatenate([score_unsup['max']['n'], score_unsup['max']['s']])
  all_mean_anomaly_score = np.concatenate([score_unsup['mean']['n'], score_unsup['mean']['s']])
  true_label = [0]*score_unsup['mean']['n'].shape[0]+[1]*score_unsup['mean']['s'].shape[0]
  
  export_score(all_max_anomaly_score, all_mean_anomaly_score, path)

  if minmax:
    all_max_anomaly_score = minmax_scaling(all_max_anomaly_score)
    score_unsup['max']['n'] =  all_max_anomaly_score[:score_unsup['max']['n'].shape[0]]
    score_unsup['max']['s'] =  all_max_anomaly_score[score_unsup['max']['n'].shape[0]:]

    all_mean_anomaly_score = minmax_scaling(all_mean_anomaly_score)
    score_unsup['mean']['n'] =  all_mean_anomaly_score[:score_unsup['mean']['n'].shape[0]]
    score_unsup['mean']['s'] =  all_mean_anomaly_score[score_unsup['mean']['n'].shape[0]:]

  plot_score_distribution(score_unsup['mean']['n'], score_unsup['mean']['s'], path, name)
  plot_score_scatter(score_unsup['max']['n'], score_unsup['max']['s'], score_unsup['mean']['n'], score_unsup['mean']['s'], path, name)

  log_name = os.path.join(path, 'result_log.txt')
  msg = ''
  with open(log_name, "w") as log_file:
    msg += f"=============== All small image mean & std =============\n"
    msg += f"Normal mean: {score_unsup['all']['n'].mean()}\n"
    msg += f"Normal std: {score_unsup['all']['n'].std()}\n"
    msg += f"Smura mean: {score_unsup['all']['s'].mean()}\n"
    msg += f"Smura std: {score_unsup['all']['s'].std()}\n"
    msg += f"=============== Anomaly max prediction =================\n"    
    msg += unsup_calc_metric(true_label, all_max_anomaly_score, path, f"{name}_max")
    msg += f"=============== Anomaly mean prediction ================\n"
    msg += unsup_calc_metric(true_label, all_mean_anomaly_score, path, f"{name}_mean")
    msg += f"=============== Anomaly max & mean prediction ==========\n"
    msg += unsup_find_param_max_mean(true_label, all_max_anomaly_score, all_mean_anomaly_score, path, f"{name}_max_mean")
    
    log_file.write(msg)  

def unsupervised_model_prediction(config):
  res_unsup = defaultdict(dict)
  for l in ['all','max','mean', 'fn']:
    for t in ['n','s']:
      res_unsup[l][t] = None

  if config['pos_normalized']:
    for idx, data_path in enumerate([config['data_loader']['test_data_root_normal']]):
      config['data_loader']['test_data_root'] = data_path
      print("Start to compute normal mean and std")
      print(f"Now path: {data_path}")

      tester = Tester(config)
      n_pos_mean, n_pos_std = tester.position_normalize()
  else:
    n_pos_mean = n_pos_std = []

  dataset_type_list = [config['data_loader']['test_data_root_normal'], config['data_loader']['test_data_root_smura']]
  for idx, data_path in enumerate(dataset_type_list):
    config['data_loader']['test_data_root'] = data_path
    print(f"Now path: {data_path}")

    tester = Tester(config)
    big_imgs_scores, big_imgs_scores_max, big_imgs_scores_mean, big_imgs_fn = tester.test(n_pos_mean, n_pos_std) 
    
    if idx == 0: # normal
      res_unsup['all']['n'] = big_imgs_scores.copy() # all 小圖
      res_unsup['max']['n'] = big_imgs_scores_max.copy() # max
      res_unsup['mean']['n'] = big_imgs_scores_mean.copy() # mean
      res_unsup['fn']['n'] = big_imgs_fn.copy()
    elif idx == 1: # smura
      res_unsup['all']['s'] = big_imgs_scores.copy() # all 小圖
      res_unsup['max']['s'] = big_imgs_scores_max.copy() # max
      res_unsup['mean']['s'] = big_imgs_scores_mean.copy() # mean
      res_unsup['fn']['s'] = big_imgs_fn.copy()

      # 找特例
      # print(res_unsup['mean']['n'].max())
      # print(res_unsup['mean']['n'].argmax())
      # print(res_unsup['fn']['n'][res_unsup['mean']['n'].argmax()])

  return res_unsup

def unsupervised_model_prediction_position(config):
  df = pd.read_csv(config['type_c_plus_path'])
  print(f"Mura 最大 :\n{df.iloc[(df['h']+df['w']).argmax()][['fn','w','h']]}")
  print(f"Mura 最小 :\n{df.iloc[(df['h']+df['w']).argmin()][['fn','w','h']]}")

  res_unsup = defaultdict(dict)

  if config['pos_normalized']:
    for idx, data_path in enumerate([config['data_loader']['test_data_root_normal']]):
      config['data_loader']['test_data_root'] = data_path
      print(f"Now path: {data_path}")

      tester = Tester(config)
      n_pos_mean, n_pos_std = tester.position_normalize()

  for idx, data_path in enumerate([config['data_loader']['test_data_root_smura']]):
    config['data_loader']['test_data_root'] = data_path
    print(f"Now path: {data_path}")

    tester = Tester(config)

    if config['pos_normalized']:
      big_imgs_scores = tester.test_position(df, n_pos_mean, n_pos_std)
    else:
      big_imgs_scores = tester.test_position(df)

    res_unsup['all']['s'] = big_imgs_scores.copy() # all 小圖
  
  return res_unsup

if __name__ == '__main__':
  with_sup_model = False
  config, gpu = initail_setting(with_sup_model)  
  
  if config['test_type'] == 'normal':
    res_unsup = unsupervised_model_prediction(config)
    result_name = f"{config['data_loader']['name']}_crop{config['data_loader']['crop_size']}_{config['anomaly_score']}_epoch{config['model_epoch']}"
    show_and_save_result(res_unsup, config['minmax'], config['result_path'], result_name)
  
  elif config['test_type'] == 'position':
    res_unsup = unsupervised_model_prediction_position(config)
    result_name = f"{config['data_loader']['name']}_crop{config['data_loader']['crop_size']}_{config['anomaly_score']}_epoch{config['model_epoch']}"
    print(f"Smura mean: {res_unsup['all']['s'].mean()}")
    print(f"Smura std: {res_unsup['all']['s'].std()}")