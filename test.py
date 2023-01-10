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
from core.utils_howard import mkdir, \
                              plot_score_distribution, plot_score_scatter, \
                              unsup_calc_metric, unsup_find_param_max_mean

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
  config['using_record'] = args.using_record

  # ===== Path setting =====
  if args.csv_path is not None:
    config['data_loader']['csv_path'] = args.csv_path

  config['save_dir'] = os.path.join(config['save_dir'], f"{config['model']['version']}")

  if config['pos_normalized']:
    result_path = os.path.join(config['result_dir'], '{}/{}/{}_pn'.format(config['model']['version'], config['data_loader']['name'], config['anomaly_score']))
  else:
    result_path = os.path.join(config['result_dir'], '{}/{}/{}'.format(config['model']['version'], config['data_loader']['name'], config['anomaly_score']))
  mkdir(result_path)
  
  config['result_path'] = result_path
  
  # ===== GPU setting =====
  gpu = args.gpu_id 
  torch.cuda.set_device(gpu)
  print(f'using GPU device {gpu} for testing ... ')

  # ===== Seed setting =====
  set_seed(config['seed'])

  return config, gpu

def export_score(score_unsup, path):
  unsup_name = score_unsup['fn']['n'] + score_unsup['fn']['s']
  unsup_label = [0]*len(score_unsup['mean']['n'])+[1]*len(score_unsup['mean']['s'])

  unsup_score_max = np.concatenate([score_unsup['max']['n'], score_unsup['max']['s']])
  df_unsup_max = pd.DataFrame(list(zip(unsup_name,unsup_score_max,unsup_label)), columns=['name', 'score_max', 'label'])
  df_unsup_max.to_csv(os.path.join(path, 'unsup_score_max.csv'), index=False)

  unsup_score_mean = np.concatenate([score_unsup['mean']['n'], score_unsup['mean']['s']])
  df_unsup_mean = pd.DataFrame(list(zip(unsup_name,unsup_score_mean,unsup_label)), columns=['name', 'score_mean', 'label'])
  df_unsup_mean.to_csv(os.path.join(path, 'unsup_score_mean.csv'), index=False)

  unsup_score_all = np.concatenate([score_unsup['all']['n'], score_unsup['all']['s']])
  unsup_label_all = [0]*len(score_unsup['all']['n'])+[1]*len(score_unsup['all']['s'])
  df_unsup_all = pd.DataFrame(list(zip(unsup_score_all,unsup_label_all)), columns=['score', 'label'])
  df_unsup_all.to_csv(os.path.join(path, 'unsup_score_all.csv'), index=False)
  
  print("save conf score finished!")

def show_and_save_result(score_unsup, path, name):
  all_max_anomaly_score = np.concatenate([score_unsup['max']['n'], score_unsup['max']['s']])
  all_mean_anomaly_score = np.concatenate([score_unsup['mean']['n'], score_unsup['mean']['s']])
  true_label = [0]*len(score_unsup['mean']['n'])+[1]*len(score_unsup['mean']['s'])

  plot_score_distribution(score_unsup['mean']['n'], score_unsup['mean']['s'], path, name)
  plot_score_scatter(score_unsup['max']['n'], score_unsup['max']['s'], score_unsup['mean']['n'], score_unsup['mean']['s'], path, name)
  
  log_name = os.path.join(path, 'result_log.txt')
  msg = ''
  with open(log_name, "w") as log_file:
    msg += f"=============== All small image mean & std =============\n"
    msg += f"Normal mean: {np.array(score_unsup['all']['n']).mean()}\n"
    msg += f"Normal std: {np.array(score_unsup['all']['n']).std()}\n"
    msg += f"Smura mean: {np.array(score_unsup['all']['s']).mean()}\n"
    msg += f"Smura std: {np.array(score_unsup['all']['s']).std()}\n"
    msg += f"=============== Anomaly max prediction =================\n"    
    msg += unsup_calc_metric(true_label, all_max_anomaly_score, path, f"{name}_max")
    msg += f"=============== Anomaly mean prediction ================\n"
    msg += unsup_calc_metric(true_label, all_mean_anomaly_score, path, f"{name}_mean")
    msg += f"=============== Anomaly max & mean prediction ==========\n"
    msg += unsup_find_param_max_mean(true_label, all_max_anomaly_score, all_mean_anomaly_score, path, f"{name}_max_mean")
    
    log_file.write(msg) 

  # log_name = os.path.join(path, 'res_log.txt')
  # msg = ''
  # with open(log_name, "w") as log_file:
  #   msg += f"=============== All small image mean & std =============\n"
  #   msg += f"Normal mean: {score_unsup['all']['n'].mean()}\n"
  #   msg += f"Normal std: {score_unsup['all']['n'].std()}\n"
  #   msg += f"Smura mean: {score_unsup['all']['s'].mean()}\n"
  #   msg += f"Smura std: {score_unsup['all']['s'].std()}\n"
    
  #   log_file.write(msg) 

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

  dataset_type_list = [[config['data_loader']['test_data_root_normal'], config['data_loader']['test_normal_num']], [config['data_loader']['test_data_root_smura'],config['data_loader']['test_smura_num']]]
  
  for idx, (data_path, data_num) in enumerate(dataset_type_list):
    config['data_loader']['test_data_root'] = data_path
    config['data_loader']['test_num'] = data_num
    print(f"Now path: {data_path}")

    tester = Tester(config)
    
    export = True
    big_imgs_scores, big_imgs_scores_max, big_imgs_scores_mean, big_imgs_fn = tester.test(n_pos_mean, n_pos_std, export) 
    
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

def model_prediction_using_record(config):
    res_unsup = defaultdict(dict)
    for l in ['max', 'mean', 'labels', 'fn']:
        for t in ['n','s']:
            res_unsup[l][t] = None

    max_df = pd.read_csv(os.path.join(config['result_path'], 'unsup_score_max.csv'))
    mean_df = pd.read_csv(os.path.join(config['result_path'], 'unsup_score_mean.csv'))
    merge_df = max_df.merge(mean_df, left_on='name', right_on='name')
    
    normal_filter = (merge_df['label_x']==0) & (merge_df['label_y']==0)
    smura_filter = (merge_df['label_x']==1) & (merge_df['label_y']==1)
    for l, c in zip(['max', 'mean', 'labels', 'fn'],['score_max', 'score_mean', 'label_y','name']):
        for t, f in zip(['n', 's'],[normal_filter, smura_filter]):
            res_unsup[l][t] = merge_df[c][f].tolist()

    all_df = pd.read_csv(os.path.join(config['result_path'], 'unsup_score_all.csv'))
    normal_filter = (all_df['label']==0)
    smura_filter = (all_df['label']==1)
    res_unsup['all']['n'] = all_df['score'][normal_filter].tolist()
    res_unsup['all']['s'] = all_df['score'][smura_filter].tolist()

    return res_unsup

if __name__ == '__main__':
  with_sup_model = False
  config, gpu = initail_setting(with_sup_model)  
  
  if config['test_type'] == 'normal':
    if config['using_record']:
      res_unsup = model_prediction_using_record(config)
    else:
      res_unsup = unsupervised_model_prediction(config)
      
    result_name = f"{config['anomaly_score']}_epoch{config['model_epoch']}"
    show_and_save_result(res_unsup, config['result_path'], result_name)

    if not config['using_record']:
      export_score(res_unsup, config['result_path'])

  elif config['test_type'] == 'position':
    res_unsup = unsupervised_model_prediction_position(config)
    result_name = f"{config['data_loader']['name']}_crop{config['data_loader']['crop_size']}_{config['anomaly_score']}_epoch{config['model_epoch']}"
    print(f"Smura mean: {res_unsup['all']['s'].mean()}")
    print(f"Smura std: {res_unsup['all']['s'].std()}")