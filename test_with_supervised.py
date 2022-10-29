# -*- coding: utf-8 -*-

import os
import argparse
import json

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn

from supervised_model.wei_dataloader import make_test_dataloader
from core.utils import set_seed
from core.tester import Tester
from core.utils_howard import mkdir, minmax_scaling, \
                              plot_score_distribution, plot_sup_unsup_scatter, \
                              get_data_info, evaluate, \
                              sup_unsup_prediction_spec_th, sup_unsup_prediction_spec_multi_th, \
                              sup_unsup_prediction_auto_th, sup_unsup_prediction_auto_multi_th, sup_unsup_svm\
                    

# ===== CLI Params =====
parser = argparse.ArgumentParser(description="MGP")
parser.add_argument("-c", "--config", type=str, required=True)
# parser.add_argument("-l", "--level",  type=int, required=True)
# parser.add_argument("-l", "--level",  type=int)
parser.add_argument("-mn", "--model_name", type=str, required=True)
parser.add_argument("-m", "--mask", default=None, type=str)
parser.add_argument("-s", "--size", default=None, type=int)
# parser.add_argument("-p", "--port", type=str, default="23451")
parser.add_argument("-me", "--model_epoch", type=int, default=-1)
parser.add_argument("-as", "--anomaly_score", type=str, default='MSE', help='MSE | Mask_MSE')
parser.add_argument("-dn", "--dataset_name", type=str, default=None)
parser.add_argument("-np", "--normal_path", type=str, default=None)
parser.add_argument("-sp", "--smura_path", type=str, default=None)
parser.add_argument("-t", "--test_type", type=str, default='normal', help='normal | position')
parser.add_argument("-pn", "--pos_normalized", action='store_true', help='Use for typecplus')
parser.add_argument("-mm", "--minmax", action='store_true', help='Use for combine supervised')
parser.add_argument("-gpu", "--gpu_id", type=int, default=0)

args = parser.parse_args()
# =======================

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
  
  # ===== Test setting =====
  config['model_epoch'] = args.model_epoch
  config['test_type'] = args.test_type
  config['anomaly_score'] = args.anomaly_score
  config['pos_normalized'] = args.pos_normalized
  config['minmax'] = args.minmax
  config['distributed'] = False

  # ===== Path setting =====
  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
    config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w']))

  if with_sup_model:
    if config['pos_normalized']:
      result_path = os.path.join(config['save_dir'], 'results_{}_{}_with_sup_pn'.format(str(config['model_epoch']).zfill(5), config['anomaly_score']))
    else:
      result_path = os.path.join(config['save_dir'], 'results_{}_{}_with_sup'.format(str(config['model_epoch']).zfill(5), config['anomaly_score']))
  else:
    if config['pos_normalized']:
      result_path = os.path.join(config['save_dir'], 'results_{}_{}_pn'.format(str(config['model_epoch']).zfill(5), config['anomaly_score']))
    else:
      result_path = os.path.join(config['save_dir'], 'results_{}_{}'.format(str(config['model_epoch']).zfill(5), config['anomaly_score']))
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
  log_name = os.path.join(path, 'conf_log.txt')
  np.savetxt(log_name, conf_sup, delimiter=",")
  log_name = os.path.join(path, 'score_log.txt')
  np.savetxt(log_name, score_unsup, delimiter=",")
  # df_res = pd.DataFrame(list(zip(conf_sup['preds_res']['all'], score_unsup)), columns=["conf", "score"])

  # df_res.to_csv(f"{path}/model_conf_score.csv", index=None)
  print("save conf score finished!")

def show_and_save_result(conf_sup, score_unsup, minmax, path, name):
  all_conf_sup = np.concatenate([conf_sup['preds_res']['n'], conf_sup['preds_res']['s']])
  all_score_unsup = np.concatenate([score_unsup['mean']['n'], score_unsup['mean']['s']])
  true_label = [0]*score_unsup['mean']['n'].shape[0]+[1]*score_unsup['mean']['s'].shape[0]
  
  export_conf_score(all_conf_sup, all_score_unsup, path) # 記錄下來，防止每次都要重跑

  if minmax:
    all_mean_anomaly_score = minmax_scaling(all_mean_anomaly_score)
    score_unsup['mean']['n'] =  all_mean_anomaly_score[:score_unsup['mean']['n'].shape[0]]
    score_unsup['mean']['s'] =  all_mean_anomaly_score[score_unsup['mean']['n'].shape[0]:]

  plot_score_distribution(conf_sup['preds_res']['n'], conf_sup['preds_res']['s'], path, f"{name}_sup")
  plot_score_distribution(score_unsup['mean']['n'], score_unsup['mean']['s'], path, f"{name}_unsup")
  plot_sup_unsup_scatter(conf_sup, score_unsup, path, name)
  
  # ===== spec line =====
  # log_name = os.path.join(path, 'result_log.txt')
  # msg = ''
  # with open(log_name, "w") as log_file:
  #   msg += f"=============== Spec one line ===================\n"
  #   msg += sup_unsup_prediction_spec_th(true_label, all_conf_sup, all_score_unsup, path)
  #   msg += f"=============== Spec two lines ===================\n"
  #   msg += sup_unsup_prediction_spec_multi_th(true_label, all_conf_sup, all_score_unsup, path)
  #   log_file.write(msg)
  # plot_sup_unsup_scatter(conf_sup, score_unsup, path, f"{name}_one_line_spec", [[[0.2, 1],[1, 0.2]],[[0.2, 1],[1, 0.35]]]) # one line
  # plot_sup_unsup_scatter(conf_sup, score_unsup, path, f"{name}_two_line_combine_spec", [[[0, 0.58],[0.6, 0.6]], [[0.58, 0.75],[0.6, 0]]]) # two line combine

  # ===== Auto line =====
  sup_unsup_prediction_auto_th(true_label, all_conf_sup, all_score_unsup, path)
  sup_unsup_prediction_auto_multi_th(true_label, all_conf_sup, all_score_unsup, path)
  sup_unsup_svm(true_label, all_conf_sup, all_score_unsup, path)
  # plot_sup_unsup_scatter(conf_sup, score_unsup, path, f"{name}_one_line_auto", [[[0.141025641025641,1],[1, 0.33]],[[0.157142857142857,1],[1,0.41]]])
  # plot_sup_unsup_scatter(conf_sup, score_unsup, path, f"{name}_two_line_auto", [[[0.141025641025641,1],[1, 0.33]],[[0.157142857142857,1],[1,0.41]], [[0.9,0.9],[0,1]]])
  # plot_sup_unsup_scatter(conf_sup, score_unsup, path, f"{name}_svm", [[[0, 0.8109007882649589], [0.7620793414417739, 0]]]) # two line combine   

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
      ds_sup[x][y] = get_data_info(x, label, image_info, csv_path)

  dataloaders = make_test_dataloader(ds_sup)
  model_sup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
  model_sup.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid()
    )
  model_sup.load_state_dict(torch.load(config['supervised']['model_path'], map_location=torch.device(f"cuda:{gpu}")))  
  
  return evaluate(model_sup, dataloaders) 

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

# save time use
def model_prediction_using_record(config):
  res_sup = defaultdict(dict)
  for l in ['preds_res','labels_res','files_res']:
    for t in ['n', 's']:
      res_sup[l][t] = None

  res_unsup = defaultdict(dict)
  for l in ['all','max','mean']:
    for t in ['n','s']:
      res_unsup[l][t] = None
  
  all_conf_sup = np.loadtxt(os.path.join(config['result_path'], 'conf_log.txt'))
  res_sup['preds_res']['n'] = all_conf_sup[:541]
  res_sup['preds_res']['s'] = all_conf_sup[541:]

  all_score_unsup = np.loadtxt(os.path.join(config['result_path'], 'score_log.txt'))
  res_unsup['mean']['n'] = all_score_unsup[:541]
  res_unsup['mean']['s'] = all_score_unsup[541:]

  return res_sup, res_unsup

if __name__ == '__main__':
  
  with_sup_model = True
  config, gpu = initail_setting(with_sup_model)  
  
  # res_sup, res_unsup = model_prediction_using_record(config)
  
  # # ===== supervised =====
  res_sup = supervised_model_prediction(config, gpu)
  
  # # ===== unsupervised =====
  res_unsup = unsupervised_model_prediction(config)

  result_name = f"{config['data_loader']['name']}_crop{config['data_loader']['crop_size']}_{config['anomaly_score']}_epoch{config['model_epoch']}_with_seresnext101"
  show_and_save_result(res_sup, res_unsup, config['minmax'], config['result_path'], result_name)