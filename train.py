import argparse
import numpy as np
import os
import json 

import torch
import torch.multiprocessing as mp

from core.philly import ompi_size, ompi_local_size, ompi_rank, ompi_local_rank
from core.philly import get_master_ip, gpu_indices, ompi_universe_size
from core.utils import set_seed
from core.trainer import Trainer

from datetime import date
# Input Parameters
parser = argparse.ArgumentParser(description="Pconv")
parser.add_argument('-c', '--config', type=str, default=None, required=True)
parser.add_argument('-mn', '--name', default='pennet', type=str)
parser.add_argument('-m', '--mask', default=None, type=str) 
parser.add_argument('-s', '--size', default=None, type=int)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('-e', '--exam', action='store_true')
parser.add_argument('-cont', '--continue_train', action='store_true')
args = parser.parse_args()

# world_size : GPU num -> 1
# local_rank : GPU device -> 0
# global_rank : GPU device -> 0

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  
  set_seed(config['seed']) # core/utils.py
  
  # create save directory
  
  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
    config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w']))
  # if (not config['distributed']) or config['global_rank'] == 0:
  if not config['distributed']:
    os.makedirs(config['save_dir'], exist_ok=True)
    print('[**] create folder {}'.format(config['save_dir']))
  
  trainer = Trainer(config, debug=args.exam) # Default debug = False
  trainer.train()


if __name__ == "__main__":
  # print('check if the gpu resource is well arranged on philly') # core/phyilly.py
  # assert ompi_size() == ompi_local_size() * ompi_universe_size() # 從環境變數取值並比較，但目前變量未設定，應為 None
  
  # loading configs 
  config = json.load(open(args.config)) # dict
  if args.mask is not None:
    config['data_loader']['mask'] = args.mask # square
  if args.size is not None:
    config['data_loader']['w'] = config['data_loader']['h'] = args.size # 512
  if args.continue_train is not None:
    config['data_loader']['continue'] = args.continue_train # 512
  config['model_name'] = args.name # pennet
  config['config'] = args.config # config path
  
  gpu_device = 1
 
  # 只使用單 GPU
  config['distributed'] = False
  main_worker(gpu_device, 1, config) # GPU device, GPU num, config
