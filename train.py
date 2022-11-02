
import os
import json 

import torch

from opt.option import get_train_parser
from core.utils import set_seed
from core.trainer import Trainer

# Input Parameters

args = get_train_parser()

def main_worker(gpu, config):
  torch.cuda.set_device(gpu)
  
  set_seed(config['seed']) # core/utils.py
  
  # create save directory
  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}'.format(config['model_name'], 
    config['model']['version'], config['data_loader']['mask'], config['data_loader']['w']))
  
  os.makedirs(config['save_dir'], exist_ok=True)
  print('[**] create folder {}'.format(config['save_dir']))
  
  trainer = Trainer(config)
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
  
  gpu_device = 0
 
  # 只使用單 GPU
  main_worker(gpu_device, config) # GPU device, GPU num, config
