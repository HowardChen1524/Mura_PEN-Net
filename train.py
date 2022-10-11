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
args = parser.parse_args()

# world_size : GPU num -> 1
# local_rank : GPU device -> 0
# global_rank : GPU device -> 0

def main_worker(gpu, ngpus_per_node, config):
  torch.cuda.set_device(gpu)
  # if 'local_rank' not in config:
  #   config['local_rank'] = config['global_rank'] = gpu # local_rank -> GPU device
    
  # if config['distributed']:
  #   torch.cuda.set_device(int(config['local_rank']))
  #   print('using GPU {} for training'.format(int(config['local_rank'])))
  #   torch.distributed.init_process_group(backend = 'nccl', 
  #     init_method = config['init_method'],
  #     world_size = config['world_size'], 
  #     rank = config['global_rank'],
  #     group_name='mtorch'
  #   )
  
  set_seed(config['seed']) # core/utils.py
  
  # create save directory
  # config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}_{}'.format(config['model_name'], 
  #   config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w'], date.today()))
  config['save_dir'] = os.path.join(config['save_dir'], '{}_{}_{}{}_{}'.format(config['model_name'], 
    config['data_loader']['name'], config['data_loader']['mask'], config['data_loader']['w'], "2022-09-29"))
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

  config['model_name'] = args.name # pennet
  config['config'] = args.config # config path
  
  gpu_device = 0
  # setup distributed parallel training environments
  # world_size = ompi_size() # 1

  # ngpus_per_node = torch.cuda.device_count() # 2

  # if world_size > 1:
  #   config['world_size'] = world_size
  #   config['init_method'] = 'tcp://' + get_master_ip() + args.port
  #   config['distributed'] = True
  #   config['local_rank'] = ompi_local_rank()
  #   config['global_rank'] = ompi_rank()
  #   main_worker(0, 1, config)
  # elif ngpus_per_node > 1: # GPU 數量 > 1
  #   config['world_size'] = ngpus_per_node
  #   config['init_method'] = 'tcp://127.0.0.1:'+ args.port 
  #   config['distributed'] = True
  #   # torch.multiprocessing 使用 spawn 可以在不同 process share cuda tensor
  #   mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config)) # 用 subprocess 同時用 2 GPU 
  # else:
  #   config['world_size'] = 1 
  #   config['distributed'] = False
  #   main_worker(0, 1, config)
  
  # 只使用單 GPU
  config['distributed'] = False
  main_worker(gpu_device, 1, config) # GPU device, GPU num, config
