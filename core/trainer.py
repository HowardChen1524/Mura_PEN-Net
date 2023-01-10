import os
import time
import glob
import importlib

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from core.dataset import AUO_Dataset
from core.utils import set_seed, set_device, Progbar, tensor2im
from core.loss import AdversarialLoss


class Trainer():
  def __init__(self, config):
    self.config = config
    self.epoch = 0
    self.total_iteration = 0

    # setup data set and data loader
    self.train_dataset = AUO_Dataset(config['data_loader'], split='train')
    worker_init_fn = partial(set_seed, base=config['seed']) # 將 set_seed 的 base param 固定為 config['seed']，並命名為 worker_init_fn
    
    self.train_loader = DataLoader(self.train_dataset, 
                                  batch_size=config['trainer']['batch_size'],
                                  shuffle=True, 
                                  num_workers=config['trainer']['num_workers'],
                                  pin_memory=True, 
                                  worker_init_fn=worker_init_fn)
    # set up losses and metrics
    self.adversarial_loss = set_device(AdversarialLoss(type=self.config['losses']['gan_type']))
    self.l1_loss = nn.L1Loss()
    self.dis_writer = None
    self.gen_writer = None
    self.summary = {}
   
    # 建立實體。資料存放在：'save_path/dis(gen)'
    # 接下來要寫入任何資料都是呼叫 writer.add_某功能()
    self.dis_writer = SummaryWriter(os.path.join(config['save_dir'], 'dis')) 
    self.gen_writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))
    self.train_args = self.config['trainer']
    
    net = importlib.import_module('model.'+config['model_name']) # 根據不同項目的配置，動態導入對應的模型
    self.netG = set_device(net.InpaintGenerator())
    self.netD = set_device(net.Discriminator(in_channels=3, use_sigmoid=config['losses']['gan_type'] != 'hinge'))
    self.optimG = torch.optim.Adam(self.netG.parameters(), lr=config['trainer']['lr'],
      betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
    self.optimD = torch.optim.Adam(self.netD.parameters(), lr=config['trainer']['lr'] * config['trainer']['d2glr'],
      betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
    self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimG, T_max=self.train_args['epochs'], eta_min=0)
    self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimD, T_max=self.train_args['epochs'], eta_min=0)

    self.load()

    self.one_epoch_iter = (len(self.train_dataset)*self.config['data_loader']['rand_crop_num']*1) // \
                          (self.config['data_loader']['rand_crop_num']*self.train_args["batch_size"])
    print(self.one_epoch_iter)
    self.log_name = os.path.join(self.config['save_dir'], 'loss_log.txt')
    with open(self.log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
  
  # get current learning rate
  def get_lr(self, type='G'):
    if type == 'G':
      return self.optimG.param_groups[0]['lr']
    return self.optimD.param_groups[0]['lr']
  
  # learning rate scheduler, step
  # def adjust_learning_rate(self):
  #   # decay 固定為 1，等於固定 lr
  #   decay = 0.1**(min(self.total_iteration, self.config['trainer']['niter_steady']) // self.config['trainer']['niter']) 
  #   new_lr = self.config['trainer']['lr'] * decay
  #   if new_lr != self.get_lr():
  #     for param_group in self.optimG.param_groups:
  #       param_group['lr'] = new_lr
  #     for param_group in self.optimD.param_groups:
  #      param_group['lr'] = new_lr

  # load netG and netD
  def load(self):
    model_path = self.config['save_dir']
    if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
      latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
    else:
      ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(os.path.join(model_path, '*.pth'))]
      ckpts.sort()
      latest_epoch = ckpts[-1] if len(ckpts)>0 else None
    if latest_epoch is not None:
      gen_path = os.path.join(model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
      dis_path = os.path.join(model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
      opt_path = os.path.join(model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
      # if self.config['global_rank'] == 0:
      print('Loading model from {}...'.format(gen_path))
      data = torch.load(gen_path, map_location = lambda storage, loc: set_device(storage)) 
      self.netG.load_state_dict(data['netG'])
      data = torch.load(dis_path, map_location = lambda storage, loc: set_device(storage)) 
      self.netD.load_state_dict(data['netD'])
      data = torch.load(opt_path, map_location = lambda storage, loc: set_device(storage)) 
      self.optimG.load_state_dict(data['optimG'])
      self.optimD.load_state_dict(data['optimD'])
      self.epoch = data['epoch']
      self.total_iteration = data['iteration']
    else:
      # if self.config['global_rank'] == 0:
      print('Warnning: There is no trained model found. An initialized model will be used.')

  # save parameters every eval_epoch
  def save(self, it):
    gen_path = os.path.join(self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
    dis_path = os.path.join(self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
    opt_path = os.path.join(self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
    print('\nsaving model to {} ...'.format(gen_path))
    
    netG, netD = self.netG, self.netD
    torch.save({'netG': netG.state_dict()}, gen_path)
    torch.save({'netD': netD.state_dict()}, dis_path)
    torch.save({'epoch': self.epoch, 
                'iteration': self.total_iteration,
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict()}, opt_path)
    os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.config['save_dir'], 'latest.ckpt')))

  def add_summary(self, writer, name, val):
    if name not in self.summary: # 如 key (name) 不在 dict 裡
      self.summary[name] = 0
    self.summary[name] += val # val = lr or loss
    
    if writer is not None and self.total_iteration % 100 == 0:
      # writer.add_scalar('myscalar', value, iteration)
      writer.add_scalar(name, self.summary[name]/100, self.total_iteration)
      self.summary[name] = 0

  # process input and calculate loss every training epoch
  def _train_epoch(self):
    progbar = Progbar(self.one_epoch_iter, width=20, stateful_metrics=['epoch', 'iter'])
    mae = 0
    iteration = 0
    for images, masks, image_name in self.train_loader:
      
      iteration += 1
      self.total_iteration += 1
      if iteration >= self.config['trainer']['fix_step']:
        print('Limit Step 5000')
        break
      # self.adjust_learning_rate()
      end = time.time()

      # delete first dim
      bs, ncrops, c, h, w = images.size()
      images = images.view(-1, c, h, w)
      bs, ncrops, c, h, w = masks.size()
      masks = masks.view(-1, c, h, w)

      images, masks = set_device([images, masks])
      images_masked = (images * (1 - masks).float()) + masks

      inputs = torch.cat((images_masked, masks), dim=1)
      feats, pred_img = self.netG(inputs, masks)                        # in: [rgb(3) + edge(1)]

      comp_img = (1 - masks)*images + masks*pred_img

      # check training image
      check_img = tensor2im(torch.unsqueeze(images[0],0))
      check_img.save(f"./ori.png")
      
      check_img = tensor2im(torch.unsqueeze(pred_img[0],0))
      check_img.save(f"./pred.png")

      check_img = tensor2im(torch.unsqueeze(comp_img[0],0))
      check_img.save(f"./comp.png")

      self.add_summary(self.dis_writer, 'lr/dis_lr', self.get_lr(type='D'))
      self.add_summary(self.gen_writer, 'lr/gen_lr', self.get_lr(type='G'))
    
      gen_loss = 0
      dis_loss = 0

      # image discriminator loss
      dis_real_feat = self.netD(images)                   
      dis_fake_feat = self.netD(comp_img.detach())     
      dis_real_loss = self.adversarial_loss(dis_real_feat, True, True)
      dis_fake_loss = self.adversarial_loss(dis_fake_feat, False, True)
      dis_loss += (dis_real_loss + dis_fake_loss) / 2
      self.add_summary(self.dis_writer, 'loss/dis_fake_loss', dis_fake_loss.item())
      self.optimD.zero_grad()
      dis_loss.backward()
      self.optimD.step()
      
      # generator adversarial loss
      gen_fake_feat = self.netD(comp_img)                    # in: [rgb(3)]
      gen_fake_loss = self.adversarial_loss(gen_fake_feat, True, False) 
      gen_loss += gen_fake_loss * self.config['losses']['adversarial_weight']
      self.add_summary(self.gen_writer, 'loss/gen_fake_loss', gen_fake_loss.item())
      
      # generator l1 loss
      hole_loss = self.l1_loss(pred_img*masks, images*masks) / torch.mean(masks) 
      gen_loss += hole_loss * self.config['losses']['hole_weight']
      self.add_summary(self.gen_writer, 'loss/hole_loss', hole_loss.item())
      valid_loss = self.l1_loss(pred_img*(1-masks), images*(1-masks)) / torch.mean(1-masks) 
      gen_loss += valid_loss * self.config['losses']['valid_weight']
      self.add_summary(self.gen_writer, 'loss/valid_loss', valid_loss.item())
      if feats is not None:
        pyramid_loss = 0 
        for _, f in enumerate(feats):
          pyramid_loss += self.l1_loss(f, F.interpolate(images, size=f.size()[2:4], mode='bilinear', align_corners=True))
        gen_loss += pyramid_loss * self.config['losses']['pyramid_weight']
        self.add_summary(self.gen_writer, 'loss/pyramid_loss', pyramid_loss.item())
      # generator backward
      self.optimG.zero_grad()
      gen_loss.backward()
      self.optimG.step()      
      
      # logs
      # new_mae = (torch.mean(torch.abs(images - pred_img)) / torch.mean(masks)).item()
      # mae = new_mae if mae == 0 else (new_mae+mae)/2 # first iter mae = new_mae, else mae = 前後平均 ?
      mae = self.l1_loss(pred_img, images).item()
      # speed = images.size(0)/(time.time() - end)*self.config['world_size'] # img size 0 = 64
      speed = images.size(0)/(time.time() - end) # img size 0 = 64
      logs = [("epoch", self.epoch), ("iter", self.total_iteration), ("lr", self.get_lr()),
              ('mae', mae), ('mask_mae', hole_loss.item()), ('pyramid_mae', pyramid_loss.item()),
              ('gen_loss', gen_loss.item()), ('dis_loss', dis_loss.item()), ('samples/s', speed)]
      # if self.config['global_rank'] == 0:
      progbar.add(1, values=logs if self.train_args['verbosity'] else [x for x in logs if not x[0].startswith('l_')])

      # print log each 10 iteration
      if iteration % 50 == 0:
        message = '('
        for msg in logs:
          message += f"{msg[0]}: {msg[1]}, "
        message = message[:-2] + ')'
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
      # saving and evaluating
      # if iteration % (self.one_epoch_iter*self.train_args['save_freq']) == 0:
      # if iteration % 5000 == 0:
      #   self.save(int(self.epoch))

    self.schedulerG.step()
    self.schedulerD.step()

  # def _test_epoch(self, it):
  #   if self.config['global_rank'] == 0:
  #     print('[**] Testing in backend ...')
  #     model_path = self.config['save_dir']
  #     result_path = '{}/results_{}_level_03'.format(model_path, str(it).zfill(5))
  #     log_path = os.path.join(model_path, 'valid.log')
  #     try: 
  #       os.popen('python test.py -c {} -n {} -l 3 -m {} -s {} > valid.log;'
  #         'CUDA_VISIBLE_DEVICES=1 python eval.py -r {} >> {};'
  #         'rm -rf {}'.format(self.config['config'], self.config['model_name'], self.config['data_loader']['mask'], self.config['data_loader']['w'], 
  #          result_path, log_path, result_path))
  #     except (BrokenPipeError, IOError):
  #       pass

  def train(self):
    while True: # opt.epoch_count default 1
      if self.epoch >= self.train_args['epochs']:
        break
      self.epoch += 1
      self._train_epoch()
      if self.epoch % 10 == 0:
        self.save(int(self.epoch))
    print('\nEnd training....')