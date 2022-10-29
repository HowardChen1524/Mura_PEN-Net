import os
import cv2
import io
import sys
import glob
import time
import zipfile
import subprocess
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from contextlib import contextmanager
import torch
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.transform import resize
# tensor to image
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor): 
        image_tensor = input_image.detach().cpu()
    else:
        return input_image

    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToPILImage()])
    image = transform((image_tensor[0]+1) / 2.0)
    return image

# set random seed 
def set_seed(seed, base=0, is_set=True):
  seed += base # 2022 + 0
  assert seed >=0, '{} >= {}'.format(seed, 0)
  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
  torch.manual_seed(seed) # #为CPU设置种子用于生成随机数，以使得结果是确定的
  torch.cuda.manual_seed_all(seed) #为所有GPU设置随机种子；torch.cuda.manual_seed() 为當前GPU设置随机种子
  random.seed(seed) # seed固定的話，生成同一个随机数
  # torch.backends.cudnn.deterministic是啥？顾名思义，将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。cudnn中包含很多卷积算法。基于 GEMM (General Matrix Multiply) 的，基于 FFT 的，基于 Winograd 算法的等等。
  '''
  cudnn中卷积算法
  static const algo_t algos[] = {
         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,#默认
         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
  '''
  torch.backends.cudnn.deterministic = True
  # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
  torch.backends.cudnn.benchmark = True

class ZipReader(object):
  file_dict = dict()
  def __init__(self):
    super(ZipReader, self).__init__()

  @staticmethod
  def build_file_dict(path):
    file_dict = ZipReader.file_dict
    if path in file_dict:
      return file_dict[path]
    else:
      file_handle = zipfile.ZipFile(path, mode='r', allowZip64=True)
      file_dict[path] = file_handle
      return file_dict[path]

  @staticmethod
  def imread(path, image_name):
    zfile = ZipReader.build_file_dict(path)
    data = zfile.read(image_name)
    im = Image.open(io.BytesIO(data))
    return im

# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args

def postprocess(img):
  img = (img+1)/2*255
  img = img.permute(0,2,3,1)
  img = img.int().cpu().numpy().astype(np.uint8)
  return img


class Progbar(object):
  """Displays a progress bar.

  Arguments:
    target: Total number of steps expected, None if unknown.
    width: Progress bar width on screen.
    verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
    stateful_metrics: Iterable of string names of metrics that
      should *not* be averaged over time. Metrics in this list
      will be displayed as-is. All others will be averaged
      by the progbar before display.
    interval: Minimum visual progress update interval (in seconds).
  """

  def __init__(self, target, width=25, verbose=1, interval=0.05, stateful_metrics=None):
    self.target = target # 4835
    self.width = width # 20
    self.verbose = verbose # 2
    self.interval = interval # 0.05
    if stateful_metrics: # epoch, iter
      self.stateful_metrics = set(stateful_metrics)
    else:
      self.stateful_metrics = set()

    self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
      sys.stdout.isatty()) or 'ipykernel' in sys.modules or 'posix' in sys.modules)
    self._total_width = 0
    self._seen_so_far = 0
    # We use a dict + list to avoid garbage collection
    # issues found in OrderedDict
    self._values = {}
    self._values_order = []
    self._start = time.time() # epoch 開始時間
    self._last_update = 0

  def update(self, current, values=None):
    """Updates the progress bar.
    Arguments:
      current: Index of current step.
      values: List of tuples:
        `(name, value_for_last_step)`.
        If `name` is in `stateful_metrics`,
        `value_for_last_step` will be displayed as-is.
        Else, an average of the metric over time will be displayed.
    """
    values = values or []
    for k, v in values: # key value, epoch 1
      if k not in self._values_order: # 第一個 iter 觸發
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        if k not in self._values: # 第一個 iter 觸發
          # 目前 (current - self._seen_so_far) = 1
          self._values[k] = [v * (current - self._seen_so_far), current - self._seen_so_far]
        else:
          self._values[k][0] += v * (current - self._seen_so_far)
          self._values[k][1] += (current - self._seen_so_far)
      else:
        self._values[k] = v # epoch, iter
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start) # 目前 total iter 所花時間
    if self.verbose == 1:
      if (now - self._last_update < self.interval and 
        self.target is not None and current < self.target):
          return

      prev_total_width = self._total_width
      if self._dynamic_display:
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')
      else:
        sys.stdout.write('\n')

      if self.target is not None:
        numdigits = int(np.floor(np.log10(self.target))) + 1
        barstr = '%%%dd/%d [' % (numdigits, self.target)
        bar = barstr % current
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += ('=' * (prog_width - 1))
          if current < self.target:
            bar += '>'
          else:
            bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
      else:
        bar = '%7d/Unknown' % current
      self._total_width = len(bar)
      sys.stdout.write(bar)
      if current:
        time_per_unit = (now - self._start) / current
      else:
        time_per_unit = 0
      if self.target is not None and current < self.target:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta
        info = ' - ETA: %s' % eta_format
      else:
        if time_per_unit >= 1:
          info += ' %.0fs/step' % time_per_unit
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/step' % (time_per_unit * 1e3)
        else:
          info += ' %.0fus/step' % (time_per_unit * 1e6)

      for k in self._values_order:
        info += ' - %s:' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))
      if self.target is not None and current >= self.target:
        info += '\n'
      sys.stdout.write(info)
      sys.stdout.flush()
    elif self.verbose == 2:
      if self.target is None or current >= self.target:
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        info += '\n'
        sys.stdout.write(info)
        sys.stdout.flush()
    self._last_update = now

  def add(self, n, values=None):
    self.update(self._seen_so_far + n, values)


# #####################################
# ############ painter ################
# #####################################

# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce

import numpy as np
import cv2 as cv

# built-in modules
import os
import itertools as it
from contextlib import contextmanager

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']

class Bunch(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __str__(self):
        return str(self.__dict__)

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def homotrans(H, x, y):
    xs = H[0, 0]*x + H[0, 1]*y + H[0, 2]
    ys = H[1, 0]*x + H[1, 1]*y + H[1, 2]
    s  = H[2, 0]*x + H[2, 1]*y + H[2, 2]
    return xs/s, ys/s

def to_rect(a):
    a = np.ravel(a)
    if len(a) == 2:
        a = (0, 0, a[0], a[1])
    return np.array(a, np.float64).reshape(2, 2)

def rect2rect_mtx(src, dst):
    src, dst = to_rect(src), to_rect(dst)
    cx, cy = (dst[1] - dst[0]) / (src[1] - src[0])
    tx, ty = dst[0] - src[0] * (cx, cy)
    M = np.float64([[ cx,  0, tx],
                    [  0, cy, ty],
                    [  0,  0,  1]])
    return M


def lookat(eye, target, up = (0, 0, 1)):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= anorm(fwd)
    right = np.cross(fwd, up)
    right /= anorm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -np.dot(R, eye)
    return R, tvec

def mtx2rvec(R):
    w, u, vt = cv.SVDecomp(R - np.eye(3))
    p = vt[0] + u[:,0]*w[0]    # same as np.dot(R, vt[0])
    c = np.dot(vt[0], p)
    s = np.dot(vt[1], p)
    axis = np.cross(vt[0], vt[1])
    return axis * np.arctan2(s, c)

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

class Sketcher:
    def __init__(self, windowname, dests, colors_func, thick):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        self.thick = thick
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, self.thick)
            self.dirty = True
            self.prev_pt = pt
            self.show()


# palette data from matplotlib/_cm.py
_jet_data =   {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1),
                         (1, 0.5, 0.5)),
               'green': ((0., 0, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1),
                         (0.91,0,0), (1, 0, 0)),
               'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0),
                         (1, 0, 0))}

cmap_data = { 'jet' : _jet_data }

def make_cmap(name, n=256):
    data = cmap_data[name]
    xs = np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x+eps]
            yp += [y1, y2]
        ch = np.interp(xs, xp, yp)
        channels.append(ch)
    return np.uint8(np.array(channels).T*255)

def nothing(*arg, **kw):
    pass

def clock():
    return cv.getTickCount() / cv.getTickFrequency()

@contextmanager
def Timer(msg):
    print(msg, '...',)
    start = clock()
    try:
        yield
    finally:
        print("%.2f ms" % ((clock()-start)*1000))

class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v

class RectSelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            return
        if self.drag_start:
            if flags & cv.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)
    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True
    @property
    def dragging(self):
        return self.drag_rect is not None


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    if PY3:
        output = it.zip_longest(fillvalue=fillvalue, *args)
    else:
        output = it.izip_longest(fillvalue=fillvalue, *args)
    return output

def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    if PY3:
        img0 = next(imgs)
    else:
        img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))

def getsize(img):
    h, w = img.shape[:2]
    return w, h

def mdot(*args):
    return reduce(np.dot, args)

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv.circle(vis, (int(x), int(y)), 2, color)

def enhance_img(fp,factor=5):
  img = Image.open(fp)
  enh_con = ImageEnhance.Contrast(img)
  new_img = enh_con.enhance(factor=factor)
  return new_img

################# Style loss #########################
######################################################
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # print(self.enc_1)
        # print(self.enc_2)
        # print(self.enc_3)
        # raise
        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):

        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

def gram_matrix(feat):
    (batch, ch, h, w) = feat.size()
    feat = feat.view(batch, ch, h*w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


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
           'tnr0.996_precision': [],
           'tnr0.996_recall': [],
           'tnr0.998_precision': [],
           'tnr0.998_recall': [],
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
    
    results['tnr0.996_recall'].append((((curve_df[curve_df['tnr'] > 0.996].iloc[0]) + (curve_df[curve_df['tnr'] < 0.996].iloc[-1])) / 2).recall)
    results['tnr0.996_precision'].append((((curve_df[curve_df['tnr'] > 0.996].iloc[0]) + (curve_df[curve_df['tnr'] < 0.996].iloc[-1])) / 2).precision)
    results['tnr0.998_recall'].append((((curve_df[curve_df['tnr'] > 0.998].iloc[0]) + (curve_df[curve_df['tnr'] < 0.998].iloc[-1])) / 2).recall)
    results['tnr0.998_precision'].append((((curve_df[curve_df['tnr'] > 0.998].iloc[0]) + (curve_df[curve_df['tnr'] < 0.998].iloc[-1])) / 2).precision)

    # fill empty slot
    for k, v in results.items():
        if len(v) == 0:
            results[k].append(-1)

    model_report = pd.DataFrame(results).T
    
    return model_report, curve_df
    
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
    # model.eval()
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
          # inputs = inputs
          # labels = labels
          
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
    print(np.array(res['preds_res']['n']))
    print(np.array(res['preds_res']['s']))
    
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
  
    model_report, curve_df = calc_matrix(res['labels_res']['all'], res['preds_res']['all'])
    model_report.to_csv(os.path.join(save_path, "model_report.csv"))
    curve_df.to_csv(os.path.join(save_path, "model_precision_recall_curve.csv"))
    print("model report record finished!")
    return res

def roc(labels, scores, path, name):

    fpr, tpr, th = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    
    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]

    plot_roc_curve(roc_auc, fpr, tpr, path, name)
    
    return roc_auc, optimal_th

def plot_roc_curve(roc_auc, fpr, tpr, path, name):
    plt.clf()
    plt.plot(fpr, tpr, color='orange', label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{path}/{name}_roc.png")
    plt.clf()

def plot_distance_distribution(n_scores, s_scores, path, name):
    plt.clf()
    # bins = np.linspace(0.000008,0.00005) # Mask MSE
    plt.hist(n_scores, bins=50, alpha=0.5, density=True, label="normal")
    plt.hist(s_scores, bins=50, alpha=0.5, density=True, label="smura")
    if "_sup" in name:
      plt.xlabel('Confidence')
    else:
      plt.xlabel('Anomaly Score')
    plt.title('Score Distribution')
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_dist.png")
    plt.clf()

def sup_unsup_prediction(labels, all_conf_sup, all_score_unsup, path, name):
    result_msg = ''
    best_a, best_b = 0, 0
    best_auc = 0
    for ten_a in range(0, 10, 1):
        a = ten_a/10.0
        for ten_b in range(0, 10, 1):
            b = ten_b/10.0            
            scores = a*all_conf_sup + b*all_score_unsup
            fpr, tpr, th = roc_curve(labels, scores)
            current_auc = auc(fpr, tpr)
            if current_auc >= best_auc:
                best_auc = current_auc
                best_a = a
                best_b = b         

    result_msg += f"Param a: {best_a}, b: {best_b}\n"

    best_scores = best_a*all_conf_sup + best_b*all_score_unsup
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

def sup_unsup_prediction_spec_th(labels, all_conf_sup, all_score_unsup, path, name):
    result_msg = ''
    
    # y = ax + b
    # a = -1.33 b = 0.8
    # 1.333x + y - 0.8 = 0
    # pink = [[0, 0.8],[0.6, 0]]
   
    # a = -0.3 b = 0.8 
    # 0.3x + y - 0.8 = 0
    # green = [[0, 0.8],[1, 0.5]]
  
    # a = -1 b = 1.2
    # x+y-1.2=0
    # purple = [[0.2, 1],[1, 0.2]]
    
    th_list = [[1.333, -0.8], [0.3, -0.8], [1, -1.2]]  
    for th in th_list:
      pred_labels = [] 
      combined_scores = th[0]*all_score_unsup + all_conf_sup + th[1]
      for score in combined_scores:
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
      result_msg += f"Threshold line: {th[0]}x+y{th[1]}=0\n"
      result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
      result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
      result_msg += f"TNR: {TN/(FP+TN)}\n"
      result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
      result_msg += f"NPV: {TN/(FN+TN)}\n"
      result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
      result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
      result_msg += f"F1-Score: {f1_score(labels, pred_labels)}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
      result_msg += f"===================================\n"

    return result_msg

def sup_unsup_prediction_auto_th(labels, all_conf_sup, all_score_unsup, path, name):
    result_msg = ''
    
    for m in range(1, 11, 1):
      m = m/10
      for b in range(-1, -11, -1):
        pred_labels = [] 
        combined_scores = m*all_score_unsup + all_conf_sup + b
        for score in combined_scores:
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
        current_tnr = TN/(FP+TN)
        if current_tnr >= 0.995:
          result_msg += f"Confusion Matrix (row1: TN,FP | row2: FN,TP):\n{cm}"
          result_msg += f"Threshold line: {m}x+1y{b}=0\n"
          result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
          result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
          result_msg += f"TNR: {TN/(FP+TN)}\n"
          result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
          result_msg += f"NPV: {TN/(FN+TN)}\n"
          result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
          result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
          result_msg += f"F1-Score: {f1_score(labels, pred_labels)}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
          result_msg += f"===================================\n"

    return result_msg

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

    # spec line
    pink = [[0, 0.6],[0.8, 0]] # [x1, x2] [y1, y2]
    green = [[0, 1],[0.8, 0.5]]
    purple = [[0.2, 1],[1, 0.2]]
    plt.plot(pink[0], pink[1], color='pink')    
    plt.plot(green[0], green[1], color='green')
    plt.plot(purple[0], purple[1], color='purple')    
    
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_all_scatter.png")
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

