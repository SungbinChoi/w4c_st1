import random
from random import shuffle
import numpy as np
from datetime import datetime
import time
import queue
import threading
import logging
from PIL import Image
import itertools
import re
import os
import glob
import shutil
import sys
import copy
import h5py
from netCDF4 import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from typing import Any, List, Tuple

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
target_city = 'R5'
input_folder_path_list = ['v0_0/' + target_city + '_0',
                          'v1_0/' + target_city + '_1',
                          'v2_0/' + target_city + '_2',
                          'v3_0/' + target_city + '_3',
                            ]
input_folder_path_list_2 = ['',
                            '',
                            'v2_1/' + target_city + '_2',
                            'v3_1/' + target_city + '_3',
                            ]
input_folder_path_list_3 = ['',
                            'v1_1/' + target_city + '_1',
                            '',
                            'v3_2/' + target_city + '_3',
                            ]
out_dir = 'submit2' + '/' + target_city + '/' + 'test'
input_data_folder_path    = '../0_data/' + target_city
input_n_data_folder_path  = '../0_data/' + target_city + 'n'

num_frame_per_day  = 96
num_frame_before   =  4    
num_frame_out      = 32     
num_frame_sequence = 36  
height=256
width =256
num_channel_1 = 9    
num_channel_2_src = 16 
num_channel_2 = 107 + num_channel_2_src
num_channel = (num_channel_1*2 + num_channel_2)   
num_channel_out= 4
NUM_INPUT_CHANNEL  = num_channel     * num_frame_before   
NUM_OUTPUT_CHANNEL = num_channel_out * num_frame_out

SEED = 0
EPS = 1e-12
np.set_printoptions(precision=4)

def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data=data, dtype=np.uint16, compression='gzip', compression_opts=9)
    f.close()

if __name__ == '__main__':

  COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)
  COMMON_STRING += '\tset random seed\n'
  COMMON_STRING += '\t\tSEED = %d\n'%SEED
  
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  
  torch.backends.cudnn.enabled       = True
  torch.backends.cudnn.benchmark     = True  
  torch.backends.cudnn.deterministic = True

  COMMON_STRING += '\tset cuda environment\n'
  COMMON_STRING += '\t\ttorch.__version__              = %s\n'%torch.__version__
  COMMON_STRING += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
  COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
  try:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
  except Exception:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
        NUM_CUDA_DEVICES = 1
  COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()

  print(COMMON_STRING)

  try:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
  except Exception:
            print('out_dir not made')
            exit(-1)


  prediction_filename_list = []
  for prediction_filename in os.listdir(input_folder_path_list[0]):
    prediction_filename_list.append(prediction_filename)
  assert len(prediction_filename_list) == 36
    
  num_day_done = 0
  for prediction_filename in prediction_filename_list:
    day_name = prediction_filename.split('.')[0] 
    out_file_path = os.path.join(out_dir, day_name + '.h5')
                                 
    pred_out = []
    for c in range(4):
      prediction = np.load(os.path.join(input_folder_path_list[c], prediction_filename))
      pred_out.append(np.moveaxis(prediction['prediction'], 0, 1))    
    pred_out = np.concatenate(pred_out, axis=1) 
    assert pred_out.dtype   == np.float32

    num_pred_list = np.ones((4),np.int32)
    
    pred_out_2 = []
    for c in range(4):
      input_folder_path = input_folder_path_list_2[c]
      if input_folder_path == '':
        pred_out_2.append(np.zeros((32,1,256,256),np.float32))
        continue
      prediction = np.load(os.path.join(input_folder_path, prediction_filename))
      pred_out_2.append(np.moveaxis(prediction['prediction'], 0, 1))    
      num_pred_list[c] += 1
    pred_out_2 = np.concatenate(pred_out_2, axis=1)                     
    assert pred_out_2.dtype == np.float32
    
    pred_out_3 = []
    for c in range(4):
      input_folder_path = input_folder_path_list_3[c]
      if input_folder_path == '':
        pred_out_3.append(np.zeros((32,1,256,256),np.float32))
        continue
      prediction = np.load(os.path.join(input_folder_path, prediction_filename))
      pred_out_3.append(np.moveaxis(prediction['prediction'], 0, 1))   
      num_pred_list[c] += 1
    pred_out_3 = np.concatenate(pred_out_3, axis=1)                     
    assert pred_out_3.dtype == np.float32
    
    pred_out += pred_out_2
    pred_out += pred_out_3
    for c in range(4):
      pred_out[:,c,:,:] = pred_out[:,c,:,:] / float(num_pred_list[c])

    pred_out[:,0,:,:] *= 22000  
    pred_out[:,1,:,:] *= 500    
    pred_out[:,2,:,:] *= 100    
    
    pred_out_binary = pred_out[:,3,:,:].copy()
    pred_out_binary[pred_out_binary>0.5]  = 1
    pred_out_binary[pred_out_binary<=0.5] = 0
    pred_out[:,3,:,:] = pred_out_binary 

    pred_out = np.rint(pred_out)    
    pred_out = pred_out.astype(np.uint16)  
    write_data(pred_out, out_file_path)
    num_day_done += 1
    
  print('num_day_done:',   num_day_done,   '\t', )
  exit(1)
  
  