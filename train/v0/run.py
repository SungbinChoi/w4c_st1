import random
from random import shuffle
import numpy as np
from datetime import datetime
from typing import Any, List, Tuple
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
target_city = 'R3'
      
TRAIN_VAL_RATIO = 5
TRAIN_VAL_INDEX = 4

global_step_start  = 0
initial_checkpoint = None
initial_checkpoint_optimizer = None
#initial_checkpoint           = 'model' + ('/%09d_model.pth'     % (global_step_start))
#initial_checkpoint_optimizer = 'model' + ('/%09d_optimizer.pth' % (global_step_start))

LEARNING_RATE  = 1e-4
BATCH_SIZE     = 1
BATCH_SIZE_VAL = 1       
VAL_INTERVAL   = 8000
num_thread=2
SEED = int(time.time())

loss_weight_np = np.array([0.03163512, 0.00024158, 0.00703378, 0.19160305], np.float32)
loss_weight_np = 1.0/loss_weight_np
loss_weight_np *=(4.0/np.sum(loss_weight_np))
loss_weight_np *=10

input_data_folder_path    = '../../0_data/' + target_city
input_n_data_folder_path  = '../../0_data/' + target_city + 'n'
out_dir                   = 'output'

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
num_groups = 8 
EPS = 1e-12
np.set_printoptions(precision=6)

other_city_list = ['R1', 'R2', 'R3']
other_city_list.remove(target_city)
assert len(other_city_list) == 2

class Deconv3x3Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Deconv3x3Block, self).__init__()
        self.add_module('deconv', nn.ConvTranspose2d(in_size, h_size, kernel_size=3, stride=2, padding=1, bias=True))
        self.add_module('elu',  nn.ELU(inplace=True))                                        
        self.add_module('norm', nn.GroupNorm(num_groups=num_groups, num_channels=h_size))    

class Conv1x1Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Conv1x1Block, self).__init__()
        self.add_module('conv', nn.Conv2d(in_size, h_size, kernel_size=1, stride=1, padding=0, bias=True))

class Conv3x3Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Conv3x3Block, self).__init__()
        self.add_module('conv', nn.Conv2d(in_size, h_size, kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('elu',  nn.ELU(inplace=True))                                       
        self.add_module('norm', nn.GroupNorm(num_groups=num_groups, num_channels=h_size))  

class AvgBlock(nn.Sequential):
    def __init__(self, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int) -> None:
        super(AvgBlock, self).__init__()
        self.add_module('pool', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))    
        
class MaxBlock(nn.Sequential):
    def __init__(self, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int) -> None:
        super(MaxBlock, self).__init__()
        self.add_module('pool', nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))    

class DownBlock(nn.Module):
  

    def __init__(self, 
                 in_size: int, 
                 h_size: int, 
                 out_size: int, 
                 do_pool: int = True):
        
        super(DownBlock, self).__init__()     
        
        self.do_pool = do_pool

        self.pool = None
        if self.do_pool:
          self.pool = AvgBlock(kernel_size=2, stride=2, padding=0)
          
        in_size_cum = in_size  
        
        self.conv_1 = Conv3x3Block( in_size=in_size_cum, h_size=h_size)
        in_size_cum += h_size
        
        self.conv_3 = Conv3x3Block( in_size=in_size_cum, h_size=h_size)
        in_size_cum += h_size
        
        self.conv_2 = Conv1x1Block( in_size=in_size_cum,  h_size=out_size)
        
        
    def forward(self, x):
        
        batch_size = len(x)

        
        if self.do_pool:
          x = self.pool(x)
        
        x_list = []
        x_list.append(x)
        
        x = self.conv_1(x)
        x_list.append(x)
        x = torch.cat(x_list, 1)
        
        x = self.conv_3(x)
        x_list.append(x)
        x = torch.cat(x_list, 1)
        
        x = self.conv_2(x)
        
        return x

    def cuda(self, ):
        super(DownBlock, self).cuda()   
        
        self.conv_1.cuda()
        self.conv_3.cuda()
        self.conv_2.cuda()
        
        return self
        
        
        
        
class UpBlock(nn.Module):
  

    def __init__(self, 
                 in_size:   int, 
                 in_size_2: int, 
                 h_size:    int, 
                 out_size:  int, 
                 ):
        
        super(UpBlock, self).__init__()     
        
        self.deconv   = Deconv3x3Block( in_size=in_size, h_size=h_size)
        self.out_conv = Conv3x3Block( in_size=h_size + in_size_2, h_size=out_size)

    def forward(self, x1, x2):

        x1 = self.deconv(x1)
        x1 = F.interpolate(x1, size=x2.size()[2:4], scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None)

        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)

    def cuda(self, ):
        super(UpBlock, self).cuda()   
        self.deconv.cuda()
        self.out_conv.cuda()
        
        return self
        
        


class NetA(nn.Module):

    def __init__(self,):
        super(NetA, self).__init__()


        self.block0 = DownBlock(in_size=NUM_INPUT_CHANNEL, h_size=128, out_size=128, do_pool=False)
        self.block1 = DownBlock(in_size=128, h_size=128, out_size=128,)
        self.block2 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block3 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block4 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block5 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block6 = DownBlock(in_size=128, h_size=128, out_size=128,)
        
        self.block20 = Conv3x3Block(in_size=128, h_size=128)
        
        
        self.block15 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,) 
        self.block14 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block13 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block12 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block11 = UpBlock(in_size=128, in_size_2=128 , h_size=128,  out_size=128,) 
        self.block10 = UpBlock(in_size=128, in_size_2=128 , h_size=128,  out_size=128,)
        
        self.out_conv  = nn.Sequential(
           nn.Conv2d(128*1, NUM_OUTPUT_CHANNEL, kernel_size=3, stride=1, padding=1, bias=True)
        )
        
        if 1:

          for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                  nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                  nn.init.constant_(m.bias, 0)


    def forward(self, x):

        batch_size = len(x)


        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        
        x  = self.block20(x6)
        
        x  = self.block15(x, x5)
        x  = self.block14(x, x4)
        x  = self.block13(x, x3)
        x  = self.block12(x, x2)
        x  = self.block11(x, x1)
        x  = self.block10(x, x0)
        
        x  = self.out_conv(x)
        
        x = torch.reshape(x, (batch_size, num_channel_out, 1, num_frame_out, height, width))
        
        
        return x[:,0,:,:,:,:], x[:,1,:,:,:,:], x[:,2,:,:,:,:], x[:,3,:,:,:,:]
      
    def cuda(self, ):
        super(NetA, self).cuda()
        
        self.block0.cuda()
        self.block1.cuda()
        self.block2.cuda()
        self.block3.cuda()
        self.block4.cuda()
        self.block5.cuda()
        self.block6.cuda()
        
        self.block20.cuda()
        
        self.block15.cuda()
        self.block14.cuda()
        self.block13.cuda()
        self.block12.cuda()
        self.block11.cuda()
        self.block10.cuda()
        
        self.out_conv.cuda()
        
        
        return self  
  
  
other_city_train_days_list = []  

for other_city in other_city_list: 

  asii_frame_file_name_prefix     = 'S_NWC_ASII-TF_MSG4_Europe-VISIR_'
  asii_frame_file_name_prefix_len = len(asii_frame_file_name_prefix)
  
  
  other_city_day_dict   = {}   
  
  
  other_data_folder_path    = '../../0_data/' + other_city
  other_n_data_folder_path  = '../../0_data/' + other_city + 'n'
  
  input_folder_path_list = []
  input_folder_path_list.append(other_data_folder_path + '/' + 'training')
  input_folder_path_list.append(other_data_folder_path + '/' + 'validation')
  
  for input_folder_path in input_folder_path_list:
   for day_folder_name in os.listdir(input_folder_path):
    
    day_folder_path = os.path.join(input_folder_path, day_folder_name)
    if os.path.isdir(day_folder_path) == False:
      continue
    
    day = int(day_folder_name)
    assert day not in other_city_day_dict
    for frame_file_name in os.listdir(os.path.join(day_folder_path, 'ASII')):
      if frame_file_name.split('.')[-1] != 'nc':
        continue
      
      assert frame_file_name[asii_frame_file_name_prefix_len-1] == '_'
      assert frame_file_name[asii_frame_file_name_prefix_len+8] == 'T'
      
      ymd = frame_file_name[asii_frame_file_name_prefix_len : (asii_frame_file_name_prefix_len+8)]
      other_city_day_dict[day] = (ymd, input_folder_path)
      break

  all_days   = sorted(list(other_city_day_dict.keys()))
  other_city_train_days_list.append(all_days)




      
        
        

day_dict   = {}        
train_days = []
val_days   = []    

if 1:  

  asii_frame_file_name_prefix     = 'S_NWC_ASII-TF_MSG4_Europe-VISIR_'
  asii_frame_file_name_prefix_len = len(asii_frame_file_name_prefix)
  
  input_folder_path_list = []
  input_folder_path_list.append(input_data_folder_path + '/' + 'training')
  input_folder_path_list.append(input_data_folder_path + '/' + 'validation')
  
  for input_folder_path in input_folder_path_list:
   for day_folder_name in os.listdir(input_folder_path):
    
    day_folder_path = os.path.join(input_folder_path, day_folder_name)
    if os.path.isdir(day_folder_path) == False:
      continue
    
    day = int(day_folder_name)
    assert day not in day_dict
    for frame_file_name in os.listdir(os.path.join(day_folder_path, 'ASII')):
      if frame_file_name.split('.')[-1] != 'nc':
        continue
      
      assert frame_file_name[asii_frame_file_name_prefix_len-1] == '_'
      assert frame_file_name[asii_frame_file_name_prefix_len+8] == 'T'
      
      ymd = frame_file_name[asii_frame_file_name_prefix_len : (asii_frame_file_name_prefix_len+8)]
      day_dict[day] = (ymd, input_folder_path)
      break
      
  all_days   = sorted(list(day_dict.keys()))
  
  num_val_case = len(all_days) // TRAIN_VAL_RATIO    
  num_val_case_begin = TRAIN_VAL_INDEX     * num_val_case
  num_val_case_end   = (TRAIN_VAL_INDEX+1) * num_val_case
  if TRAIN_VAL_INDEX == (TRAIN_VAL_RATIO-1):
    num_val_case_end = len(all_days)
  for i, day in enumerate(all_days):
    if i < num_val_case_begin or i >= num_val_case_end:
      train_days.append(day)
    else:
      val_days.append(day)


continuous_data_info_list = np.zeros((num_channel_1, 3), np.float32)
if 1:
  
  continuous_data_info_filepath = os.path.join('../../0_data', 'continuous_data_info_all.txt')

  c=0
  with open(continuous_data_info_filepath) as info_file:
    content = info_file.readlines()

    for line in content:
      cols = line.strip().split('\t')
      
      d_min    = int(  cols[0])
      d_max    = int(  cols[1])
      d_avg    = float(cols[2])
      continuous_data_info_list[c,:] = (d_min,d_max,d_avg)
      
      c += 1
      
  assert c == num_channel_1    
  print(continuous_data_info_filepath, '\t', 'num_line:', c, '\t', )

continuous_data_info_list_min = continuous_data_info_list[np.newaxis,:, 0, np.newaxis,np.newaxis,]
continuous_data_info_list_max = continuous_data_info_list[np.newaxis,:, 1, np.newaxis,np.newaxis,]


continuous_output_info_list = np.zeros((3, 2), np.float32)    
continuous_output_info_list[0,:] = (130, 350)     
continuous_output_info_list[1,:] = (0,    50)      
continuous_output_info_list[2,:] = (0,   100)      
continuous_output_info_list = continuous_output_info_list[np.newaxis, :, :, np.newaxis,np.newaxis,]



 
discrete_data_info_list = np.zeros((num_channel_2_src, ), np.uint8)
if 1:
  
  discrete_data_info_filepath   = os.path.join(input_n_data_folder_path, 'discrete_data_info.txt')
  
  c=0
  with open(discrete_data_info_filepath) as info_file:
    content = info_file.readlines()

    for line in content:
      cols = line.strip().split('\t')
      
      num_flag = int(cols[0])
      
      discrete_data_info_list[c] = (num_flag+1)
      
      c += 1
      
  assert c == num_channel_2_src    
  assert np.sum(discrete_data_info_list) == num_channel_2


cum_num_flag_list = np.zeros((num_channel_2_src, 2), np.uint8)
cc = 0
for c in range(num_channel_2_src):
  cum_num_flag_list[c,0] = cc
  cc+=discrete_data_info_list[c]
  cum_num_flag_list[c,1] = cc
assert cc < 256  






if __name__ == '__main__':
  
  if initial_checkpoint == None:
    assert global_step_start == 0 
  else:
    assert global_step_start > 0 
    
    
    
  COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)
  COMMON_STRING += '\tset random seed\n'
  COMMON_STRING += '\t\tSEED = %d\n'%SEED
  
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  
  torch.backends.cudnn.enabled       = True
  torch.backends.cudnn.benchmark     = True  
  torch.backends.cudnn.deterministic = False


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


  net = NetA().cuda()
  
  loss_weight = torch.from_numpy(loss_weight_np).float().cuda() 
  asii_logit_m = -torch.logit(torch.from_numpy(np.array(0.003,np.float32)).float().cuda())
  
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=LEARNING_RATE)

  loss_func2 = nn.MSELoss(reduction='none')
  loss_func3 = nn.BCEWithLogitsLoss(reduction='none')    
  
  if initial_checkpoint is not None:
    print('Loading ', initial_checkpoint)
    
    state_dict_0 = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict_0, strict=True)

    optimizer_state_dict_ = torch.load(initial_checkpoint_optimizer, map_location=lambda storage, loc: storage)
    optimizer_state_dict = optimizer_state_dict_['optimizer']
    optimizer.load_state_dict(optimizer_state_dict)


  train_list = [] 
  for day in train_days:           
    for frame_start in range(num_frame_per_day):
      if (frame_start + num_frame_sequence) > num_frame_per_day:
        if (day+1) not in train_days:
          continue
      train_list.append( (day, frame_start) )

  other_train_list_list = []
  for other_city_train_days in other_city_train_days_list:
   other_train_list = []
   for day in other_city_train_days:           
    for frame_start in range(num_frame_per_day):
      if (frame_start + num_frame_sequence) > num_frame_per_day:
        if (day+1) not in other_city_train_days:
          continue
      other_train_list.append( (day, frame_start) )
   other_train_list_list.append(other_train_list)
  
  num_iteration_per_epoch = (len(train_list) + len(other_train_list_list[0]) +  len(other_train_list_list[1]) ) // BATCH_SIZE
  
  
  val_list = []
  for day in val_days:           
    for frame_start in range(0, num_frame_per_day, num_frame_sequence//4):
      if (frame_start+num_frame_sequence) > num_frame_per_day:
        continue
      
      val_list.append( (day, frame_start) )
  num_val_iteration_per_epoch = int(len(val_list) / BATCH_SIZE_VAL)     

  print('len(train_list):', len(train_list))
  print('len(val_list):',   len(val_list))
  
  print('BATCH_SIZE:', BATCH_SIZE,)
  print('num_iteration_per_epoch:', num_iteration_per_epoch)
  print('BATCH_SIZE_VAL:', BATCH_SIZE_VAL,)
  print('num_val_iteration_per_epoch:', num_val_iteration_per_epoch)
  
  np.random.shuffle(train_list) 
  np.random.shuffle(other_train_list_list[0])
  np.random.shuffle(other_train_list_list[1])
  
  global_step = global_step_start
  epoch = float(global_step*BATCH_SIZE) / float(len(train_list) + len(other_train_list_list[0]) +  len(other_train_list_list[1]) )
  
  
  
  index_list2 = np.arange(num_frame_before * height * width)
  
  def get_data_and_label_from_other_city(day, frame_start, other_n_data_folder_path, ):

    input_data_1     = np.zeros((num_frame_before,    num_channel_1,        height, width), np.float32)
    input_data_2     = np.zeros((num_frame_before,    num_channel_2_src,    height, width), np.uint16)

    d = day
    f_start = frame_start
    f_end   = f_start + num_frame_before
    
    do_next_day = False
    if f_end > num_frame_per_day:
      do_next_day = True
      f_end = num_frame_per_day
    
    for f in range(f_start, f_end):
      np_filepath = os.path.join(other_n_data_folder_path, str(d) + '_' + str(f) +'.npz')
      day_np = np.load(np_filepath)
      input_data_1[f-f_start, :, :,:]  = np.array(day_np['data_1']).astype(np.uint16) 
      input_data_2[f-f_start, :, :,:]  = np.array(day_np['data_2']).astype(np.uint8) 
    if do_next_day:
      ff = f_end - f_start
      d += 1
      f_end   = num_frame_before - (f_end - f_start)
      f_start = 0
      for f in range(f_start, f_end):
        np_filepath = os.path.join(other_n_data_folder_path, str(d) + '_' + str(f) +'.npz')
        day_np = np.load(np_filepath)
        input_data_1[ff, :, :,:]  = np.array(day_np['data_1']).astype(np.uint16) 
        input_data_2[ff, :, :,:]  = np.array(day_np['data_2']).astype(np.uint8) 
        ff += 1

    input_data_out_3 = np.ones((num_frame_before,    num_channel_1,  height, width), np.float32)
    
    input_data_out_1 = \
              (input_data_1 
             - continuous_data_info_list_min)\
            / (continuous_data_info_list_max
             - continuous_data_info_list_min)
    
    input_data_out_1[input_data_1==65535] = 0
    input_data_out_3[input_data_1==65535] = 0
    
    input_data_out_1 = np.moveaxis(input_data_out_1, 1, 0)
    input_data_out_3 = np.moveaxis(input_data_out_3, 1, 0)
    
    
    
    input_data_2 += 1
    input_data_2[input_data_2==256] = 0
    one_hot_list = np.zeros((num_frame_before * height * width, num_channel_2), np.uint8)
    for c in range(num_channel_2_src):
      one_hot_list[index_list2, cum_num_flag_list[c,0] + input_data_2[:,c,:,:].reshape(-1)] = 1
    input_data_out_2 = np.moveaxis(one_hot_list, -1, 0).reshape(num_channel_2,   num_frame_before,    height, width)
                                  
    input_data_out = np.concatenate([input_data_out_1, input_data_out_3, input_data_out_2, ], axis=0)

    chunk_size = 16
    f_start = f_end
    f_end   = f_start + num_frame_out
    do_next_day = False
    if f_end > num_frame_per_day:
      do_next_day = True
      f_end = num_frame_per_day
    label_chunk_list = []
    fg_start        = f_start  //chunk_size
    fg_end          = (f_end-1)//chunk_size
    fg_start_offset = f_start - (fg_start*chunk_size)
    for fg in range(fg_start, fg_end+1):
      np_filepath = os.path.join(other_n_data_folder_path, 'l_' + str(d) + '_' + str(fg) +'.npz')
      day_np = np.load(np_filepath)
      label_chunk_list.append(np.array(day_np['label']).astype(np.uint16))
       
    if do_next_day:
      d += 1
      f_end   = num_frame_out - (f_end - f_start)
      f_start = 0
      
      fg_start        = f_start  //chunk_size
      fg_end          = (f_end-1)//chunk_size
      
      for fg in range(fg_start, fg_end+1):
        np_filepath = os.path.join(other_n_data_folder_path, 'l_' + str(d) + '_' + str(fg) +'.npz')
        day_np = np.load(np_filepath)
        label_chunk_list.append(np.array(day_np['label']).astype(np.uint16))
        
    label_data = np.concatenate(label_chunk_list, axis=0)[fg_start_offset:(fg_start_offset+num_frame_out),:,:,:]
    label_mask        = np.ones((num_frame_out,  num_channel_out,  height, width), np.uint8)
    label_mask[label_data==65535] = 0
    label_data = label_data.astype(np.float32)
    label_data[:,0:3,:,:] = \
              ( label_data[:,0:3,:,:]
             - continuous_output_info_list[:,:,0,:,:])\
            / (continuous_output_info_list[:,:,1,:,:] 
             - continuous_output_info_list[:,:,0,:,:])
    label_data[label_mask==0] = 0

    label_data = np.moveaxis(label_data, 0, 1)
    label_mask = np.moveaxis(label_mask, 0, 1).astype(np.float32)

    input_data_out = input_data_out.reshape(-1, height, width)

    return (input_data_out, label_data, label_mask)
  
  
  
  train_output_queue = queue.Queue()
  
  def load_train_multithread():

    tt = 0
    tt_list = np.zeros((2),np.int32)
    
    train_list_copied            = copy.deepcopy(train_list)
    other_train_list_list_copied = copy.deepcopy(other_train_list_list)
    
    while True:
      
      if train_output_queue.qsize() > 8:
        time.sleep(0.05)
        continue
      
      random_city_i = np.random.randint(0,3)
      if random_city_i == 2:
        
        (day, frame_start) = train_list_copied[tt]
        (input_data_one, label_data_one, label_mask_one) = get_data_and_label_from_other_city(day, frame_start, input_n_data_folder_path)
        
        if np.random.randint(0,2)==0:
          input_data_one = input_data_one[:,:,::-1]
          label_data_one = label_data_one[:,:,:,::-1]
          label_mask_one = label_mask_one[:,:,:,::-1]
        if np.random.randint(0,2)==0:
          input_data_one = input_data_one[:,::-1,:]
          label_data_one = label_data_one[:,:,::-1,:]
          label_mask_one = label_mask_one[:,:,::-1,:]
        if np.random.randint(0,2)==0:
          input_data_one = np.moveaxis(input_data_one, 1, -1)
          label_data_one = np.moveaxis(label_data_one, 2, -1)
          label_mask_one = np.moveaxis(label_mask_one, 2, -1)
        
        train_output_queue.put( (input_data_one, label_data_one, label_mask_one))
        tt += 1
        if tt == len(train_list_copied):
          np.random.shuffle(train_list_copied)  
          tt = 0
          
      else:
        
        (day, frame_start) = other_train_list_list_copied[random_city_i][tt_list[random_city_i]]
        
        other_city = other_city_list[random_city_i]
        other_n_data_folder_path  = '../../0_data/' + other_city + 'n'
        (input_data_one, label_data_one, label_mask_one) = get_data_and_label_from_other_city(day, frame_start, other_n_data_folder_path, )
                
        if np.random.randint(0,2)==0:
          input_data_one = input_data_one[:,:,::-1]
          label_data_one = label_data_one[:,:,:,::-1]
          label_mask_one = label_mask_one[:,:,:,::-1]
        if np.random.randint(0,2)==0:
          input_data_one = input_data_one[:,::-1,:]
          label_data_one = label_data_one[:,:,::-1,:]
          label_mask_one = label_mask_one[:,:,::-1,:]
        if np.random.randint(0,2)==0:
          input_data_one = np.moveaxis(input_data_one, 1, -1)
          label_data_one = np.moveaxis(label_data_one, 2, -1)
          label_mask_one = np.moveaxis(label_mask_one, 2, -1)
        
        train_output_queue.put( (input_data_one, label_data_one, label_mask_one))
        tt_list[random_city_i] += 1
        if tt_list[random_city_i] == len(other_train_list_list_copied[random_city_i]):
          np.random.shuffle(other_train_list_list_copied[random_city_i])  
          tt_list[random_city_i] = 0

  thread_list = []
  for i in range(num_thread):

    t = threading.Thread(
                        target=load_train_multithread, 
                        )
    t.start()

  
  if 1:
    
    print('batch_size =',BATCH_SIZE)
    print(datetime.now(), '\t',)
    print('---------------------------------------------------------------------------------------------------------------')
    print('[iter]  [epoch]  |   [loss(train   val)]  |   ')
    print('---------------------------------------------------------------------------------------------------------------')
    input_data         = np.zeros((BATCH_SIZE,     num_channel     * num_frame_before,     height, width), np.float32)
    label_data         = np.zeros((BATCH_SIZE,     num_channel_out,  num_frame_out,    height, width), np.float32)
    label_mask         = np.zeros((BATCH_SIZE,     num_channel_out,  num_frame_out,    height, width), np.float32)
    val_input_data     = np.zeros((BATCH_SIZE_VAL, num_channel     * num_frame_before,     height, width), np.float32)       
    val_label_data     = np.zeros((BATCH_SIZE_VAL, num_channel_out,  num_frame_out,    height, width), np.float32)
    val_label_mask     = np.zeros((BATCH_SIZE_VAL, num_channel_out,  num_frame_out,    height, width), np.float32)
    sum_train_freq = 0
    sum_train_loss = 0.0
    train_loss_1_sum_list = np.zeros((num_channel_out), np.float32)
    optimizer.zero_grad()
    net.train()

    while True:
  
      if global_step % VAL_INTERVAL == 0:

        state_dict_0 = copy.deepcopy(net.state_dict())
        torch.save(state_dict_0, out_dir + '/%09d_model.pth' % (global_step))
        
        torch.save(
              {
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
              }, 
              out_dir + '/%09d_optimizer.pth' % (global_step))  

        val_loss_list = []
        val_loss_1_sum_list = np.zeros((num_channel_out), np.float32)
        
        net.eval()
        with torch.no_grad():
          
          v = 0
          for vi in range(num_val_iteration_per_epoch):
            
            b = 0
            for _ in range(BATCH_SIZE_VAL):
              
              (day, frame_start) = val_list[v]
              v += 1
              assert (frame_start + num_frame_sequence) <= num_frame_per_day

              (val_input_data_one, val_label_data_one, val_label_mask_one) = get_data_and_label_from_other_city(day, frame_start, input_n_data_folder_path)

              val_input_data[b, :, :, :] = val_input_data_one
              val_label_data[b, :, :, :, :] = val_label_data_one
              val_label_mask[b, :, :, :, :] = val_label_mask_one
              
              b += 1

            if b == 0:
              continue
            input  = torch.from_numpy(val_input_data[:b, :, :, :]).float().cuda() 
            target = torch.from_numpy(val_label_data[:b, :, :, :, :]).float().cuda() 
            mask   = torch.from_numpy(val_label_mask[:b, :, :, :, :]).float().cuda()
            
            target[:,2,:,:,:] = (torch.logit(torch.clamp(target[:,2,:,:,:], min=0.003, max=0.997), eps=1e-6) + asii_logit_m) / (asii_logit_m*2)


            logit0, logit1, logit2, logit3 = net(input)
            logit = torch.cat(
                [
                  torch.sigmoid(logit0),  
                  torch.sigmoid(logit1),  
                  (torch.logit(torch.clamp(torch.sigmoid(logit2), min=0.003, max=0.997), eps=1e-6) + asii_logit_m) / (asii_logit_m*2),
                  (torch.sigmoid(logit3)>0.5).float(),
                ],              1)
            loss_per_var = torch.sum(loss_func2(logit, target) * mask, axis=(0,2,3,4)) / (torch.sum(mask, axis=(0,2,3,4)) + EPS)
            loss_w       = loss_weight * loss_per_var
            loss         = torch.mean(loss_w)

            val_loss_1_sum_list += loss_w.cpu().detach().numpy()
            val_loss_list.append(loss.item())

        avg_val_loss                    = np.mean(val_loss_list)                
        val_loss_1_sum_list /= len(val_loss_list) 

        avg_train_loss   = sum_train_loss       /(sum_train_freq+EPS)   
        avg_train_loss_1 = train_loss_1_sum_list/(sum_train_freq+EPS)
        
        sum_train_freq = 0
        sum_train_loss = 0.0
        train_loss_1_sum_list[:] = 0

        epoch = float(global_step*BATCH_SIZE) / float(len(train_list) + len(other_train_list_list[0]) +  len(other_train_list_list[1]) )

        print(global_step, 
                  ('\t%.1f' % epoch),                  '\t',
                  ('\t%.6f' % float(avg_train_loss)), 
                  ('\t%.6f' % float(avg_val_loss)),    
                  ('\t%.6f\t%.6f\t%.6f\t%.6f' % (avg_train_loss_1[0], avg_train_loss_1[1], avg_train_loss_1[2], avg_train_loss_1[3],)),
                  ('\t%.6f\t%.6f\t%.6f\t%.6f' % (val_loss_1_sum_list[0], val_loss_1_sum_list[1], val_loss_1_sum_list[2], val_loss_1_sum_list[3],)),
                  '\t', datetime.now(), 
                  )
        net.train()    

      for b in range(BATCH_SIZE):
        
        while train_output_queue.empty():
          time.sleep(0.05)
            
        (input_data_one, label_data_one, label_mask_one) = train_output_queue.get()
        input_data[b, :, :, :]    = input_data_one
        label_data[b, :, :, :, :] = label_data_one  
        label_mask[b, :, :, :, :] = label_mask_one
      optimizer.zero_grad()

      input  = torch.from_numpy(input_data).float().cuda() 
      target = torch.from_numpy(label_data).float().cuda() 
      mask   = torch.from_numpy(label_mask).float().cuda()
      target[:,2,:,:,:] = (torch.logit(torch.clamp(target[:,2,:,:,:], min=0.003, max=0.997), eps=1e-6) + asii_logit_m) / (asii_logit_m*2)

      logit0, logit1, logit2, logit3 = net(input)

      logit0123 = torch.cat(
                [
                  torch.sigmoid(logit0), 
                  torch.sigmoid(logit1),
                  (torch.logit(torch.sigmoid(logit2), eps=1e-6) + asii_logit_m) / (asii_logit_m*2),
                  logit3, #torch.sigmoid(logit3),
                ],              1)

      loss_per_var = (torch.sum(\
        loss_func2(logit0123, target) * mask        
        #loss_func3(logit0123, target) * mask            
        , axis=(0,2,3,4))        )  / (torch.sum(mask, axis=(0,2,3,4)) + EPS)

      loss_w       = loss_weight * loss_per_var
      loss         = loss_w[0] * 4.0

      sum_train_freq += 1
      sum_train_loss += loss.item()
      train_loss_1_sum_list += loss_w.cpu().detach().numpy()
      
      loss.backward()
      optimizer.step()
      global_step += 1

  print('\nsucess!')
  exit(1)      
