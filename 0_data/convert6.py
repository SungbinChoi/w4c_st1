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
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from typing import Any, List, Tuple


target_city = 'R6'



input_data_folder_path = target_city
out_dir                = target_city + 'n'
num_frame_per_day  = 96   
num_frame_before   =  4         
num_frame_out      = 32         
num_frame_sequence = 36  
height=256
width =256
num_channel=116                                 
num_channel_out=4  
NUM_INPUT_CHANNEL  = num_channel     * num_frame_before
NUM_OUTPUT_CHANNEL = num_channel_out * num_frame_out
SEED = int(time.time())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

day_dict   = {}        
all_days   = []
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
  print('all_days:',   len(all_days),   '\t', )


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



  
  
  
  
  num_channel_1 = 9    
  num_channel_2 = 16    
  
  discrete_data_info_out   = open(os.path.join(out_dir, 'discrete_data_info.txt'),   'w')
  
  data_min_range    = np.zeros((num_channel_1), np.uint16)
  data_min_range[:] = 65535
  data_max_range    = np.zeros((num_channel_1), np.uint16)
  
  data_sum_list  = np.zeros((num_channel_1), np.float32)
  data_freq_list = np.zeros((num_channel_1), np.float32)
  
  continuous_data_info_list = np.zeros((num_channel_1, 7), np.float32)     
  discrete_data_info_list   = np.zeros((num_channel_2, 1), np.float32)    
  
  continuous_data_info_done_list = np.zeros((num_channel_1, ), np.uint8)
  discrete_data_info_done_list   = np.zeros((num_channel_2, ), np.uint8)


  for d in all_days:        
    
    out_data_1 = np.zeros((num_frame_per_day, num_channel_1, height, width), np.uint16)
    out_data_2 = np.zeros((num_frame_per_day, num_channel_2, height, width), np.uint8)
    
    out_data_1[:,:,:,:] = 65535
    out_data_2[:,:,:,:] = 255
    
    
    for f in range(num_frame_per_day):
    
      assert d in day_dict, ('Not in day_dict:  (d f)  %d %d' % (d, f))
      (ymd, input_folder_path) = day_dict[d]
      
    
      hour    = '%02d' % ((15 * f) // 60)
      minute  = '%02d' % ((15 * f) %  60)
      second  = '00'
    

      c1 = 0
      c2 = 0

      # CTTH              9
      if 1:
        dtype = np.dtype(np.uint16)
        filepath = os.path.join(input_folder_path, str(d), 'CTTH', 'S_NWC_CTTH_MSG4_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')
        if os.path.exists(filepath) == False:
          filepath = os.path.join(input_folder_path, str(d), 'CTTH', 'S_NWC_CTTH_MSG2_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')

        
        if os.path.exists(filepath) == False:
          c_type = 0
          c1    += 6
          c2    += 3

        else:
        
          ds = Dataset(filepath, 'r')

          key_list = ['temperature',
                      'ctth_tempe',
                      'ctth_pres',
                      'ctth_alti',
                      'ctth_effectiv',
                      'ctth_method',
                      'ctth_quality',
                      'ishai_skt',
                      'ishai_quality',
                      ]
          assert len(key_list) == 9
          
          for key in key_list:
            
            
            d_arr = np.array(ds[key]).astype(dtype)
            
            
            has_flag = False
            flags    = None
            try:
              flags = np.array(ds[key].flag_values, dtype) 
              has_flag = True
            except:
              pass
              

            
            
            if has_flag:
              
              for flag_i, flag in enumerate(flags):
                out_data_2[f, c2, d_arr == flag] = flag_i
                
                
                if discrete_data_info_done_list[c2] == 0:
                  discrete_data_info_list[c2,0]    = len(flags)
                  discrete_data_info_done_list[c2] = 1

              c2 += 1
              
              
            else:
              
              
              
              has_scale  = False
              has_offset = False
              scale_factor = 1.0
              offset       = 0.0
              #
              try:
                scale_factor = ds[key].scale_factor
                has_scale    = True
              except:
                pass
              try:
                offset       = ds[key].add_offset
                has_offset   = True
              except:
                pass
              assert scale_factor != 0
              
              
              
              has_valid_range   = False
              valid_min         = 0
              valid_max         = 0
              try:
                valid_min = ds[key].valid_range[0]
                valid_max = ds[key].valid_range[1]
                has_valid_range = True
              except:
                pass
              
              
              
              if continuous_data_info_done_list[c1] == 0:
                  continuous_data_info_list[c1,0]    = 1 if has_scale else 0
                  continuous_data_info_list[c1,1]    = scale_factor
                  continuous_data_info_list[c1,2]    = 1 if has_offset else 0
                  continuous_data_info_list[c1,3]    = offset
                  continuous_data_info_list[c1,4]    = 1 if has_valid_range else 0
                  continuous_data_info_list[c1,5]    = valid_min
                  continuous_data_info_list[c1,6]    = valid_max
                  
                  continuous_data_info_done_list[c1] = 1
                  
                  
                  
              
              
              
              
              mask_arr = np.ones((height, width), np.uint8)
              mask_arr[d_arr == ds[key]._FillValue] = 0                 
              
              d_arr_float = d_arr.astype(np.float32)
              if has_offset:
                d_arr_float -= offset
              if has_scale:
                d_arr_float /= scale_factor
              # 
              if has_valid_range:
                mask_arr[d_arr_float < valid_min] = 0                  
                mask_arr[d_arr_float > valid_max] = 0
              
              
              
                
              is_valid    = (mask_arr == 1)  
              is_invalid  = (mask_arr == 0)

              out_data_1[f, c1, :, :] = d_arr   
              out_data_1[f, c1, is_invalid]  = 65535   
              
              if np.any(is_valid):
                valid_data = d_arr[is_valid]
                data_min_range[c1] = min(np.min(valid_data), data_min_range[c1])
                data_max_range[c1] = max(np.max(valid_data), data_max_range[c1])
                data_sum_list[c1]  += np.sum(valid_data)
                data_freq_list[c1] += np.sum(is_valid)
              
              c1 += 1
  

      
      
      

        
      
      #CRR             4
      if 1:
        dtype = np.dtype(np.uint16)
        filepath = os.path.join(input_folder_path, str(d), 'CRR', 'S_NWC_CRR_MSG4_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')
        if os.path.exists(filepath) == False:
          filepath = os.path.join(input_folder_path, str(d), 'CRR', 'S_NWC_CRR_MSG2_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')
          
        
        if os.path.exists(filepath) == False:
          c_type = 1
          c1    += 2
          c2    += 2

        else:
        
          ds = Dataset(filepath, 'r')


          key_list = ['crr',
                      'crr_intensity',
                      'crr_accum',
                      'crr_quality',
                      ]
          assert len(key_list) == 4
          
          for key in key_list:
            
            
            d_arr = np.array(ds[key]).astype(dtype)  
            
            
            has_flag = False
            flags    = None
            try:
              flags = np.array(ds[key].flag_values, dtype)  
              has_flag = True
            except:
              pass
              

            
            
            if has_flag:
              
              for flag_i, flag in enumerate(flags):
                out_data_2[f, c2, d_arr == flag] = flag_i
                
                
                if discrete_data_info_done_list[c2] == 0:
                  discrete_data_info_list[c2,0]    = len(flags)           
                  discrete_data_info_done_list[c2] = 1

              c2 += 1
              
              
            else:
              
              
              
              has_scale  = False
              has_offset = False
              scale_factor = 1.0
              offset       = 0.0
              #
              try:
                scale_factor = ds[key].scale_factor
                has_scale    = True
              except:
                pass
              try:
                offset       = ds[key].add_offset
                has_offset   = True
              except:
                pass
              assert scale_factor != 0
              
              
              
              has_valid_range   = False
              valid_min         = 0
              valid_max         = 0
              try:
                valid_min = ds[key].valid_range[0]
                valid_max = ds[key].valid_range[1]
                has_valid_range = True
              except:
                pass
              
              
              
              if continuous_data_info_done_list[c1] == 0:
                  continuous_data_info_list[c1,0]    = 1 if has_scale else 0
                  continuous_data_info_list[c1,1]    = scale_factor
                  continuous_data_info_list[c1,2]    = 1 if has_offset else 0
                  continuous_data_info_list[c1,3]    = offset
                  continuous_data_info_list[c1,4]    = 1 if has_valid_range else 0
                  continuous_data_info_list[c1,5]    = valid_min
                  continuous_data_info_list[c1,6]    = valid_max
                  
                  continuous_data_info_done_list[c1] = 1
                  
                  
                  
              
              
              
              
              mask_arr = np.ones((height, width), np.uint8)
              mask_arr[d_arr == ds[key]._FillValue] = 0                 
              
              d_arr_float = d_arr.astype(np.float32)
              if has_offset:
                d_arr_float -= offset
              if has_scale:
                d_arr_float /= scale_factor
              # 
              if has_valid_range:
                mask_arr[d_arr_float < valid_min] = 0                   
                mask_arr[d_arr_float > valid_max] = 0
              
              
              
                
              is_valid    = (mask_arr == 1)  
              is_invalid  = (mask_arr == 0)

              out_data_1[f, c1, :, :] = d_arr   
              out_data_1[f, c1, is_invalid]  = 65535   
              
              if np.any(is_valid):
                valid_data = d_arr[is_valid]
                data_min_range[c1] = min(np.min(valid_data), data_min_range[c1])
                data_max_range[c1] = max(np.max(valid_data), data_max_range[c1])
                data_sum_list[c1]  += np.sum(valid_data)
                data_freq_list[c1] += np.sum(is_valid)
              
              c1 += 1
  
  

      
      
      #ASII            2
      if 1:
        dtype = np.dtype(np.uint8)
        filepath = os.path.join(input_folder_path, str(d), 'ASII', 'S_NWC_ASII-TF_MSG4_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')
        if os.path.exists(filepath) == False:
          filepath = os.path.join(input_folder_path, str(d), 'ASII', 'S_NWC_ASII-TF_MSG2_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')

        
        if os.path.exists(filepath) == False:
          c_type = 2
          c1    += 1
          c2    += 1

        else:
        
          ds = Dataset(filepath, 'r')


          key_list = ['asii_turb_trop_prob',
                      'asiitf_quality',
                      ]
          assert len(key_list) == 2
          
          for key in key_list:
            
            
            d_arr = np.array(ds[key]).astype(dtype) 
            
            
            has_flag = False
            flags    = None
            try:
              flags = np.array(ds[key].flag_values, dtype)  
              has_flag = True
            except:
              pass
              

            
            
            if has_flag:
              
              for flag_i, flag in enumerate(flags):
                out_data_2[f, c2, d_arr == flag] = flag_i
                
                
                if discrete_data_info_done_list[c2] == 0:
                  discrete_data_info_list[c2,0]    = len(flags)           
                  discrete_data_info_done_list[c2] = 1

              c2 += 1
              
              
            else:
              
              
              
              has_scale  = False
              has_offset = False
              scale_factor = 1.0
              offset       = 0.0
              #
              try:
                scale_factor = ds[key].scale_factor
                has_scale    = True
              except:
                pass
              try:
                offset       = ds[key].add_offset
                has_offset   = True
              except:
                pass
              assert scale_factor != 0
              
              
              
              has_valid_range   = False
              valid_min         = 0
              valid_max         = 0
              try:
                valid_min = ds[key].valid_range[0]
                valid_max = ds[key].valid_range[1]
                has_valid_range = True
              except:
                pass
              
              
              
              if continuous_data_info_done_list[c1] == 0:
                  continuous_data_info_list[c1,0]    = 1 if has_scale else 0
                  continuous_data_info_list[c1,1]    = scale_factor
                  continuous_data_info_list[c1,2]    = 1 if has_offset else 0
                  continuous_data_info_list[c1,3]    = offset
                  continuous_data_info_list[c1,4]    = 1 if has_valid_range else 0
                  continuous_data_info_list[c1,5]    = valid_min
                  continuous_data_info_list[c1,6]    = valid_max
                  
                  continuous_data_info_done_list[c1] = 1
                  
                  
                  
              
              
              
              
              mask_arr = np.ones((height, width), np.uint8)
              mask_arr[d_arr == ds[key]._FillValue] = 0               
              
              d_arr_float = d_arr.astype(np.float32)
              if has_offset:
                d_arr_float -= offset
              if has_scale:
                d_arr_float /= scale_factor
              # 
              if has_valid_range:
                mask_arr[d_arr_float < valid_min] = 0                 
                mask_arr[d_arr_float > valid_max] = 0
              
              
              
                
              is_valid    = (mask_arr == 1)  
              is_invalid  = (mask_arr == 0)

              out_data_1[f, c1, :, :] = d_arr   
              out_data_1[f, c1, is_invalid]  = 65535  
              
              if np.any(is_valid):
                valid_data = d_arr[is_valid]
                data_min_range[c1] = min(np.min(valid_data), data_min_range[c1])
                data_max_range[c1] = max(np.max(valid_data), data_max_range[c1])
                data_sum_list[c1]  += np.sum(valid_data)
                data_freq_list[c1] += np.sum(is_valid)
              
              c1 += 1
  
      
      
      
      
          
      #CMA             6
      if 1:
        dtype = np.dtype(np.uint8)
        filepath = os.path.join(input_folder_path, str(d), 'CMA', 'S_NWC_CMA_MSG4_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')
        if os.path.exists(filepath) == False:
          filepath = os.path.join(input_folder_path, str(d), 'CMA', 'S_NWC_CMA_MSG2_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')

        
        
        if os.path.exists(filepath) == False:
          c_type = 3
          c1    += 0
          c2    += 6

        else:
        
          ds = Dataset(filepath, 'r')


          key_list = ['cma_cloudsnow',
                      'cma',
                      'cma_dust',
                      'cma_volcanic',
                      'cma_smoke',
                      'cma_quality',
                      ]
          assert len(key_list) == 6
          
          for key in key_list:
            
            
            d_arr = np.array(ds[key]).astype(dtype) 
            
            
            has_flag = False
            flags    = None
            try:
              flags = np.array(ds[key].flag_values, dtype)  # []
              has_flag = True
            except:
              pass
              

            
            
            if has_flag:
              
              for flag_i, flag in enumerate(flags):
                out_data_2[f, c2, d_arr == flag] = flag_i
                
                
                if discrete_data_info_done_list[c2] == 0:
                  discrete_data_info_list[c2,0]    = len(flags)           
                  discrete_data_info_done_list[c2] = 1

              c2 += 1
              
              
            else:
              
              
              
              has_scale  = False
              has_offset = False
              scale_factor = 1.0
              offset       = 0.0
              #
              try:
                scale_factor = ds[key].scale_factor
                has_scale    = True
              except:
                pass
              try:
                offset       = ds[key].add_offset
                has_offset   = True
              except:
                pass
              assert scale_factor != 0
              
              
              
              has_valid_range   = False
              valid_min         = 0
              valid_max         = 0
              try:
                valid_min = ds[key].valid_range[0]
                valid_max = ds[key].valid_range[1]
                has_valid_range = True
              except:
                pass
              
              
              
              if continuous_data_info_done_list[c1] == 0:
                  continuous_data_info_list[c1,0]    = 1 if has_scale else 0
                  continuous_data_info_list[c1,1]    = scale_factor
                  continuous_data_info_list[c1,2]    = 1 if has_offset else 0
                  continuous_data_info_list[c1,3]    = offset
                  continuous_data_info_list[c1,4]    = 1 if has_valid_range else 0
                  continuous_data_info_list[c1,5]    = valid_min
                  continuous_data_info_list[c1,6]    = valid_max
                  
                  continuous_data_info_done_list[c1] = 1
                  
                  
                  
              
              
              
              
              mask_arr = np.ones((height, width), np.uint8)
              mask_arr[d_arr == ds[key]._FillValue] = 0               
              
              d_arr_float = d_arr.astype(np.float32)
              if has_offset:
                d_arr_float -= offset
              if has_scale:
                d_arr_float /= scale_factor
              # 
              if has_valid_range:
                mask_arr[d_arr_float < valid_min] = 0               
                mask_arr[d_arr_float > valid_max] = 0
              
              
              
                
              is_valid    = (mask_arr == 1)  
              is_invalid  = (mask_arr == 0)

              out_data_1[f, c1, :, :] = d_arr   
              out_data_1[f, c1, is_invalid]  = 65535   
              
              if np.any(is_valid):
                valid_data = d_arr[is_valid]
                data_min_range[c1] = min(np.min(valid_data), data_min_range[c1])
                data_max_range[c1] = max(np.max(valid_data), data_max_range[c1])
                data_sum_list[c1]  += np.sum(valid_data)
                data_freq_list[c1] += np.sum(is_valid)
              
              c1 += 1
  
      

      
      #CT              4
      if 1:
        dtype = np.dtype(np.uint8)
        filepath = os.path.join(input_folder_path, str(d), 'CT', 'S_NWC_CT_MSG4_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')
        if os.path.exists(filepath) == False:
          filepath = os.path.join(input_folder_path, str(d), 'CT', 'S_NWC_CT_MSG2_Europe-VISIR_' + ymd + 'T' + hour + minute + second + 'Z.nc')
        
        
        if os.path.exists(filepath) == False:
          c_type = 4
          c1    += 0
          c2    += 4

        else:
        
          ds = Dataset(filepath, 'r')


          key_list = ['ct',
                      'ct_cumuliform',
                      'ct_multilayer',
                      'ct_quality',
                      ]
          assert len(key_list) == 4
          
          for key in key_list:
            
            
            d_arr = np.array(ds[key]).astype(dtype)   
            
            
            has_flag = False
            flags    = None
            try:
              flags = np.array(ds[key].flag_values, dtype)  # []
              has_flag = True
            except:
              pass
              

            
            
            if has_flag:
              
              for flag_i, flag in enumerate(flags):
                out_data_2[f, c2, d_arr == flag] = flag_i
                
                
                if discrete_data_info_done_list[c2] == 0:
                  discrete_data_info_list[c2,0]    = len(flags)          
                  discrete_data_info_done_list[c2] = 1

              c2 += 1
              
              
            else:
              
              
              
              has_scale  = False
              has_offset = False
              scale_factor = 1.0
              offset       = 0.0
              #
              try:
                scale_factor = ds[key].scale_factor
                has_scale    = True
              except:
                pass
              try:
                offset       = ds[key].add_offset
                has_offset   = True
              except:
                pass
              assert scale_factor != 0
              
              
              
              has_valid_range   = False
              valid_min         = 0
              valid_max         = 0
              try:
                valid_min = ds[key].valid_range[0]
                valid_max = ds[key].valid_range[1]
                has_valid_range = True
              except:
                pass
              
              
              
              if continuous_data_info_done_list[c1] == 0:
                  continuous_data_info_list[c1,0]    = 1 if has_scale else 0
                  continuous_data_info_list[c1,1]    = scale_factor
                  continuous_data_info_list[c1,2]    = 1 if has_offset else 0
                  continuous_data_info_list[c1,3]    = offset
                  continuous_data_info_list[c1,4]    = 1 if has_valid_range else 0
                  continuous_data_info_list[c1,5]    = valid_min
                  continuous_data_info_list[c1,6]    = valid_max
                  
                  continuous_data_info_done_list[c1] = 1
                  
                  
                  
              
              
              
              
              mask_arr = np.ones((height, width), np.uint8)
              mask_arr[d_arr == ds[key]._FillValue] = 0                 
              
              d_arr_float = d_arr.astype(np.float32)
              if has_offset:
                d_arr_float -= offset
              if has_scale:
                d_arr_float /= scale_factor
              # 
              if has_valid_range:
                mask_arr[d_arr_float < valid_min] = 0                  
                mask_arr[d_arr_float > valid_max] = 0
              
              is_valid    = (mask_arr == 1)  
              is_invalid  = (mask_arr == 0)

              out_data_1[f, c1, :, :] = d_arr   
              out_data_1[f, c1, is_invalid]  = 65535 
              
              if np.any(is_valid):
                valid_data = d_arr[is_valid]
                data_min_range[c1] = min(np.min(valid_data), data_min_range[c1])
                data_max_range[c1] = max(np.max(valid_data), data_max_range[c1])
                data_sum_list[c1]  += np.sum(valid_data)
                data_freq_list[c1] += np.sum(is_valid)
              
              c1 += 1

      assert c1 ==num_channel_1
      assert c2 ==num_channel_2
      

    
    if 1:
      for f in range(num_frame_per_day):
        np.savez_compressed(os.path.join(out_dir, str(d) + '_' + str(f)), data_1=out_data_1[f,:,:,:], data_2=out_data_2[f,:,:,:])
      chunk_size = 16
      assert num_frame_per_day % chunk_size == 0
      
      out_label = \
        np.concatenate(
        [out_data_1[:,0, :,:][:,np.newaxis,:,:],         
         out_data_1[:,6, :,:][:,np.newaxis,:,:],         
         out_data_1[:,8, :,:][:,np.newaxis,:,:],         
         out_data_2[:,7, :,:][:,np.newaxis,:,:],         
        ], axis=1).astype(np.uint16)
      if 1: 
        cma_label = out_label[:,3,:,:]
        cma_label[cma_label==255] = 65535
        out_label[:,3,:,:] = cma_label
      for fg in range(0, num_frame_per_day//chunk_size):
        np.savez_compressed(os.path.join(out_dir, 'l_' + str(d) + '_' + str(fg)), label=out_label[(fg*chunk_size):((fg+1)*chunk_size),:,:,:])

  for c in range(num_channel_2):  
    discrete_data_info_out.write('%d'  % (int(discrete_data_info_list[c,0])))
    discrete_data_info_out.write('\n')
  discrete_data_info_out.close()

  print('\nsucess!')
  exit(1)      
