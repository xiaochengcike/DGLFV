# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:47:13 2018

@author: Ashiki
"""

import os
import numpy as np
from skimage.filters import threshold_niblack
from skimage import io
from skimage.filters import roberts
from skimage.color import rgb2gray
import shutil
import math

#ori_path = '/home/user/test/Ashiki_Mask_RCNN/ROI_result'
#data_path = 'SDU_ROI_data_v'

#ori_path = 'SDU_data/SDU_ROI_rotate'
#data_path = 'SDU_data/SDU_ROI_train'

#ori_path = '/home/user/test/Ashiki_RCAE_Net/enhance_SDU'
#data_path = 'SDU_enhancement/SDU_ROI_train_enhancement'



if not os.path.exists(data_path):
    os.mkdir(data_path)

train_path = os.path.join(data_path,'train')
test_path = os.path.join(data_path,'test')
notrain_path = os.path.join(data_path,'notrain')

if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(notrain_path)):
    os.mkdir(train_path)
    os.mkdir(test_path)
    os.mkdir(notrain_path)

top_path = os.listdir(ori_path)
top_path.sort()

for i in range(len(top_path)):
    temp = top_path[i].split('_')
    if i < 536*6 :     # 400 train label + 136 train no label     
        doc = math.floor(i/6)
        if i < 400*6:  # 400 train label
            if int(temp[3].split('.')[0]) <= 4:
#                file_name = temp[0]+'_'+temp[1]+'_'+temp[2]
                file_name = str(doc).zfill(3)
                temp_name = os.path.join(train_path,file_name)
                if not os.path.exists(temp_name):
                    os.mkdir(temp_name)
                shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
            else : 
#                file_name = temp[0]+'_'+temp[1]+'_'+temp[2]
                file_name = str(doc).zfill(3)
                temp_name = os.path.join(test_path,file_name)
                if not os.path.exists(temp_name):
                    os.mkdir(temp_name)
                shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
        else :         # 136 train no label
            if int(temp[3].split('.')[0]) <= 4:
                file_name = '400'
                temp_name = os.path.join(train_path,file_name)
                if not os.path.exists(temp_name):
                    os.mkdir(temp_name)
                shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
            else : 
                file_name = '400'
                temp_name = os.path.join(test_path,file_name)
                if not os.path.exists(temp_name):
                    os.mkdir(temp_name)
                shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
    else :
        file_name = '400'
        temp_name = os.path.join(notrain_path,file_name)
        if not os.path.exists(temp_name):
            os.mkdir(temp_name)
        shutil.copy(os.path.join(ori_path,top_path[i]),temp_name)
        
